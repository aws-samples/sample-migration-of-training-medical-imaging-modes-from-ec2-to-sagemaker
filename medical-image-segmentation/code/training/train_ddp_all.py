"""
DDP training with WandB + MLFlow + TensorBoard (All-in-One)
"""
import os
import logging
import argparse
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import monai
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandFlipd, RandShiftIntensityd,
    Activations, AsDiscrete, Transpose
)
from monai.visualize import plot_2d_or_3d_image

try:
    from model_def import ModelDefinition
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from model_def import ModelDefinition

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SwinUNETR")
    parser.add_argument("--data", type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument("--out_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './output'))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    # WandB
    parser.add_argument("--use_wandb", type=lambda x: x.lower() == 'true', default=False, help="Enable WandB tracking")
    parser.add_argument("--wandb_project", type=str, default="medical-segmentation-ddp")
    parser.add_argument("--wandb_api_key", type=str, default=os.environ.get('WANDB_API_KEY', ''))
    # MLFlow
    parser.add_argument("--use_mlflow", type=lambda x: x.lower() == 'true', default=False, help="Enable MLFlow tracking")
    parser.add_argument("--mlflow_tracking_uri", type=str, nargs='?', default="", const="")
    parser.add_argument("--mlflow_experiment_name", type=str, default="medical-segmentation-ddp")
    return parser.parse_args()


def setup(rank, world_size):
    # Use SageMaker's environment variables
    os.environ.setdefault('MASTER_ADDR', 'algo-1')
    os.environ.setdefault('MASTER_PORT', '7777')
    os.environ.setdefault('RANK', str(rank))
    os.environ.setdefault('WORLD_SIZE', str(world_size))
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'WARN'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))


def cleanup():
    dist.destroy_process_group()


def create_image_mask_pairs(image_mask_dir):
    subj_list = os.listdir(image_mask_dir)
    return [{"image": os.path.join(image_mask_dir, s, 'img.nii.gz'),
             "mask": os.path.join(image_mask_dir, s, 'label.nii.gz')} for s in subj_list]


def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Resized(keys=["image", "mask"], spatial_size=(128, 128, 64)),
        ScaleIntensityRanged(keys="image", a_min=-100, a_max=500, b_min=0, b_max=1, clip=True),
        RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5)
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(keys="image", a_min=-100, a_max=500, b_min=0, b_max=1, clip=True),
        Resized(keys=["image", "mask"], spatial_size=(128, 128, 64))
    ])
    return train_transforms, val_transforms


def create_data_loaders(data_path, batch_size, rank, world_size):
    train_transforms, val_transforms = get_transforms()
    train_files = create_image_mask_pairs(os.path.join(data_path, 'train'))
    val_files = create_image_mask_pairs(os.path.join(data_path, 'valid'))
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_ds, rank=rank, num_replicas=world_size, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
    
    return train_ds, train_loader, val_ds, val_loader, train_sampler


def log_metrics(metrics, step, use_wandb, use_mlflow, writer):
    """Unified logging to all enabled trackers"""
    for key, value in metrics.items():
        # TensorBoard (always enabled)
        writer.add_scalar(key, value, step)
        
        # WandB
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({key: value, "step": step})
        
        # MLFlow
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metric(key.replace("/", "_"), value, step=step)


def train_epoch(model, train_loader, optimizer, rank, epoch, sampler, use_wandb, use_mlflow, writer):
    model.train()
    sampler.set_epoch(epoch)
    ddp_loss = torch.zeros(2).to(rank)
    step = 0
    
    for batch in tqdm(train_loader, desc=f"Rank {rank} - Epoch {epoch}", disable=(rank != 0)):
        step += 1
        images = batch["image"].to(rank)
        masks = batch["mask"].to(rank)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = monai.losses.DiceLoss(sigmoid=True)(outputs, masks)
        loss.backward()
        optimizer.step()
        
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(images)
        
        if rank == 0:
            log_metrics({"train/loss_step": loss.item()}, 
                       epoch * len(train_loader) + step, use_wandb, use_mlflow, writer)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    avg_loss = ddp_loss[0] / ddp_loss[1]
    
    if rank == 0:
        log_metrics({"train/loss_epoch": avg_loss, "epoch": epoch}, 
                   epoch, use_wandb, use_mlflow, writer)
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate(model, val_loader, rank, epoch, use_wandb, use_mlflow, writer):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    first_batch_images = None
    first_batch_masks = None
    first_batch_outputs = None
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc=f"Rank {rank} - Validation", disable=(rank != 0))):
            images = batch["image"].to(rank)
            masks = batch["mask"].to(rank)
            outputs = model(images)
            outputs_post = post_trans(outputs)
            dice_metric(y_pred=outputs_post, y=masks)
            
            if idx == 0 and rank == 0:
                first_batch_images = images
                first_batch_masks = masks
                first_batch_outputs = outputs_post
    
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    
    metric_tensor = torch.tensor(metric).to(rank)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
    
    if rank == 0:
        log_metrics({"val/dice": metric_tensor.item()}, epoch, use_wandb, use_mlflow, writer)
        logger.info(f"Epoch {epoch} - Val Dice: {metric_tensor.item():.4f}")
        
        # TensorBoard visualizations
        if first_batch_images is not None:
            plot_2d_or_3d_image(Transpose((0, 1, 4, 3, 2))(first_batch_images), 
                              epoch, writer, index=0, tag="val/image")
            plot_2d_or_3d_image(Transpose((0, 1, 4, 3, 2))(first_batch_masks), 
                              epoch, writer, index=0, tag="val/mask")
            plot_2d_or_3d_image(Transpose((0, 1, 4, 3, 2))(first_batch_outputs), 
                              epoch, writer, index=0, tag="val/prediction")
    
    return metric_tensor.item()


def ddp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Initialize all trackers (rank 0 only)
    writer = None
    if rank == 0:
        # TensorBoard (always enabled)
        writer = SummaryWriter(log_dir=args.out_dir)
        logger.info(f"✓ TensorBoard enabled: {args.out_dir}")
        
        # WandB
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(project=args.wandb_project, config=vars(args))
            logger.info(f"✓ WandB enabled: {args.wandb_project}")
        elif args.use_wandb:
            logger.warning("✗ WandB requested but not available")
        
        # MLFlow
        if args.use_mlflow and MLFLOW_AVAILABLE:
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.mlflow_experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "world_size": world_size
            })
            logger.info(f"✓ MLFlow enabled: {args.mlflow_experiment_name}")
        elif args.use_mlflow:
            logger.warning("✗ MLFlow requested but not available")
        
        logger.info(f"Training {args.model_name} with DDP")
    
    train_ds, train_loader, val_ds, val_loader, train_sampler = create_data_loaders(
        args.data, args.batch_size, rank, world_size
    )
    
    model = ModelDefinition(args.model_name).get_model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    best_metric = -1
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, rank, epoch, train_sampler, 
                                args.use_wandb, args.use_mlflow, writer)
        
        if epoch % 2 == 0:
            val_metric = validate(model, val_loader, rank, epoch, args.use_wandb, args.use_mlflow, writer)
            
            if rank == 0 and val_metric > best_metric:
                best_metric = val_metric
                model_path = os.path.join(args.out_dir, "best_model.pth")
                torch.save(model.module.state_dict(), model_path)
                
                log_metrics({"val/best_dice": best_metric}, epoch, args.use_wandb, args.use_mlflow, writer)
                
                # MLFlow artifact logging
                if args.use_mlflow and MLFLOW_AVAILABLE:
                    mlflow.log_artifact(model_path)
    
    # Cleanup all trackers
    if rank == 0:
        writer.close()
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        if args.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()
    
    cleanup()


def main():
    args = parse_args()
    
    # SageMaker handles process spawning via mpirun
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', torch.cuda.device_count()))
    
    if world_size < 2:
        logger.warning("DDP requires at least 2 GPUs")
        return
    
    if rank == 0:
        logger.info(f"Starting DDP training with {world_size} GPUs")
        logger.info(f"Integrations: TensorBoard=Always, WandB={args.use_wandb}, MLFlow={args.use_mlflow}")
    
    ddp_main(rank, world_size, args)


if __name__ == '__main__':
    main()
