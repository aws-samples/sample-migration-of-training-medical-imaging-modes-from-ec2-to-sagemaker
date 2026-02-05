"""
FSDP training with TensorBoard integration
"""
import os
import logging
import argparse
import functools
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
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
from model_def import ModelDefinition

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
    return parser.parse_args()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    return train_ds, train_loader, val_ds, val_loader, train_sampler


def train_epoch(model, train_loader, optimizer, rank, epoch, sampler, writer):
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
            writer.add_scalar("train/loss_step", loss.item(), epoch * len(train_loader) + step)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    avg_loss = ddp_loss[0] / ddp_loss[1]
    
    if rank == 0:
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate(model, val_loader, rank, epoch, writer):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # Store first batch for visualization
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
            
            # Save first batch for visualization
            if idx == 0 and rank == 0:
                first_batch_images = images
                first_batch_masks = masks
                first_batch_outputs = outputs_post
    
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    
    metric_tensor = torch.tensor(metric).to(rank)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
    
    if rank == 0:
        writer.add_scalar("val/dice", metric_tensor.item(), epoch)
        logger.info(f"Epoch {epoch} - Val Dice: {metric_tensor.item():.4f}")
        
        # Visualize predictions
        if first_batch_images is not None:
            plot_2d_or_3d_image(Transpose((0, 1, 4, 3, 2))(first_batch_images), 
                              epoch, writer, index=0, tag="val/image")
            plot_2d_or_3d_image(Transpose((0, 1, 4, 3, 2))(first_batch_masks), 
                              epoch, writer, index=0, tag="val/mask")
            plot_2d_or_3d_image(Transpose((0, 1, 4, 3, 2))(first_batch_outputs), 
                              epoch, writer, index=0, tag="val/prediction")
    
    return metric_tensor.item()


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Initialize TensorBoard only on rank 0
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=args.out_dir)
        logger.info(f"Training {args.model_name} with FSDP + TensorBoard")
        logger.info(f"TensorBoard logs: {args.out_dir}")
    
    train_ds, train_loader, val_ds, val_loader, train_sampler = create_data_loaders(
        args.data, args.batch_size, rank, world_size
    )
    
    model = ModelDefinition(args.model_name).get_model().to(rank)
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    if rank == 0:
        writer.add_text("config/model", args.model_name)
        writer.add_text("config/hyperparameters", 
                       f"batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    
    best_metric = -1
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, rank, epoch, train_sampler, writer)
        
        if epoch % 2 == 0:
            val_metric = validate(model, val_loader, rank, epoch, writer)
            if rank == 0 and val_metric > best_metric:
                best_metric = val_metric
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
                writer.add_scalar("val/best_dice", best_metric, epoch)
    
    if rank == 0:
        writer.close()
        logger.info(f"TensorBoard logs saved to: {args.out_dir}")
    
    cleanup()


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        logger.warning("FSDP requires at least 2 GPUs")
        return
    
    mp.spawn(fsdp_main, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
