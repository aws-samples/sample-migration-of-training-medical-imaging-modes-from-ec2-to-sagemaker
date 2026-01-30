"""
DDP (Distributed Data Parallel) training for medical image segmentation
"""
import os
import logging
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import monai
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandFlipd, RandShiftIntensityd,
    Activations, AsDiscrete
)
from model_def import ModelDefinition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SegResNet")
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


def train_epoch(model, train_loader, optimizer, rank, epoch, sampler):
    model.train()
    sampler.set_epoch(epoch)
    ddp_loss = torch.zeros(2).to(rank)
    
    for batch in tqdm(train_loader, desc=f"Rank {rank} - Epoch {epoch}", disable=(rank != 0)):
        images = batch["image"].to(rank)
        masks = batch["mask"].to(rank)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = monai.losses.DiceLoss(sigmoid=True)(outputs, masks)
        loss.backward()
        optimizer.step()
        
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(images)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    avg_loss = ddp_loss[0] / ddp_loss[1]
    
    if rank == 0:
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate(model, val_loader, rank, epoch):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Rank {rank} - Validation", disable=(rank != 0)):
            images = batch["image"].to(rank)
            masks = batch["mask"].to(rank)
            outputs = model(images)
            outputs = post_trans(outputs)
            dice_metric(y_pred=outputs, y=masks)
    
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    
    metric_tensor = torch.tensor(metric).to(rank)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
    
    if rank == 0:
        logger.info(f"Epoch {epoch} - Val Dice: {metric_tensor.item():.4f}")
    
    return metric_tensor.item()


def ddp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        logger.info(f"Training {args.model_name} with DDP")
        logger.info(f"World size: {world_size}")
    
    train_ds, train_loader, val_ds, val_loader, train_sampler = create_data_loaders(
        args.data, args.batch_size, rank, world_size
    )
    
    # Create model and wrap with DDP
    model = ModelDefinition(args.model_name).get_model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        logger.info("Model wrapped with DDP")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    best_metric = -1
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, rank, epoch, train_sampler)
        
        if epoch % 2 == 0:
            val_metric = validate(model, val_loader, rank, epoch)
            if rank == 0 and val_metric > best_metric:
                best_metric = val_metric
                torch.save(model.module.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
                logger.info(f"New best model saved! Dice: {best_metric:.4f}")
    
    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.out_dir, "final_model.pth"))
    
    cleanup()


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        logger.warning("DDP requires at least 2 GPUs")
        return
    
    logger.info(f"Starting DDP training with {world_size} GPUs")
    mp.spawn(ddp_main, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
