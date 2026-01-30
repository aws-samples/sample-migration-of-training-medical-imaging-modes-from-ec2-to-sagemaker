"""
Simple single-GPU training script for medical image segmentation
"""
import os
import sys
import logging
import argparse
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument("--model_name", type=str, default="SegResNet")
    parser.add_argument("--data", type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument("--out_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './output'))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def create_image_mask_pairs(image_mask_dir):
    """Create list of image-mask pairs from directory"""
    subj_list = os.listdir(image_mask_dir)
    data_list = []
    for subj in subj_list:
        data = {
            "image": os.path.join(image_mask_dir, subj, 'img.nii.gz'),
            "mask": os.path.join(image_mask_dir, subj, 'label.nii.gz')
        }
        data_list.append(data)
    return data_list


def get_transforms():
    """Define training and validation transforms"""
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


def create_data_loaders(data_path, batch_size):
    """Create training and validation data loaders"""
    train_transforms, val_transforms = get_transforms()
    
    train_files = create_image_mask_pairs(os.path.join(data_path, 'train'))
    val_files = create_image_mask_pairs(os.path.join(data_path, 'valid'))
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    
    return train_ds, train_loader, val_ds, val_loader


def train_epoch(model, train_loader, optimizer, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        step += 1
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = monai.losses.DiceLoss(sigmoid=True)(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        writer.add_scalar("train/loss", loss.item(), epoch * len(train_loader) + step)
    
    return epoch_loss / step


def validate(model, val_loader, device, epoch, writer):
    """Validate the model"""
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = model(images)
            outputs = post_trans(outputs)
            dice_metric(y_pred=outputs, y=masks)
    
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    
    writer.add_scalar("val/dice", metric, epoch)
    return metric


def main():
    args = parse_args()
    
    logger.info(f"Training {args.model_name} model")
    logger.info(f"Hyperparameters: LR={args.lr}, Batch Size={args.batch_size}, Epochs={args.epochs}")
    logger.info(f"Data path: {args.data}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_ds, train_loader, val_ds, val_loader = create_data_loaders(args.data, args.batch_size)
    logger.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    # Create model
    model = ModelDefinition(args.model_name).get_model()
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and writer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    writer = SummaryWriter(log_dir=args.out_dir)
    
    # Training loop
    best_metric = -1
    best_epoch = -1
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Validate
        if epoch % args.val_interval == 0:
            val_metric = validate(model, val_loader, device, epoch, writer)
            logger.info(f"Validation Dice: {val_metric:.4f}")
            
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
                logger.info(f"New best model saved! Dice: {best_metric:.4f}")
    
    logger.info(f"\\nTraining completed!")
    logger.info(f"Best Dice: {best_metric:.4f} at epoch {best_epoch}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pth"))
    writer.close()


if __name__ == '__main__':
    main()
