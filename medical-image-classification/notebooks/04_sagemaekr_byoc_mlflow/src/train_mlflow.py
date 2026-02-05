"""
Script to perform hyperparameter optimization on a model
"""
from model_def import ModelDef
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import monai
from monai.transforms.adaptors import adaptor
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    Resize,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from torch.utils.tensorboard import SummaryWriter
import json
import argparse
import logging
import os
import sys
from PIL import ImageFile
from model_def import ModelDef
from utils import generate_list_labels, MedNISTDataset, create_data_loaders
import mlflow
import mlflow.pytorch

writer = SummaryWriter()
# Parse command line arguments from SageMaker SDK

#Mapping training and test data locations from S3 to traning container environment
# parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
# parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

#Mapping hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--model_name", type=str, default="DenseNet121")
parser.add_argument("--num_classes", type=int, default=8)
parser.add_argument("--val_interval", type=int, default=1)

args = parser.parse_args()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Arguments: %s", args)
val_interval =1

# Initialize MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment("medical-image-classification")
mlflow.start_run()

# Log parameters
mlflow.log_param("batch_size", args.batch_size)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("learning_rate", args.learning_rate)
mlflow.log_param("model_name", args.model_name)
mlflow.log_param("num_classes", args.num_classes)
mlflow.log_param("val_interval", args.val_interval) 
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info(os.listdir(os.environ["SM_CHANNEL_TRAIN"]))
logger.info(os.listdir(os.environ["SM_CHANNEL_TEST"]))
subj_dir = os.path.join(os.environ["SM_CHANNEL_TRAIN"], \
    os.listdir(os.environ["SM_CHANNEL_TRAIN"])[0])
logger.info(os.listdir(subj_dir))
train_loader, _ = create_data_loaders(os.environ["SM_CHANNEL_TRAIN"], args.batch_size)
test_loader, _ = create_data_loaders(os.environ["SM_CHANNEL_TEST"], args.batch_size)
num_classes = _
model = ModelDef(num_classes, args.model_name).get_model().to(device)


optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

def train(train_loader, model, loss_fn, optimizer):
    size = len(train_loader.dataset)
    model.train()
    running_loss = 0
    running_corrects = 0
    running_corrects = 0
    running_samples = 0
    
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if args.model_name == 'DenseNet121':
            outputs = model(inputs[:, :, :, :, 0])
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs[0], labels)
            _, preds = torch.max(outputs[0], 1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples += len(inputs)

        if running_samples % 100 == 0:
            current_loss = running_loss / running_samples
            current_acc = running_corrects / running_samples
            logger.info(f"Step {step}: Loss: {current_loss}, Accuracy: {current_acc}")
            logger.info("Images [{}/{}]: Loss: {:.4f}, Accuracy: {:.4f}".format(
                running_samples, len(train_loader.dataset), current_loss, current_acc))
            writer.add_scalar("Accuracy/train", current_acc, step)
            mlflow.log_metric("train_accuracy", current_acc, step=step)
        writer.add_scalar("Loss/train", loss.item(), step)
        mlflow.log_metric("train_loss", loss.item(), step=step)
    epoch_loss = running_loss / running_samples
    epoch_acc = running_corrects / running_samples
    logger.info(f"Training Loss: {epoch_loss}")
    logger.info(f"Training Accuracy: {epoch_acc}")
    writer.add_scalar("Loss/train", epoch_loss, step)
    writer.add_scalar("Accuracy/train", epoch_acc, step)
    mlflow.log_metric("epoch_train_loss", epoch_loss, step=epoch)
    mlflow.log_metric("epoch_train_accuracy", epoch_acc, step=epoch)
    logger.info(f"Epoch [{epoch + 1}/{args.epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc
    
    
def test(test_loader, model, criterion, device):
    running_loss = 0
    running_corrects = 0
    print("Validation")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if args.model_name == 'DenseNet121':
                outputs = model(inputs[:, :, :, :, 0])
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            else:
                outputs = model(inputs)
                loss = criterion(outputs[0], labels)
                
                _, preds = torch.max(outputs[0], 1)  # Update to use outputs[0]
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects / len(test_loader.dataset)

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    mlflow.log_metric("val_loss", total_loss, step=epoch)
    mlflow.log_metric("val_accuracy", total_acc, step=epoch)
    return total_loss, total_acc

for epoch in range(int(args.epochs)):
    logger.info(f"Epoch {epoch + 1}/{args.epochs}")
    train(train_loader, model, loss_fn, optimizer)
    if epoch % val_interval == 0:
        test(test_loader, model, loss_fn, device)
        
path = os.path.join(args.model_dir, "model.pth")
checkpoint  = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
}
torch.save(checkpoint, path)
logger.info(f"Model saved to {path}")

# Log model to MLflow
mlflow.pytorch.log_model(model, "model")
mlflow.log_artifact(path)

writer.close()
mlflow.end_run()
logger.info("Training complete.")