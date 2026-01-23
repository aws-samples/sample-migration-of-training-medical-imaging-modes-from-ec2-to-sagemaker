"""
Script to perform hyperparameter optimization on a model
"""
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

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
from monai.networks.nets import DenseNet121

#Generate a log directory path based on timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Create a directory for logs
log_dir = os.path.join(os.getcwd(), 'logs', 'log_' + timestamp)

# Create the log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

os.environ['SM_CHANNEL_TRAINING'] = '/home/ubuntu/data/vindr-spinexr-subset' 
os.environ['SM_MODEL_DIR']= '/home/ubuntu/data/spine-model'
os.environ['SM_OUTPUT_DATA_DIR'] = '/home/ubuntu/data/spine-output'

img_size = (256, 256, 1)


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def generate_list_labels(data_folder):
    """
    Generate a pair of label, image list
    :TODO
    """
    list_labels = os.listdir(data_folder)
    num_classes = len(list_labels)
    count = 0
    # label_dict = {}
    # for _label in list_labels:
    #     label_dict[_label] = count
    #     count += 1
    #read label_dict from a json file
    with open("./label_dict.json", "r", encoding='utf-8') as fid:
        label_dict = json.load(fid)
        
    label_list = []
    image_list = []
    for _label in label_dict:
        label_data = os.path.join(data_folder, _label)
        file_list = os.listdir(label_data)
        for _file in file_list:
            image_list.append(os.path.join(label_data, _file))
            label_list.append(label_dict[_label])

    c = list(zip(image_list, label_list))

    random.shuffle(c)
    a, b = zip(*c)
    return a, b, num_classes


def test(model, test_loader, criterion, device):
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
    return total_loss, total_acc


def train(model,
          train_loader,
          validation_loader,
          criterion, optimizer, device, args): 
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.model_name, 'logs'))
    logger.info("Training the model")
    logger.info("Using device: {}".format(device))
    logger.info("Using MONAI version: {}".format(monai.__version__))

    print("MONAI Version ", monai.__version__)
    for epoch in range(0, args.num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0
        print("Epoch {}/{}".format(epoch, args.num_epochs))

        for step, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if args.model_name == 'DenseNet121':
                outputs = model(inputs[:, :, :, :, 0])
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            else:
                outputs = model(inputs)
                loss = criterion(outputs[0], labels)
                _, preds = torch.max(outputs[0], 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_samples += len(inputs)
            
            if running_samples % 4000 == 0:
                accuracy = running_corrects / running_samples
                print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(image_dataset["train"].dataset),
                        100.0 * (running_samples / len(image_dataset["train"].dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0 * accuracy,
                    )
                )
                writer.add_scalar('training accuracy', accuracy, epoch * len(train_loader) + step)

            # NOTE: Comment lines below to train and test on whole dataset
            # if running_samples > (0.2 * len(image_dataset["phase"].dataset)):
            #     break
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + step)
            
        epoch_loss = running_loss / running_samples
        epoch_acc = running_corrects / running_samples

        if epoch % args.val_interval == 0:
            model.eval()
            validation_loss, validation_accuracy = test(model, validation_loader, criterion, device)
            if validation_loss < best_loss:
                best_loss = validation_loss
                # Save checkpoint
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name, 'best_model_' + str(epoch) + '.pth'))
            else:
                loss_counter += 1
            logger.info('Training loss: {:.4f}, Epoch: {}, acc: {:.4f}, best loss: {:.4f}, validation_loss: {:.4f}'.format(
                                                                                    epoch_loss,
                                                                                    epoch,
                                                                                    epoch_acc,
                                                                                    best_loss,
                                                                                    validation_loss))
        
            writer.add_scalar('validation loss', validation_loss, epoch)
            writer.add_scalar('validation accuracy', validation_accuracy, epoch)
        if loss_counter == args.early_stopping_rounds:
            logger.info('Early stopping')
            break
    writer.close()
    return model

from utils import DropChannel
def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transforms = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Resize(spatial_size=img_size),
        ScaleIntensity(),])
    val_transforms = Compose([LoadImage(), EnsureChannelFirst(), \
                Resize(spatial_size=img_size),
                ScaleIntensity()])
    train_image, train_labels, _ = generate_list_labels(train_data_path)
    val_image, val_labels, _ = generate_list_labels(validation_data_path)
    test_image, test_labels, _ = generate_list_labels(test_data_path)

    train_ds = MedNISTDataset(train_image, train_labels, train_transforms)
    val_ds = MedNISTDataset(val_image, val_labels, val_transforms)
    test_ds = MedNISTDataset(test_image, test_labels, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=16)
    validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    return train_loader, test_loader, validation_loader, _


def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}, Early Stopping: {args.early_stopping_rounds}')
    logger.info(f'Data Paths: {args.data}')
    
    os.makedirs(os.path.join(args.model_dir, args.model_name), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader, validation_loader, num_classes = create_data_loaders(args.data,
                                                                       args.batch_size)
    model = ModelDef(num_classes, args.model_name).get_model()
    logger.info(f'Model: {args.model_name}')
    logger.info(f'Number of classes: {num_classes}')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model = model.to(device)

    logger.info("Training the model")
    model = train(model,
                  train_loader,
                  validation_loader,
                  criterion,
                  optimizer,
                  device,
                  args)

    logger.info("Testing the model")
    #test(model, test_loader, criterion, device)

    logger.info("Saving the model")
    out_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--batch-size',
                        type=int,
                        default=32)
    parser.add_argument('--early-stopping-rounds',
                        type=int,
                        default=10)
    parser.add_argument('--data', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_name', 
                        type=str,
                        default='DenseNet121')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3)
    parser.add_argument('--val_interval',
                        type=int,
                        default=10)

    args = parser.parse_args()
    print(args)

    main(args)