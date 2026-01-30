import os
import json
import random
from monai.data import decollate_batch, DataLoader
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
import torch
from torch.utils.data import Dataset

img_size = (256, 256, 1)
def generate_list_labels(data_folder):
    """
    Generate a pair of label, image list
    :TODO
    """
    list_labels = os.listdir(data_folder)
    num_classes = len(list_labels)
    count = 0
    #read label_dict from a json file
    fid = open("./label_dict.json", "r")
    label_dict = json.load(fid)
    fid.close()
        
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


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

def create_data_loaders(train_data_path, batch_size):
    train_transforms = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Resize(spatial_size=img_size),
        ScaleIntensity(),])
    train_image, train_labels, _ = generate_list_labels(train_data_path)
    train_ds = MedNISTDataset(train_image, train_labels, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=16)
    return train_loader, _

def create_data_loaders_archive(train_data_path, valid_data_path, batch_size):
    
    train_transforms = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Resize(spatial_size=img_size),
        ScaleIntensity(),])
    val_transforms = Compose([LoadImage(), EnsureChannelFirst(), \
                Resize(spatial_size=img_size),
                ScaleIntensity()])
    train_image, train_labels, _ = generate_list_labels(train_data_path)
    val_image, val_labels, _ = generate_list_labels(valid_data_path)
    test_image, test_labels, _ = generate_list_labels(test_data_path)

    train_ds = MedNISTDataset(train_image, train_labels, train_transforms)
    val_ds = MedNISTDataset(val_image, val_labels, val_transforms)
    test_ds = MedNISTDataset(test_image, test_labels, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=16)
    validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    return train_loader, test_loader, validation_loader, _
