
import os
import json
import logging
import shutil
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().info("Starting preprocessing script")
"""
The dataset is distributed in the following way. The source directory contains all the images in individual class folders. 
We need to split the data into train, test and validation sets. In each of the train, test and validation sets, we need to have the same distribution of classes.

"""


def main():
    data_src = '/opt/ml/processing/input'
    data_dest = '/opt/ml/processing/output'
    
    list_classes = os.listdir(data_src)
    list_classes.sort()
    logging.info(f"Classes found: {list_classes}")
    for class_name in list_classes:
        os.makedirs(os.path.join(data_dest, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dest, 'test', class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dest, 'val', class_name), exist_ok=True)
        
    # Split the data into train, test and validation sets
    for class_name in list_classes:
        list_images = os.listdir(os.path.join(data_src, class_name))
        list_images.sort()
        train_images = list_images[:int(len(list_images)*0.7)]
        test_images = list_images[int(len(list_images)*0.7):int(len(list_images)*0.9)]
        val_images = list_images[int(len(list_images)*0.9):]
        for image in train_images:
            # copy the image to the train folder
            shutil.copy(os.path.join(data_src, class_name, image), os.path.join(data_dest, 'train', class_name, image))
        for image in test_images:
            shutil.copy(os.path.join(data_src, class_name, image), os.path.join(data_dest, 'test', class_name, image))
        for image in val_images:
            shutil.copy(os.path.join(data_src, class_name, image), os.path.join(data_dest, 'val', class_name, image))

if __name__ == "__main__":
    main()
