
import os
import logging
import shutil
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().info("Starting preprocessing script")


def main():
    data_src = '/opt/ml/processing/input'
    data_dest = '/opt/ml/processing/output'

    # Only include dirs that contain image files (skip 'code', 'mednist-processed', etc.)
    list_classes = [
        d for d in sorted(os.listdir(data_src))
        if os.path.isdir(os.path.join(data_src, d))
        and any(os.path.isfile(os.path.join(data_src, d, f)) for f in os.listdir(os.path.join(data_src, d)))
    ]
    logging.info(f"Classes found: {list_classes}")

    for class_name in list_classes:
        os.makedirs(os.path.join(data_dest, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dest, 'test', class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dest, 'val', class_name), exist_ok=True)

    for class_name in list_classes:
        list_images = sorted([
            f for f in os.listdir(os.path.join(data_src, class_name))
            if os.path.isfile(os.path.join(data_src, class_name, f))
        ])
        train_images = list_images[:int(len(list_images)*0.7)]
        test_images = list_images[int(len(list_images)*0.7):int(len(list_images)*0.9)]
        val_images = list_images[int(len(list_images)*0.9):]
        for image in train_images:
            shutil.copy(os.path.join(data_src, class_name, image), os.path.join(data_dest, 'train', class_name, image))
        for image in test_images:
            shutil.copy(os.path.join(data_src, class_name, image), os.path.join(data_dest, 'test', class_name, image))
        for image in val_images:
            shutil.copy(os.path.join(data_src, class_name, image), os.path.join(data_dest, 'val', class_name, image))

if __name__ == "__main__":
    main()
