"""
nnU-Net v2 Stage 1: Plan and Preprocess

Copies dataset into nnUNet_raw, then runs nnUNetv2_plan_and_preprocess CLI.
Expects data already in nnU-Net format (imagesTr/, labelsTr/, dataset.json).
Copies preprocessed output + metadata to SM_MODEL_DIR for downstream stages.

Recommended instance: ml.c5.4xlarge (CPU-only, no GPU needed)
"""
import os

os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")

import shutil
import argparse
import subprocess

DATASET_ID = 1
DATASET_NAME = "MedicalSegmentation"
CONFIGURATION = "3d_fullres"


def parse_args():
    parser = argparse.ArgumentParser(description="nnU-Net v2 Stage 1: Plan and Preprocess")
    parser.add_argument("--data", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
                        help="Input data directory (nnU-Net format)")
    parser.add_argument("--out_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./output"),
                        help="Output directory (SM_MODEL_DIR)")
    parser.add_argument("--n_proc", type=int, default=8,
                        help="Number of processes for preprocessing")
    return parser.parse_args()


def run(args):
    raw_dir = os.environ["nnUNet_raw"]
    preprocessed_dir = os.environ["nnUNet_preprocessed"]

    dataset_dir = os.path.join(raw_dir, f"Dataset{DATASET_ID:03d}_{DATASET_NAME}")
    shutil.copytree(args.data, dataset_dir, dirs_exist_ok=True)

    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(DATASET_ID),
        "-c", CONFIGURATION,
        "--verify_dataset_integrity",
        "-np", str(args.n_proc),
    ], check=True)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(preprocessed_dir):
        shutil.copytree(preprocessed_dir, os.path.join(out_dir, "nnUNet_preprocessed"), dirs_exist_ok=True)
    if os.path.isdir(raw_dir):
        shutil.copytree(raw_dir, os.path.join(out_dir, "nnUNet_raw"), dirs_exist_ok=True)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
