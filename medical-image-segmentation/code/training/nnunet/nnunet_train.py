"""
nnU-Net v2 Stage 2: Training

Restores preprocessed data from Stage 1, runs nnUNetv2_train CLI.
Copies results to SM_MODEL_DIR for downstream stages.

Recommended instance: ml.g5.xlarge (1 GPU, 24GB)
"""
import os

os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")

import shutil
import argparse
import subprocess

DATASET_ID = 1
CONFIGURATION = "3d_fullres"


def parse_args():
    parser = argparse.ArgumentParser(description="nnU-Net v2 Stage 2: Training")
    parser.add_argument("--preprocessed", type=str,
                        default=os.environ.get("SM_CHANNEL_PREPROCESSED", ""),
                        help="Preprocessed data from Stage 1")
    parser.add_argument("--out_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./output"),
                        help="Output directory (SM_MODEL_DIR)")
    return parser.parse_args()


def _restore_preprocessed(preprocessed_dir):
    for subdir, target in [
        ("nnUNet_raw", os.environ["nnUNet_raw"]),
        ("nnUNet_preprocessed", os.environ["nnUNet_preprocessed"]),
    ]:
        src = os.path.join(preprocessed_dir, subdir)
        if os.path.isdir(src):
            shutil.copytree(src, target, dirs_exist_ok=True)


def run(args):
    if args.preprocessed and os.path.isdir(args.preprocessed):
        _restore_preprocessed(args.preprocessed)

    subprocess.run([
        "nnUNetv2_train", str(DATASET_ID), CONFIGURATION, "all",
    ], check=True)

    results_dir = os.environ["nnUNet_results"]
    preprocessed_dir = os.environ["nnUNet_preprocessed"]
    raw_dir = os.environ["nnUNet_raw"]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(results_dir):
        shutil.copytree(results_dir, os.path.join(out_dir, "nnUNet_results"), dirs_exist_ok=True)
    if os.path.isdir(preprocessed_dir):
        shutil.copytree(preprocessed_dir, os.path.join(out_dir, "nnUNet_preprocessed"), dirs_exist_ok=True)
    if os.path.isdir(raw_dir):
        shutil.copytree(raw_dir, os.path.join(out_dir, "nnUNet_raw"), dirs_exist_ok=True)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
