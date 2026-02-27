"""
nnU-Net v2 Stage 3: Find Best Configuration / Evaluate

Restores trained model results from Stage 2, runs nnUNetv2_find_best_configuration CLI.
Copies results to SM_MODEL_DIR.

Can run on CPU instance (ml.c5.xlarge).
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
    parser = argparse.ArgumentParser(description="nnU-Net v2 Stage 3: Evaluate")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("SM_CHANNEL_MODEL", ""),
                        help="Model directory from Stage 2")
    parser.add_argument("--out_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./output"),
                        help="Output directory (SM_MODEL_DIR)")
    return parser.parse_args()


def _restore_model(model_dir):
    for subdir, target in [
        ("nnUNet_results", os.environ["nnUNet_results"]),
        ("nnUNet_preprocessed", os.environ["nnUNet_preprocessed"]),
        ("nnUNet_raw", os.environ["nnUNet_raw"]),
    ]:
        src = os.path.join(model_dir, subdir)
        if os.path.isdir(src):
            shutil.copytree(src, target, dirs_exist_ok=True)


def run(args):
    if hasattr(args, "model") and args.model and os.path.isdir(args.model):
        _restore_model(args.model)

    results_dir = os.environ["nnUNet_results"]

    subprocess.run([
        "nnUNetv2_find_best_configuration", str(DATASET_ID),
        "-c", CONFIGURATION,
        "-f", "all",
    ], check=True)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(results_dir):
        shutil.copytree(results_dir, os.path.join(out_dir, "nnUNet_results"), dirs_exist_ok=True)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
