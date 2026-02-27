"""
nnU-Net v2 Stage 4: Inference / Prediction

Restores trained model, runs nnUNetv2_predict CLI,
copies predictions to SM_MODEL_DIR.

Recommended instance: ml.g5.xlarge (1 GPU)
"""
import os

os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")

import shutil
import argparse
import subprocess
from pathlib import Path

DATASET_ID = 1
CONFIGURATION = "3d_fullres"


def parse_args():
    parser = argparse.ArgumentParser(description="nnU-Net v2 Stage 4: Predict")
    parser.add_argument("--data", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
                        help="Input images for prediction (nnU-Net format)")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("SM_CHANNEL_MODEL", ""),
                        help="Model directory from Stage 2/3")
    parser.add_argument("--out_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./output"),
                        help="Output directory (SM_MODEL_DIR)")
    return parser.parse_args()


def _restore_model(model_dir):
    for subdir, target in [
        ("nnUNet_results", os.environ["nnUNet_results"]),
        ("nnUNet_raw", os.environ["nnUNet_raw"]),
        ("nnUNet_preprocessed", os.environ["nnUNet_preprocessed"]),
    ]:
        src = os.path.join(model_dir, subdir)
        if os.path.isdir(src):
            shutil.copytree(src, target, dirs_exist_ok=True)


def run(args):
    if hasattr(args, "model") and args.model and os.path.isdir(args.model):
        _restore_model(args.model)

    predict_output = "/tmp/nnunet_predictions"
    os.makedirs(predict_output, exist_ok=True)

    subprocess.run([
        "nnUNetv2_predict",
        "-i", args.data,
        "-o", predict_output,
        "-d", str(DATASET_ID),
        "-c", CONFIGURATION,
        "-f", "all",
    ], check=True)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    dst_predictions = os.path.join(out_dir, "predictions")
    if os.path.isdir(predict_output):
        shutil.copytree(predict_output, dst_predictions, dirs_exist_ok=True)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
