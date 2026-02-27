"""
nnU-Net v2 Combined Pipeline Wrapper

Runs any subset of stages (preprocess, train, evaluate, predict) in canonical order.
Each stage's run(args) function is called directly, sharing /tmp/ state.

Usage:
    python nnunet_pipeline.py --stages preprocess,train,evaluate \\
        --data /path/to/data --out_dir /path/to/output
"""
import os

os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")

import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGE_ORDER = ["preprocess", "train", "evaluate", "predict"]


def build_parser():
    """Build combined argument parser with args from all stages."""
    parser = argparse.ArgumentParser(
        description="nnU-Net v2 Pipeline: run multiple stages sequentially"
    )

    parser.add_argument("--stages", type=str, default="preprocess,train,evaluate",
                        help="Comma-separated stages to run (preprocess,train,evaluate,predict)")
    parser.add_argument("--data", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
                        help="Input data directory (nnU-Net format)")
    parser.add_argument("--out_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./output"),
                        help="Output directory (SM_MODEL_DIR)")
    parser.add_argument("--n_proc", type=int, default=8,
                        help="Number of processes for preprocessing")
    parser.add_argument("--preprocessed", type=str,
                        default=os.environ.get("SM_CHANNEL_PREPROCESSED", ""),
                        help="Preprocessed data from Stage 1 (optional)")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("SM_CHANNEL_MODEL", ""),
                        help="Model directory (for standalone evaluate/predict)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    requested = [s.strip().lower() for s in args.stages.split(",")]
    stages = [s for s in STAGE_ORDER if s in requested]

    invalid = set(requested) - set(STAGE_ORDER)
    if invalid:
        print(f"ERROR: Unknown stages: {invalid}. Valid stages: {STAGE_ORDER}")
        sys.exit(1)

    print("=" * 60)
    print(f"nnU-Net Pipeline: stages={stages}")
    print("=" * 60)

    for stage_name in stages:
        print(f"\n{'#' * 60}")
        print(f"# Starting stage: {stage_name}")
        print(f"{'#' * 60}\n")

        if stage_name == "preprocess":
            from nnunet_preprocess import run as preprocess_run
            preprocess_run(args)

        elif stage_name == "train":
            from nnunet_train import run as train_run
            train_run(args)

        elif stage_name == "evaluate":
            from nnunet_evaluate import run as evaluate_run
            evaluate_run(args)

        elif stage_name == "predict":
            from nnunet_predict import run as predict_run
            predict_run(args)

        print(f"Stage '{stage_name}' completed\n")

    print("=" * 60)
    print("Pipeline completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
