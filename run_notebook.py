#!/usr/bin/env python3
"""
Run notebooks with private config values injected from config_pvt.yaml.

The notebooks on GitHub have placeholders like YOUR_BUCKET_NAME, YOUR_TRAINING_JOB, etc.
This script substitutes real values from config_pvt.yaml at runtime and executes
the notebook via papermill, without modifying the source .ipynb files.

Usage:
    python run_notebook.py <notebook_path> [--output <output_path>] [--config <config_path>]

Examples:
    python run_notebook.py medical-image-segmentation/notebooks/lab7_model_deployment.ipynb
    python run_notebook.py medical-image-segmentation/notebooks/lab1_single_gpu_training.ipynb
    python run_notebook.py medical-image-classification/notebooks/02_sm_script_mode/train.ipynb

    # Custom output location
    python run_notebook.py lab7_model_deployment.ipynb --output /tmp/lab7_output.ipynb

    # Use a different config file
    python run_notebook.py lab7_model_deployment.ipynb --config my_config.yaml

The executed notebook (with outputs) is saved to ./notebook_outputs/ by default.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml


def load_config(config_path: str = None) -> dict:
    """Load private configuration."""
    if config_path is None:
        root = Path(__file__).parent
        candidates = [root / "config_pvt.yaml", root / "config.local.yaml"]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break
        else:
            print("ERROR: No config_pvt.yaml or config.local.yaml found.")
            print("Copy config.yaml to config_pvt.yaml and fill in your values.")
            sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")
    return config


def build_substitutions(config: dict) -> dict:
    """Build a mapping of placeholder -> real value from config."""
    bucket = config["aws"]["s3_bucket"]
    seg = config["segmentation"]
    cls = config["classification"]

    # Extract test NIfTI subject IDs from the S3 path
    test_nifti = seg.get("test_nifti_s3", "")

    subs = {
        # Bucket placeholders
        "YOUR_BUCKET_NAME": bucket,
        "<your_bucket_name>": bucket,
        "<your_input_bucket_name>": bucket,

        # Region placeholders
        "<your_region>": config["aws"]["region"],

        # Account ID placeholders
        "<account_id>": config["aws"].get("account_id", ""),
        "<Your_account_id>": config["aws"].get("account_id", ""),

        # Dataset name for classification
        "<your_datasets_name>": "vindr-spinexr-subset",

        # EC2 local data path
        "/home/ubuntu/data/YOUR_DATASET": "/home/ubuntu/data/vindr-spinexr-subset",

        # Segmentation model artifact
        "s3://YOUR_BUCKET_NAME/segmentation_data/output/YOUR_TRAINING_JOB/output/model.tar.gz": seg.get("model_s3", ""),
        "YOUR_TRAINING_JOB": _extract_job_name(seg.get("model_s3", "")),

        # Test NIfTI paths
        "s3://YOUR_BUCKET_NAME/segmentation_data/test/SUBJECT_ID_2/img.nii.gz": _second_test_subject(config),
        "s3://YOUR_BUCKET_NAME/segmentation_data/test/SUBJECT_ID/img.nii.gz": test_nifti,
        "SUBJECT_ID_2": _extract_subject_id(_second_test_subject(config)),
        "SUBJECT_ID": _extract_subject_id(test_nifti),
    }

    return subs


def _extract_job_name(model_s3: str) -> str:
    """Extract training job name from model artifact path."""
    # s3://bucket/segmentation_data/output/JOB_NAME/output/model.tar.gz
    if not model_s3:
        return "YOUR_TRAINING_JOB"
    parts = model_s3.split("/")
    try:
        output_idx = parts.index("output")
        return parts[output_idx + 1]
    except (ValueError, IndexError):
        return "YOUR_TRAINING_JOB"


def _extract_subject_id(s3_path: str) -> str:
    """Extract subject ID from test NIfTI path."""
    # s3://bucket/segmentation_data/test/SUBJECT_ID/img.nii.gz
    if not s3_path:
        return "SUBJECT_ID"
    parts = s3_path.rstrip("/").split("/")
    try:
        test_idx = parts.index("test")
        return parts[test_idx + 1]
    except (ValueError, IndexError):
        return "SUBJECT_ID"


def _second_test_subject(config: dict) -> str:
    """Get a second test subject path (for batch inference demos)."""
    # If you have multiple test subjects, add them to config.
    # For now, derive from the first one by convention or reuse it.
    test_nifti = config["segmentation"].get("test_nifti_s3", "")
    if not test_nifti:
        return ""
    # Try to find a second subject — use a different known ID or same path
    # Users can customize this in config_pvt.yaml by adding test_nifti_s3_2
    second = config["segmentation"].get("test_nifti_s3_2", "")
    if second:
        return second
    # Default: replace the subject folder with a second known one
    # This is specific to the segmentation dataset structure
    parts = test_nifti.rsplit("/", 2)
    if len(parts) >= 3:
        # Try incrementing the subject ID
        subject_id = parts[-2]
        # Just reuse the same subject for now — users can add test_nifti_s3_2
        return test_nifti
    return test_nifti


def inject_config_into_notebook(nb_path: str, substitutions: dict) -> str:
    """
    Create a temporary notebook with placeholders replaced by real values.
    Returns path to the temporary notebook.
    """
    with open(nb_path, "r") as f:
        content = f.read()

    # Apply substitutions (order matters — longer/more specific patterns first)
    # Sort by key length descending to avoid partial replacements
    sorted_subs = sorted(substitutions.items(), key=lambda x: len(x[0]), reverse=True)

    for placeholder, value in sorted_subs:
        if value:  # Only substitute if we have a real value
            content = content.replace(placeholder, value)

    # Write to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="nb_run_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(nb_path))
    with open(tmp_path, "w") as f:
        f.write(content)

    return tmp_path


def run_notebook(nb_path: str, output_path: str, config_path: str = None):
    """Execute a notebook with config values injected."""
    try:
        import papermill as pm
    except ImportError:
        print("ERROR: papermill is required. Install with:")
        print("  pip install papermill")
        sys.exit(1)

    config = load_config(config_path)
    substitutions = build_substitutions(config)

    print(f"\nNotebook: {nb_path}")
    print(f"Output:   {output_path}")
    print(f"\nSubstitutions applied:")
    for k, v in sorted(substitutions.items()):
        if v and v != k:
            print(f"  {k} -> {v}")

    # Create temp notebook with values injected
    tmp_nb = inject_config_into_notebook(nb_path, substitutions)

    print(f"\nExecuting notebook...")
    print("=" * 60)

    try:
        pm.execute_notebook(
            tmp_nb,
            output_path,
            kernel_name="python3",
            cwd=str(Path(nb_path).parent.resolve()),
        )
        print("=" * 60)
        print(f"\nSUCCESS. Output saved to: {output_path}")
    except pm.PapermillExecutionError as e:
        print("=" * 60)
        print(f"\nFAILED at cell {e.cell_index}:")
        print(f"  {e.ename}: {e.evalue}")
        print(f"\nPartial output saved to: {output_path}")
        sys.exit(1)
    finally:
        # Cleanup temp file
        os.unlink(tmp_nb)
        os.rmdir(os.path.dirname(tmp_nb))


def main():
    parser = argparse.ArgumentParser(
        description="Run notebooks with config_pvt.yaml values injected",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("notebook", help="Path to the notebook to run")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for executed notebook (default: ./notebook_outputs/<name>)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config YAML (default: config_pvt.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show substitutions without executing the notebook",
    )

    args = parser.parse_args()

    nb_path = args.notebook
    if not os.path.exists(nb_path):
        print(f"ERROR: Notebook not found: {nb_path}")
        sys.exit(1)

    # Default output directory
    if args.output is None:
        output_dir = Path(__file__).parent / "notebook_outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / os.path.basename(nb_path))
    else:
        output_path = args.output

    if args.dry_run:
        config = load_config(args.config)
        substitutions = build_substitutions(config)
        print(f"\nNotebook: {nb_path}")
        print(f"\nSubstitutions that would be applied:")
        for k, v in sorted(substitutions.items()):
            if v and v != k:
                print(f"  {k} -> {v}")
        print("\n(Dry run — notebook not executed)")
        return

    run_notebook(nb_path, output_path, args.config)


if __name__ == "__main__":
    main()
