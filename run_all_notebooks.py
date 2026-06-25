#!/usr/bin/env python3
"""
Run all notebooks in the repo with config_pvt.yaml values injected.

Usage:
    python run_all_notebooks.py                    # Run all notebooks
    python run_all_notebooks.py --seg              # Run segmentation notebooks only
    python run_all_notebooks.py --cls              # Run classification notebooks only
    python run_all_notebooks.py --dry-run          # Show what would run without executing
    python run_all_notebooks.py --stop-on-failure  # Stop at first failure
    python run_all_notebooks.py --skip lab6 lab3   # Skip notebooks matching these patterns

Notebooks are executed in dependency order:
  1. Classification: EC2 training -> preprocessing -> script mode -> BYOC -> BYOC+MLflow -> deployment
  2. Segmentation: single GPU -> FSDP multi-GPU -> WandB tracking -> DDP tracking -> HPO -> nnUNet -> deployment

Executed notebooks are saved to ./notebook_outputs/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Notebook execution order — structured by dependency
SEGMENTATION_NOTEBOOKS = [
    "medical-image-segmentation/notebooks/lab1_single_gpu_training.ipynb",
    "medical-image-segmentation/notebooks/train_fsdp_sagemaker.ipynb",
    "medical-image-segmentation/notebooks/lab2_fsdp_multi_gpu.ipynb",
    "medical-image-segmentation/notebooks/lab3_wandb_experiment_tracking.ipynb",
    "medical-image-segmentation/notebooks/lab4_ddp_unified_tracking.ipynb",
    "medical-image-segmentation/notebooks/lab5_hyperparameter_optimization.ipynb",
    "medical-image-segmentation/notebooks/lab6_nnunet_pipeline.ipynb",
    "medical-image-segmentation/notebooks/lab7_model_deployment.ipynb",
]

CLASSIFICATION_NOTEBOOKS = [
    "medical-image-classification/notebooks/00_ec2_training/00-ec2-training.ipynb",
    "medical-image-classification/notebooks/01_data_preprocessing/01-data-preprocessing.ipynb",
    "medical-image-classification/notebooks/02_sm_script_mode/train.ipynb",
    "medical-image-classification/notebooks/03_sagemaker_byoc/train.ipynb",
    "medical-image-classification/notebooks/04_sagemaekr_byoc_mlflow/train.ipynb",
    "medical-image-classification/notebooks/05_sagemaker_deployment/deploy.ipynb",
    "medical-image-classification/notebooks/05_sagemaker_deployment/deploy_realtime.ipynb",
]

ALL_NOTEBOOKS = CLASSIFICATION_NOTEBOOKS + SEGMENTATION_NOTEBOOKS


def _extract_error_from_notebook(nb_path: str) -> str:
    """Try to extract a meaningful error message from a failed notebook output."""
    import re
    try:
        with open(nb_path) as f:
            nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell['cell_type'] != 'code':
                continue
            meta = cell.get('metadata', {}).get('papermill', {})
            if not meta.get('exception', False):
                continue
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'error':
                    ename = output.get('ename', '')
                    evalue = re.sub(r'\x1b\[[0-9;]*m', '', output.get('evalue', ''))
                    return f"{ename}: {evalue[:200]}"
                if output.get('output_type') == 'display_data':
                    text = output.get('data', {}).get('text/plain', '')
                    if isinstance(text, list):
                        text = ''.join(text)
                    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
                    # Find error lines
                    for line in reversed(text.split('\n')):
                        line = line.strip().strip('│').strip()
                        if 'Error:' in line or 'Exception:' in line:
                            return line[:200]
            break
    except Exception:
        pass
    return ""


def run_all(notebooks, config_path=None, dry_run=False, stop_on_failure=False, skip_patterns=None):
    """Run a list of notebooks, collecting results."""
    # Import run_notebook module from same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from run_notebook import load_config, build_substitutions, inject_config_into_notebook

    try:
        import papermill as pm
    except ImportError:
        print("ERROR: papermill is required. Install with:")
        print("  pip install papermill")
        sys.exit(1)

    config = load_config(config_path)
    substitutions = build_substitutions(config)

    root = Path(__file__).parent
    output_dir = root / "notebook_outputs"
    output_dir.mkdir(exist_ok=True)

    # Filter out skipped notebooks
    if skip_patterns:
        filtered = []
        for nb in notebooks:
            if any(pat.lower() in nb.lower() for pat in skip_patterns):
                print(f"  SKIP: {nb}")
            else:
                filtered.append(nb)
        notebooks = filtered

    # Check all notebooks exist
    missing = [nb for nb in notebooks if not (root / nb).exists()]
    if missing:
        print("ERROR: Missing notebooks:")
        for nb in missing:
            print(f"  {nb}")
        sys.exit(1)

    if dry_run:
        print(f"\nWould run {len(notebooks)} notebooks:\n")
        for i, nb in enumerate(notebooks, 1):
            print(f"  {i:2d}. {nb}")
        print(f"\nSubstitutions:")
        for k, v in sorted(substitutions.items()):
            if v and v != k:
                print(f"    {k} -> {v}")
        print("\n(Dry run — no notebooks executed)")
        return

    # Execute notebooks
    results = []
    total = len(notebooks)
    start_all = time.time()

    print(f"\n{'='*70}")
    print(f"  Running {total} notebooks")
    print(f"{'='*70}\n")

    for i, nb_rel in enumerate(notebooks, 1):
        nb_path = str(root / nb_rel)
        # Use parent folder name as prefix to avoid collisions (e.g., multiple train.ipynb)
        nb_basename = os.path.basename(nb_rel)
        parent_dir = Path(nb_rel).parent.name
        output_name = f"{parent_dir}--{nb_basename}" if parent_dir != "notebooks" else nb_basename
        output_path = str(output_dir / output_name)

        print(f"\n[{i}/{total}] {nb_rel}")
        print(f"  Output: {output_path}")

        start = time.time()

        # Create temp notebook with substitutions
        tmp_nb = inject_config_into_notebook(nb_path, substitutions)

        try:
            pm.execute_notebook(
                tmp_nb,
                output_path,
                kernel_name="python3",
                cwd=str(Path(nb_path).parent.resolve()),
            )
            elapsed = time.time() - start
            results.append(("PASS", nb_rel, elapsed))
            print(f"  PASS ({elapsed:.0f}s)")

        except pm.PapermillExecutionError as e:
            elapsed = time.time() - start
            err_msg = f"{e.ename}: {e.evalue}" if e.ename else str(e)
            results.append(("FAIL", nb_rel, elapsed, err_msg))
            print(f"  FAIL ({elapsed:.0f}s): {err_msg[:120]}")
            if stop_on_failure:
                print("\n  --stop-on-failure set, aborting.")
                break

        except Exception as e:
            elapsed = time.time() - start
            # Try to extract meaningful error from the output notebook
            err_msg = _extract_error_from_notebook(output_path) or str(e)
            results.append(("ERROR", nb_rel, elapsed, err_msg))
            print(f"  ERROR ({elapsed:.0f}s): {err_msg[:120]}")
            if stop_on_failure:
                print("\n  --stop-on-failure set, aborting.")
                break

        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_nb)
                os.rmdir(os.path.dirname(tmp_nb))
            except OSError:
                pass

    # Summary
    total_time = time.time() - start_all
    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] in ("FAIL", "ERROR"))

    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {total_time:.0f}s total")
    print(f"{'='*70}\n")

    for status, nb, elapsed, *err in results:
        icon = "✓" if status == "PASS" else "✗"
        line = f"  {icon} {nb} ({elapsed:.0f}s)"
        if err:
            line += f" — {err[0][:80]}"
        print(line)

    print(f"\n  Output directory: {output_dir}/")

    if failed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run all notebooks with config_pvt.yaml values injected",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config YAML (default: config_pvt.yaml)",
    )
    parser.add_argument(
        "--seg", action="store_true",
        help="Run only segmentation notebooks",
    )
    parser.add_argument(
        "--cls", action="store_true",
        help="Run only classification notebooks",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Stop at the first notebook failure",
    )
    parser.add_argument(
        "--skip", nargs="+", default=None,
        metavar="PATTERN",
        help="Skip notebooks matching these patterns (e.g., --skip lab6 ec2)",
    )

    args = parser.parse_args()

    if args.seg and args.cls:
        notebooks = ALL_NOTEBOOKS
    elif args.seg:
        notebooks = SEGMENTATION_NOTEBOOKS
    elif args.cls:
        notebooks = CLASSIFICATION_NOTEBOOKS
    else:
        notebooks = ALL_NOTEBOOKS

    run_all(
        notebooks,
        config_path=args.config,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure,
        skip_patterns=args.skip,
    )


if __name__ == "__main__":
    main()
