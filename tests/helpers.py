"""
Shared helpers and constants for notebook integration tests.

This module contains utility functions and path constants used across
all test modules. Import from here (not from conftest.py).
"""

import os
import nbformat
from pathlib import Path
from typing import Optional


# ============================================================================
# Path Constants
# ============================================================================

ROOT_DIR = Path(__file__).parent.parent
CLASSIFICATION_NOTEBOOKS = ROOT_DIR / "medical-image-classification" / "notebooks"
SEGMENTATION_NOTEBOOKS = ROOT_DIR / "medical-image-segmentation" / "notebooks"

NOTEBOOK_PATHS = {
    # Classification notebooks
    "ec2_training": CLASSIFICATION_NOTEBOOKS / "00_ec2_training" / "00-ec2-training.ipynb",
    "data_preprocessing": CLASSIFICATION_NOTEBOOKS / "01_data_preprocessing" / "01-data-preprocessing.ipynb",
    "script_mode": CLASSIFICATION_NOTEBOOKS / "02_sm_script_mode" / "train.ipynb",
    "byoc": CLASSIFICATION_NOTEBOOKS / "03_sagemaker_byoc" / "train.ipynb",
    "byoc_mlflow": CLASSIFICATION_NOTEBOOKS / "04_sagemaekr_byoc_mlflow" / "train.ipynb",
    "deploy_async": CLASSIFICATION_NOTEBOOKS / "05_sagemaker_deployment" / "deploy.ipynb",
    "deploy_realtime": CLASSIFICATION_NOTEBOOKS / "05_sagemaker_deployment" / "deploy_realtime.ipynb",
    # Segmentation notebooks
    "seg_single_gpu": SEGMENTATION_NOTEBOOKS / "lab1_single_gpu_training.ipynb",
    "seg_fsdp": SEGMENTATION_NOTEBOOKS / "lab2_fsdp_multi_gpu.ipynb",
    "seg_wandb": SEGMENTATION_NOTEBOOKS / "lab3_wandb_experiment_tracking.ipynb",
    "seg_ddp_tracking": SEGMENTATION_NOTEBOOKS / "lab4_ddp_unified_tracking.ipynb",
    "seg_hpo": SEGMENTATION_NOTEBOOKS / "lab5_hyperparameter_optimization.ipynb",
    "seg_nnunet": SEGMENTATION_NOTEBOOKS / "lab6_nnunet_pipeline.ipynb",
    "seg_deploy": SEGMENTATION_NOTEBOOKS / "lab7_model_deployment.ipynb",
    "seg_fsdp_standalone": SEGMENTATION_NOTEBOOKS / "train_fsdp_sagemaker.ipynb",
}


# ============================================================================
# Notebook Utilities
# ============================================================================


def load_notebook(path: Path) -> nbformat.NotebookNode:
    """Load a notebook from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def get_code_cells(notebook: nbformat.NotebookNode) -> list:
    """Extract code cells from a notebook."""
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def get_cell_source(notebook: nbformat.NotebookNode, index: int) -> str:
    """Get source code of a specific cell by index."""
    code_cells = get_code_cells(notebook)
    if index < len(code_cells):
        return code_cells[index].source
    return ""


def execute_notebook(
    notebook_path: Path,
    parameters: Optional[dict] = None,
    output_path: Optional[Path] = None,
    timeout: int = 1800,
    kernel_name: str = "python3",
) -> Path:
    """Execute a notebook using papermill.

    Args:
        notebook_path: Path to the input notebook.
        parameters: Dictionary of parameters to inject into the notebook.
        output_path: Path to save the executed notebook. Defaults to /tmp.
        timeout: Cell execution timeout in seconds.
        kernel_name: Jupyter kernel to use.

    Returns:
        Path to the executed output notebook.
    """
    import papermill as pm

    if output_path is None:
        output_path = Path("/tmp") / f"test_output_{notebook_path.stem}.ipynb"

    pm.execute_notebook(
        str(notebook_path),
        str(output_path),
        parameters=parameters or {},
        kernel_name=kernel_name,
        request_save_on_cell_execute=True,
        progress_bar=False,
        cwd=str(notebook_path.parent),
    )
    return output_path


# ============================================================================
# AWS Wait Helpers
# ============================================================================


def wait_for_training_job(sagemaker_client, job_name: str, timeout: int = 1800) -> str:
    """Wait for a SageMaker training job to complete.

    Returns:
        Final status: 'Completed', 'Failed', or 'Stopped'
    """
    import time

    start = time.time()
    while time.time() - start < timeout:
        desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            return status
        time.sleep(30)
    raise TimeoutError(f"Training job {job_name} did not complete within {timeout}s")


def wait_for_endpoint(sagemaker_client, endpoint_name: str, timeout: int = 900) -> str:
    """Wait for a SageMaker endpoint to be InService.

    Returns:
        Final status: 'InService', 'Failed', etc.
    """
    import time

    start = time.time()
    while time.time() - start < timeout:
        desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        if status in ("InService", "Failed", "RollingBack"):
            return status
        time.sleep(30)
    raise TimeoutError(f"Endpoint {endpoint_name} did not become InService within {timeout}s")
