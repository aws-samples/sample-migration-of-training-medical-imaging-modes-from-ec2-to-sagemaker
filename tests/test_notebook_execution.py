"""
Full notebook execution tests using papermill.

These tests run entire notebooks end-to-end using papermill, which:
- Executes every cell in order
- Captures cell outputs and errors
- Can inject parameters to override defaults
- Saves the executed notebook for debugging

This is the most comprehensive integration test - it validates the exact
user experience of running the notebook.

Run with: pytest -m "integration and notebook_exec" --timeout=3600

Environment variables:
    AWS_TEST_REGION: AWS region (default: us-east-1)
    CLASSIFICATION_DATA_S3: S3 path to classification data
    SEGMENTATION_DATA_S3: S3 path to segmentation data
    NOTEBOOK_KERNEL: Kernel to use (default: python3)
"""

import os
import pytest
from pathlib import Path

from helpers import NOTEBOOK_PATHS, execute_notebook


# ============================================================================
# Configuration
# ============================================================================

NOTEBOOK_KERNEL = os.environ.get("NOTEBOOK_KERNEL", "python3")
OUTPUT_DIR = Path("/tmp/notebook_test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Classification Notebook Execution
# ============================================================================


@pytest.mark.integration
@pytest.mark.classification
@pytest.mark.expensive
class TestClassificationNotebookExecution:
    """Execute classification notebooks end-to-end with papermill."""

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_DATA_S3"),
        reason="CLASSIFICATION_DATA_S3 not set",
    )
    @pytest.mark.timeout(1200)
    def test_execute_data_preprocessing_notebook(self):
        """Execute the data preprocessing notebook."""
        notebook_path = NOTEBOOK_PATHS["data_preprocessing"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "01_data_preprocessing_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=600,
        )
        assert output.exists(), "Output notebook not created"

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_DATA_S3"),
        reason="CLASSIFICATION_DATA_S3 not set",
    )
    @pytest.mark.timeout(2400)
    def test_execute_script_mode_notebook(self):
        """Execute the Script Mode training notebook."""
        notebook_path = NOTEBOOK_PATHS["script_mode"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "02_script_mode_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=1800,
        )
        assert output.exists()

    @pytest.mark.skipif(
        not os.environ.get("BYOC_IMAGE_URI"),
        reason="BYOC_IMAGE_URI not set",
    )
    @pytest.mark.timeout(2400)
    def test_execute_byoc_notebook(self):
        """Execute the BYOC training notebook."""
        notebook_path = NOTEBOOK_PATHS["byoc"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "03_byoc_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=1800,
        )
        assert output.exists()

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_MODEL_S3"),
        reason="CLASSIFICATION_MODEL_S3 not set",
    )
    @pytest.mark.timeout(1800)
    def test_execute_deployment_notebook(self):
        """Execute the async deployment notebook."""
        notebook_path = NOTEBOOK_PATHS["deploy_async"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "05_deploy_async_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=1200,
        )
        assert output.exists()


# ============================================================================
# Segmentation Notebook Execution
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.expensive
class TestSegmentationNotebookExecution:
    """Execute segmentation notebooks end-to-end with papermill."""

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_DATA_S3"),
        reason="SEGMENTATION_DATA_S3 not set",
    )
    @pytest.mark.timeout(2400)
    def test_execute_single_gpu_training(self):
        """Execute Lab 1: Single GPU training notebook."""
        notebook_path = NOTEBOOK_PATHS["seg_single_gpu"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "seg_lab1_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=1800,
        )
        assert output.exists()

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_DATA_S3"),
        reason="SEGMENTATION_DATA_S3 not set",
    )
    @pytest.mark.timeout(3600)
    def test_execute_fsdp_multi_gpu(self):
        """Execute Lab 2: FSDP multi-GPU training notebook."""
        notebook_path = NOTEBOOK_PATHS["seg_fsdp"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "seg_lab2_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=2400,
        )
        assert output.exists()

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_DATA_S3") or not os.environ.get("WANDB_API_KEY"),
        reason="SEGMENTATION_DATA_S3 or WANDB_API_KEY not set",
    )
    @pytest.mark.timeout(2400)
    def test_execute_wandb_tracking(self):
        """Execute Lab 3: W&B experiment tracking notebook."""
        notebook_path = NOTEBOOK_PATHS["seg_wandb"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "seg_lab3_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=1800,
        )
        assert output.exists()

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_DATA_S3"),
        reason="SEGMENTATION_DATA_S3 not set",
    )
    @pytest.mark.timeout(3600)
    def test_execute_hpo_notebook(self):
        """Execute Lab 5: Hyperparameter optimization notebook."""
        notebook_path = NOTEBOOK_PATHS["seg_hpo"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "seg_lab5_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=3000,
        )
        assert output.exists()

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_MODEL_S3"),
        reason="SEGMENTATION_MODEL_S3 not set",
    )
    @pytest.mark.timeout(1800)
    def test_execute_deployment_notebook(self):
        """Execute Lab 7: Model deployment notebook."""
        notebook_path = NOTEBOOK_PATHS["seg_deploy"]
        if not notebook_path.exists():
            pytest.skip("Notebook not found")

        output = execute_notebook(
            notebook_path,
            output_path=OUTPUT_DIR / "seg_lab7_output.ipynb",
            kernel_name=NOTEBOOK_KERNEL,
            timeout=1200,
        )
        assert output.exists()
