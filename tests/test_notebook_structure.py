"""
Smoke tests for notebook structure validation.

These tests verify notebooks are well-formed without executing them or
creating any AWS resources. They run fast and catch common issues like
syntax errors, missing imports, and structural problems.

Run with: pytest -m smoke
"""

import ast
import pytest
import nbformat
from pathlib import Path

from helpers import (
    NOTEBOOK_PATHS,
    CLASSIFICATION_NOTEBOOKS,
    SEGMENTATION_NOTEBOOKS,
    load_notebook,
    get_code_cells,
)


# ============================================================================
# Parametrize over all notebooks
# ============================================================================

ALL_NOTEBOOKS = list(NOTEBOOK_PATHS.items())


@pytest.mark.smoke
class TestNotebookExists:
    """Verify all expected notebooks exist on disk."""

    @pytest.mark.parametrize("name,path", ALL_NOTEBOOKS)
    def test_notebook_file_exists(self, name, path):
        assert path.exists(), f"Notebook not found: {path}"

    @pytest.mark.parametrize("name,path", ALL_NOTEBOOKS)
    def test_notebook_is_valid_json(self, name, path):
        """Notebooks are valid JSON (nbformat)."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        assert nb.nbformat >= 4, f"Notebook {name} uses outdated format"

    @pytest.mark.parametrize("name,path", ALL_NOTEBOOKS)
    def test_notebook_has_code_cells(self, name, path):
        """Every notebook should have at least one code cell."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        code_cells = get_code_cells(nb)
        assert len(code_cells) > 0, f"Notebook {name} has no code cells"


@pytest.mark.smoke
class TestNotebookSyntax:
    """Validate Python syntax in all code cells."""

    @pytest.mark.parametrize("name,path", ALL_NOTEBOOKS)
    def test_code_cells_have_valid_syntax(self, name, path):
        """All code cells should parse as valid Python."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        code_cells = get_code_cells(nb)

        errors = []
        for i, cell in enumerate(code_cells):
            source = cell.source
            # Skip cells with magic commands or shell commands
            if source.strip().startswith(("%", "!", "%%")):
                continue
            # Strip IPython magics from mixed cells
            lines = []
            for line in source.split("\n"):
                stripped = line.strip()
                if stripped.startswith(("%", "!")):
                    continue
                lines.append(line)
            clean_source = "\n".join(lines)
            if not clean_source.strip():
                continue

            try:
                ast.parse(clean_source)
            except SyntaxError as e:
                errors.append(f"Cell {i}: {e.msg} (line {e.lineno})")

        assert not errors, f"Syntax errors in {name}:\n" + "\n".join(errors)


@pytest.mark.smoke
class TestNotebookImports:
    """Verify expected imports are present in notebooks."""

    @pytest.mark.parametrize("name,path", [
        n for n in ALL_NOTEBOOKS
        if n[0] not in ("ec2_training",)
    ])
    def test_sagemaker_import(self, name, path):
        """All SageMaker notebooks should import sagemaker."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        all_source = "\n".join(c.source for c in get_code_cells(nb))
        assert "import sagemaker" in all_source or "from sagemaker" in all_source, \
            f"Notebook {name} does not import sagemaker"

    @pytest.mark.parametrize("name,path", [
        n for n in ALL_NOTEBOOKS
        if n[0] not in ("ec2_training",)
    ])
    def test_boto3_or_aws_import(self, name, path):
        """All SageMaker notebooks should import boto3 directly or via utility."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        all_source = "\n".join(c.source for c in get_code_cells(nb))
        has_boto3 = "import boto3" in all_source or "from boto3" in all_source
        # Some notebooks use a shared utility that imports boto3 internally
        has_utils = "from utils import" in all_source or "import utils" in all_source
        assert has_boto3 or has_utils, \
            f"Notebook {name} does not import boto3 or a utility that wraps it"

    @pytest.mark.parametrize("name,path", [
        n for n in ALL_NOTEBOOKS
        if "seg" in n[0] and "deploy" not in n[0]
    ])
    def test_segmentation_uses_pytorch_estimator(self, name, path):
        """Segmentation training notebooks should use PyTorch estimator."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        all_source = "\n".join(c.source for c in get_code_cells(nb))
        has_estimator = (
            "PyTorch(" in all_source
            or "Estimator(" in all_source
            or "HyperparameterTuner(" in all_source
        )
        assert has_estimator, f"Notebook {name} doesn't create an Estimator"


@pytest.mark.smoke
class TestNotebookMetadata:
    """Verify notebook metadata and kernel specifications."""

    @pytest.mark.parametrize("name,path", ALL_NOTEBOOKS)
    def test_has_kernel_spec(self, name, path):
        """Notebooks should specify a kernel."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        kernel = nb.metadata.get("kernelspec", {})
        assert kernel.get("language") == "python", \
            f"Notebook {name} does not have a Python kernel specified"

    @pytest.mark.parametrize("name,path", ALL_NOTEBOOKS)
    def test_has_markdown_header(self, name, path):
        """Notebooks should start with a markdown title/description."""
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        nb = load_notebook(path)
        first_cell = nb.cells[0] if nb.cells else None
        assert first_cell is not None, f"Notebook {name} is empty"
        # Some notebooks start with a setup code cell; check first 3 cells
        has_markdown = any(
            c.cell_type == "markdown" for c in nb.cells[:3]
        )
        assert has_markdown, \
            f"Notebook {name} should have a markdown cell in the first 3 cells"


@pytest.mark.smoke
class TestSupportingFiles:
    """Verify supporting files (scripts, Dockerfiles, requirements) exist."""

    def test_classification_script_mode_has_train_script(self):
        train_script = CLASSIFICATION_NOTEBOOKS / "02_sm_script_mode" / "code" / "train.py"
        assert train_script.exists(), "Script mode train.py not found"

    def test_classification_script_mode_has_model_def(self):
        model_def = CLASSIFICATION_NOTEBOOKS / "02_sm_script_mode" / "code" / "model_def.py"
        assert model_def.exists(), "Script mode model_def.py not found"

    def test_classification_byoc_has_dockerfile(self):
        dockerfile = CLASSIFICATION_NOTEBOOKS / "03_sagemaker_byoc" / "Dockerfile"
        assert dockerfile.exists(), "BYOC Dockerfile not found"

    def test_classification_byoc_has_train_script(self):
        train_script = CLASSIFICATION_NOTEBOOKS / "03_sagemaker_byoc" / "src" / "train.py"
        assert train_script.exists(), "BYOC train.py not found"

    def test_classification_deployment_has_inference_script(self):
        inference = CLASSIFICATION_NOTEBOOKS / "05_sagemaker_deployment" / "inference.py"
        assert inference.exists(), "Deployment inference.py not found"

    def test_segmentation_has_deploy_inference_script(self):
        inference = SEGMENTATION_NOTEBOOKS / "deploy" / "inference.py"
        assert inference.exists(), "Segmentation deploy/inference.py not found"

    def test_segmentation_has_requirements(self):
        reqs = SEGMENTATION_NOTEBOOKS / "requirements.txt"
        assert reqs.exists(), "Segmentation requirements.txt not found"

    def test_preprocessing_has_dockerfile(self):
        dockerfile = CLASSIFICATION_NOTEBOOKS / "01_data_preprocessing" / "Dockerfile"
        assert dockerfile.exists(), "Preprocessing Dockerfile not found"

    def test_preprocessing_has_script(self):
        script = CLASSIFICATION_NOTEBOOKS / "01_data_preprocessing" / "preprocessing.py"
        assert script.exists(), "Preprocessing script not found"
