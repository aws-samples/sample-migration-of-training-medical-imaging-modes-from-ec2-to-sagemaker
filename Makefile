# ============================================================================
# Integration Test Makefile
# ============================================================================
#
# Usage:
#   make test-smoke          Run smoke tests (no AWS, fast)
#   make test-infra          Validate AWS prerequisites
#   make test-classification Run classification integration tests
#   make test-segmentation   Run segmentation integration tests
#   make test-all            Run all integration tests
#   make test-cheap          Run non-expensive integration tests
#
# Required environment variables for integration tests:
#   AWS_SAGEMAKER_ROLE       - SageMaker execution role ARN or name
#   CLASSIFICATION_DATA_S3   - S3 URI to classification dataset
#   SEGMENTATION_DATA_S3     - S3 URI to segmentation dataset
#
# Optional:
#   AWS_TEST_REGION          - AWS region (default: us-east-1)
#   AWS_TEST_BUCKET          - S3 bucket (default: SageMaker default)
#   CLASSIFICATION_MODEL_S3  - Model artifact for deployment tests
#   SEGMENTATION_MODEL_S3    - Model artifact for deployment tests
#   BYOC_IMAGE_URI           - Custom container image URI
#   WANDB_API_KEY            - Weights & Biases API key
#   TEST_NIFTI_S3            - S3 path to test NIfTI file
# ============================================================================

.PHONY: install test-smoke test-infra test-classification test-segmentation \
        test-all test-cheap test-training test-deployment clean help

PYTEST_OPTS ?= -v --tb=short
TEST_DIR = tests

# ============================================================================
# Setup
# ============================================================================

install:
	pip install -r tests/requirements.txt

# ============================================================================
# Test Targets
# ============================================================================

## Run smoke tests only (validates notebook structure, no AWS calls)
test-smoke:
	pytest $(TEST_DIR)/test_notebook_structure.py -m smoke $(PYTEST_OPTS)

## Validate AWS infrastructure prerequisites
test-infra:
	pytest $(TEST_DIR)/test_infrastructure.py -m integration $(PYTEST_OPTS)

## Run classification notebook integration tests
test-classification:
	pytest $(TEST_DIR)/test_classification_integration.py -m "integration and classification" $(PYTEST_OPTS) --timeout=3600

## Run segmentation notebook integration tests
test-segmentation:
	pytest $(TEST_DIR)/test_segmentation_integration.py -m "integration and segmentation" $(PYTEST_OPTS) --timeout=3600

## Run training tests only (skip deployment)
test-training:
	pytest -m "integration and training" $(PYTEST_OPTS) --timeout=2400

## Run deployment tests only (skip training)
test-deployment:
	pytest -m "integration and deployment" $(PYTEST_OPTS) --timeout=1800

## Run all integration tests (expensive!)
test-all:
	pytest -m integration $(PYTEST_OPTS) --timeout=3600

## Run integration tests that are not marked expensive
test-cheap:
	pytest -m "integration and not expensive" $(PYTEST_OPTS)

## Run full notebook execution via papermill
test-notebooks:
	pytest $(TEST_DIR)/test_notebook_execution.py -m integration $(PYTEST_OPTS) --timeout=3600

## Run run.py CLI smoke tests (no AWS, fast)
test-cli-smoke:
	pytest $(TEST_DIR)/test_run_cli.py -m smoke $(PYTEST_OPTS)

## Run run.py CLI integration tests
test-cli:
	pytest $(TEST_DIR)/test_run_cli.py -m integration $(PYTEST_OPTS) --timeout=3600

## Clean up test outputs
clean:
	rm -rf /tmp/notebook_test_outputs/
	rm -rf /tmp/test_output_*.ipynb
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Available targets:"
	@echo ""
	@echo "  install            Install test dependencies"
	@echo "  test-smoke         Run smoke tests (fast, no AWS)"
	@echo "  test-infra         Validate AWS prerequisites"
	@echo "  test-classification  Run classification integration tests"
	@echo "  test-segmentation  Run segmentation integration tests"
	@echo "  test-training      Run only training tests"
	@echo "  test-deployment    Run only deployment tests"
	@echo "  test-all           Run ALL integration tests (expensive)"
	@echo "  test-cheap         Run non-expensive integration tests"
	@echo "  test-notebooks     Full notebook execution via papermill"
	@echo "  test-cli-smoke     Run run.py config/structure tests (no AWS)"
	@echo "  test-cli           Run run.py integration tests"
	@echo "  clean              Remove test artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  AWS_SAGEMAKER_ROLE          SageMaker execution role"
	@echo "  CLASSIFICATION_DATA_S3      S3 URI for classification data"
	@echo "  SEGMENTATION_DATA_S3        S3 URI for segmentation data"
	@echo "  CLASSIFICATION_MODEL_S3     Model artifact for deploy tests"
	@echo "  SEGMENTATION_MODEL_S3       Model artifact for deploy tests"
	@echo "  BYOC_IMAGE_URI              Custom container image URI"
	@echo "  WANDB_API_KEY               W&B API key"
	@echo "  TEST_NIFTI_S3               S3 path to test NIfTI file"
