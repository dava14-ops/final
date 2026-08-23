.PHONY: help install install-dev test test-cov test-fast test-integration test-consistency test-severity lint format type-check security clean pre-commit-install pre-commit-run train-model api-server docs-serve all
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
install:
	pip install -r requirements.txt
install-dev:
	pip install -r requirements-dev.txt
	pre-commit install
test:
	python -m pytest tests/ -v
test-cov:
	python -m pytest tests/ -v --cov=prediction_engine --cov=premium_engine --cov=severity_model --cov-report=term-missing --cov-report=html
test-fast:
	python -m pytest tests/ -v -m "not slow"
test-integration:
	python -m pytest tests/test_integration.py -v
test-consistency:
	python -m pytest tests/test_consistency.py -v
test-severity:
	python -m pytest tests/test_severity_integration.py -v
lint:
	ruff check .
	ruff format --check .
format:
	ruff check . --fix
	ruff format .
	black .
	isort .
type-check:
	mypy prediction_engine.py premium_engine.py severity_model.py || true
security:
	bandit -r . -x tests,*.md,docs
	safety check --full-report
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .eggs/
pre-commit-install:
	pre-commit install
pre-commit-run:
	pre-commit run --all-files
train-model:
	python train_model.py
api-server:
	uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
docs-serve:
	@echo "Документация в docs/"
	@ls -la docs/
all: install-dev format lint type-check test-cov
