.PHONY: help install install-dev test test-cov lint format clean build docker docker-dev run-docker benchmark

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	poetry install --no-dev

install-dev: ## Install all dependencies including dev
	poetry install

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=freerouter --cov-report=term-missing --cov-report=html

test-integration: ## Run integration tests
	poetry run pytest tests/test_integration.py

lint: ## Run linting tools
	poetry run black --check freerouter tests
	poetry run isort --check-only freerouter tests
	poetry run flake8 freerouter tests
	poetry run mypy freerouter

format: ## Format code
	poetry run black freerouter tests
	poetry run isort freerouter tests

security: ## Run security checks
	poetry run safety check
	poetry run bandit -r freerouter/

clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build package
	poetry build

docker: ## Build Docker image
	docker build -t freerouter:latest .

docker-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t freerouter:dev .

run-docker: ## Run Docker container
	docker-compose up freerouter

run-docker-dev: ## Run development Docker container
	docker-compose up freerouter-dev

benchmark: ## Run benchmarks
	python scripts/benchmark.py

benchmark-quick: ## Run quick benchmark
	python scripts/benchmark.py --quick

demo: ## Run demo queries
	poetry run freerouter ask "Write a Python function to calculate factorial" --explain
	poetry run freerouter ask "Tell me a story about a robot" --explain
	poetry run freerouter ask "What is machine learning?" --explain

models: ## Show available models
	poetry run freerouter models --refresh --test

stats: ## Show router statistics  
	poetry run freerouter stats

setup-dev: install-dev ## Set up development environment
	poetry run pre-commit install
	@echo "Development environment set up successfully!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make demo' to see FreeRouter in action"

# Docker shortcuts
up: ## Start services with docker-compose
	docker-compose up -d

down: ## Stop services with docker-compose
	docker-compose down

logs: ## View logs from docker-compose
	docker-compose logs -f

# Release helpers
version: ## Show current version
	@poetry version

bump-patch: ## Bump patch version
	poetry version patch

bump-minor: ## Bump minor version
	poetry version minor

bump-major: ## Bump major version
	poetry version major

release-patch: bump-patch build ## Release patch version
	git add pyproject.toml
	git commit -m "Bump version to $(shell poetry version -s)"
	git tag v$(shell poetry version -s)
	git push origin main --tags

release-minor: bump-minor build ## Release minor version
	git add pyproject.toml
	git commit -m "Bump version to $(shell poetry version -s)"
	git tag v$(shell poetry version -s)
	git push origin main --tags

release-major: bump-major build ## Release major version
	git add pyproject.toml
	git commit -m "Bump version to $(shell poetry version -s)"
	git tag v$(shell poetry version -s)
	git push origin main --tags