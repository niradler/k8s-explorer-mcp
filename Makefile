.PHONY: help install install-dev test lint format clean run example check sync

help:
	@echo "K8s Explorer MCP - Makefile Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install package with dependencies (using uv)"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make sync         - Sync dependencies from pyproject.toml"
	@echo ""
	@echo "Development:"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make check        - Run all checks (format, lint, type)"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo ""
	@echo "Running:"
	@echo "  make run          - Start MCP server"
	@echo "  make example      - Run example usage"
	@echo ""
	@echo "Publishing:"
	@echo "  make build        - Build distribution packages"
	@echo "  make publish      - Build and publish to PyPI (requires PyPI token)"
	@echo "  make test-publish - Build and publish to TestPyPI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo ""

install:
	@echo "📦 Installing k8s-explorer-mcp with uv..."
	uv pip install -e .
	@echo "✅ Installation complete!"

install-dev:
	@echo "📦 Installing k8s-explorer-mcp with dev dependencies using uv..."
	uv pip install -e ".[dev]"
	@echo "✅ Development installation complete!"

sync:
	@echo "🔄 Syncing dependencies with uv..."
	uv pip sync pyproject.toml
	@echo "✅ Dependencies synced!"

test:
	@echo "🧪 Running tests..."
	pytest

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest --cov=k8s_explorer --cov-report=html --cov-report=term tests/

lint:
	@echo "🔍 Linting code..."
	ruff check k8s_explorer/ tests/ server.py

format:
	@echo "✨ Formatting code..."
	black k8s_explorer/ tests/ server.py

type-check:
	@echo "🔎 Type checking..."
	mypy k8s_explorer/

check: format lint type-check
	@echo "✅ All checks passed!"

run:
	@echo "🚀 Starting MCP server..."
	python server.py

example:
	@echo "🎯 Running example..."
	python examples/example_usage.py

build: clean
	@echo "📦 Building distribution packages..."
	uv build
	@echo "✅ Build complete! Packages in dist/"

publish: build
	@echo "🚀 Publishing to PyPI..."
	@echo "⚠️  Make sure you have set PYPI_TOKEN or configured uv with PyPI credentials"
	uv publish
	@echo "✅ Published to PyPI!"

clean:
	@echo "🧹 Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

.DEFAULT_GOAL := help

