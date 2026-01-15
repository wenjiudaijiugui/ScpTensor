# Contributing to ScpTensor

Thank you for your interest in contributing to ScpTensor! This document provides a quick overview of how to contribute.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ScpTensor.git`
3. Install dependencies: `uv pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`
5. Make your changes
6. Run tests: `uv run pytest tests/`
7. Submit a pull request

## Development Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/

# Run linting
uv run ruff check .
uv run mypy scptensor
```

## Code Style

- Python 3.12+
- Type hints required for all public APIs
- NumPy-style docstrings
- English-only documentation
- Line length: 100 characters

## Testing

- Write tests for all new features
- Use pytest for testing
- Aim for >90% coverage
- Add docstring examples

## Documentation

See [Developer Guide](docs/DEVELOPER_GUIDE.md) for comprehensive documentation on:

- Project setup
- Code organization
- Adding new features
- Testing conventions
- Performance guidelines

## Questions?

- Open an issue for bugs or feature requests
- Check [docs/](docs/) for detailed documentation
- See [docs/design/](docs/design/) for architecture specifications
