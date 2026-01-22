# ScpTensor Documentation

This directory contains all project documentation organized by category.

## Directory Structure

- `api/` - Sphinx-generated API documentation (RST format)
- `benchmarks/` - Benchmark test reports and results
- `design/` - Design documents with progressive loader system
- `guides/` - Technical guides and implementation details
- `notebooks/` - Example Jupyter notebooks
- `reference/` - Reference materials (plans, management, reports)
- `tutorials/` - Tutorial notebooks for learning ScpTensor

## Quick Links

- [Design Documents](design/INDEX.md) - Start here for design docs
- [API Reference](api/index.rst) - API documentation
- [Tutorials](tutorials/README.md) - Learning tutorials

## Building Documentation

To build the Sphinx documentation:

```bash
cd scripts
./build_docs.sh
```

Or from the project root:

```bash
python -m sphinx -b html docs/ docs/_build/html
```
