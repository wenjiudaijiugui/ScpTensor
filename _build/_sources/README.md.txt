# ScpTensor Documentation

This directory contains the Sphinx documentation for ScpTensor.

## Building the Documentation

### Quick Build

```bash
# Using the build script (recommended)
./build.sh

# Or using make
make html

# Or using sphinx-build directly
uv run sphinx-build -b html . _build/html
```

### Viewing the Documentation

After building, open `_build/html/index.html` in your browser, or serve it locally:

```bash
cd _build/html
python3 -m http.server 8000
# Visit http://localhost:8000
```

### Live Reload (Optional)

For live reloading during development:

```bash
# Install sphinx-autobuild first
uv pip install sphinx-autobuild

# Then run
make livehtml
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── quickstart.rst       # Quick start guide
├── api/                 # API reference
│   ├── index.rst        # API documentation index
│   ├── core.rst         # Core data structures
│   ├── normalization.rst  # Normalization methods
│   ├── impute.rst       # Imputation methods
│   ├── integration.rst  # Batch correction methods
│   ├── dim_reduction.rst # Dimensionality reduction
│   ├── cluster.rst      # Clustering algorithms
│   ├── qc.rst           # Quality control
│   ├── viz.rst          # Visualization
│   └── utils.rst        # Utilities
├── design/              # Design documentation (not for API docs)
└── _build/              # Generated output (not in git)
```

## Adding Documentation

### For New Modules

1. Add a new RST file in `docs/api/`
2. Reference it in `docs/api/index.rst`
3. Use `automodule` directive:

```rst
.. automodule:: scptensor.your_module
   :members:
   :undoc-members:
   :show-inheritance:
```

### For New Functions/Classes

Add NumPy-style docstrings to your code:

```python
def your_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Longer description if needed.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> your_function("test", 5)
    True

    Notes
    -----
    Additional notes or implementation details.
    """
    pass
```

## Dependencies

Documentation dependencies are managed via `uv`:

```bash
# Install documentation dependencies
uv sync --group docs
```

See `pyproject.toml` `[dependency-groups.docs]` section for the full list.

## Theme

This documentation uses the `furo` theme, which is modern, clean, and responsive.

## Troubleshooting

### Import Errors

If you see import errors when building:

1. Make sure you've installed the package: `uv pip install -e .`
2. Make sure documentation dependencies are installed: `uv sync --group docs`
3. Check that the module path in the RST file matches the actual Python module

### Warnings

Some warnings are expected:
- Design docs in `docs/design/` may have formatting warnings (these are separate from API docs)
- Cross-reference warnings for missing modules (e.g., tsne if not implemented yet)

### Build Fails

If the build fails:

1. Clean the build directory: `rm -rf _build`
2. Check the error message carefully
3. Make sure all Python modules are importable
4. Verify docstring formatting (NumPy style)

## Output Formats

By default, HTML documentation is built. Other formats are available:

```bash
# PDF (requires LaTeX)
make latexpdf

# EPUB
make epub

# Plain text
make text
```

See `make help` for all available formats.
