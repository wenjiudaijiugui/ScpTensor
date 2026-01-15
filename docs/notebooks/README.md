# ScpTensor Tutorial Notebooks

This directory contains interactive Jupyter notebooks demonstrating the ScpTensor single-cell proteomics analysis framework.

## Prerequisites

### Installation

To run these notebooks, you need:

1. **Python 3.11+** with ScpTensor installed:
   ```bash
   # Clone the repository
   git clone https://github.com/your-org/ScpTensor.git
   cd ScpTensor

   # Install with uv (recommended)
   uv pip install -e .

   # Or with pip
   pip install -e .
   ```

2. **JupyterLab or Jupyter Notebook**:
   ```bash
   # Install JupyterLab
   pip install jupyterlab

   # Or classic Jupyter Notebook
   pip install notebook
   ```

3. **Required dependencies** (installed with ScpTensor):
   - numpy
   - polars
   - scipy
   - scikit-learn
   - matplotlib
   - scienceplots

## Running the Notebooks

### Option 1: JupyterLab (Recommended)

```bash
# From the project root
jupyter lab
```

Then navigate to `docs/notebooks/` and open a notebook.

### Option 2: Jupyter Notebook

```bash
# From the project root
jupyter notebook
```

### Option 3: VS Code

1. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
2. Open any `.ipynb` file in VS Code
3. Select a kernel (Python with ScpTensor installed)

## Available Tutorials

| Notebook | Topics Covered | Duration |
|----------|----------------|----------|
| **01_basic_workflow.ipynb** | Data loading, QC, normalization, imputation, PCA, clustering, visualization | 20 min |
| **02_batch_correction.ipynb** | Batch effect detection, ComBat correction, integration verification | 25 min |

### Notebook Details

#### 01_basic_workflow.ipynb
A complete end-to-end analysis pipeline:
- Generate synthetic SCP data with realistic missing patterns
- Quality control with completeness visualization
- Log normalization for variance stabilization
- KNN imputation for missing values
- PCA for dimensionality reduction
- K-means clustering
- Publication-ready visualizations

**Best for:** New users learning ScpTensor basics

#### 02_batch_correction.ipynb
Advanced workflow for handling batch effects:
- Generate data with strong batch effects
- Detect and visualize batch effects
- Apply ComBat empirical Bayes correction
- Verify integration quality
- Before/after comparison visualizations

**Best for:** Users working with multi-batch data or integrating multiple datasets

## Notebook Tips

1. **Run cells sequentially**: Each notebook builds on previous cells
2. **Expected outputs**: Descriptions of expected outputs are provided
3. **Synthetic data**: Notebooks generate reproducible synthetic data - no external files needed
4. **Publication-ready plots**: All figures use SciencePlots style with 300 DPI

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'scptensor'`:
```bash
# Reinstall ScpTensor in development mode
pip install -e .
```

### Missing SciencePlots

If you see a style error:
```bash
pip install scienceplots
```

### Kernel Issues

If Jupyter can't find the Python kernel:
```bash
# Install ipykernel
pip install ipykernel

# Register your environment
python -m ipykernel install --user --name=scptensor
```

## Additional Resources

- **API Documentation**: [API Reference](../api/)
- **Design Documents**: [Design Docs](../design/)
- **GitHub Repository**: [ScpTensor GitHub](https://github.com/your-org/ScpTensor)

## Citation

If you use ScpTensor in your research, please cite:
```
ScpTensor: A Framework for Single-Cell Proteomics Analysis
[Authors et al., Year]
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the [Design Documents](../design/) for architecture details
- See [ISSUES_AND_LIMITATIONS.md](../ISSUES_AND_LIMITATIONS.md) for known issues
