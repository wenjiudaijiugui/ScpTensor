# ScpTensor Tutorials

This directory contains interactive Jupyter notebook tutorials for learning ScpTensor - a Python library for single-cell proteomics (SCP) data analysis.

## Tutorial Overview

| Tutorial | Topics Covered | Prerequisites | Duration |
|----------|---------------|---------------|----------|
| **Tutorial 01: Getting Started** | Data loading, inspection, basic operations | Basic Python | 20 min |
| **Tutorial 02: QC and Normalization** | Quality control, filtering, normalization | Tutorial 01 | 30 min |
| **Tutorial 03: Imputation and Integration** | Missing value imputation, batch correction | Tutorial 02 | 40 min |
| **Tutorial 04: Clustering and Visualization** | Dimensionality reduction, clustering, plotting | Tutorial 03 | 40 min |

## Tutorial Details

### Tutorial 01: Getting Started with ScpTensor

**What you'll learn:**
- The ScpTensor data structure (`ScpContainer` -> `Assay` -> `ScpMatrix`)
- Loading built-in example datasets
- Inspecting data and metadata
- Understanding the mask code system for tracking missing values
- Saving and loading data

**Key functions:**
- `load_toy_example()`, `load_simulated_scrnaseq_like()`
- `ScpContainer`, `Assay`, `ScpMatrix`
- `count_mask_codes()`, `save_csv()`, `save_npz()`

---

### Tutorial 02: Quality Control and Normalization

**What you'll learn:**
- Calculating and visualizing QC metrics
- Detecting and filtering low-quality samples and features
- Applying various normalization methods
- Comparing normalization approaches

**Key functions:**
- `calculate_qc_metrics()`, `detect_outliers()`
- `filter_samples_by_missing_rate()`, `filter_features_by_variance()`
- `log_normalize()`, `sample_median_normalization()`, `tmm_normalization()`
- `zscore()`, `upper_quartile_normalization()`

---

### Tutorial 03: Imputation and Batch Integration

**What you'll learn:**
- Understanding missing value patterns (MCAR vs MNAR)
- Applying various imputation methods (KNN, PPCA, SVD, MissForest)
- Detecting and assessing batch effects
- Applying batch correction methods (ComBat, Harmony, MNN)

**Key functions:**
- `knn()`, `ppca()`, `svd_impute()`, `missforest()`
- `combat()`, `harmony()`, `mnn_correct()`
- `pca()` for batch effect visualization

---

### Tutorial 04: Clustering and Visualization

**What you'll learn:**
- Dimensionality reduction (PCA, UMAP)
- Running clustering algorithms (K-Means)
- Visualizing results with publication-quality plots
- Evaluating clustering quality

**Key functions:**
- `pca()`, `umap()`
- `run_kmeans()`
- `embedding()`, `heatmap()`, `violin()`, `qc_completeness()`

---

## Getting Started

### Prerequisites

Ensure you have ScpTensor installed with all dependencies:

```bash
# Install ScpTensor
pip install scptensor

# Or install with development dependencies
pip install scptensor[dev]
```

### Running the Tutorials

1. **Start Jupyter** in the tutorials directory:

```bash
cd docs/tutorials
jupyter notebook
```

2. **Open a tutorial** (start with `tutorial_01_getting_started.ipynb`)

3. **Run cells** sequentially (Shift+Enter or Run button)

### Alternative: Using JupyterLab

```bash
jupyter lab
```

### Alternative: Using VS Code

1. Open the tutorial notebook in VS Code
2. Ensure the Jupyter extension is installed
3. Select a Python kernel with ScpTensor installed

---

## Example Dataset Sizes

The tutorials use built-in synthetic datasets with the following characteristics:

| Dataset | Samples | Features | Batches | Cell Types | Missing Rate |
|---------|---------|----------|---------|------------|--------------|
| `load_toy_example()` | 100 | 50 | 3 | 3 | 20% |
| `load_simulated_scrnaseq_like()` | 500 | 200 | 3 | 4 | 30% |
| `load_example_with_clusters()` | 300 | 100 | 3 | 5 | 25% |

---

## Expected Output

Each tutorial saves visualization outputs to the `tutorial_output/` directory (created automatically). Plots are saved as PNG files at 300 DPI for publication quality.

---

## Common Issues

### Issue: ModuleNotFoundError

**Solution:** Ensure ScpTensor is installed in your active Python environment:

```bash
pip install scptensor
```

### Issue: SciencePlots style not found

**Solution:** Install SciencePlots:

```bash
pip install SciencePlots
```

### Issue: Memory errors with large datasets

**Solution:** Use the `load_toy_example()` dataset which is smaller, or reduce the number of features/samples in your analysis.

---

## Additional Resources

- **API Reference**: `../design/API_REFERENCE.md`
- **Architecture Guide**: `../design/ARCHITECTURE.md`
- **GitHub Repository**: [Link to repo]
- **Issue Tracker**: [Link to issues]

---

## Citation

If you use ScpTensor in your research, please cite:

```bibtex
@software{scptensor2024,
  title = {ScpTensor: Single-Cell Proteomics Analysis Framework},
  author = {ScpTensor Team},
  year = {2024},
  version = {0.1.0-alpha},
  url = {https://github.com/your-org/scptensor}
}
```

---

## Feedback and Contributions

We welcome feedback, bug reports, and contributions! Please see the main repository's CONTRIBUTING.md for guidelines.

---

**Last Updated:** 2025-01-14
**Tutorials Version:** 0.1.0-alpha
