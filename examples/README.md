# ScpTensor Examples

This directory contains example scripts demonstrating various ScpTensor workflows and analyses.

## Available Scripts

### MVP Analysis Script (`mvp_analysis.py`)

A comprehensive analysis pipeline demonstrating the complete ScpTensor workflow from data loading through dimensionality reduction and visualization.

**Features:**
- Data loading from multiple formats (DIA-NN BGS, DIA-NN TSV, DIA-NN Parquet, Spectronaut)
- Dataset summary and statistics
- Logarithmic transformation
- Quality control visualizations
- Principal Component Analysis (PCA)
- Result export to NPZ format
- Analysis summary report generation

**Usage:**
```bash
# Basic analysis with DIA-NN BGS format
python examples/mvp_analysis.py --data scptensor/datasets/pride/PXD064564/report.tsv --format diann-bgs

# Custom output directory with verbose logging
python examples/mvp_analysis.py --data data/report.tsv --format diann-parquet --output my_analysis --verbose

# Skip visualization for faster processing
python examples/mvp_analysis.py --data data/report.tsv --format diann-tsv --no-viz

# Custom transformation parameters
python examples/mvp_analysis.py --data data/report.tsv --format diann-bgs --log-base 2.0 --n-components 20
```

**Arguments:**
- `--data`: Path to input data file (required)
- `--format`: Data format (choices: diann-bgs, diann-tsv, diann-parquet, spectronaut-tsv, default: diann-bgs)
- `--assay-name`: Name for the assay (default: proteins)
- `--output`: Output directory for results (default: mvp_results)
- `--log-base`: Base for logarithmic transformation (default: 2.0)
- `--log-offset`: Offset for logarithmic transformation (default: 1.0)
- `--n-components`: Number of PCA components (default: 15)
- `--no-viz`: Skip visualization generation
- `--verbose`: Enable verbose output

**Output Files:**
- `container.npz`: Processed data container with all analysis results
- `analysis_summary.txt`: Text summary of the analysis
- `qc_missing_heatmap.png`: Missing values visualization
- `pca_embedding.png`: PCA scatter plot

## Example Datasets

ScpTensor includes example datasets from PRIDE (ProteomeXchange) for testing and demonstration:

| Dataset ID | Description | Format |
|------------|-------------|--------|
| PXD064564 | Single-cell hPSC proteomics (33 samples) | DIA-NN BGS |
| PXD049412 | HeLa single-cell proteomics (3 samples) | DIA-NN BGS |
| PXD049211 | SCP-EvoCHIP HeLa cells | Spectronaut long format |

These datasets are located in `scptensor/datasets/pride/`.

## Getting Started

1. **Install ScpTensor:**
   ```bash
   pip install scptensor
   # or with uv
   uv pip install scptensor
   ```

2. **Run the MVP analysis:**
   ```bash
   python examples/mvp_analysis.py --data scptensor/datasets/pride/PXD064564/report.tsv --format diann-bgs
   ```

3. **Check the results:**
   ```bash
   ls mvp_results/
   cat mvp_results/analysis_summary.txt
   ```

## Advanced Usage

### Custom Analysis Pipeline

```python
from scptensor import (
    read_pivot_report,
    log_transform,
    reduce_pca,
    cluster_kmeans,
    save_npz
)

# Load data
container = read_pivot_report("data/report.tsv", assay_name="proteins")

# Log transformation
container = log_transform(container, assay_name="proteins", source_layer="X")

# PCA
container = reduce_pca(container, assay_name="proteins", base_layer="log", n_components=15)

# Clustering
container = cluster_kmeans(container, n_clusters=5)

# Save results
save_npz(container, "results.npz")
```

## Additional Resources

- **Documentation**: [ScpTensor Docs](https://github.com/your-org/scptensor)
- **API Reference**: See individual module documentation
- **Tutorials**: Coming soon

## Contributing Examples

To contribute new examples:
1. Add your script to this directory
2. Follow the existing code style (PEP 8, type hints, docstrings)
3. Include usage examples in help text
4. Test with provided example datasets
5. Update this README with your script description

## License

These examples are part of ScpTensor and follow the same license terms.
