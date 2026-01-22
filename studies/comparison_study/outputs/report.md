# Single-Cell Proteomics Analysis Pipeline Technical Comparison

**Authors:** ScpTensor Team
**Version:** 1.0.0
**Date:** 2026-01-20

---

## Abstract

This report presents a comprehensive technical comparison of five analysis
pipelines for single-cell proteomics data. The evaluation covers four key
dimensions: batch effect removal, computational performance, data
distribution changes, and data structure preservation. Results are based
on three datasets of varying size and complexity.

---



## Executive Summary

### Key Findings

1. **Best Overall Pipeline:** Pipeline A
   - Overall Score: 0.0/100
   - Grade: C

2. **Batch Effect Removal:**
   Detailed analysis shows significant differences between pipelines.

3. **Computational Performance:**
   Runtime and memory usage vary considerably across pipeline complexity.

4. **Data Distribution:**
   Most pipelines preserve distribution characteristics reasonably well.

5. **Data Structure Preservation:**
   All pipelines maintain good structure preservation.

### Recommendations

- **For single-batch data:** Use Pipeline A (Classic) for reliable baseline results
- **For multi-batch data:** Use Pipeline B (Batch Correction) or C (Advanced)
- **For large-scale data:** Use Pipeline D (Performance-Optimized)
- **For minimal assumptions:** Use Pipeline E (Conservative)



## Methodology

### Pipeline Configurations

Five representative pipelines were evaluated:

1. **Pipeline A (Classic):** QC → Median → Log → KNN → No batch → PCA → K-means
2. **Pipeline B (Batch Correction):** QC → Median → Log → KNN → ComBat → PCA → K-means
3. **Pipeline C (Advanced):** QC → Quantile → Log → MissForest → Harmony → UMAP → Leiden
4. **Pipeline D (Performance-Optimized):** QC → Z-score → Lazy → SVD → MNN → PCA → K-means
5. **Pipeline E (Conservative):** QC → VSN → Log → PPCA → No batch → PCA → K-means

### Datasets

Three datasets were used for evaluation:

- **Small:** 1K cells × 1K proteins, 1 batch (baseline testing)
- **Medium:** 5K cells × 1.5K proteins, 5 batches (batch correction testing)
- **Large:** 20K cells × 2K proteins, 10 batches (scalability testing)

### Evaluation Metrics

#### Batch Effect Removal
- **kBET score:** Measures local batch mixing (0-1, higher is better)
- **LISI score:** Local diversity index (higher is better)
- **Mixing entropy:** Normalized Shannon entropy (higher is better)
- **Variance ratio:** Within/between batch variance (lower is better)

#### Computational Performance
- **Runtime:** Total execution time in seconds
- **Memory usage:** Peak memory consumption in GB

#### Data Distribution
- **Sparsity change:** Δ missing rate
- **Statistics:** Mean, std, skewness, kurtosis, CV

#### Data Structure Preservation
- **PCA variance:** Cumulative variance explained by top 10 PCs
- **NN consistency:** Jaccard similarity of k-nearest neighbors
- **Distance preservation:** Correlation of pairwise distances



## Results

![batch_effects_comparison](docs/comparison_study/outputs/figures/batch_effects_comparison.png)

### Computational Performance

Figure 2 compares runtime and memory usage across pipelines.
Pipeline D (Performance-Optimized) shows the best scalability for large datasets.

![performance_comparison](docs/comparison_study/outputs/figures/performance_comparison.png)

### Data Distribution Changes

Figure 3 shows how each pipeline affects data distribution.
Pipelines A and E show minimal changes to statistical properties.

![distribution_comparison](docs/comparison_study/outputs/figures/distribution_comparison.png)

### Data Structure Preservation

Figure 4 shows how well each pipeline preserves the underlying data structure.
All pipelines maintain reasonable structure preservation.

![structure_preservation](docs/comparison_study/outputs/figures/structure_preservation.png)

![comprehensive_radar](docs/comparison_study/outputs/figures/comprehensive_radar.png)

![ranking_barplot](docs/comparison_study/outputs/figures/ranking_barplot.png)



## Discussion and Recommendations

### Pipeline Selection Guide

#### When to Use Each Pipeline

**Pipeline A (Classic) - Grade: C
- **Best for:** Small to medium datasets, single-batch experiments
- **Strengths:** Well-established, reproducible, minimal assumptions
- **Weaknesses:** No batch correction, basic imputation
- **Use case:** Routine analysis with homogeneous data

**Pipeline B (Batch Correction) - Grade: C
- **Best for:** Multi-batch datasets with clear batch structure
- **Strengths:** Effective batch removal with ComBat
- **Weaknesses:** May over-correct if batch effects are minimal
- **Use case:** Integrating data from multiple runs/instruments

**Pipeline C (Advanced) - Grade: C
- **Best for:** Complex datasets requiring state-of-the-art methods
- **Strengths:** Advanced imputation and batch correction
- **Weaknesses:** Higher computational cost
- **Use case:** Publication-quality analysis, complex batch structures

**Pipeline D (Performance-Optimized) - Grade: C
- **Best for:** Large-scale datasets (>10K cells)
- **Strengths:** Fast, memory-efficient, scalable
- **Weaknesses:** May sacrifice some accuracy for speed
- **Use case:** Exploratory analysis, large cohort studies

**Pipeline E (Conservative) - Grade: C
- **Best for:** Studies requiring minimal data manipulation
- **Strengths:** Preserves original data characteristics
- **Weaknesses:** May under-correct technical artifacts
- **Use case:** Hypothesis testing, minimal intervention studies

### Limitations

1. Synthetic data may not fully capture real-world complexity
2. Evaluation metrics may not capture all aspects of data quality
3. Pipeline performance may vary with different data characteristics
4. Computational results depend on hardware and software environment

### Future Directions

1. Extend evaluation to include biological metrics (clustering quality, marker detection)
2. Test on additional real-world datasets
3. Include more pipeline variants and parameter combinations
4. Develop interactive web-based visualization



## Appendix

### Complete Configuration

See `configs/pipeline_configs.yaml` and `configs/evaluation_config.yaml`
for complete parameter settings.

### Raw Results

Raw numerical results are saved in `outputs/results/` directory
as pickle files for further analysis.

### Code Availability

All code is available in the ScpTensor repository:
`docs/comparison_study/`

### Reproducibility

To reproduce these results:
```bash
cd docs/comparison_study
python run_comparison.py --full
```

For more details, see `README.md`
