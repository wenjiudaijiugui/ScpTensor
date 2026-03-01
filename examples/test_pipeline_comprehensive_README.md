# ScpTensor Comprehensive Pipeline Test Script

## Overview

This script performs sequential testing of all major ScpTensor modules:
1. **Normalization** (4 methods): log_transform, norm_mean, norm_median, norm_quantile
2. **Imputation** (6 methods): knn, bpca, mf, lls, qrilc, minprob
3. **Integration** (5 methods): combat, harmony, mnn, scanorama, nonlinear
4. **Dimensionality Reduction** (2 methods): pca, umap
5. **Clustering** (2 methods): kmeans, leiden

## Usage

```bash
# Run with default settings (uses real data if available, otherwise synthetic)
uv run python examples/test_pipeline_comprehensive.py

# Run with synthetic data only
python examples/test_pipeline_comprehensive.py --synthetic-only
```

## Output

All test results are saved to `tmp/pipeline_test_report/pipeline_test_report.md` with:
- Test summary statistics
- Module-level breakdown
- Individual test results with status (PASS/FAIL/SKIP)
- Duration and key metrics for each test
- Error messages for failed/skipped tests

## Test Status

| Status | Description |
|--------|-------------|
| ✅ PASS | Test completed successfully |
| ❌ FAIL | Test failed with error |
| ⚠️ SKIP | Test skipped due to missing dependencies |

## Dependencies

Some tests require optional dependencies:
- `harmonypy` for Harmony integration
- `scanorama` for Scanorama integration
- `igraph` and `leidenalg` for Leiden clustering

Install with:
```bash
pip install harmonypy scanorama igraph leidenalg
```

## Data Loading

The script automatically:
1. Tries to load real data from `data/dia/diann/PXD054343/1_SC_LF_report.tsv`
2. Falls back to synthetic data generation if real data is not available
3. Adds synthetic batch information for integration testing

## Example Report

```markdown
# ScpTensor 综合管道测试报告

## 测试概要

| 指标 | 数值 |
|------|------|
| 总测试数 | 19 |
| 通过 | 15 |
| 失败 | 0 |
| 跳过 | 4 |
| 通过率 | 78.9% |
| 总耗时 | 7.28 秒 |

## 模块汇总

| 模块 | 通过 | 失败 | 跳过 | 耗时 (秒) |
|------|------|------|------|----------|
| normalization | 4 | 0 | 0 | 0.12 |
| imputation | 6 | 0 | 0 | 0.00 |
| integration | 2 | 0 | 3 | 0.13 |
| dim_reduction | 2 | 0 | 0 | 5.36 |
| clustering | 1 | 0 | 1 | 0.00 |
```

## Notes

- Tests are run sequentially, with each test updating the container state
- PCA components are automatically adjusted based on data size
- Clustering uses PCA results if available, otherwise uses original data
- Missing value rates are tracked for imputation tests
