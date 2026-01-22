# Accuracy Assessment Report

Generated: 2026-01-18 22:23:15

---

## Method-by-Method Accuracy Analysis

### PCA

| Metric | Value |
|--------|-------|
| MSE | 471.013285 |
| MAE | 11.431066 |
| Correlation | 0.5997 |
| Relative Error | 5.6110 |

**Assessment**: **Fair** - Moderate agreement, some differences expected

### K-Means

| Metric | Value |
|--------|-------|
| MSE | 7.571000 |
| MAE | 2.657000 |
| Correlation | -0.4184 |
| Relative Error | 920000002.3959 |

**Assessment**: **Poor** - Low correlation, implementations differ significantly

### Z-Score Normalize

| Metric | Value |
|--------|-------|
| MSE | 1.139790 |
| MAE | 0.808347 |
| Correlation | 0.0000 |
| Relative Error | 0.8083 |

**Assessment**: **Poor** - Low correlation, implementations differ significantly

---

## Overall Summary

- **Average Correlation**: 0.6727
- **High Agreement Methods**: 1/3 (correlation ≥ 0.95)

**Conclusion**: ScpTensor and Scanpy show **moderate agreement**; differences may be due to algorithmic variations.
