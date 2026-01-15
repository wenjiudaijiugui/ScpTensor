# ScpTensor Visualization Report Design

**Date:** 2026-01-15
**Version:** 1.0
**Status:** ✅ **COMPLETED** (2026-01-15)
**Target Release:** v0.2.0

---

## Implementation Summary

**All features successfully implemented!**

- ✅ ReportTheme dataclass with presets (default, dark, colorblind)
- ✅ generate_analysis_report() main function
- ✅ 8 panel rendering functions (overview, QC, missing, embedding, feature, cluster, batch, DE)
- ✅ 13 tests passing
- ✅ Full integration with scptensor.viz module

**Files Created:**
- `scptensor/viz/recipes/report.py` (320 lines, 82% coverage)
- `tests/test_viz_report.py` (13 tests)

**Commits:**
- `3043671` feat(viz): add ReportTheme dataclass with presets
- `66e954e` feat(viz): add generate_analysis_report skeleton function
- `c93c793` feat(viz): implement overview panel with summary table
- `38706bd` feat(viz): implement remaining report panels
- `de1327a` docs(viz): add exports to viz module and update ROADMAP

---

## Overview

Design a single-page comprehensive analysis report generator for ScpTensor. The report will cover the entire analysis workflow from QC to differential expression, enabling users to quickly understand their single-cell proteomics data.

---

## Layout Structure

### Grid Layout (3×3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      ScpTensor Analysis Report                   │
│                    [Dataset: XXX | Date: XXX]                    │
├──────────────────────┬──────────────────────┬───────────────────┤
│   [1] 数据概览        │   [2] QC指标分布      │   [3] 缺失率热图    │
│   样本/特征统计表     │   检测数小提琴图      │   样本×特征热图    │
├──────────────────────┴──────────────────────┴───────────────────┤
│                 [4] 降维嵌入分析 (占宽2列)                        │
│                 PCA + UMAP 并排散点图                            │
├──────────────┬──────────────────────────────┬──────────────────┤
│ [5] 特征统计  │        [6] 聚类分析         │  [7] 批次效应      │
│ 方差/均值散点 │        聚类热图             │  PCA按批次着色    │
├──────────────┴──────────────────────────────┴───────────────────┤
│                 [8] 差异表达分析 (占宽3列)                        │
│                 火山图 + Top10 差异特征表                         │
└─────────────────────────────────────────────────────────────────┘
```

**Specifications:**
- Implementation: `matplotlib.gridspec.GridSpec`
- Default size: 16×12 inches
- DPI: 300 (publication quality)
- Configurable panel visibility

---

## Panel Details

### Panel [1]: Data Overview

Text-based summary table displaying:
- Number of samples
- Number of features
- Missing rate percentage
- Number of batches
- Group information

**Implementation:** `plt.table()` with centered alignment

---

### Panel [2]: QC Distribution

Violin plot showing detection distribution by batch.

**Features:**
- Reuses `qc_completeness()` logic
- Jittered points for individual samples
- Median and IQR annotations

---

### Panel [3]: Missing Rate Heatmap

Sparse heatmap showing missing rates (samples × features).

**Features:**
- Reuses `qc_matrix_spy()` logic
- Displays top 100 high-variance features
- Clustered rows and columns
- Color scale: White (0%) → Red (100%)

---

### Panel [4]: Dimensionality Reduction

Side-by-side PCA and UMAP scatter plots.

**Features:**
- Reuses `embedding()` function
- PCA shows variance explained
- UMAP optional contour lines
- Configurable color grouping

---

### Panel [5]: Feature Statistics

Mean vs Variance scatter plot with HVG highlighting.

**Features:**
- HVGs highlighted in red
- LOESS fitting curve
- R² statistic displayed

---

### Panel [6]: Cluster Analysis

Heatmap showing cluster centers.

**Features:**
- Reuses existing heatmap functions
- Hierarchical clustering dendrograms
- Side bars for true labels and cluster assignments

---

### Panel [7]: Batch Effect

PCA colored by batch to assess batch effect strength.

**Features:**
- Optional before/after comparison
- R²/PC1 vs Batch metrics

---

### Panel [8]: Differential Expression

Combined volcano plot, top features table, and effect size distribution.

**Features:**
- Reuses `volcano()` function
- Left: Volcano scatter plot
- Right: Top 10 DE features table
- Bottom: Cohen's d distribution histogram

---

## API Design

### Main Function

```python
def generate_analysis_report(
    container: ScpContainer,
    assay_name: str = "proteins",
    group_col: str = "group",
    batch_col: str | None = "batch",
    diff_expr_groups: tuple[str, str] | None = None,
    output_path: str | None = None,
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 300,
    style: str = "science",
    panels: list[str] | None = None,
    theme: ReportTheme | None = None,
) -> plt.Figure:
    """Generate a comprehensive analysis report as a single-page figure."""
```

### Module Organization

```
scptensor/viz/
├── recipes/
│   └── report.py          # New: Report generation functions
└── base/
    └── __init__.py         # Update: Export report functions
```

### Usage Example

```python
from scptensor.viz import generate_analysis_report

# Basic usage
fig = generate_analysis_report(container)
fig.savefig("report.png")

# Full configuration
fig = generate_analysis_report(
    container=container,
    assay_name="proteins",
    group_col="treatment",
    batch_col="batch",
    diff_expr_groups=("Control", "Treated"),
    output_path="analysis_report.pdf",
    theme=ReportTheme.dark(),
)
```

---

## Theme System

### Theme Configuration

```python
@dataclass
class ReportTheme:
    # Layout
    figsize: tuple[float, float] = (16, 12)
    dpi: int = 300
    panel_spacing: float = 0.3

    # Colors
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    success_color: str = "#2ca02c"
    danger_color: str = "#d62728"
    neutral_color: str = "#7f7f7f"

    # Fonts
    title_fontsize: int = 14
    label_fontsize: int = 10
    tick_fontsize: int = 8

    # Elements
    linewidth: float = 1.0
    marker_size: float = 20
    alpha: float = 0.7

    # Colormaps
    cmap_missing: str = "Reds"
    cmap_cluster: str = "viridis"
```

### Preset Themes

| Theme | Use Case | Description |
|-------|----------|-------------|
| `science` | Publication | SciencePlots style, B&W compatible |
| `dark` | Presentation | Dark background, high contrast |
| `colorblind` | Accessibility | Colorblind-friendly palette |
| `nature` | Natural | Earth tones, soft colors |
| `vibrant` | Screen viewing | High saturation colors |

---

## Error Handling

| Error Case | Handling |
|------------|----------|
| Assay not found | Raise `ValueError` |
| group_col not found | Fallback to "All" group |
| Empty data | Raise `ValidationError` |
| diff_expr_groups not specified | Skip DE panel with placeholder |
| Panel generation fails | Show error message in panel, continue |

---

## Implementation Plan

1. **Phase 1:** Core structure and grid layout
2. **Phase 2:** Individual panel rendering functions
3. **Phase 3:** Theme system integration
4. **Phase 4:** Testing and documentation

**Estimated Effort:** 8-12 person-days

---

## Dependencies

- Existing: `matplotlib`, `numpy`, `polars`, `scipy`
- New: None (reuses existing visualization functions)

---

## Open Questions

1. Should we support multi-page PDF reports as an optional format?
2. Should the report include statistical test results (ANOVA p-values, etc.)?
3. Should we add a "summary statistics" text block at the bottom?
