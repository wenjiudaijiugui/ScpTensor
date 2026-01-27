# Visualization Module - Quick Start Guide

## Installation

Dependencies are already installed in the project environment:

```bash
# Already installed via project setup
# matplotlib, numpy, pandas, scienceplots
```

## Basic Usage

### 1. Import the Module

```python
from docs.comparison_study.visualization import (
    ComparisonPlotter,
    generate_all_figures,
    ReportGenerator,
    calculate_overall_scores,
)
```

### 2. Load Your Configuration

```python
import yaml

with open("docs/comparison_study/configs/evaluation_config.yaml") as f:
    config = yaml.safe_load(f)
```

### 3. Load Your Results

Results should follow this structure:

```python
results = {
    "small_pipeline_a": {
        "batch_effects": {"kbet": 0.85, "lisi": 0.78, ...},
        "performance": {"runtime_seconds": 120.5, "memory_gb": 4.2},
        "distribution": {"sparsity_change": 0.05, "mean_change": 0.12, ...},
        "structure": {"pca_variance_cumulative": 0.75, "nn_consistency": 0.88, ...}
    },
    # ... more dataset_pipeline combinations
}
```

### 4. Generate Individual Figures

```python
# Initialize plotter
plotter = ComparisonPlotter(
    config,
    output_dir="outputs/figures",
    dpi=300  # publication quality
)

# Generate specific figures
batch_effects_path = plotter.plot_batch_effects_comparison(results)
performance_path = plotter.plot_performance_comparison(results)
distribution_path = plotter.plot_distribution_comparison(results)
structure_path = plotter.plot_structure_preservation(results)
radar_path = plotter.plot_comprehensive_radar(results)

print(f"Figures saved to: {batch_effects_path}, {performance_path}, ...")
```

### 5. Generate All Figures at Once

```python
# Generate all 6 figures
figures = generate_all_figures(
    results,
    config,
    output_dir="outputs/figures"
)

print(f"Generated {len(figures)} figures:")
for fig_path in figures:
    print(f"  - {fig_path}")
```

### 6. Calculate Pipeline Rankings

```python
# Calculate weighted scores
scores = calculate_overall_scores(results, config)

# Display rankings
print("\nPipeline Rankings:")
for pipeline, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    grade = "A" if score >= 80 else "B" if score >= 60 else "C"
    print(f"  {pipeline}: {score:.1f}/100 (Grade {grade})")

# Generate ranking plot
ranking_path = plotter.plot_ranking_barplot(scores)
print(f"\nRanking plot: {ranking_path}")
```

### 7. Generate Complete Report

```python
# Initialize report generator
generator = ReportGenerator(
    config,
    output_dir="outputs"
)

# Generate comprehensive report
pdf_path = generator.generate_report(
    results,
    figures,
    save_path="outputs/comparison_report.pdf"
)

print(f"\nReport generated: {pdf_path}")
print("Note: Markdown saved to outputs/report.md")
print("Convert to PDF: pandoc outputs/report.md -o outputs/comparison_report.pdf")
```

## Complete Example

```python
"""Complete example of visualization module usage."""

import yaml
from docs.comparison_study.visualization import (
    generate_all_figures,
    ReportGenerator,
)

# Load configuration
with open("docs/comparison_study/configs/evaluation_config.yaml") as f:
    config = yaml.safe_load(f)

# Load results (from your evaluation module)
# results = load_evaluation_results("outputs/results.pkl")

# Generate all figures
figures = generate_all_figures(results, config, output_dir="outputs/figures")
print(f"Generated {len(figures)} figures")

# Generate report
generator = ReportGenerator(config, output_dir="outputs")
pdf_path = generator.generate_report(results, figures)
print(f"Report: {pdf_path}")
```

## Customization

### Change Output Directory

```python
plotter = ComparisonPlotter(
    config,
    output_dir="custom_output_path"
)
```

### Adjust DPI

```python
plotter = ComparisonPlotter(
    config,
    output_dir="outputs/figures",
    dpi=600  # Higher quality for print
)
```

### Custom Color Scheme

Edit `configs/evaluation_config.yaml`:

```yaml
visualization:
  figure:
    colors:
      pipeline_a: "#FF5733"
      pipeline_b: "#33FF57"
      # ... etc
```

### Custom Font Settings

Edit `configs/evaluation_config.yaml`:

```yaml
visualization:
  font:
    family: "Times New Roman"
    title_size: 18
    label_size: 14
    legend_size: 12
```

## Testing the Module

Run the test suite:

```bash
cd /home/shenshang/projects/ScpTensor
uv run python docs/comparison_study/visualization/test_visualization.py
```

This will:
- Create synthetic evaluation results
- Test all plotting functions
- Test score calculation
- Test report generation
- Save outputs to `test_outputs/`

## Troubleshooting

### Issue: SciencePlots style not found

**Solution**: The module automatically falls back to default style if SciencePlots is not installed.

```python
# No action needed - automatic fallback
```

### Issue: Figures not saving

**Solution**: Check that output directory exists and is writable.

```python
from pathlib import Path
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
```

### Issue: Missing data in results

**Solution**: The module handles missing data gracefully. Check your results structure matches the expected format.

### Issue: Type checking errors

**Solution**: Ensure you're using Python 3.12+ with latest type stubs.

```bash
uv run mypy docs/comparison_study/visualization/
```

## Advanced Usage

### Custom Plotting

```python
# Extend ComparisonPlotter for custom plots
class CustomPlotter(ComparisonPlotter):
    def plot_custom_metric(self, results, save_path=None):
        """Your custom plotting logic."""
        fig, ax = plt.subplots(figsize=(10, 6))
        # ... your plotting code ...
        plt.savefig(save_path or self.output_dir / "custom.png", dpi=self.dpi)
        plt.close()
        return str(save_path)
```

### Batch Processing

```python
# Process multiple result sets
result_sets = ["experiment1", "experiment2", "experiment3"]

for exp_name in result_sets:
    results = load_results(f"outputs/{exp_name}/results.pkl")
    figures = generate_all_figures(
        results,
        config,
        output_dir=f"outputs/{exp_name}/figures"
    )
    print(f"{exp_name}: {len(figures)} figures generated")
```

## Output Files

After running the visualization module, you'll have:

```
outputs/
├── figures/
│   ├── batch_effects_comparison.png
│   ├── performance_comparison.png
│   ├── distribution_comparison.png
│   ├── structure_preservation.png
│   ├── comprehensive_radar.png
│   └── ranking_barplot.png
└── report.md  # Convert to PDF with pandoc
```

## Next Steps

1. Run your pipeline evaluation
2. Load results into the visualization module
3. Generate all figures
4. Create comprehensive report
5. Convert report to PDF (optional)
6. Incorporate figures into your manuscript/presentation

For more details, see:
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation details
- Test suite in `test_visualization.py` - Working examples
- Configuration in `configs/evaluation_config.yaml` - Customization options
