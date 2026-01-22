# Pipeline Comparison Study Runner

Main execution script for running comprehensive single-cell proteomics analysis pipeline comparison experiments.

## Overview

The `run_comparison.py` script executes the complete pipeline comparison study, including:
- Loading or generating synthetic datasets
- Running multiple analysis pipelines
- Evaluating performance across four dimensions
- Generating publication-quality visualizations
- Creating comprehensive comparison reports

## Quick Start

### Test Mode (Quick Test)
```bash
# Quick test with small dataset (recommended for first run)
python docs/comparison_study/run_comparison.py --test --verbose
```

### Default Mode (Medium Dataset)
```bash
# Run with medium dataset, single repeat
python docs/comparison_study/run_comparison.py --verbose
```

### Full Experiment
```bash
# Complete experiment with all datasets and multiple repeats
python docs/comparison_study/run_comparison.py --full --repeats 3 --verbose
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--full` | Run complete experiment (all datasets, multiple repeats) | False |
| `--test` | Quick test mode (small dataset only) | False |
| `--config PATH` | Path to custom configuration file | None (use defaults) |
| `--output DIR` | Output directory for results | `docs/comparison_study/outputs` |
| `--no-cache` | Regenerate datasets even if cached | False |
| `--repeats N` | Number of repeats per experiment | 3 |
| `--verbose` | Enable verbose output | False |

## Usage Examples

### Basic Usage

```bash
# Run with default settings (medium dataset, 1 repeat)
python docs/comparison_study/run_comparison.py
```

### Testing and Debugging

```bash
# Quick test to verify setup
python docs/comparison_study/run_comparison.py --test --verbose

# Test with custom output directory
python docs/comparison_study/run_comparison.py --test --output /tmp/test_output
```

### Production Runs

```bash
# Full experiment with default 3 repeats
python docs/comparison_study/run_comparison.py --full --verbose

# Full experiment with 5 repeats for more robust statistics
python docs/comparison_study/run_comparison.py --full --repeats 5 --verbose

# Full experiment with custom configuration
python docs/comparison_study/run_comparison.py --full --config my_config.yaml --verbose
```

### Data Management

```bash
# Regenerate datasets (ignore cache)
python docs/comparison_study/run_comparison.py --full --no-cache --verbose

# Use custom output directory
python docs/comparison_study/run_comparison.py --output /path/to/results --verbose
```

## Output Structure

After running, the output directory will contain:

```
outputs/
├── data_cache/              # Cached datasets (for fast re-runs)
│   ├── small.pkl
│   ├── medium.pkl
│   └── large.pkl
├── results/                 # Individual experiment results
│   ├── small_pipeline_a_r0.pkl
│   ├── small_pipeline_a_r1.pkl
│   ├── small_pipeline_b_r0.pkl
│   ├── ...
│   └── complete_results.pkl  # Complete results dictionary
├── figures/                 # Generated visualizations (300 DPI)
│   ├── batch_effects_comparison.png
│   ├── performance_comparison.png
│   ├── distribution_comparison.png
│   ├── structure_comparison.png
│   ├── ranking_barplot.png
│   └── radar_plot.png
├── report.md                # Markdown report
└── report.pdf               # PDF report (requires manual conversion)
```

## Experiment Modes

### Test Mode (`--test`)
- **Datasets**: Small dataset only
- **Repeats**: 1
- **Runtime**: ~2-5 minutes
- **Purpose**: Verify setup, quick testing, debugging

### Default Mode (no flags)
- **Datasets**: Medium dataset only
- **Repeats**: 1
- **Runtime**: ~5-15 minutes
- **Purpose**: Typical development workflow

### Full Mode (`--full`)
- **Datasets**: All datasets (small, medium, large)
- **Repeats**: 3 (configurable with `--repeats`)
- **Runtime**: ~30-60 minutes (depends on hardware)
- **Purpose**: Complete evaluation for publication

## Pipelines Evaluated

The script evaluates 5 different analysis pipelines:

1. **Pipeline A** (Classic): Most common approach in literature
2. **Pipeline B** (Batch Correction): Optimized for multi-batch data
3. **Pipeline C** (Advanced): Latest methods and techniques
4. **Pipeline D** (Performance-Optimized): For large-scale data
5. **Pipeline E** (Conservative): Minimal assumptions approach

## Evaluation Dimensions

Each pipeline is evaluated across four dimensions:

1. **Batch Effect Removal**
   - kBET score (mixing quality)
   - LISI score (local inversion)
   - Mixing entropy
   - Variance ratio

2. **Computational Performance**
   - Runtime
   - Memory usage
   - Efficiency score
   - Complexity estimate

3. **Data Distribution Changes**
   - Sparsity patterns
   - Statistical moments
   - Distribution tests
   - Quantile comparisons

4. **Data Structure Preservation**
   - PCA variance
   - Nearest neighbor consistency
   - Distance preservation
   - Global structure metrics

## Configuration

### Default Configuration

The script uses default configurations from:
- `configs/pipeline_configs.yaml`: Pipeline-specific settings
- `configs/evaluation_config.yaml`: Evaluation metric weights

### Custom Configuration

Create a custom YAML file and use with `--config`:

```yaml
# custom_config.yaml
pipeline:
  # Override pipeline settings here

evaluation:
  # Override evaluation settings here
  weights:
    batch_effects: 0.3
    performance: 0.3
    distribution: 0.2
    structure: 0.2
```

## Interpreting Results

### Console Output

In verbose mode, you'll see:
- Progress for each experiment
- Runtime and memory usage
- Success/failure status
- Final summary with statistics

### Results Files

- **Individual results**: One `.pkl` file per experiment
- **Complete results**: `complete_results.pkl` contains all data

```python
# Load results
import pickle

with open("outputs/results/complete_results.pkl", "rb") as f:
    data = pickle.load(f)

results = data["results"]          # Individual results
aggregated = data["aggregated"]    # Aggregated statistics
config = data["config"]            # Configuration used
total_runtime = data["total_runtime"]  # Total runtime
```

### Visualizations

All figures are generated at 300 DPI for publication quality:
- **batch_effects_comparison.png**: Batch correction performance
- **performance_comparison.png**: Runtime and memory comparison
- **distribution_comparison.png**: Distribution change metrics
- **structure_comparison.png**: Structure preservation metrics
- **ranking_barplot.png**: Overall pipeline rankings
- **radar_plot.png**: Multi-dimensional comparison

### Report

The `report.md` file contains:
- Experiment configuration
- Results summary
- Pipeline recommendations
- Detailed metric tables
- Figure references

## Performance Considerations

### Runtime Estimates

| Mode | Datasets | Repeats | Estimated Time |
|------|----------|---------|----------------|
| Test | Small | 1 | 2-5 min |
| Default | Medium | 1 | 5-15 min |
| Full | All | 3 | 30-60 min |

**Note**: Times vary based on hardware and dataset sizes.

### Memory Usage

- **Small dataset**: ~1-2 GB
- **Medium dataset**: ~2-4 GB
- **Large dataset**: ~4-8 GB

### Caching

Datasets are automatically cached in `outputs/data_cache/` for faster re-runs. Use `--no-cache` to regenerate.

## Troubleshooting

### Common Issues

1. **ImportError**
   ```bash
   # Ensure all dependencies are installed
   uv pip install -e ".[dev]"
   ```

2. **FileNotFoundError (no cache)**
   ```bash
   # First run or regenerate datasets
   python docs/comparison_study/run_comparison.py --no-cache
   ```

3. **Memory Error**
   ```bash
   # Use smaller dataset or reduce repeats
   python docs/comparison_study/run_comparison.py --test
   ```

4. **Pipeline Failure**
   - Check verbose output for error details
   - Individual results are saved even if some fail
   - Review error messages in summary

### Debug Mode

Use test mode with verbose output:
```bash
python docs/comparison_study/run_comparison.py --test --verbose
```

## Advanced Usage

### Parallel Execution

For large experiments, consider running different datasets in parallel:
```bash
# Terminal 1: Small dataset
python docs/comparison_study/run_comparison.py --test --output outputs/small

# Terminal 2: Medium dataset
python docs/comparison_study/run_comparison.py --output outputs/medium --no-cache

# Terminal 3: Large dataset
python docs/comparison_study/run_comparison.py --full --output outputs/large --no-cache
```

### Result Analysis

Load and analyze results in Python:
```python
import pickle
import pandas as pd

# Load results
with open("outputs/results/complete_results.pkl", "rb") as f:
    data = pickle.load(f)

# Convert to DataFrame for analysis
results_df = pd.DataFrame.from_dict(data["results"], orient="index")
print(results_df.head())
```

## Citation

If you use this comparison study in your research, please cite the ScpTensor framework.

## Contact

For issues or questions, please use the project's issue tracker.

## Version History

- **v1.0.0** (2026-01-20): Initial release
  - Complete pipeline comparison framework
  - 5 analysis pipelines
  - 4-dimensional evaluation
  - Automated visualization and reporting
