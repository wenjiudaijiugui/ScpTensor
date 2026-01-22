# Pipeline Comparison Runner - Quick Reference

## Quick Commands

```bash
# Test setup (2-5 min)
python docs/comparison_study/run_comparison.py --test --verbose

# Default mode (5-15 min)
python docs/comparison_study/run_comparison.py --verbose

# Full experiment (30-60 min)
python docs/comparison_study/run_comparison.py --full --repeats 3 --verbose
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--full` | Run all datasets, multiple repeats | False |
| `--test` | Quick test with small dataset | False |
| `--config PATH` | Custom config file | None |
| `--output DIR` | Output directory | `docs/comparison_study/outputs` |
| `--no-cache` | Regenerate datasets | False |
| `--repeats N` | Number of repeats | 3 |
| `--verbose` | Detailed output | False |

## Common Workflows

### First Time Setup
```bash
python docs/comparison_study/run_comparison.py --test --verbose
```

### Development Workflow
```bash
python docs/comparison_study/run_comparison.py --verbose
```

### Publication Experiment
```bash
python docs/comparison_study/run_comparison.py --full --repeats 5 --verbose
```

### Custom Output Location
```bash
python docs/comparison_study/run_comparison.py --full --output /path/to/results
```

## Output Structure

```
outputs/
├── data_cache/          # Cached datasets (.pkl)
├── results/             # Experiment results (.pkl)
├── figures/             # Generated figures (300 DPI)
├── report.md            # Markdown report
└── report.pdf           # PDF report
```

## Python API

```python
from docs.comparison_study.run_comparison import (
    load_config,
    load_datasets,
    initialize_pipelines,
    run_complete_experiment
)

# Load
config = load_config()
datasets = load_datasets(use_cache=True)
pipelines = initialize_pipelines(config)

# Run
results = run_complete_experiment(
    datasets=datasets,
    pipelines=pipelines,
    config=config,
    output_dir="outputs",
    dataset_names=["small"],
    n_repeats=3,
    verbose=True
)
```

## Loading Results

```python
import pickle

with open("docs/comparison_study/outputs/results/complete_results.pkl", "rb") as f:
    data = pickle.load(f)

results = data["results"]          # Individual results
aggregated = data["aggregated"]    # Statistics
config = data["config"]            # Configuration
```

## Troubleshooting

### ImportError
```bash
uv pip install -e ".[dev]"
```

### No Cache Found
```bash
python docs/comparison_study/run_comparison.py --no-cache
```

### Memory Error
```bash
python docs/comparison_study/run_comparison.py --test
```

## Performance Estimates

| Mode | Datasets | Repeats | Time | Memory |
|------|----------|---------|------|--------|
| Test | Small | 1 | 2-5 min | 1-2 GB |
| Default | Medium | 1 | 5-15 min | 2-4 GB |
| Full | All | 3 | 30-60 min | 4-8 GB |

## More Information

- **Full Documentation**: `README.md`
- **Implementation Report**: `RUNNER_IMPLEMENTATION_REPORT.md`
- **Examples**: `examples/runner_example.py`
- **Help**: `python docs/comparison_study/run_comparison.py --help`
