# Start Here - Pipeline Comparison Study

**Last Updated:** 2026-01-20
**Status:** ✅ Ready to Use

---

## Quick Start

### 1. Verify Setup (30 seconds)

```bash
python docs/comparison_study/verify_setup.py
```

Expected output:
```
✓ ALL CHECKS PASSED - Setup is ready!
```

### 2. Run Quick Test (2-5 minutes)

```bash
python docs/comparison_study/run_comparison.py --test --verbose
```

### 3. View Results

```bash
# View results directory
ls -lh docs/comparison_study/outputs/

# View generated figures
ls -lh docs/comparison_study/outputs/figures/

# Read report
cat docs/comparison_study/outputs/report.md
```

---

## Documentation Guide

### For Users

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Essential commands and common workflows
2. **[README.md](README.md)** - Complete usage guide with examples
3. **[examples/runner_example.py](examples/runner_example.py)** - 5 working examples

### For Developers

1. **[RUNNER_IMPLEMENTATION_REPORT.md](RUNNER_IMPLEMENTATION_REPORT.md)** - Implementation details and design decisions
2. **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** - Complete delivery summary
3. **[run_comparison.py](run_comparison.py)** - Main runner script (534 lines)

### Module Documentation

1. **[data/README.md](data/README.md)** - Dataset generation and loading
2. **[pipelines/README.md](pipelines/)** - 5 analysis pipelines
3. **[evaluation/README.md](evaluation/README.md)** - Evaluation metrics
4. **[visualization/QUICK_START.md](visualization/QUICK_START.md)** - Visualization tools

---

## Common Tasks

### Quick Test (Verify Setup)

```bash
python docs/comparison_study/run_comparison.py --test --verbose
```

**Runtime:** 2-5 minutes
**Datasets:** Small only
**Repeats:** 1

### Default Mode (Development)

```bash
python docs/comparison_study/run_comparison.py --verbose
```

**Runtime:** 5-15 minutes
**Datasets:** Medium only
**Repeats:** 1

### Full Experiment (Publication)

```bash
python docs/comparison_study/run_comparison.py --full --repeats 3 --verbose
```

**Runtime:** 30-60 minutes
**Datasets:** All (small, medium, large)
**Repeats:** 3

### Custom Configuration

```bash
python docs/comparison_study/run_comparison.py --config custom.yaml --verbose
```

### Regenerate Datasets

```bash
python docs/comparison_study/run_comparison.py --full --no-cache --verbose
```

---

## Getting Help

### Command-Line Help

```bash
python docs/comparison_study/run_comparison.py --help
```

### Example Scripts

```bash
# View usage examples
python docs/comparison_study/examples/runner_example.py

# View data generation examples
python docs/comparison_study/examples/demo_data_generation.py
```

### Troubleshooting

See **[README.md](README.md)** - Troubleshooting section

Common issues:
- **ImportError**: Run `uv pip install -e ".[dev]"`
- **No cache found**: Run with `--no-cache`
- **Memory error**: Run with `--test`

---

## Output Structure

After running, you'll find:

```
docs/comparison_study/outputs/
├── data_cache/          # Cached datasets (.pkl files)
├── results/             # Experiment results (.pkl files)
├── figures/             # Generated figures (300 DPI PNGs)
├── report.md            # Markdown report
└── report.pdf           # PDF report (if generated)
```

---

## Python API Usage

```python
from docs.comparison_study.run_comparison import (
    load_config,
    load_datasets,
    initialize_pipelines,
    run_complete_experiment
)

# Load configuration
config = load_config()

# Load datasets (cached if available)
datasets = load_datasets(use_cache=True, verbose=True)

# Initialize pipelines
pipelines = initialize_pipelines(config)

# Run experiment
results = run_complete_experiment(
    datasets=datasets,
    pipelines=pipelines,
    config=config,
    output_dir="outputs/my_experiment",
    dataset_names=["small"],
    n_repeats=3,
    verbose=True
)

# Use results
print(f"Completed {len(results)} experiments")
```

See **[examples/runner_example.py](examples/runner_example.py)** for more examples.

---

## What's Included

### Components

1. **Runner Script** (`run_comparison.py`)
   - Command-line interface
   - Python API
   - Automatic caching
   - Performance monitoring
   - Error recovery

2. **Data Module** (`data/`)
   - Synthetic dataset generation
   - Dataset loading
   - Caching system

3. **Pipelines Module** (`pipelines/`)
   - 5 analysis pipelines
   - Classic, Batch Correction, Advanced, Performance-Optimized, Conservative

4. **Evaluation Module** (`evaluation/`)
   - 4-dimensional evaluation
   - Batch effects, Performance, Distribution, Structure

5. **Visualization Module** (`visualization/`)
   - 6 publication-quality figures (300 DPI)
   - Comprehensive report generation

### Documentation

- **[START_HERE.md](START_HERE.md)** - This file (overview)
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Essential commands
- **[README.md](README.md)** - Complete guide
- **[RUNNER_IMPLEMENTATION_REPORT.md](RUNNER_IMPLEMENTATION_REPORT.md)** - Technical details
- **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** - Delivery checklist

### Examples

- **[examples/runner_example.py](examples/runner_example.py)** - Runner usage examples
- **[examples/demo_data_generation.py](examples/demo_data_generation.py)** - Data generation examples

### Configuration

- **[configs/pipeline_configs.yaml](configs/pipeline_configs.yaml)** - Pipeline settings
- **[configs/evaluation_config.yaml](configs/evaluation_config.yaml)** - Evaluation weights

---

## Performance Estimates

| Mode | Datasets | Repeats | Time | Memory | Purpose |
|------|----------|---------|------|--------|---------|
| Test | Small | 1 | 2-5 min | 1-2 GB | Quick verification |
| Default | Medium | 1 | 5-15 min | 2-4 GB | Development |
| Full | All (3) | 3 | 30-60 min | 4-8 GB | Complete evaluation |

---

## Next Steps

### First Time Users

1. ✅ Run verification: `python docs/comparison_study/verify_setup.py`
2. ✅ Run quick test: `python docs/comparison_study/run_comparison.py --test --verbose`
3. ✅ Read results in `outputs/report.md`
4. ✅ View figures in `outputs/figures/`

### Researchers

1. ✅ Review **[README.md](README.md)** for complete guide
2. ✅ Run full experiment: `python docs/comparison_study/run_comparison.py --full --repeats 3 --verbose`
3. ✅ Analyze results in `outputs/report.md`
4. ✅ Use figures for publications

### Developers

1. ✅ Review **[RUNNER_IMPLEMENTATION_REPORT.md](RUNNER_IMPLEMENTATION_REPORT.md)**
2. ✅ Study **[run_comparison.py](run_comparison.py)**
3. ✅ Run **[examples/runner_example.py](examples/runner_example.py)**
4. ✅ Extend with custom pipelines or metrics

---

## Support

### Documentation

- Quick help: **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
- Full guide: **[README.md](README.md)**
- Technical details: **[RUNNER_IMPLEMENTATION_REPORT.md](RUNNER_IMPLEMENTATION_REPORT.md)**

### Verification

```bash
python docs/comparison_study/verify_setup.py
```

### Examples

```bash
python docs/comparison_study/examples/runner_example.py
```

### Command-Line Help

```bash
python docs/comparison_study/run_comparison.py --help
```

---

## Summary

The Pipeline Comparison Study runner is **ready to use** for:

- ✅ Quick testing and development
- ✅ Comprehensive pipeline evaluation
- ✅ Publication-quality analysis
- ✅ Performance benchmarking
- ✅ Method validation

**Start here:** Run the verification script, then the quick test!

```bash
python docs/comparison_study/verify_setup.py
python docs/comparison_study/run_comparison.py --test --verbose
```

---

**Status:** ✅ Complete and Ready
**Version:** 1.0.0
**Date:** 2026-01-20
