# Scripts

Utility scripts for ScpTensor development.

## Available Scripts

### Performance Benchmark

```bash
python scripts/performance_benchmark.py
```

Benchmark key operations across ScpTensor to identify performance bottlenecks and track optimizations.

Options:
- `--html output.html`: Generate HTML report
- See `--help` for more options

Example:
```bash
uv run python scripts/performance_benchmark.py --html benchmark.html
```
