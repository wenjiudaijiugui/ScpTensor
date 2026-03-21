# Tutorial Index

`tutorial/` stores runnable tutorial notebooks and their generated tutorial outputs.

## Current Files

Notebook:

- `tutorial.ipynb`: main walkthrough for the stable `I/O -> aggregation -> QC -> raw -> log -> norm -> imputed` path, plus an explicit experimental downstream demo
- `autoselect_tutorial.ipynb`: AutoSelect workflow tutorial for stable `normalize / impute / integrate` selection, with optional `reduce / cluster` experimental helpers

Outputs:

- `_tutorial_outputs/`: generated tutorial artifacts and toy input files

## Boundary

- tutorials are runnable examples, not frozen implementation contracts
- vendor I/O contracts, literature reviews, and implementation boundaries stay in `docs/`
- benchmark scripts and benchmark outputs stay in `benchmark/`
- stable preprocessing release 的主线终点仍是 protein-level quantitative matrix；tutorial 中出现的 `reduce / cluster` 只作 exploratory downstream 示例
- tutorial 输出中若出现 `raw_norm_median`、`raw_norm_median_knn` 等名字，应按 AutoSelect artifact naming 理解，不把它们当作 canonical `norm / imputed` 终态层名
