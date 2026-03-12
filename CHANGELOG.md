# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `.gitignore` now explicitly re-includes `.github/workflows/*.yml` to keep CI/CD workflow files version-controlled.
- `scptensor.standardization.zscore` was restored and re-exported in top-level package API (`from scptensor import zscore`).
- Dimensionality reduction and clustering were downgraded to experimental release scope and are now documented under `scptensor.experimental`.
- Visualization APIs were standardized around clear `plot_*` naming with backward-compatible aliases, and targeted workflow plots were added for aggregation/normalization/integration diagnostics.

### Removed
- Unimplemented placeholder module `scptensor/core/reader.py` (it was not exported and only raised `NotImplementedError`).
- Differential-expression visualization entry `volcano` was removed from public APIs and legacy recipe module `scptensor/viz/recipes/stats.py` was deleted.
- Deprecated wrapper module `scptensor/normalization/log_transform.py` was removed (use `scptensor.transformation.log_transform`).
- Deprecated integration aliases `harmony()` and `integrate_nonlinear` were removed (use `integrate_harmony`).

## [0.1.0] - 2026-03-06

### Added
- `reduce_tsne` support in `scptensor.dim_reduction` and public API exports.
- Additional dimensionality-reduction tests for PCA / UMAP / t-SNE behavior and edge cases.

### Changed
- Dimensionality-reduction implementations were hardened with stricter input validation.
- Project documentation was cleaned to match the current DIA preprocessing scope and module layout.
- Release quality gates were hardened in CI/CD (lint, format, typing, and test checks before publish paths).

### Removed
- Differential expression module `scptensor/diff_expr/`.
- Differential-expression-specific visualization recipe `scptensor/viz/recipes/differential.py`.
- Differential expression test suite under `tests/diff_expr/`.
- Legacy skipped QC test modules for removed APIs: `tests/qc/test_missing.py`, `tests/qc/test_sensitivity.py`.
- Unimplemented `reader` export from public package API.

## [0.1.0-alpha] - 2025-01-05

### Added
- Initial alpha release of ScpTensor with core DIA preprocessing workflow foundations.

[Unreleased]: https://github.com/wenjiudaijiugui/ScpTensor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wenjiudaijiugui/ScpTensor/compare/v0.1.0-alpha...v0.1.0
[0.1.0-alpha]: https://github.com/wenjiudaijiugui/ScpTensor/releases/tag/v0.1.0-alpha
