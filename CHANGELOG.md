# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `reduce_tsne` support in `scptensor.dim_reduction` and public API exports.
- Additional dimensionality-reduction tests for PCA / UMAP / t-SNE behavior and edge cases.

### Changed
- Dimensionality-reduction implementations were hardened with stricter input validation.
- Project documentation was cleaned to match the current DIA preprocessing scope and module layout.

### Removed
- Differential expression module `scptensor/diff_expr/`.
- Differential-expression-specific visualization recipe `scptensor/viz/recipes/differential.py`.
- Differential expression test suite under `tests/diff_expr/`.

## [0.1.0-alpha] - 2025-01-05

### Added
- Initial alpha release of ScpTensor with core DIA preprocessing workflow foundations.

[Unreleased]: https://github.com/wenjiudaijiugui/ScpTensor/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/wenjiudaijiugui/ScpTensor/releases/tag/v0.1.0-alpha
