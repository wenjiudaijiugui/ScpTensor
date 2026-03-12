# Docs Index

`docs/` stores user-facing documentation only.
Benchmark scripts and benchmark outputs should stay under `benchmark/`.

## Current Files

- `../tutorial/README.md`: tutorial index
- `../tutorial/tutorial.ipynb`: main tutorial
- `../tutorial/autoselect_tutorial.ipynb`: AutoSelect tutorial
- `io_input_spec_diann_spectronaut.md`: DIA-NN / Spectronaut I/O contract
- `aggregation_methods_from_literature.md`: peptide -> protein aggregation literature notes
- `dia_sc_imputation_literature_review_20260304.md`: DIA single-cell imputation review notes
- `dia_sc_batch_correction_literature_review_20260305.md`: DIA single-cell batch-correction review notes
- `dia_sc_normalization_literature_review_20260307.md`: DIA single-cell normalization review notes
- `dia_sc_qc_filtering_literature_review_20260312.md`: DIA single-cell QC and filtering review notes
- `dia_sc_batch_diagnostics_metrics_review_20260312.md`: DIA single-cell batch diagnostics metrics review notes
- `dia_sc_autoselect_scoring_framework_review_20260312.md`: AutoSelect scoring and reporting framework review notes
- `dia_sc_missingness_semantics_review_20260312.md`: DIA single-cell missingness semantics and detection-state review notes
- `dia_sc_log_transform_scale_contract_review_20260312.md`: DIA single-cell log-transform and scale-contract review notes
- `dia_sc_public_benchmark_datasets_and_task_design_review_20260312.md`: DIA single-cell public benchmark datasets and task-design review notes
- `dia_sc_masked_value_benchmark_design_review_20260312.md`: DIA single-cell masked-value imputation benchmark design review notes
- `dia_sc_batch_confounding_benchmark_design_review_20260312.md`: DIA single-cell batch-confounding benchmark design review notes
- `dia_sc_io_state_mapping_contract_review_20260312.md`: DIA-NN / Spectronaut importer state-mapping contract review notes
- `dia_sc_aggregation_benchmark_design_review_20260312.md`: DIA single-cell aggregation benchmark design review notes
- `dia_sc_state_aware_completeness_metrics_review_20260312.md`: DIA single-cell state-aware completeness and uncertainty metrics review notes

## Evidence Taxonomy

All `dia_sc_*review_*.md` reviews now use the same source typing contract:

- `论文证据`
  - journal articles, benchmark papers, method papers
- `数据入口`
  - stable dataset pages or accession landing pages
- `模块规范 / 软件文档`
  - official software manuals, API docs, benchmark module pages
- `资源包`
  - reusable package/resource distribution pages
- `本地实现上下文`
  - ScpTensor local code/docs used only to check alignment with external evidence

Entry rule:

- do not use a paper page as a substitute for a dataset entrypoint
- do not use a software manual as a substitute for benchmark evidence
- do not use a package page as a substitute for a public dataset

Machine-readable index:

- `review_resource_manifest_20260312.json`: typed manifest for all `dia_sc_*review_*.md` reviews

## Notes

- Keep `docs/` focused on readable documentation.
- Keep executable benchmarks, benchmark reports, and generated benchmark assets in `benchmark/`.
- Do not track notebook runtime artifacts under `docs/`.
- Release boundary: dimensionality reduction and clustering are documented as
  experimental downstream helpers, not core preprocessing deliverables.
- Prefer `scptensor.experimental` imports when documenting `reduce_*` and
  `cluster_*` workflows.
- Prefer `scptensor.viz` canonical `plot_*` APIs in documentation examples.
