# Docs Index

`docs/` stores user-facing documentation only.
Benchmark scripts and benchmark outputs should stay under `benchmark/`.

## Current Files

### Review

- `review_imputation_20260304.md`: DIA single-cell imputation review notes
- `review_batch_correction_20260305.md`: DIA single-cell batch-correction review notes
- `review_normalization_20260307.md`: DIA single-cell normalization review notes
- `review_qc_filtering_20260312.md`: DIA single-cell QC and filtering review notes
- `review_batch_diagnostics_20260312.md`: DIA single-cell batch diagnostics metrics review notes
- `review_autoselect_scoring_20260312.md`: AutoSelect scoring and reporting framework review notes
- `review_missingness_20260312.md`: DIA single-cell missingness semantics and detection-state review notes
- `review_log_scale_20260312.md`: DIA single-cell log-transform and scale-contract review notes
- `review_zscore_standardization_20260313.md`: DIA single-cell z-score standardization and downstream scale-contract review notes
- `review_public_benchmark_data_20260312.md`: DIA single-cell public benchmark datasets and task-design review notes
- `review_masked_imputation_20260312.md`: DIA single-cell masked-value imputation benchmark design review notes
- `review_batch_confounding_20260312.md`: DIA single-cell batch-confounding benchmark design review notes
- `review_io_state_mapping_20260312.md`: DIA-NN / Spectronaut importer state-mapping contract review notes
- `review_aggregation_benchmark_20260312.md`: DIA single-cell aggregation benchmark design review notes
- `review_state_metrics_20260312.md`: DIA single-cell state-aware completeness and uncertainty metrics review notes

### Registry

- `review_manifest_20260312.json`: typed manifest for all `review_*.md` reviews only; contract docs are indexed in `Contract`
- `references/citations.json`: canonical metadata for shared high-frequency citations
- `references/citation_usage.json`: file-level usage map for shared citations

### Contract

- `core_data_contract.md`: ScpTensor core container contract inspired by AnnData / MuData
- `core_compute_contract.md`: low-level compute, sparse, and JIT implementation contract
- `io_diann_spectronaut.md`: DIA-NN / Spectronaut I/O contract
- `aggregation_contract.md`: peptide/precursor -> protein aggregation implementation contract
- `transformation_contract.md`: log-transform implementation contract
- `normalization_contract.md`: normalization implementation contract
- `standardization_contract.md`: z-score representation-layer implementation contract
- `imputation_contract.md`: imputation implementation contract
- `integration_contract.md`: batch-correction / integration implementation contract
- `qc_contract.md`: protein-level QC implementation contract
- `qc_psm_contract.md`: experimental peptide/PSM QC helper contract
- `autoselect_contract.md`: AutoSelect implementation and reporting contract
- `utils_contract.md`: utility package public-surface and helper-boundary contract
- `experimental_downstream_contract.md`: experimental dim-reduction / clustering boundary contract
- `viz_contract.md`: visualization API, return-type, and data-boundary contract

### Background

- `aggregation_literature.md`: peptide -> protein aggregation literature background notes

### Active Plan

- `experimental_downstream_alignment_plan.md`: non-contract convergence record for experimental helper asymmetries; current mandatory gaps are closed, keep it as a resolution log until an archive slot is introduced
- `optimization_checklist.md`: document-driven execution checklist for future code optimization; defines authority-doc priority, PR gates, staged execution order, and stop conditions for contract drift
- `compatibility_policy.md`: stable-surface compatibility rules; defines which aliases and facades remain allowed, which compatibility patterns are explicitly banned, and how tests/docs should treat canonical APIs
- `runtime_baseline.md`: PR-0 runtime baseline spec for stable preprocessing paths; separates engineering runtime regression checks from scientific method benchmarks and distinguishes full-chain gates from normalization-only micro-gates
- `repo_file_tiering.md`: repository file importance tiers and cleanup rules; defines what is release-critical, support-only, reference-only, and local-generated noise

### Tutorial

- `../tutorial/README.md`: tutorial index
- `../tutorial/tutorial.ipynb`: main tutorial
- `../tutorial/autoselect_tutorial.ipynb`: AutoSelect tutorial

## Evidence Taxonomy

All `review_*.md` reviews now use the same source typing contract:

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

- `review_manifest_20260312.json`: typed manifest for all `review_*.md` reviews only; it does not enumerate contract docs
- `references/citations.json`: canonical citation registry for repeated paper/module/package entries
- `references/citation_usage.json`: file-level usage map for the shared citation registry

Repository-wide review status:

- all `review_*.md` files have completed the `registry-first` tail-template migration
- the standardized tail section is `Shared Citation Registry Coverage`
- shared citation metadata should be corrected in `references/citations.json` first, then synced back to affected reviews
- frozen implementation contracts now cover `core-data / core-compute / aggregation / transformation / normalization / standardization / imputation / integration / qc / experimental-qc-psm / autoselect / utils / experimental-downstream / viz`

Canonical citation policy:

- prefer DOI URLs for papers when DOI is verified
- prefer accession landing pages for datasets
- prefer stable documentation landing pages for software/manuals/modules
- prefer release landing pages for reusable resource packages
- edit the citation registry first when correcting shared metadata, then sync affected reviews

## Notes

- Keep `docs/` focused on readable documentation.
- Keep executable benchmarks, benchmark reports, and generated benchmark assets in `benchmark/`.
- Do not track notebook runtime artifacts under `docs/`.
- Keep shared paper metadata in `docs/references/` to reduce cross-review citation drift.
- Keep background notes separate from implementation contracts; `aggregation_literature.md`
  is background context, not a frozen source-of-truth contract.
- Keep remediation / convergence notes separate from contracts;
  `experimental_downstream_alignment_plan.md` is not a source-of-truth
  contract, even after its mandatory gaps are closed.
- Release boundary: dimensionality reduction and clustering are documented as
  experimental downstream helpers, not core preprocessing deliverables.
- `qc_psm` is documented as an experimental pre-aggregation helper; use
  `scptensor.experimental.qc_psm` semantics in docs rather than treating it as
  part of stable `scptensor.qc`.
- Core object docs prefer the canonical naming set
  `proteins / peptides` and `raw / log / norm / imputed / zscore`; treat
  `protein / peptide`, implementation-local names such as `normalized`,
  `trqn_norm`, `log2`, and layer `X` as compatibility-facing or
  implementation-local names rather than repository-wide defaults.
- `raw` is the canonical imported quantitative layer name in repository docs;
  whether that layer is vendor-normalized must be stated through provenance
  fields such as `is_vendor_normalized`, not by introducing a second default
  layer-name family.
- Treat `Assay.X` as a compatibility-facing shortcut only; document explicit
  assay-level access patterns such as `container.assay_shape(assay_name)` and
  `container.assays[assay_name].layers[source_layer]` for stable preprocessing.
- Prefer `scptensor.experimental` imports when documenting `reduce_*` and
  `cluster_*` workflows.
- Prefer `from scptensor.experimental import qc_psm` when documenting
  peptide/PSM experimental QC helpers.
- Prefer `scptensor.viz` canonical `plot_*` APIs in documentation examples.
