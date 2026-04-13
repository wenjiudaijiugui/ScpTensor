# Docs Map

`docs/` stores the stable user mainline, implementation contracts, and
research / audit notes that explain why those contracts look the way they do.
Benchmark scripts and benchmark outputs should stay under `benchmark/`.

## Mainline

Read these first when the goal is the canonical DIA preprocessing path:

- `user_workflows.md`: stable `DIA-NN / Spectronaut -> protein matrix` workflow guide
- `io_diann_spectronaut.md`: vendor I/O contract
- `aggregation_contract.md`: only stable peptide/precursor -> protein conversion stage
- `transformation_contract.md`: log-transform stage contract
- `normalization_contract.md`: normalization stage contract
- `imputation_contract.md`: imputation stage contract
- `integration_contract.md`: integration stage contract
- `qc_contract.md`: protein-level QC contract

## Contract

Implementation-facing contracts and boundary documents:

- `core_data_contract.md`: `ScpContainer -> Assay -> ScpMatrix` contract
- `core_compute_contract.md`: compute, sparse, and JIT contract
- `io_diann_spectronaut.md`: DIA-NN / Spectronaut I/O contract
- `aggregation_contract.md`: aggregation contract
- `transformation_contract.md`: transformation contract
- `normalization_contract.md`: normalization contract
- `standardization_contract.md`: `zscore` representation-layer contract
- `imputation_contract.md`: imputation contract
- `integration_contract.md`: integration contract
- `qc_contract.md`: protein-level QC contract
- `qc_psm_contract.md`: experimental peptide/PSM QC helper contract
- `autoselect_contract.md`: AutoSelect contract
- `utils_contract.md`: utility package contract
- `experimental_downstream_contract.md`: experimental downstream boundary contract
- `viz_contract.md`: visualization contract

## Research And Audit

Review registry and evidence/audit surfaces:

- `internal/review_manifest_20260312.json`: machine-readable index for all `internal/review_*.md` files
- `references/citations.json`: canonical citation registry
- `references/citation_usage.json`: citation usage map
- `article/README.md`: inadmissible-source record rules and index

Retained review surfaces are intentionally narrower now:

- importer and detection-state semantics:
  `internal/review_io_state_mapping_20260312.md`
- preprocessing-stage evidence:
  `internal/review_normalization_20260307.md`, `internal/review_log_scale_20260312.md`,
  `internal/review_qc_filtering_20260312.md`
- benchmark/task-design evidence:
  `internal/review_public_benchmark_data_20260312.md`,
  `internal/review_masked_imputation_20260312.md`,
  `internal/review_state_metrics_20260312.md`,
  `internal/review_batch_correction_20260305.md`,
  `internal/review_aggregation_benchmark_20260312.md`,
  `internal/review_autoselect_scoring_20260312.md`

Representation-layer `zscore` guidance is now folded back into
`standardization_contract.md` instead of living in a separate review file.

## Maintainer Notes

Governance, background, and convergence notes:

- `aggregation_literature.md`: aggregation background note, not a frozen contract
- `internal/optimization_checklist.md`: authority-doc priority and PR execution checklist
- `internal/compatibility_policy.md`: stable-surface compatibility rules
- `internal/runtime_baseline.md`: runtime baseline spec for stable preprocessing paths
- `internal/repo_file_tiering.md`: repository file cleanup and importance tiers

## Tutorial

- `user_workflows.md`: canonical workflow guide
- `../tutorial/tutorial.ipynb`: main tutorial
- `../tutorial/autoselect_tutorial.ipynb`: AutoSelect tutorial

## Evidence Taxonomy

All `internal/review_*.md` reviews use the same source typing contract:

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

- `internal/review_manifest_20260312.json`: typed manifest for all `internal/review_*.md` reviews only; it does not enumerate contract docs
- `references/citations.json`: canonical citation registry for repeated paper/module/package entries
- `references/citation_usage.json`: file-level usage map for the shared citation registry

Canonical citation policy:

- prefer DOI URLs for papers when DOI is verified
- prefer accession landing pages for datasets
- prefer stable documentation landing pages for software/manuals/modules
- prefer release landing pages for reusable resource packages
- edit the citation registry first when correcting shared metadata, then sync affected reviews

## Notes

- Keep `docs/` focused on readable documentation with one clear user mainline.
- Keep executable benchmarks, benchmark reports, and generated benchmark assets in `benchmark/`.
- Do not track notebook runtime artifacts under `docs/`.
- Keep inadmissible literature records under `docs/article/`; they can inform
  backlog but cannot be treated as reproducible benchmark evidence.
- Keep shared paper metadata in `docs/references/` to reduce cross-review citation drift.
- Keep background notes separate from implementation contracts; `aggregation_literature.md`
  is background context, not a frozen source-of-truth contract.
- Completed experimental-helper convergence decisions should be folded back into
  `experimental_downstream_contract.md` or `qc_psm_contract.md`, not kept as a
  long-lived parallel plan file.
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
