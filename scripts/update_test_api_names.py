#!/usr/bin/env python3
"""Update test files to use new API names.

This script updates all test files to use the new function names after
the deprecated aliases were removed.
"""

from pathlib import Path

# Mapping of old function names to new ones
API_REPLACEMENTS = {
    # Normalization
    "log_normalize": "norm_log",
    "zscore": "norm_zscore",
    "median_scaling": "norm_median_scale",
    "global_median_normalization": "norm_global_median",
    "sample_mean_normalization": "norm_sample_mean",
    "sample_median_normalization": "norm_sample_median",
    "upper_quartile_normalization": "norm_quartile",
    "tmm_normalization": "norm_tmm",
    "median_centering": "norm_median_center",
    # Imputation
    "knn": "impute_knn",
    "ppca": "impute_ppca",
    "svd_impute": "impute_svd",
    "missforest": "impute_mf",
    # QC (function names)
    "basic_qc": "qc_basic",
    "compute_quality_score": "qc_score",
    "detect_contaminant_proteins": "detect_contaminants",
    "compute_batch_metrics": "qc_batch_metrics",
    # Integration
    "combat": "integrate_combat",
    "harmony": "integrate_harmony",
    "mnn_correct": "integrate_mnn",
    "scanorama_integrate": "integrate_scanorama",
    # Clustering (deprecated aliases)
    "run_kmeans": "cluster_kmeans",
    # Dim Reduction
    "pca": "reduce_pca",
    "umap": "reduce_umap",
}

# Import statement replacements (for from imports)
# Note: Using a list of tuples instead of dict to avoid ordering issues
IMPORT_REPLACEMENTS_LIST = [
    # Normalization imports
    (
        "from scptensor.normalization.log import log_normalize",
        "from scptensor.normalization.log import norm_log as log_normalize",
    ),  # Keep alias for test consistency
    (
        "from scptensor.normalization.zscore import zscore",
        "from scptensor.normalization.zscore import norm_zscore as zscore",
    ),
    (
        "from scptensor.normalization.median_scaling import median_scaling",
        "from scptensor.normalization.median_scaling import norm_median_scale as median_scaling",
    ),
    (
        "from scptensor.normalization.global_median import global_median_normalization",
        "from scptensor.normalization.global_median import norm_global_median as global_median_normalization",
    ),
    (
        "from scptensor.normalization.sample_mean import sample_mean_normalization",
        "from scptensor.normalization.sample_mean import norm_sample_mean as sample_mean_normalization",
    ),
    (
        "from scptensor.normalization.sample_median import sample_median_normalization",
        "from scptensor.normalization.sample_median import norm_sample_median as sample_median_normalization",
    ),
    (
        "from scptensor.normalization.upper_quartile import upper_quartile_normalization",
        "from scptensor.normalization.upper_quartile import norm_quartile as upper_quartile_normalization",
    ),
    (
        "from scptensor.normalization.tmm import tmm_normalization",
        "from scptensor.normalization.tmm import norm_tmm as tmm_normalization",
    ),
    (
        "from scptensor.normalization.median_centering import median_centering",
        "from scptensor.normalization.median_centering import norm_median_center as median_centering",
    ),
    # Imputation imports
    ("from scptensor.impute.knn import knn", "from scptensor.impute.knn import impute_knn as knn"),
    (
        "from scptensor.impute.ppca import ppca",
        "from scptensor.impute.ppca import impute_ppca as ppca",
    ),
    (
        "from scptensor.impute.svd import svd_impute",
        "from scptensor.impute.svd import impute_svd as svd_impute",
    ),
    (
        "from scptensor.impute.missforest import missforest",
        "from scptensor.impute.missforest import impute_mf as missforest",
    ),
    # QC imports
    (
        "from scptensor.qc.basic import basic_qc",
        "from scptensor.qc.basic import qc_basic as basic_qc",
    ),
    (
        "from scptensor.qc.basic import compute_quality_score",
        "from scptensor.qc.basic import qc_score as compute_quality_score",
    ),
    # Integration imports
    (
        "from scptensor.integration.combat import combat",
        "from scptensor.integration.combat import integrate_combat as combat",
    ),
    (
        "from scptensor.integration.harmony import harmony",
        "from scptensor.integration.harmony import integrate_harmony as harmony",
    ),
    (
        "from scptensor.integration.mnn import mnn_correct",
        "from scptensor.integration.mnn import integrate_mnn as mnn_correct",
    ),
    (
        "from scptensor.integration.scanorama import scanorama_integrate",
        "from scptensor.integration.scanorama import integrate_scanorama as scanorama_integrate",
    ),
    # Dim reduction imports
    (
        "from scptensor.dim_reduction import pca",
        "from scptensor.dim_reduction import reduce_pca as pca",
    ),
    (
        "from scptensor.dim_reduction import umap",
        "from scptensor.dim_reduction import reduce_umap as umap",
    ),
    # Cluster imports
    (
        "from scptensor.cluster import run_kmeans",
        "from scptensor.cluster import cluster_kmeans as run_kmeans",
    ),
]


def update_file(file_path: Path) -> bool:
    """Update a single file with new API names."""
    content = file_path.read_text()
    original_content = content

    # First handle import statements - use aliases to keep test code working
    for old_import, new_import in IMPORT_REPLACEMENTS_LIST:
        content = content.replace(old_import, new_import)

    # Handle special cases for log_normalize in tests (it's from normalization.log module)
    # In tests, we keep using the old names via aliases for consistency
    # So we add "as <old_name>" to imports

    # Update action names in assertions (history logging)
    # These need to be updated to match the new action names
    action_replacements = {
        '"log_normalize"': '"norm_log"',
        "'log_normalize'": "'norm_log'",
        '"zscore"': '"norm_zscore"',
        "'zscore'": "'norm_zscore'",
        '"median_scaling"': '"norm_median_scale"',
        "'median_scaling'": "'norm_median_scale'",
        '"global_median_normalization"': '"norm_global_median"',
        "'global_median_normalization'": "'norm_global_median'",
        '"sample_mean_normalization"': '"norm_sample_mean"',
        "'sample_mean_normalization'": "'norm_sample_mean'",
        '"sample_median_normalization"': '"norm_sample_median"',
        "'sample_median_normalization'": "'norm_sample_median'",
        '"upper_quartile_normalization"': '"norm_quartile"',
        "'upper_quartile_normalization'": "'norm_quartile'",
        '"tmm_normalization"': '"norm_tmm"',
        "'tmm_normalization'": "'norm_tmm'",
        '"median_centering"': '"norm_median_center"',
        "'median_centering'": "'norm_median_center'",
        '"knn"': '"impute_knn"',
        "'knn'": "'impute_knn'",
        '"ppca"': '"impute_ppca"',
        "'ppca'": "'impute_ppca'",
        '"svd_impute"': '"impute_svd"',
        "'svd_impute'": "'impute_svd'",
        '"missforest"': '"impute_mf"',
        "'missforest'": "'impute_mf'",
        '"basic_qc"': '"qc_basic"',
        "'basic_qc'": "'qc_basic'",
        '"combat"': '"integrate_combat"',
        "'combat'": "'integrate_combat'",
        '"harmony"': '"integrate_harmony"',
        "'harmony'": "'integrate_harmony'",
        '"mnn_correct"': '"integrate_mnn"',
        "'mnn_correct'": "'integrate_mnn'",
        '"scanorama_integrate"': '"integrate_scanorama"',
        "'scanorama_integrate'": "'integrate_scanorama'",
        '"pca"': '"reduce_pca"',
        "'pca'": "'reduce_pca'",
        '"umap"': '"reduce_umap"',
        "'umap'": "'reduce_umap'",
        '"cluster_kmeans"': '"cluster_kmeans"',  # This stays the same
    }

    for old_action, new_action in action_replacements.items():
        content = content.replace(old_action, new_action)

    # Handle the test_qc.py special case where the action check uses qc_basic
    # The basic_qc function now logs as "qc_basic"
    if file_path.name == "test_qc.py":
        # The action assertion already checks for qc_basic, which is correct
        pass

    # Handle the test_normalization.py special case - keep using old names via aliases
    # The functions use the old names in tests but import with new names as aliases
    if file_path.name == "test_normalization.py":
        # Add the import aliases at the bottom
        if "from scptensor.normalization" in content:
            # Replace direct imports with aliased imports
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                if (
                    "from scptensor.normalization." in line
                    and "import" in line
                    and " as " not in line
                ):
                    # This is a normalization import that needs aliasing
                    parts = line.split(" import ")
                    if len(parts) == 2:
                        module = parts[0]
                        func_name = parts[1].strip()
                        new_line = f"{module} import {func_name} as {func_name}"
                        # Now apply the mapping
                        for old, new in API_REPLACEMENTS.items():
                            if func_name == old:
                                new_line = f"{module} import {new} as {old}"
                                break
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            content = "\n".join(new_lines)

    # Handle test_performance.py special case - directly imports from modules
    if file_path.name == "test_performance.py":
        content = content.replace(
            "from scptensor.normalization.log import log_normalize",
            "from scptensor.normalization.log import norm_log as log_normalize",
        )

    # Handle test_cluster.py - run_kmeans import
    if file_path.name == "test_cluster.py":
        content = content.replace(
            "from scptensor.cluster.kmeans import run_kmeans  # noqa: F401 (used in tests)",
            "from scptensor.cluster.kmeans import cluster_kmeans as run_kmeans  # noqa: F401 (used in tests)",
        )

    # Handle test_error_handling.py - imports from submodules
    if file_path.name == "test_error_handling.py":
        content = content.replace(
            "from scptensor.normalization.log import log_normalize",
            "from scptensor.normalization.log import norm_log as log_normalize",
        )
        content = content.replace(
            "from scptensor.normalization.zscore import zscore",
            "from scptensor.normalization.zscore import norm_zscore as zscore",
        )
        content = content.replace(
            "from scptensor.impute.knn import knn",
            "from scptensor.impute.knn import impute_knn as knn",
        )
        content = content.replace(
            "from scptensor.impute.missforest import missforest",
            "from scptensor.impute.missforest import impute_mf as missforest",
        )
        content = content.replace(
            "from scptensor.impute.ppca import ppca",
            "from scptensor.impute.ppca import impute_ppca as ppca",
        )
        content = content.replace(
            "from scptensor.impute.svd import svd_impute",
            "from scptensor.impute.svd import impute_svd as svd_impute",
        )
        content = content.replace(
            "from scptensor.integration.combat import combat",
            "from scptensor.integration.combat import integrate_combat as combat",
        )
        content = content.replace(
            "from scptensor.integration.mnn import mnn_correct",
            "from scptensor.integration.mnn import integrate_mnn as mnn_correct",
        )
        content = content.replace(
            "from scptensor.integration.scanorama import scanorama_integrate",
            "from scptensor.integration.scanorama import integrate_scanorama as scanorama_integrate",
        )
        content = content.replace(
            "from scptensor.qc.basic import basic_qc",
            "from scptensor.qc.basic import qc_basic as basic_qc",
        )

    # Handle test_workflows.py in integration tests
    if "test_workflows.py" in str(file_path):
        content = content.replace(
            "from scptensor.cluster import run_kmeans",
            "from scptensor.cluster import cluster_kmeans as run_kmeans",
        )
        content = content.replace(
            "from scptensor.dim_reduction import pca",
            "from scptensor.dim_reduction import reduce_pca as pca",
        )
        content = content.replace(
            "from scptensor.impute import knn, ppca",
            "from scptensor.impute import impute_knn as knn, impute_ppca as ppca",
        )
        content = content.replace(
            "from scptensor.integration import combat",
            "from scptensor.integration import integrate_combat as combat",
        )
        content = content.replace(
            "from scptensor.normalization import log_normalize",
            "from scptensor.normalization import norm_log as log_normalize",
        )
        content = content.replace(
            "from scptensor.qc import basic_qc", "from scptensor.qc import qc_basic as basic_qc"
        )
        # Update action name assertions
        content = content.replace(
            'assert last_log.action == "log_normalize"', 'assert last_log.action == "norm_log"'
        )
        content = content.replace(
            'assert last_log.action == "knn"', 'assert last_log.action == "impute_knn"'
        )
        content = content.replace(
            'assert last_log.action == "integration_combat"',
            'assert last_log.action == "integrate_combat"',
        )
        content = content.replace(
            'assert last_log.action == "basic_qc"', 'assert last_log.action == "qc_basic"'
        )

    # Handle test_container_basic.py
    if file_path.name == "test_container_basic.py":
        content = content.replace(
            "from scptensor.integration import combat, harmony",
            "from scptensor.integration import integrate_combat as combat, integrate_harmony as harmony",
        )
        content = content.replace(
            "from scptensor.qc import basic_qc, detect_outliers",
            "from scptensor.qc import qc_basic as basic_qc, detect_outliers",
        )

    # Handle test_impute.py
    if file_path.name == "test_impute.py":
        # Update action name assertions
        content = content.replace(
            'assert result.history[-1].action == "impute_knn"',
            'assert result.history[-1].action == "impute_knn"',
        )
        content = content.replace(
            'assert result.history[-1].action == "impute_ppca"',
            'assert result.history[-1].action == "impute_ppca"',
        )
        content = content.replace(
            'assert result.history[-1].action == "impute_svd"',
            'assert result.history[-1].action == "impute_svd"',
        )
        # These should already be correct

    # Handle test_viz.py
    if file_path.name == "test_viz.py":
        # Update any old API references
        pass

    # Only write if content changed
    if content != original_content:
        file_path.write_text(content)
        return True
    return False


def main():
    """Update all test files."""
    project_root = Path("/home/shenshang/projects/ScpTensor")
    tests_dir = project_root / "tests"

    # Files to update
    files_to_update = [
        "tests/core/test_container_basic.py",
        "tests/core/test_container_edge_cases.py",
        "tests/core/test_matrix_edge_cases.py",
        "tests/test_benchmark.py",
        "tests/test_cluster.py",
        "tests/test_diff_expr.py",
        "tests/test_error_handling.py",
        "tests/test_feature_selection.py",
        "tests/test_filtering.py",
        "tests/test_impute.py",
        "tests/test_io_container_methods.py",
        "tests/test_io_export.py",
        "tests/test_normalization.py",
        "tests/test_qc.py",
        "tests/test_utils_data_generator.py",
        "tests/test_viz.py",
        "tests/test_performance.py",
        "tests/integration/test_workflows.py",
    ]

    updated_count = 0
    for file_path_str in files_to_update:
        file_path = project_root / file_path_str
        if file_path.exists():
            if update_file(file_path):
                print(f"Updated: {file_path}")
                updated_count += 1
            else:
                print(f"No changes: {file_path}")
        else:
            print(f"Not found: {file_path}")

    print(f"\nUpdated {updated_count} files.")


if __name__ == "__main__":
    main()
