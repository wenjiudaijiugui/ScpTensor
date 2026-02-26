"""Pipeline executor - configuration-driven pipeline execution.

This module provides a simple, configuration-driven approach to running
analysis pipelines without the overhead of abstract base classes.
"""

from __future__ import annotations

import time
from typing import Any

from scptensor.cluster import cluster_kmeans
from scptensor.core.structures import ScpContainer
from scptensor.dim_reduction import reduce_pca
from scptensor.impute import impute_knn, impute_lls, impute_missforest
from scptensor.integration import integrate_combat, integrate_harmony, integrate_mnn
from scptensor.normalization import log_transform, norm_mean, norm_median
from scptensor.qc.qc_sample import filter_low_quality_samples

# Step function registry
STEP_FUNCTIONS = {
    "qc": lambda c, a, **kw: filter_low_quality_samples(c, a, **kw),
    "norm_median": lambda c, a, **kw: norm_median(c, a, **kw),
    "norm_mean": lambda c, a, **kw: norm_mean(c, a, **kw),
    "log_transform": lambda c, a, **kw: log_transform(c, a, **kw),
    "impute_knn": lambda c, a, **kw: impute_knn(c, a, **kw),
    "impute_missforest": lambda c, a, **kw: impute_missforest(c, a, **kw),
    "impute_lls": lambda c, a, **kw: impute_lls(c, a, **kw),
    "batch_combat": lambda c, a, **kw: integrate_combat(c, a, **kw),
    "batch_harmony": lambda c, a, **kw: integrate_harmony(c, a, **kw),
    "batch_mnn": lambda c, a, **kw: integrate_mnn(c, a, **kw),
    "pca": lambda c, a, **kw: reduce_pca(c, a, **kw),
    "kmeans": lambda c, a, **kw: cluster_kmeans(c, a, **kw),
}

# Predefined pipeline configurations
PIPELINE_CONFIGS = {
    "classic": {
        "name": "Classic Pipeline",
        "steps": [
            ("qc", {}),
            ("norm_median", {}),
            ("log_transform", {}),
            ("impute_knn", {}),
            ("pca", {"n_components": 50}),
            ("kmeans", {"n_clusters": 5}),
        ],
    },
    "batch_corrected": {
        "name": "Batch Corrected Pipeline",
        "steps": [
            ("qc", {}),
            ("norm_median", {}),
            ("log_transform", {}),
            ("impute_knn", {}),
            ("batch_combat", {}),
            ("pca", {"n_components": 50}),
            ("kmeans", {"n_clusters": 5}),
        ],
    },
    "advanced": {
        "name": "Advanced Pipeline",
        "steps": [
            ("qc", {}),
            ("norm_median", {}),
            ("log_transform", {}),
            ("impute_missforest", {}),
            ("batch_harmony", {}),
            ("pca", {"n_components": 50}),
            ("kmeans", {"n_clusters": 5}),
        ],
    },
    "fast": {
        "name": "Fast Pipeline",
        "steps": [
            ("qc", {}),
            ("norm_median", {}),
            ("log_transform", {}),
            ("impute_lls", {}),
            ("pca", {"n_components": 30}),
            ("kmeans", {"n_clusters": 5}),
        ],
    },
    "conservative": {
        "name": "Conservative Pipeline",
        "steps": [
            ("qc", {}),
            ("norm_median", {}),
            ("log_transform", {}),
            # No imputation
            # No batch correction
            ("pca", {"n_components": 50}),
            ("kmeans", {"n_clusters": 5}),
        ],
    },
}


def run_pipeline(
    container: ScpContainer,
    pipeline_name: str | None = None,
    steps: list[tuple[str, dict]] | None = None,
    assay_name: str = "proteins",
    verbose: bool = False,
) -> tuple[ScpContainer, dict[str, Any]]:
    """
    Execute an analysis pipeline on a container.

    Parameters
    ----------
    container : ScpContainer
        Input data container
    pipeline_name : str, optional
        Name of predefined pipeline ("classic", "batch_corrected", "advanced", "fast", "conservative")
    steps : list of tuples, optional
        Custom steps as [(step_name, params_dict), ...]
    assay_name : str
        Name of assay to process
    verbose : bool
        Print progress information

    Returns
    -------
    tuple[ScpContainer, dict]
        Processed container and execution log

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> container = create_test_container()
    >>> result, log = run_pipeline(container, "classic")
    >>> result, log = run_pipeline(container, steps=[("qc", {}), ("norm_median", {})])
    """
    # Get pipeline configuration
    if steps is None:
        if pipeline_name is None:
            pipeline_name = "classic"
        if pipeline_name not in PIPELINE_CONFIGS:
            raise ValueError(
                f"Unknown pipeline: {pipeline_name}. Available: {list(PIPELINE_CONFIGS.keys())}"
            )
        config = PIPELINE_CONFIGS[pipeline_name]
        steps = config["steps"]
        pipeline_label = config["name"]
    else:
        pipeline_label = "Custom Pipeline"

    # Check assay exists
    if assay_name not in container.assays:
        available = list(container.assays.keys())
        if available:
            assay_name = available[0]
            if verbose:
                print(f"Using assay: {assay_name}")
        else:
            raise ValueError("No assays found in container")

    # Execution log
    execution_log = {
        "pipeline_name": pipeline_label,
        "assay_name": assay_name,
        "steps": [],
        "total_time": 0.0,
    }

    start_time = time.time()

    # Execute steps
    for step_name, params in steps:
        if step_name not in STEP_FUNCTIONS:
            raise ValueError(f"Unknown step: {step_name}")

        step_start = time.time()
        try:
            func = STEP_FUNCTIONS[step_name]
            container = func(container, assay_name, **params)
            step_time = time.time() - step_start

            execution_log["steps"].append(
                {
                    "name": step_name,
                    "status": "success",
                    "time": step_time,
                }
            )

            if verbose:
                print(f"  [{step_name}] completed in {step_time:.2f}s")

        except Exception as e:
            step_time = time.time() - step_start
            execution_log["steps"].append(
                {
                    "name": step_name,
                    "status": "failed",
                    "error": str(e),
                    "time": step_time,
                }
            )
            raise RuntimeError(f"Step '{step_name}' failed: {e}") from e

    execution_log["total_time"] = time.time() - start_time

    if verbose:
        print(f"Pipeline completed in {execution_log['total_time']:.2f}s")

    return container, execution_log


def get_available_pipelines() -> list[str]:
    """Return list of available pipeline names."""
    return list(PIPELINE_CONFIGS.keys())


def get_pipeline_description(pipeline_name: str) -> str:
    """Return description of a pipeline."""
    if pipeline_name not in PIPELINE_CONFIGS:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return PIPELINE_CONFIGS[pipeline_name]["name"]
