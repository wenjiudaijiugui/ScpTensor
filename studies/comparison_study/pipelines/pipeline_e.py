"""Pipeline E: Conservative Pipeline (minimal assumptions).

This pipeline makes minimal assumptions about data distribution:
QC → VSN normalization → Log transform → PPCA imputation → No batch correction → PCA → K-means

Note: VSN (Variance Stabilizing Normalization) is not directly available in ScpTensor.
This pipeline uses log normalization as a substitute.
"""

from __future__ import annotations

import warnings
from typing import Any

from scptensor.cluster import cluster_kmeans
from scptensor.core.structures import ScpContainer
from scptensor.dim_reduction import reduce_pca
from scptensor.impute import impute_ppca
from scptensor.normalization import norm_log
from scptensor.qc import qc_basic

from .base import BasePipeline, load_pipeline_config


class PipelineE(BasePipeline):
    """
    Conservative Pipeline: Minimal assumptions about data distribution.

    This pipeline uses conservative methods that make minimal assumptions
    about the underlying data distribution. VSN normalization would be
    ideal but is not available, so we use log normalization instead.

    Steps:
        1. Quality Control (basic)
        2. VSN normalization (substituted with log normalization)
        3. Log transform
        4. PPCA imputation
        5. No batch correction
        6. PCA dimensionality reduction
        7. K-means clustering

    Parameters
    ----------
    config : Dict[str, Any], optional
        Pipeline configuration. If None, loads from default config file.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from docs.comparison_study.pipelines.pipeline_e import PipelineE
    >>> container = create_test_container()
    >>> pipeline = PipelineE()
    >>> result = pipeline.run(container)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Pipeline E.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Pipeline configuration dictionary
        """
        if config is None:
            config = load_pipeline_config("pipeline_e")

        global_config = config.get("global", {})
        super().__init__(
            name=config["name"], config=config, random_seed=global_config.get("random_seed", 42)
        )

    def run(self, container: ScpContainer) -> ScpContainer:
        """
        Execute the conservative pipeline.

        Parameters
        ----------
        container : ScpContainer
            Input data container

        Returns
        -------
        ScpContainer
            Processed container with all analysis results
        """
        assay_name = "proteins"
        if assay_name not in container.assays:
            assay_name = list(container.assays.keys())[0]

        # Step 1: Quality Control
        container = self._execute_step("qc", self._run_qc, container, assay_name)

        # Step 2: Normalization (VSN - substituted with log)
        container = self._execute_step(
            "normalization", self._run_normalization, container, assay_name
        )

        # Step 3: Log transform (optional after log normalization)
        if self.config["steps"]["log_transform"]["enabled"]:
            container = self._execute_step(
                "log_transform", self._run_log_transform, container, assay_name
            )

        # Step 4: Imputation (PPCA)
        container = self._execute_step("imputation", self._run_imputation, container, assay_name)

        # Step 5: Batch correction (disabled for Pipeline E)
        # Skipped

        # Step 6: Dimensionality reduction (PCA)
        container = self._execute_step(
            "dim_reduction", self._run_dim_reduction, container, assay_name
        )

        # Step 7: Clustering (K-means)
        container = self._execute_step("clustering", self._run_clustering, container, assay_name)

        return container

    def _run_qc(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run quality control."""
        params = self.config["steps"]["qc"]["params"]
        min_features = params.get("min_features_per_cell", 200)
        min_cells = params.get("min_cells_per_feature", 3)

        return qc_basic(
            container, assay_name=assay_name, min_features=min_features, min_cells=min_cells
        )

    def _run_normalization(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run VSN normalization (substituted with log normalization).

        Note: VSN (Variance Stabilizing Normalization) is not available in
        ScpTensor. We use log normalization as a conservative alternative
        that also stabilizes variance.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Name of assay to process

        Returns
        -------
        ScpContainer
            Container with normalized data
        """
        params = self.config["steps"]["normalization"]["params"]
        source_layer = params.get("target_layer", "raw")

        warnings.warn(
            "VSN normalization is not available in ScpTensor. "
            "Using log normalization as a substitute. "
            "Results should still be comparable for the comparison study.",
            stacklevel=2,
        )

        return norm_log(
            container,
            assay_name=assay_name,
            source_layer=source_layer,
            new_layer_name="normalized",
            base=2.0,
            offset=1.0,
        )

    def _run_log_transform(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run additional log transform if needed.

        This step may be redundant if VSN normalization was already applied,
        but we include it for consistency with the pipeline specification.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Name of assay to process

        Returns
        -------
        ScpContainer
            Container with log-transformed data
        """
        params = self.config["steps"]["log_transform"]["params"]
        base = params.get("base", 2.0)
        offset = params.get("offset", 1.0)

        return norm_log(
            container,
            assay_name=assay_name,
            source_layer="normalized",
            new_layer_name="log_transformed",
            base=base,
            offset=offset,
        )

    def _run_imputation(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run PPCA imputation.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Name of assay to process

        Returns
        -------
        ScpContainer
            Container with imputed data
        """
        params = self.config["steps"]["imputation"]["params"]
        n_components = params.get("n_components", 20)

        # Use the last available layer (normalized or log_transformed)
        source_layer = (
            "log_transformed" if self.config["steps"]["log_transform"]["enabled"] else "normalized"
        )

        return impute_ppca(
            container,
            assay_name=assay_name,
            source_layer=source_layer,
            new_layer_name="imputed",
            n_components=n_components,
        )

    def _run_dim_reduction(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run PCA dimensionality reduction."""
        params = self.config["steps"]["dim_reduction"]["params"]
        n_components = params.get("n_components", 50)
        center = params.get("center", True)

        return reduce_pca(
            container,
            assay_name=assay_name,
            base_layer_name="imputed",
            n_components=n_components,
            center=center,
        )

    def _run_clustering(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run K-means clustering."""
        params = self.config["steps"]["clustering"]["params"]
        n_clusters = params.get("n_clusters", 8)

        return cluster_kmeans(
            container,
            assay_name="pca",
            base_layer="scores",
            n_clusters=n_clusters,
            random_state=self.random_seed,
        )
