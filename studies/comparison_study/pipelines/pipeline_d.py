"""Pipeline D: Performance-Optimized Pipeline (for large-scale data).

This pipeline prioritizes computational efficiency:
QC → Z-score normalization → Lazy validation → SVD imputation → MNN correction → PCA → K-means
"""

from __future__ import annotations

import warnings
from typing import Any

from scptensor.cluster import cluster_kmeans
from scptensor.core.structures import ScpContainer
from scptensor.dim_reduction import reduce_pca
from scptensor.impute import impute_svd
from scptensor.integration import integrate_mnn
from scptensor.normalization import norm_zscore
from scptensor.qc import qc_basic

from .base import BasePipeline, load_pipeline_config


class PipelineD(BasePipeline):
    """
    Performance-Optimized Pipeline: For large-scale data.

    This pipeline prioritizes computational efficiency while maintaining
    good analysis quality. It uses fast methods like Z-score normalization,
    SVD imputation, and MNN batch correction.

    Steps:
        1. Quality Control (basic)
        2. Z-score normalization
        3. Lazy validation (if enabled)
        4. SVD imputation
        5. MNN batch correction
        6. PCA dimensionality reduction
        7. K-means clustering

    Parameters
    ----------
    config : Dict[str, Any], optional
        Pipeline configuration. If None, loads from default config file.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from docs.comparison_study.pipelines.pipeline_d import PipelineD
    >>> container = create_test_container()
    >>> pipeline = PipelineD()
    >>> result = pipeline.run(container)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Pipeline D.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Pipeline configuration dictionary
        """
        if config is None:
            config = load_pipeline_config("pipeline_d")

        global_config = config.get("global", {})
        super().__init__(
            name=config["name"], config=config, random_seed=global_config.get("random_seed", 42)
        )

    def run(self, container: ScpContainer) -> ScpContainer:
        """
        Execute the performance-optimized pipeline.

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

        # Step 2: Normalization (Z-score)
        container = self._execute_step(
            "normalization", self._run_normalization, container, assay_name
        )

        # Step 3: Lazy validation (if enabled)
        if self.config["steps"].get("lazy_validation", {}).get("enabled", False):
            container = self._execute_step(
                "lazy_validation", self._run_lazy_validation, container, assay_name
            )

        # Step 4: Imputation (SVD)
        container = self._execute_step("imputation", self._run_imputation, container, assay_name)

        # Step 5: Batch correction (MNN)
        container = self._execute_step(
            "batch_correction", self._run_batch_correction, container, assay_name
        )

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
        min_features = params.get("min_features_per_feature", 200)  # Note: typo in config
        if min_features == 100:  # Fix based on config
            min_features = params.get("min_features_per_cell", 200)
        min_cells = params.get("min_cells_per_feature", 3)

        return qc_basic(
            container, assay_name=assay_name, min_features=min_features, min_cells=min_cells
        )

    def _run_normalization(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run Z-score normalization.

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

        return norm_zscore(
            container, assay_name=assay_name, source_layer=source_layer, new_layer_name="normalized"
        )

    def _run_lazy_validation(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run lazy validation setup.

        Note: Lazy validation is a performance optimization that delays
        validation until data is accessed. This is a placeholder for
        future implementation.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Name of assay to process

        Returns
        -------
        ScpContainer
            Container with lazy validation enabled
        """
        params = self.config["steps"]["lazy_validation"]["params"]
        validate_on_access = params.get("validate_on_access", True)

        if validate_on_access:
            # Placeholder: In a full implementation, this would set up
            # lazy validation for the container
            warnings.warn(
                "Lazy validation is enabled but not fully implemented. "
                "Standard validation will be used."
            )

        return container

    def _run_imputation(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run SVD imputation.

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

        return impute_svd(
            container,
            assay_name=assay_name,
            source_layer="normalized",
            new_layer_name="imputed",
            n_components=n_components,
        )

    def _run_batch_correction(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run MNN batch correction.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Name of assay to process

        Returns
        -------
        ScpContainer
            Container with batch-corrected data
        """
        params = self.config["steps"]["batch_correction"]["params"]
        batch_key = params.get("batch_key", "batch")
        k = params.get("k", 20)

        return integrate_mnn(
            container, assay_name=assay_name, base_layer="imputed", batch_key=batch_key, k=k
        )

    def _run_dim_reduction(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run PCA dimensionality reduction."""
        params = self.config["steps"]["dim_reduction"]["params"]
        n_components = params.get("n_components", 50)
        center = params.get("center", True)

        return reduce_pca(
            container,
            assay_name=assay_name,
            base_layer_name="mnn_corrected",
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
