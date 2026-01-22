"""Pipeline C: Advanced Pipeline (with latest methods).

This pipeline uses state-of-the-art methods:
QC → Quantile normalization → Log transform → MissForest imputation → Harmony → UMAP → Leiden
"""

from __future__ import annotations

from typing import Any

from scptensor.cluster import cluster_leiden
from scptensor.core.structures import ScpContainer
from scptensor.dim_reduction import reduce_umap
from scptensor.impute import impute_mf
from scptensor.integration import integrate_harmony
from scptensor.normalization import norm_log, norm_quartile
from scptensor.qc import qc_basic

from .base import BasePipeline, load_pipeline_config


class PipelineC(BasePipeline):
    """
    Advanced Pipeline: Using state-of-the-art methods.

    This pipeline implements the latest advances in single-cell proteomics
    analysis, including quantile normalization, MissForest imputation,
    Harmony batch correction, UMAP dimensionality reduction, and Leiden clustering.

    Steps:
        1. Quality Control (basic)
        2. Quantile normalization
        3. Log transform
        4. MissForest imputation
        5. Harmony batch correction
        6. UMAP dimensionality reduction
        7. Leiden clustering

    Parameters
    ----------
    config : Dict[str, Any], optional
        Pipeline configuration. If None, loads from default config file.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from docs.comparison_study.pipelines.pipeline_c import PipelineC
    >>> container = create_test_container()
    >>> pipeline = PipelineC()
    >>> result = pipeline.run(container)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Pipeline C.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Pipeline configuration dictionary
        """
        if config is None:
            config = load_pipeline_config("pipeline_c")

        global_config = config.get("global", {})
        super().__init__(
            name=config["name"], config=config, random_seed=global_config.get("random_seed", 42)
        )

    def run(self, container: ScpContainer) -> ScpContainer:
        """
        Execute the advanced pipeline.

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

        # Step 2: Normalization (Quantile)
        container = self._execute_step(
            "normalization", self._run_normalization, container, assay_name
        )

        # Step 3: Log transform
        container = self._execute_step(
            "log_transform", self._run_log_transform, container, assay_name
        )

        # Step 4: Imputation (MissForest)
        container = self._execute_step("imputation", self._run_imputation, container, assay_name)

        # Step 5: Batch correction (Harmony)
        container = self._execute_step(
            "batch_correction", self._run_batch_correction, container, assay_name
        )

        # Step 6: Dimensionality reduction (UMAP)
        container = self._execute_step(
            "dim_reduction", self._run_dim_reduction, container, assay_name
        )

        # Step 7: Clustering (Leiden)
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
        Run quantile normalization.

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

        return norm_quartile(
            container, assay_name=assay_name, source_layer=source_layer, new_layer_name="normalized"
        )

    def _run_log_transform(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run log transformation."""
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
        Run MissForest imputation.

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
        n_estimators = params.get("n_estimators", 100)
        max_iter = params.get("max_iter", 10)

        return impute_mf(
            container,
            assay_name=assay_name,
            source_layer="log_transformed",
            new_layer_name="imputed",
            n_estimators=n_estimators,
            max_iter=max_iter,
        )

    def _run_batch_correction(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run Harmony batch correction.

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
        lambda_param = params.get("lambda_param", 1.0)

        return integrate_harmony(
            container,
            assay_name=assay_name,
            base_layer="imputed",
            batch_key=batch_key,
            new_layer_name="harmony",
            lamb=lambda_param,
        )

    def _run_dim_reduction(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run UMAP dimensionality reduction.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Name of assay to process

        Returns
        -------
        ScpContainer
            Container with UMAP embeddings
        """
        params = self.config["steps"]["dim_reduction"]["params"]
        n_components = params.get("n_components", 2)
        n_neighbors = params.get("n_neighbors", 15)
        min_dist = params.get("min_dist", 0.1)

        return reduce_umap(
            container,
            assay_name=assay_name,
            base_layer="harmony",
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

    def _run_clustering(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run Leiden clustering.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Name of assay to process

        Returns
        -------
        ScpContainer
            Container with cluster assignments
        """
        params = self.config["steps"]["clustering"]["params"]
        resolution = params.get("resolution", 1.0)
        n_neighbors = params.get("n_neighbors", 15)

        return cluster_leiden(
            container,
            assay_name="umap",
            base_layer="embedding",
            resolution=resolution,
            n_neighbors=n_neighbors,
        )
