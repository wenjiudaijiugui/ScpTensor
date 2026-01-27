"""Pipeline B: Batch Correction Pipeline (for multi-batch data).

This pipeline includes ComBat batch correction:
QC → Median normalization → Log transform → KNN imputation → ComBat → PCA → K-means
"""

from __future__ import annotations

from typing import Any

from scptensor.cluster import cluster_kmeans
from scptensor.core.structures import ScpContainer
from scptensor.dim_reduction import reduce_pca
from scptensor.impute import impute_knn
from scptensor.integration import integrate_combat
from scptensor.normalization import log_transform, norm_median
from scptensor.qc.qc_sample import filter_low_quality_samples

from .base import BasePipeline, load_pipeline_config


class PipelineB(BasePipeline):
    """
    Batch Correction Pipeline: For multi-batch data.

    This pipeline extends the classic workflow with ComBat batch correction
    to handle technical batch effects in multi-batch experiments.

    Steps:
        1. Quality Control (basic)
        2. Median normalization
        3. Log transform
        4. KNN imputation
        5. ComBat batch correction
        6. PCA dimensionality reduction
        7. K-means clustering

    Parameters
    ----------
    config : Dict[str, Any], optional
        Pipeline configuration. If None, loads from default config file.

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from studies.pipelines.pipeline_b import PipelineB
    >>> container = create_test_container()
    >>> pipeline = PipelineB()
    >>> result = pipeline.run(container)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Pipeline B.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Pipeline configuration dictionary
        """
        if config is None:
            config = load_pipeline_config("pipeline_b")

        global_config = config.get("global", {})
        super().__init__(
            name=config["name"], config=config, random_seed=global_config.get("random_seed", 42)
        )

    def run(self, container: ScpContainer) -> ScpContainer:
        """
        Execute the batch correction pipeline.

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

        # Step 2: Normalization
        container = self._execute_step(
            "normalization", self._run_normalization, container, assay_name
        )

        # Step 3: Log transform
        container = self._execute_step(
            "log_transform", self._run_log_transform, container, assay_name
        )

        # Step 4: Imputation
        container = self._execute_step("imputation", self._run_imputation, container, assay_name)

        # Step 5: Batch correction
        container = self._execute_step(
            "batch_correction", self._run_batch_correction, container, assay_name
        )

        # Step 6: Dimensionality reduction
        container = self._execute_step(
            "dim_reduction", self._run_dim_reduction, container, assay_name
        )

        # Step 7: Clustering
        container = self._execute_step("clustering", self._run_clustering, container, assay_name)

        return container

    def _run_qc(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run quality control."""
        params = self.config["steps"]["qc"]["params"]
        min_features = params.get("min_features_per_cell", 200)
        min_cells = params.get("min_cells_per_feature", 3)

        return filter_low_quality_samples(
            container, assay_name=assay_name, min_features=min_features, min_cells=min_cells
        )

    def _run_normalization(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run median normalization."""
        params = self.config["steps"]["normalization"]["params"]
        source_layer = params.get("target_layer", "raw")

        return norm_median(
            container, assay_name=assay_name, source_layer=source_layer, new_layer_name="normalized"
        )

    def _run_log_transform(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run log transformation."""
        params = self.config["steps"]["log_transform"]["params"]
        base = params.get("base", 2.0)
        offset = params.get("offset", 1.0)

        return log_transform(
            container,
            assay_name=assay_name,
            source_layer="normalized",
            new_layer_name="log_transformed",
            base=base,
            offset=offset,
        )

    def _run_imputation(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run KNN imputation."""
        params = self.config["steps"]["imputation"]["params"]
        n_neighbors = params.get("n_neighbors", 5)

        return impute_knn(
            container,
            assay_name=assay_name,
            source_layer="log_transformed",
            new_layer_name="imputed",
            k=n_neighbors,
        )

    def _run_batch_correction(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """
        Run ComBat batch correction.

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

        return integrate_combat(
            container,
            assay_name=assay_name,
            base_layer="imputed",
            batch_key=batch_key,
            new_layer_name="combat",
        )

    def _run_dim_reduction(self, container: ScpContainer, assay_name: str) -> ScpContainer:
        """Run PCA dimensionality reduction."""
        params = self.config["steps"]["dim_reduction"]["params"]
        n_components = params.get("n_components", 50)

        return reduce_pca(
            container, assay_name=assay_name, base_layer_name="combat", n_components=n_components
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
