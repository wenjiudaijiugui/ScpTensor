"""
Metrics computation for benchmark evaluation.
"""

import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import numpy as np
import psutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from scptensor.core import MatrixOps, ScpContainer

from .core import BiologicalMetrics, ComputationalMetrics, TechnicalMetrics


# Constants for default values
_DEFAULT_SCALE = 1.0
_DEFAULT_BATCH_MIXING = 0.5
_EPSILON = 1e-8


@contextmanager
def _track_resources():
    """Context manager for tracking time and memory usage.

    Yields
    ------
    tuple[float, float]
        (runtime_seconds, memory_mb)
    """
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        yield
    finally:
        runtime = time.time() - start_time
        memory = process.memory_info().rss / 1024 / 1024 - start_memory


class MetricsEngine:
    """Engine for computing comprehensive benchmark metrics.

    Attributes
    ----------
    start_time : float | None
        Start time for timing measurements.
    start_memory : float | None
        Starting memory usage in MB.
    """

    __slots__ = ("start_time", "start_memory")

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.start_memory: float | None = None

    def start_timing(self) -> None:
        """Start timing a method run."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    def stop_timing(self) -> tuple[float, float]:
        """Stop timing and return runtime and memory usage.

        Returns
        -------
        tuple[float, float]
            (runtime_seconds, memory_usage_mb)

        Raises
        ------
        RuntimeError
            If timing was not started.
        """
        if self.start_time is None or self.start_memory is None:
            raise RuntimeError("Timing not started. Call start_timing() first.")

        runtime = time.time() - self.start_time
        memory = psutil.Process().memory_info().rss / 1024 / 1024 - self.start_memory

        self.start_time = None
        self.start_memory = None

        return runtime, max(memory, 0.0)

    def evaluate_technical(
        self,
        input_container: ScpContainer,
        output_container: ScpContainer,
        assay_name: str = "protein",
    ) -> TechnicalMetrics:
        """Compute technical quality metrics.

        Parameters
        ----------
        input_container : ScpContainer
            Input data container.
        output_container : ScpContainer
            Output data container after processing.
        assay_name : str
            Name of the assay to evaluate.

        Returns
        -------
        TechnicalMetrics
            Computed technical metrics.
        """
        assay = input_container.assays[assay_name]

        # Get input matrix (prefer 'raw', otherwise first layer)
        input_matrix = assay.layers.get("raw")
        if input_matrix is None:
            input_matrix = next(iter(assay.layers.values()))

        # Get output matrix (prefer last added layer)
        output_layers = list(output_container.assays[assay_name].layers.keys())
        layer_name = output_layers[-1] if len(output_layers) > 1 else output_layers[0]
        output_matrix = output_container.assays[assay_name].layers[layer_name]

        return TechnicalMetrics(
            data_recovery_rate=_compute_data_recovery(input_matrix, output_matrix),
            variance_preservation=_compute_variance_preservation(input_matrix, output_matrix),
            sparsity_preservation=_compute_sparsity_preservation(input_matrix, output_matrix),
            batch_mixing_score=_compute_batch_mixing(output_container, layer_name),
            signal_to_noise_ratio=_compute_signal_to_noise(output_matrix),
            missing_value_pattern_score=_compute_missing_pattern_score(
                input_matrix, output_matrix
            ),
        )

    def evaluate_biological(
        self,
        output_container: ScpContainer,
        ground_truth_groups: np.ndarray | None = None,
        assay_name: str = "protein",
    ) -> BiologicalMetrics:
        """Compute biological fidelity metrics.

        Parameters
        ----------
        output_container : ScpContainer
            Output data container.
        ground_truth_groups : np.ndarray | None
            Ground truth group labels.
        assay_name : str
            Name of the assay to evaluate.

        Returns
        -------
        BiologicalMetrics
            Computed biological metrics.
        """
        output_layers = list(output_container.assays[assay_name].layers.keys())
        layer_name = output_layers[-1] if len(output_layers) > 1 else output_layers[0]

        X_valid = MatrixOps.apply_mask_to_values(
            output_container.assays[assay_name].layers[layer_name],
            operation="nan",
        ).X

        if ground_truth_groups is None and "group" in output_container.obs.columns:
            ground_truth_groups = output_container.obs["group"].to_numpy()

        return BiologicalMetrics(
            group_separation=_compute_group_separation(X_valid, ground_truth_groups),
            biological_signal_preservation=_compute_signal_preservation(X_valid),
            clustering_consistency=_compute_clustering_consistency(X_valid, ground_truth_groups),
            biological_variance_explained=_compute_biological_variance_explained(
                output_container, layer_name, ground_truth_groups
            ),
        )

    @staticmethod
    def evaluate_computational(
        runtime_seconds: float, memory_usage_mb: float
    ) -> ComputationalMetrics:
        """Compute computational efficiency metrics.

        Parameters
        ----------
        runtime_seconds : float
            Runtime in seconds.
        memory_usage_mb : float
            Memory usage in MB.

        Returns
        -------
        ComputationalMetrics
            Computational metrics.
        """
        return ComputationalMetrics(
            runtime_seconds=runtime_seconds,
            memory_usage_mb=memory_usage_mb,
            scalability_factor=_DEFAULT_SCALE,
            cpu_utilization_percent=None,
        )


# =============================================================================
# Metric computation functions (extracted for better testability)
# =============================================================================


def _compute_data_recovery(input_matrix: Any, output_matrix: Any) -> float:
    """Compute proportion of missing values that were successfully handled."""
    input_missing = MatrixOps.get_missing_mask(input_matrix)
    output_valid = MatrixOps.get_valid_mask(output_matrix)
    recovered = input_missing & output_valid

    missing_count = np.sum(input_missing)
    if missing_count == 0:
        return 1.0

    return np.sum(recovered) / missing_count


def _compute_variance_preservation(input_matrix: Any, output_matrix: Any) -> float:
    """Compute how well original variance structure is preserved."""
    input_X = MatrixOps.apply_mask_to_values(input_matrix, operation="zero").X
    output_X = MatrixOps.apply_mask_to_values(output_matrix, operation="zero").X

    if input_X.shape[0] < 2 or input_X.shape[1] < 2:
        return 1.0

    try:
        n_components = min(10, input_X.shape[1], input_X.shape[0])
        pca_input = PCA(n_components=n_components)
        pca_output = PCA(n_components=min(10, output_X.shape[1], output_X.shape[0]))

        pca_input.fit(input_X)
        pca_output.fit(output_X)

        min_comp = min(len(pca_input.explained_variance_ratio_),
                       len(pca_output.explained_variance_ratio_))

        if min_comp == 0:
            return 0.0

        return np.corrcoef(
            pca_input.explained_variance_ratio_[:min_comp],
            pca_output.explained_variance_ratio_[:min_comp],
        )[0, 1]
    except Exception:
        return 0.0


def _compute_sparsity_preservation(input_matrix: Any, output_matrix: Any) -> float:
    """Compute preservation of original sparsity patterns."""
    input_valid = np.sum(MatrixOps.get_valid_mask(input_matrix)) / input_matrix.X.size
    output_valid = np.sum(MatrixOps.get_valid_mask(output_matrix)) / output_matrix.X.size

    return 1.0 - abs(input_valid - output_valid)


def _compute_batch_mixing(container: ScpContainer, layer_name: str) -> float:
    """Compute how well batches are mixed (higher = better mixing)."""
    if "batch" not in container.obs.columns:
        return _DEFAULT_BATCH_MIXING

    X = MatrixOps.apply_mask_to_values(
        container.assays["protein"].layers[layer_name],
        operation="nan",
    ).X

    batch_labels = container.obs["batch"].to_numpy()

    try:
        valid_rows = ~np.isnan(X).all(axis=1)
        X_valid = X[valid_rows]
        batch_valid = batch_labels[valid_rows]

        if len(np.unique(batch_valid)) < 2 or len(X_valid) < 2:
            return _DEFAULT_BATCH_MIXING

        score = silhouette_score(X_valid, batch_valid)
        return max(0.0, 1.0 - score)  # Convert to mixing score
    except Exception:
        return _DEFAULT_BATCH_MIXING


def _compute_signal_to_noise(matrix: Any) -> float:
    """Compute signal-to-noise ratio in processed data."""
    X = MatrixOps.apply_mask_to_values(matrix, operation="nan").X

    try:
        valid_rows = ~np.isnan(X).all(axis=1)
        X_valid = X[valid_rows]

        if len(X_valid) == 0:
            return 0.0

        signal = np.nanvar(X_valid, axis=1)
        noise = np.nanmean(
            np.abs(X_valid - np.nanmean(X_valid, axis=1, keepdims=True)),
            axis=1,
        )

        snr = signal / (noise + _EPSILON)
        return float(np.nanmean(snr))
    except Exception:
        return 0.0


def _compute_missing_pattern_score(input_matrix: Any, output_matrix: Any) -> float:
    """Compute quality of missing value pattern handling."""
    input_stats = MatrixOps.get_mask_statistics(input_matrix)
    output_stats = MatrixOps.get_mask_statistics(output_matrix)

    input_rate = input_stats["VALID"]["percentage"] / 100.0
    output_rate = output_stats["VALID"]["percentage"] / 100.0

    return min(1.0, output_rate / max(input_rate, _EPSILON))


def _compute_group_separation(X: np.ndarray, groups: np.ndarray | None) -> float:
    """Compute separation between biological groups."""
    if groups is None or len(np.unique(groups)) < 2:
        return 0.0

    try:
        valid_rows = ~np.isnan(X).all(axis=1)
        X_valid = X[valid_rows]
        groups_valid = groups[valid_rows]

        if len(np.unique(groups_valid)) < 2 or len(X_valid) < 2:
            return 0.0

        return silhouette_score(X_valid, groups_valid)
    except Exception:
        return 0.0


def _compute_signal_preservation(X: np.ndarray) -> float:
    """Compute preservation of biological signals via PCA variance."""
    try:
        n_components = min(5, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return float(np.sum(pca.explained_variance_ratio_))
    except Exception:
        return 0.0


def _compute_clustering_consistency(X: np.ndarray, groups: np.ndarray | None) -> float:
    """Compute clustering consistency with ground truth."""
    if groups is None:
        return 0.0

    try:
        valid_rows = ~np.isnan(X).all(axis=1)
        X_valid = X[valid_rows]
        groups_valid = groups[valid_rows]

        n_unique = len(np.unique(groups_valid))
        if n_unique < 2 or len(X_valid) < 2:
            return 0.0

        kmeans = KMeans(n_clusters=n_unique, random_state=42, n_init=10)
        predicted = kmeans.fit_predict(X_valid)

        return adjusted_rand_score(groups_valid, predicted)
    except Exception:
        return 0.0


def _compute_biological_variance_explained(
    container: ScpContainer, layer_name: str, groups: np.ndarray | None
) -> float:
    """Compute percentage of variance explained by biological factors."""
    if groups is None:
        return 0.0

    try:
        X = MatrixOps.apply_mask_to_values(
            container.assays["protein"].layers[layer_name],
            operation="nan",
        ).X

        valid_rows = ~np.isnan(X).all(axis=1)
        X_valid = X[valid_rows]
        groups_valid = groups[valid_rows]

        unique_groups = np.unique(groups_valid)
        if len(unique_groups) < 2:
            return 0.0

        overall_mean = np.nanmean(X_valid)
        total_n = len(X_valid)

        # Between-group variance
        between_var = 0.0
        for group in unique_groups:
            mask = groups_valid == group
            group_mean = np.nanmean(X_valid[mask])
            n_group = np.sum(mask)
            between_var += (group_mean - overall_mean) ** 2 * n_group

        between_var /= (total_n - 1)

        # Within-group variance
        within_var = 0.0
        for group in unique_groups:
            mask = groups_valid == group
            group_var = np.nanvar(X_valid[mask], ddof=1)
            n_group = np.sum(mask)
            within_var += group_var * (n_group - 1)

        within_var /= (total_n - len(unique_groups))

        total_var = between_var + within_var
        return (between_var / total_var) if total_var > 0 else 0.0
    except Exception:
        return 0.0
