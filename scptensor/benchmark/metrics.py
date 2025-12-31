"""
Metrics computation for benchmark evaluation.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.stats import entropy
from scptensor.core import ScpContainer, MatrixOps
from .core import TechnicalMetrics, BiologicalMetrics, ComputationalMetrics


class MetricsEngine:
    """Engine for computing comprehensive benchmark metrics."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None

    def start_timing(self):
        """Start timing a method run."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def stop_timing(self) -> Tuple[float, float]:
        """Stop timing and return runtime and memory usage."""
        if self.start_time is None or self.start_memory is None:
            raise RuntimeError("Timing not started. Call start_timing() first.")

        runtime = time.time() - self.start_time
        memory = psutil.Process().memory_info().rss / 1024 / 1024 - self.start_memory

        self.start_time = None
        self.start_memory = None

        return runtime, max(memory, 0)  # Ensure non-negative

    def evaluate_technical(
        self,
        input_container: ScpContainer,
        output_container: ScpContainer,
        assay_name: str = 'protein'
    ) -> TechnicalMetrics:
        """Compute technical quality metrics."""

        input_matrix = input_container.assays[assay_name].layers.get('raw', None)
        if input_matrix is None:
            # Try to get the first available layer
            input_layers = list(input_container.assays[assay_name].layers.keys())
            input_matrix = input_container.assays[assay_name].layers[input_layers[0]]

        # Find the processed layer (typically the last one added)
        output_layers = list(output_container.assays[assay_name].layers.keys())
        processed_layer = output_layers[-1] if len(output_layers) > 1 else output_layers[0]
        output_matrix = output_container.assays[assay_name].layers[processed_layer]

        return TechnicalMetrics(
            data_recovery_rate=self._compute_data_recovery(input_matrix, output_matrix),
            variance_preservation=self._compute_variance_preservation(input_matrix, output_matrix),
            sparsity_preservation=self._compute_sparsity_preservation(input_matrix, output_matrix),
            batch_mixing_score=self._compute_batch_mixing(output_container, processed_layer),
            signal_to_noise_ratio=self._compute_signal_to_noise(output_matrix),
            missing_value_pattern_score=self._compute_missing_pattern_score(input_matrix, output_matrix)
        )

    def evaluate_biological(
        self,
        output_container: ScpContainer,
        ground_truth_groups: Optional[np.ndarray] = None,
        assay_name: str = 'protein'
    ) -> BiologicalMetrics:
        """Compute biological fidelity metrics."""

        output_layers = list(output_container.assays[assay_name].layers.keys())
        processed_layer = output_layers[-1] if len(output_layers) > 1 else output_layers[0]
        output_matrix = output_container.assays[assay_name].layers[processed_layer]

        # Use MatrixOps to get valid data
        X_valid = MatrixOps.apply_mask_to_values(
            output_container.assays[assay_name].layers[processed_layer],
            operation='nan'
        ).X

        if ground_truth_groups is None and 'group' in output_container.obs.columns:
            ground_truth_groups = output_container.obs['group'].to_numpy()

        return BiologicalMetrics(
            group_separation=self._compute_group_separation(X_valid, ground_truth_groups),
            biological_signal_preservation=self._compute_signal_preservation(X_valid),
            clustering_consistency=self._compute_clustering_consistency(X_valid, ground_truth_groups),
            biological_variance_explained=self._compute_biological_variance_explained(
                output_container, processed_layer, ground_truth_groups
            )
        )

    def evaluate_computational(self, runtime_seconds: float, memory_usage_mb: float) -> ComputationalMetrics:
        """Compute computational efficiency metrics."""
        return ComputationalMetrics(
            runtime_seconds=runtime_seconds,
            memory_usage_mb=memory_usage_mb,
            scalability_factor=1.0,  # Would need multiple runs to compute this
            cpu_utilization_percent=None  # Would need system monitoring
        )

    def _compute_data_recovery(self, input_matrix, output_matrix) -> float:
        """Compute proportion of missing values that were successfully handled."""
        input_missing = MatrixOps.get_missing_mask(input_matrix)
        output_valid = MatrixOps.get_valid_mask(output_matrix)
        originally_missing_now_valid = input_missing & output_valid

        originally_missing_count = np.sum(input_missing)
        if originally_missing_count == 0:
            return 1.0

        return np.sum(originally_missing_now_valid) / originally_missing_count

    def _compute_variance_preservation(self, input_matrix, output_matrix) -> float:
        """Compute how well original variance structure is preserved."""
        input_X = MatrixOps.apply_mask_to_values(input_matrix, operation='zero').X
        output_X = MatrixOps.apply_mask_to_values(output_matrix, operation='zero').X

        # Compute variance explained by comparing principal components
        if input_X.shape[0] < 2 or input_X.shape[1] < 2:
            return 1.0

        try:
            pca_input = PCA(n_components=min(10, input_X.shape[1], input_X.shape[0]))
            pca_output = PCA(n_components=min(10, output_X.shape[1], output_X.shape[0]))

            pc_input = pca_input.fit_transform(input_X)
            pc_output = pca_output.fit_transform(output_X)

            # Compare variance explained ratios
            min_components = min(len(pca_input.explained_variance_ratio_),
                               len(pca_output.explained_variance_ratio_))

            return np.corrcoef(
                pca_input.explained_variance_ratio_[:min_components],
                pca_output.explained_variance_ratio_[:min_components]
            )[0, 1]
        except:
            return 0.0

    def _compute_sparsity_preservation(self, input_matrix, output_matrix) -> float:
        """Compute preservation of original sparsity patterns."""
        input_sparsity = np.sum(MatrixOps.get_valid_mask(input_matrix)) / input_matrix.X.size
        output_sparsity = np.sum(MatrixOps.get_valid_mask(output_matrix)) / output_matrix.X.size

        return 1.0 - abs(input_sparsity - output_sparsity)

    def _compute_batch_mixing(self, output_container, layer_name: str) -> float:
        """Compute how well batches are mixed (negative of silhouette score on batch labels)."""
        if 'batch' not in output_container.obs.columns:
            return 0.5  # Neutral score if no batch information

        X = MatrixOps.apply_mask_to_values(
            output_container.assays['protein'].layers[layer_name],
            operation='nan'
        ).X

        batch_labels = output_container.obs['batch'].to_numpy()

        try:
            # Remove rows with all NaN values
            valid_rows = ~np.isnan(X).all(axis=1)
            X_valid = X[valid_rows]
            batch_valid = batch_labels[valid_rows]

            if len(np.unique(batch_valid)) < 2 or len(X_valid) < 2:
                return 0.5

            # Compute silhouette score on batch labels (lower = better mixing)
            score = silhouette_score(X_valid, batch_valid)
            # Convert to mixing score (higher = better mixing)
            mixing_score = 1.0 - score
            return max(0.0, mixing_score)
        except:
            return 0.5

    def _compute_signal_to_noise(self, output_matrix) -> float:
        """Compute signal-to-noise ratio in processed data."""
        X = MatrixOps.apply_mask_to_values(output_matrix, operation='nan').X

        try:
            # Remove rows with all NaN values
            valid_rows = ~np.isnan(X).all(axis=1)
            X_valid = X[valid_rows]

            if len(X_valid) == 0:
                return 0.0

            # Compute SNR as ratio of variance to mean absolute deviation
            signal = np.nanvar(X_valid, axis=1)
            noise = np.nanmean(np.abs(X_valid - np.nanmean(X_valid, axis=1, keepdims=True)), axis=1)

            # Average SNR across all samples
            snr_values = signal / (noise + 1e-8)  # Add small epsilon to avoid division by zero
            return np.nanmean(snr_values)
        except:
            return 0.0

    def _compute_missing_pattern_score(self, input_matrix, output_matrix) -> float:
        """Compute quality of missing value pattern handling."""
        input_stats = MatrixOps.get_mask_statistics(input_matrix)
        output_stats = MatrixOps.get_mask_statistics(output_matrix)

        # Focus on improvement in data availability
        input_valid_rate = input_stats['VALID']['percentage'] / 100.0
        output_valid_rate = output_stats['VALID']['percentage'] / 100.0

        return min(1.0, output_valid_rate / max(input_valid_rate, 1e-8))

    def _compute_group_separation(self, X: np.ndarray, groups: Optional[np.ndarray]) -> float:
        """Compute separation between biological groups."""
        if groups is None or len(np.unique(groups)) < 2:
            return 0.0

        try:
            # Remove rows with all NaN values
            valid_rows = ~np.isnan(X).all(axis=1)
            X_valid = X[valid_rows]
            groups_valid = groups[valid_rows]

            if len(np.unique(groups_valid)) < 2 or len(X_valid) < 2:
                return 0.0

            return silhouette_score(X_valid, groups_valid)
        except:
            return 0.0

    def _compute_signal_preservation(self, X: np.ndarray) -> float:
        """Compute preservation of biological signals."""
        try:
            # Use variance explained by first few PCs as signal measure
            pca = PCA(n_components=min(5, X.shape[1], X.shape[0]))
            pca.fit(X)

            # Return cumulative variance explained by first few components
            return np.sum(pca.explained_variance_ratio_)
        except:
            return 0.0

    def _compute_clustering_consistency(self, X: np.ndarray, groups: Optional[np.ndarray]) -> float:
        """Compute clustering consistency with ground truth."""
        if groups is None:
            return 0.0

        try:
            # Remove rows with all NaN values
            valid_rows = ~np.isnan(X).all(axis=1)
            X_valid = X[valid_rows]
            groups_valid = groups[valid_rows]

            if len(np.unique(groups_valid)) < 2 or len(X_valid) < 2:
                return 0.0

            # Simple K-means clustering for comparison
            from sklearn.cluster import KMeans
            n_clusters = len(np.unique(groups_valid))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(X_valid)

            return adjusted_rand_score(groups_valid, predicted_labels)
        except:
            return 0.0

    def _compute_biological_variance_explained(
        self,
        container: ScpContainer,
        layer_name: str,
        groups: Optional[np.ndarray]
    ) -> float:
        """Compute percentage of variance explained by biological factors."""
        if groups is None:
            return 0.0

        try:
            X = MatrixOps.apply_mask_to_values(
                container.assays['protein'].layers[layer_name],
                operation='nan'
            ).X

            # Remove rows with all NaN values
            valid_rows = ~np.isnan(X).all(axis=1)
            X_valid = X[valid_rows]
            groups_valid = groups[valid_rows]

            if len(np.unique(groups_valid)) < 2:
                return 0.0

            # Compute between-group vs within-group variance
            overall_mean = np.nanmean(X_valid)
            group_means = []
            total_n = 0

            for group in np.unique(groups_valid):
                group_data = X_valid[groups_valid == group]
                group_mean = np.nanmean(group_data)
                n_group = len(group_data)
                group_means.append(group_mean)
                total_n += n_group

            # Between-group variance
            between_var = np.sum([(g - overall_mean) ** 2 * n_group
                                 for g, n_group in zip(group_means,
                                                        [len(X_valid[groups_valid == g]) for g in np.unique(groups_valid)])]) / (total_n - 1)

            # Within-group variance
            within_var = 0
            for i, group in enumerate(np.unique(groups_valid)):
                group_data = X_valid[groups_valid == group]
                group_var = np.nanvar(group_data, ddof=1)
                within_var += group_var * (len(group_data) - 1)

            within_var = within_var / (total_n - len(np.unique(groups_valid)))

            # Percentage of variance explained by groups
            total_var = between_var + within_var
            return (between_var / total_var) if total_var > 0 else 0.0
        except:
            return 0.0