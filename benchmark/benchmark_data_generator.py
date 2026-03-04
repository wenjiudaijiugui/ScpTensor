# ruff: noqa: N803, N806
# Allow uppercase X (data matrix) and M (mask matrix) - standard scientific computing convention
"""Comprehensive benchmark data generator for ScpTensor evaluation.

This module provides a flexible data generator for evaluating ScpTensor's
analysis methods across different data characteristics including:
- Multiple data scales (small, medium, large)
- Various missing rates (low, medium, high)
- Different missing patterns (MCAR, MAR, MNAR)
- Distribution characteristics (normal, log-normal, multimodal, with batch effect)

The generator creates ScpContainer objects with ground truth labels for
clustering and batch integration evaluation.

Examples
--------
>>> from benchmark.benchmark_data_generator import BenchmarkDataGenerator
>>> gen = BenchmarkDataGenerator(seed=42)
>>> container = gen.generate(n_samples=200, n_features=500, missing_rate=0.3)
>>> container = gen.quick("small_mcar")
>>> scenarios = gen.generate_scenarios([{"scale": "small", "missing_rate": "medium"}])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from scipy.special import expit

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from scptensor.core.structures import Assay, MaskCode, ScpContainer, ScpMatrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# Type Definitions
# =============================================================================

ScaleType = Literal["small", "medium", "large"]
MissingRateType = Literal["low", "medium", "high"]
MissingPatternType = Literal["mcar", "mar", "mnar"]
DistributionType = Literal["normal", "log_normal", "multimodal"]


# =============================================================================
# Constants
# =============================================================================

# Scale configurations
SCALE_CONFIG: dict[ScaleType, tuple[int, int]] = {
    "small": (50, 100),  # 50 samples x 100 features
    "medium": (200, 500),  # 200 samples x 500 features
    "large": (500, 1000),  # 500 samples x 1000 features
}

# Missing rate configurations
MISSING_RATE_CONFIG: dict[MissingRateType, tuple[float, float]] = {
    "low": (0.05, 0.10),  # 5-10% missing
    "medium": (0.20, 0.30),  # 20-30% missing
    "high": (0.40, 0.50),  # 40-50% missing
}

# Default parameters
_DEFAULT_N_CLUSTERS = 4
_DEFAULT_N_BATCHES = 3
_DEFAULT_ASSAY_NAME = "proteins"
_DEFAULT_LAYER_NAME = "raw"


# =============================================================================
# Predefined Scenarios
# =============================================================================

PREDEFINED_SCENARIOS: dict[str, dict[str, Any]] = {
    # Small scale scenarios
    "small_mcar": {
        "scale": "small",
        "missing_rate": "medium",
        "missing_pattern": "mcar",
        "distribution": "log_normal",
    },
    "small_mar": {
        "scale": "small",
        "missing_rate": "medium",
        "missing_pattern": "mar",
        "distribution": "log_normal",
    },
    "small_mnar": {
        "scale": "small",
        "missing_rate": "medium",
        "missing_pattern": "mnar",
        "distribution": "log_normal",
    },
    # Medium scale scenarios
    "medium_mcar": {
        "scale": "medium",
        "missing_rate": "medium",
        "missing_pattern": "mcar",
        "distribution": "log_normal",
    },
    "medium_mar": {
        "scale": "medium",
        "missing_rate": "medium",
        "missing_pattern": "mar",
        "distribution": "log_normal",
    },
    "medium_mnar": {
        "scale": "medium",
        "missing_rate": "medium",
        "missing_pattern": "mnar",
        "distribution": "log_normal",
    },
    "medium_batch": {
        "scale": "medium",
        "missing_rate": "medium",
        "missing_pattern": "mar",
        "distribution": "log_normal",
        "with_batch_effect": True,
        "n_batches": 3,
    },
    # Large scale scenarios
    "large_mcar": {
        "scale": "large",
        "missing_rate": "low",
        "missing_pattern": "mcar",
        "distribution": "log_normal",
    },
    "large_mar": {
        "scale": "large",
        "missing_rate": "medium",
        "missing_pattern": "mar",
        "distribution": "log_normal",
    },
    "large_mnar_high": {
        "scale": "large",
        "missing_rate": "high",
        "missing_pattern": "mnar",
        "distribution": "log_normal",
    },
    # Distribution variants
    "medium_normal": {
        "scale": "medium",
        "missing_rate": "medium",
        "missing_pattern": "mcar",
        "distribution": "normal",
    },
    "medium_multimodal": {
        "scale": "medium",
        "missing_rate": "medium",
        "missing_pattern": "mcar",
        "distribution": "multimodal",
    },
}


# =============================================================================
# Main Generator Class
# =============================================================================


class BenchmarkDataGenerator:
    """Comprehensive benchmark data generator for ScpTensor evaluation.

    This generator creates synthetic DIA-based single-cell proteomics data with
    controlled characteristics for systematic benchmarking of analysis methods.

    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    seed : int
        Random seed.
    rng : np.random.Generator
        NumPy random generator instance.

    Examples
    --------
    >>> gen = BenchmarkDataGenerator(seed=42)
    >>> # Generate a single dataset
    >>> container = gen.generate(
    ...     n_samples=200,
    ...     n_features=500,
    ...     missing_rate=0.3,
    ...     missing_pattern="mar",
    ...     distribution="log_normal",
    ... )
    >>> # Generate predefined scenario
    >>> container = gen.quick("small_mcar")
    >>> # Generate multiple scenarios
    >>> scenarios = gen.generate_scenarios([
    ...     {"scale": "small", "missing_rate": "medium", "missing_pattern": "mcar"},
    ... ])
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_samples: int = 200,
        n_features: int = 500,
        missing_rate: float = 0.3,
        missing_pattern: MissingPatternType = "mar",
        distribution: DistributionType = "log_normal",
        with_batch_effect: bool = False,
        n_batches: int = 3,
        n_clusters: int = 4,
        assay_name: str = _DEFAULT_ASSAY_NAME,
        layer_name: str = _DEFAULT_LAYER_NAME,
    ) -> ScpContainer:
        """Generate a benchmark dataset with specified characteristics.

        Parameters
        ----------
        n_samples : int, default=200
            Number of samples (cells).
        n_features : int, default=500
            Number of features (proteins).
        missing_rate : float, default=0.3
            Proportion of missing values (0.0 to 0.7).
        missing_pattern : {"mcar", "mar", "mnar"}, default="mar"
            Missing value mechanism:
            - "mcar": Missing Completely At Random
            - "mar": Missing At Random (depends on observed values)
            - "mnar": Missing Not At Random (depends on unobserved values)
        distribution : {"normal", "log_normal", "multimodal"}, default="log_normal"
            Distribution of expression values:
            - "normal": Standard normal distribution
            - "log_normal": Log-normal (typical for proteomics)
            - "multimodal": Multiple peaks (cell subpopulations)
        with_batch_effect : bool, default=False
            Whether to add batch effects.
        n_batches : int, default=3
            Number of batches (used if with_batch_effect=True).
        n_clusters : int, default=4
            Number of ground truth clusters.
        assay_name : str, default="proteins"
            Name for the assay.
        layer_name : str, default="raw"
            Name for the data layer.

        Returns
        -------
        ScpContainer
            Container with generated data including:
            - Data matrix X with missing values as NaN
            - Mask matrix M with missing codes
            - obs metadata with "true_cluster" and "batch" columns
            - var metadata with feature information

        Raises
        ------
        ValueError
            If missing_rate is not in [0, 0.7] or invalid parameters.
        """
        # Validate parameters
        if not 0.0 <= missing_rate <= 0.7:
            raise ValueError(f"missing_rate must be in [0, 0.7], got {missing_rate}")
        if missing_pattern not in ("mcar", "mar", "mnar"):
            raise ValueError(f"Invalid missing_pattern: {missing_pattern}")
        if distribution not in ("normal", "log_normal", "multimodal"):
            raise ValueError(f"Invalid distribution: {distribution}")

        # Generate cluster and batch assignments
        cluster_labels = self._assign_clusters(n_samples, n_clusters)
        batch_labels = self._assign_batches(n_samples, n_batches)

        # Generate base expression matrix
        X = self._generate_expression(
            n_samples=n_samples,
            n_features=n_features,
            distribution=distribution,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
        )

        # Add batch effects if requested
        if with_batch_effect:
            X = self._add_batch_effects(X, batch_labels, n_batches)

        # Generate missing mask based on pattern
        M = self._generate_missing_mask(
            X=X,
            missing_rate=missing_rate,
            missing_pattern=missing_pattern,
            batch_labels=batch_labels,
        )

        # Apply missing values to X
        X_missing = X.copy()
        X_missing[M != MaskCode.VALID] = np.nan

        # Create metadata DataFrames
        obs = self._create_obs_metadata(n_samples, cluster_labels, batch_labels)
        var = self._create_var_metadata(n_features)

        # Create ScpContainer
        matrix = ScpMatrix(X=X_missing, M=M)
        assay = Assay(var=var, feature_id_col="protein_id")
        assay.add_layer(layer_name, matrix)

        container = ScpContainer(obs=obs, sample_id_col="sample_id")
        container.add_assay(assay_name, assay)

        # Log provenance
        self._log_provenance(
            container=container,
            n_samples=n_samples,
            n_features=n_features,
            missing_rate=missing_rate,
            missing_pattern=missing_pattern,
            distribution=distribution,
            with_batch_effect=with_batch_effect,
            n_batches=n_batches,
            n_clusters=n_clusters,
        )

        return container

    def quick(self, scenario_name: str) -> ScpContainer:
        """Generate a predefined scenario by name.

        Parameters
        ----------
        scenario_name : str
            Name of the predefined scenario. Available scenarios:
            - "small_mcar", "small_mar", "small_mnar"
            - "medium_mcar", "medium_mar", "medium_mnar", "medium_batch"
            - "large_mcar", "large_mar", "large_mnar_high"
            - "medium_normal", "medium_multimodal"

        Returns
        -------
        ScpContainer
            Generated container for the scenario.

        Raises
        ------
        KeyError
            If scenario name is not found.

        Examples
        --------
        >>> gen = BenchmarkDataGenerator(seed=42)
        >>> container = gen.quick("small_mcar")
        >>> print(container)
        <ScpContainer n_samples=50, assays=[proteins(100)]>
        """
        if scenario_name not in PREDEFINED_SCENARIOS:
            available = list(PREDEFINED_SCENARIOS.keys())
            raise KeyError(f"Unknown scenario: '{scenario_name}'. Available scenarios: {available}")

        config = PREDEFINED_SCENARIOS[scenario_name].copy()
        return self.generate_from_config(config)

    def generate_from_config(self, config: dict[str, Any]) -> ScpContainer:
        """Generate a dataset from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - scale: {"small", "medium", "large"}
            - missing_rate: {"low", "medium", "high"} or float
            - missing_pattern: {"mcar", "mar", "mnar"}
            - distribution: {"normal", "log_normal", "multimodal"}
            - with_batch_effect: bool (optional)
            - n_batches: int (optional)
            - n_clusters: int (optional)

        Returns
        -------
        ScpContainer
            Generated container.
        """
        # Resolve scale
        scale = config.get("scale", "medium")
        n_samples, n_features = SCALE_CONFIG[scale]

        # Resolve missing rate
        mr_config = config.get("missing_rate", "medium")
        if isinstance(mr_config, str):
            mr_min, mr_max = MISSING_RATE_CONFIG[mr_config]
            missing_rate = self.rng.uniform(mr_min, mr_max)
        else:
            missing_rate = float(mr_config)

        # Get other parameters
        missing_pattern: MissingPatternType = config.get("missing_pattern", "mar")
        distribution: DistributionType = config.get("distribution", "log_normal")
        with_batch_effect: bool = config.get("with_batch_effect", False)
        n_batches: int = config.get("n_batches", _DEFAULT_N_BATCHES)
        n_clusters: int = config.get("n_clusters", _DEFAULT_N_CLUSTERS)

        return self.generate(
            n_samples=n_samples,
            n_features=n_features,
            missing_rate=missing_rate,
            missing_pattern=missing_pattern,
            distribution=distribution,
            with_batch_effect=with_batch_effect,
            n_batches=n_batches,
            n_clusters=n_clusters,
        )

    def generate_scenarios(
        self,
        scenario_configs: list[dict[str, Any]],
    ) -> list[tuple[str, ScpContainer]]:
        """Generate multiple benchmark scenarios.

        Parameters
        ----------
        scenario_configs : list of dict
            List of scenario configurations. Each config can include:
            - name: str (optional, auto-generated if not provided)
            - scale, missing_rate, missing_pattern, distribution, etc.

        Returns
        -------
        list of tuple
            List of (scenario_name, container) tuples.

        Examples
        --------
        >>> gen = BenchmarkDataGenerator(seed=42)
        >>> scenarios = gen.generate_scenarios([
        ...     {"scale": "small", "missing_rate": "medium", "missing_pattern": "mcar"},
        ...     {"scale": "medium", "missing_rate": "high", "missing_pattern": "mar"},
        ... ])
        >>> for name, container in scenarios:
        ...     print(f"{name}: {container.n_samples} samples")
        """
        results: list[tuple[str, ScpContainer]] = []

        for i, config in enumerate(scenario_configs):
            # Generate scenario name if not provided
            name = config.get(
                "name",
                f"scenario_{i}_{config.get('scale', 'medium')}_{config.get('missing_pattern', 'mar')}",
            )

            # Generate container
            container = self.generate_from_config(config)
            results.append((name, container))

        return results

    def generate_full_matrix(
        self,
        scales: list[ScaleType] | None = None,
        missing_rates: list[MissingRateType] | None = None,
        missing_patterns: list[MissingPatternType] | None = None,
        distributions: list[DistributionType] | None = None,
        with_batch_variants: bool = False,
    ) -> list[tuple[str, ScpContainer]]:
        """Generate a comprehensive matrix of benchmark scenarios.

        This generates all combinations of the specified parameters for
        thorough benchmarking.

        Parameters
        ----------
        scales : list of str, optional
            Scales to include. Default: ["small", "medium", "large"]
        missing_rates : list of str, optional
            Missing rates to include. Default: ["low", "medium", "high"]
        missing_patterns : list of str, optional
            Missing patterns to include. Default: ["mcar", "mar", "mnar"]
        distributions : list of str, optional
            Distributions to include. Default: ["log_normal"]
        with_batch_variants : bool, default=False
            Whether to also generate variants with batch effects.

        Returns
        -------
        list of tuple
            List of (scenario_name, container) tuples for all combinations.

        Examples
        --------
        >>> gen = BenchmarkDataGenerator(seed=42)
        >>> # Generate all combinations
        >>> all_scenarios = gen.generate_full_matrix()
        >>> print(f"Generated {len(all_scenarios)} scenarios")
        """
        # Set defaults
        if scales is None:
            scales = ["small", "medium", "large"]
        if missing_rates is None:
            missing_rates = ["low", "medium", "high"]
        if missing_patterns is None:
            missing_patterns = ["mcar", "mar", "mnar"]
        if distributions is None:
            distributions = ["log_normal"]

        scenarios: list[tuple[str, ScpContainer]] = []

        # Generate all combinations
        for scale in scales:
            for mr in missing_rates:
                for mp in missing_patterns:
                    for dist in distributions:
                        # Without batch effect
                        name = f"{scale}_{mp}_{mr}_{dist}"
                        config = {
                            "name": name,
                            "scale": scale,
                            "missing_rate": mr,
                            "missing_pattern": mp,
                            "distribution": dist,
                            "with_batch_effect": False,
                        }
                        container = self.generate_from_config(config)
                        scenarios.append((name, container))

                        # With batch effect if requested
                        if with_batch_variants:
                            name_batch = f"{scale}_{mp}_{mr}_{dist}_batch"
                            config_batch = config.copy()
                            config_batch["name"] = name_batch
                            config_batch["with_batch_effect"] = True
                            container_batch = self.generate_from_config(config_batch)
                            scenarios.append((name_batch, container_batch))

        return scenarios

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _assign_clusters(self, n_samples: int, n_clusters: int) -> NDArray[np.int64]:
        """Assign samples to clusters."""
        labels = np.arange(n_samples) % n_clusters
        self.rng.shuffle(labels)
        return labels

    def _assign_batches(self, n_samples: int, n_batches: int) -> NDArray[np.int64]:
        """Assign samples to batches."""
        return np.arange(n_samples) % n_batches

    def _generate_expression(
        self,
        n_samples: int,
        n_features: int,
        distribution: DistributionType,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
    ) -> NDArray[np.float64]:
        """Generate base expression matrix with specified distribution."""
        if distribution == "normal":
            X = self._generate_normal(n_samples, n_features, cluster_labels, n_clusters)
        elif distribution == "log_normal":
            X = self._generate_log_normal(n_samples, n_features, cluster_labels, n_clusters)
        elif distribution == "multimodal":
            X = self._generate_multimodal(n_samples, n_features, cluster_labels, n_clusters)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return X

    def _generate_normal(
        self,
        n_samples: int,
        n_features: int,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
    ) -> NDArray[np.float64]:
        """Generate normally distributed data."""
        # Base expression
        X = self.rng.normal(0, 1, (n_samples, n_features))

        # Add cluster-specific shifts for some features
        n_cluster_features = n_features // 4
        for c in range(n_clusters):
            mask = cluster_labels == c
            shift = self.rng.normal(0, 2, n_cluster_features)
            X[np.ix_(mask, np.arange(n_cluster_features))] += shift

        return X

    def _generate_log_normal(
        self,
        n_samples: int,
        n_features: int,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
    ) -> NDArray[np.float64]:
        """Generate log-normal distributed data (typical for proteomics)."""
        # Log-normal base (simulates protein intensities)
        # Mean ~15, Std ~2 in log2 space (typical for proteomics)
        base_mean = 15.0
        base_std = 2.0

        X = self.rng.normal(base_mean, base_std, (n_samples, n_features))

        # Add cluster-specific patterns
        n_cluster_features = n_features // 4
        for c in range(n_clusters):
            mask = cluster_labels == c
            # Each cluster has up-regulated proteins
            upregulated = self.rng.choice(n_features, n_cluster_features, replace=False)
            shift = self.rng.uniform(1.0, 3.0, len(upregulated))
            X[np.ix_(mask, upregulated)] += shift

        return X

    def _generate_multimodal(
        self,
        n_samples: int,
        n_features: int,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
    ) -> NDArray[np.float64]:
        """Generate multimodal distribution (simulating cell subpopulations)."""
        X = np.zeros((n_samples, n_features))

        # Create distinct modes for different feature groups
        n_features_per_cluster = n_features // n_clusters

        for c in range(n_clusters):
            mask = cluster_labels == c
            start_idx = c * n_features_per_cluster
            end_idx = start_idx + n_features_per_cluster if c < n_clusters - 1 else n_features

            # Primary mode: high expression for this cluster's features
            X[np.ix_(mask, np.arange(start_idx, end_idx))] = self.rng.normal(
                3.0, 1.0, (mask.sum(), end_idx - start_idx)
            )

            # Secondary mode: medium expression for other features
            other_features = (
                np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n_features)])
                if start_idx > 0 or end_idx < n_features
                else np.array([])
            )
            if len(other_features) > 0:
                X[np.ix_(mask, other_features)] = self.rng.normal(
                    0.0, 1.0, (mask.sum(), len(other_features))
                )

        return X

    def _add_batch_effects(
        self,
        X: NDArray[np.float64],
        batch_labels: NDArray[np.int64],
        n_batches: int,
    ) -> NDArray[np.float64]:
        """Add batch effects to the expression matrix."""
        n_features = X.shape[1]
        X_batched = X.copy()

        # Generate batch-specific shifts
        batch_shifts = self.rng.normal(0, 0.5, (n_batches, n_features))

        # Apply batch effects
        for b in range(n_batches):
            mask = batch_labels == b
            X_batched[mask] += batch_shifts[b]

        return X_batched

    def _generate_missing_mask(
        self,
        X: NDArray[np.float64],
        missing_rate: float,
        missing_pattern: MissingPatternType,
        batch_labels: NDArray[np.int64],
    ) -> NDArray[np.int8]:
        """Generate missing value mask based on specified pattern."""
        n_samples, n_features = X.shape
        M = np.zeros((n_samples, n_features), dtype=np.int8)

        if missing_rate == 0:
            return M

        total_elements = n_samples * n_features
        target_missing = int(total_elements * missing_rate)

        if missing_pattern == "mcar":
            M = self._apply_mcar(M, target_missing)
        elif missing_pattern == "mar":
            M = self._apply_mar(X, M, target_missing, batch_labels)
        elif missing_pattern == "mnar":
            M = self._apply_mnar(X, M, target_missing)

        return M

    def _apply_mcar(
        self,
        M: NDArray[np.int8],
        target_missing: int,
    ) -> NDArray[np.int8]:
        """Apply Missing Completely At Random pattern."""
        n_samples, n_features = M.shape
        total_elements = n_samples * n_features

        # Randomly select elements to be missing
        indices = self.rng.choice(total_elements, target_missing, replace=False)
        rows, cols = np.unravel_index(indices, (n_samples, n_features))
        M[rows, cols] = MaskCode.LOD

        return M

    def _apply_mar(
        self,
        X: NDArray[np.float64],
        M: NDArray[np.int8],
        target_missing: int,
        batch_labels: NDArray[np.int64],
    ) -> NDArray[np.int8]:
        """Apply Missing At Random pattern.

        Missing probability depends on observed values (e.g., samples with
        lower overall intensity have more missing values).
        """
        n_samples, n_features = X.shape

        # Calculate sample-wise mean intensity
        sample_means = np.nanmean(X, axis=1, keepdims=True)

        # Normalize to [0, 1]
        min_mean, max_mean = sample_means.min(), sample_means.max()
        if max_mean > min_mean:
            normalized_means = (sample_means - min_mean) / (max_mean - min_mean)
        else:
            normalized_means = np.ones_like(sample_means) * 0.5

        # Lower intensity samples have higher missing probability
        missing_prob = (1.0 - normalized_means).flatten()

        # Distribute missing values across samples
        remaining = target_missing
        for i in range(n_samples):
            if remaining <= 0:
                break

            # Number of missing for this sample
            n_missing_sample = int(target_missing * missing_prob[i] / n_samples)
            n_missing_sample = min(n_missing_sample, n_features, remaining)

            if n_missing_sample > 0:
                # Randomly select features to be missing
                missing_features = self.rng.choice(n_features, n_missing_sample, replace=False)
                M[i, missing_features] = MaskCode.LOD
                remaining -= n_missing_sample

        # Fill remaining if any
        if remaining > 0:
            valid_mask = M == MaskCode.VALID
            valid_indices = np.where(valid_mask.ravel())[0]
            if len(valid_indices) > 0:
                n_fill = min(remaining, len(valid_indices))
                fill_indices = self.rng.choice(valid_indices, n_fill, replace=False)
                rows, cols = np.unravel_index(fill_indices, (n_samples, n_features))
                M[rows, cols] = MaskCode.LOD

        return M

    def _apply_mnar(
        self,
        X: NDArray[np.float64],
        M: NDArray[np.int8],
        target_missing: int,
    ) -> NDArray[np.int8]:
        """Apply Missing Not At Random pattern.

        Missing probability depends on the (unobserved) true value.
        Lower abundance proteins are more likely to be missing.
        """
        n_samples, n_features = X.shape

        # Calculate feature-wise mean abundance
        feature_means = np.nanmean(X, axis=0)

        # Calibrate threshold to achieve target missing rate
        # The percentile represents where to set the threshold so that
        # lower abundance proteins have higher missing probability
        target_rate = target_missing / (n_samples * n_features)
        percentile_val = min(100.0, max(0.0, 100 * (1.0 - target_rate)))
        threshold = np.percentile(feature_means, percentile_val)

        # Calculate missing probability using sigmoid
        # Lower abundance -> higher missing probability
        dropout_slope = 1.5
        p_missing = expit(-dropout_slope * (feature_means - threshold))

        # Apply probabilistic dropout
        for j in range(n_features):
            random_vals = self.rng.random(n_samples)
            missing_mask = random_vals < p_missing[j]
            M[missing_mask, j] = MaskCode.LOD

        # Adjust to hit target missing rate more precisely
        current_missing = np.sum(M != MaskCode.VALID)
        if current_missing < target_missing:
            # Add more missing
            valid_mask = M == MaskCode.VALID
            valid_indices = np.where(valid_mask.ravel())[0]
            n_add = min(target_missing - current_missing, len(valid_indices))
            if n_add > 0:
                add_indices = self.rng.choice(valid_indices, n_add, replace=False)
                rows, cols = np.unravel_index(add_indices, (n_samples, n_features))
                M[rows, cols] = MaskCode.LOD
        elif current_missing > target_missing:
            # Remove some missing (convert back to valid)
            missing_mask = M != MaskCode.VALID
            missing_indices = np.where(missing_mask.ravel())[0]
            n_remove = min(current_missing - target_missing, len(missing_indices))
            if n_remove > 0:
                remove_indices = self.rng.choice(missing_indices, n_remove, replace=False)
                rows, cols = np.unravel_index(remove_indices, (n_samples, n_features))
                M[rows, cols] = MaskCode.VALID

        return M

    def _create_obs_metadata(
        self,
        n_samples: int,
        cluster_labels: NDArray[np.int64],
        batch_labels: NDArray[np.int64],
    ) -> pl.DataFrame:
        """Create sample metadata DataFrame."""
        return pl.DataFrame(
            {
                "sample_id": [f"Cell_{i:05d}" for i in range(n_samples)],
                "true_cluster": [f"Cluster_{c}" for c in cluster_labels],
                "batch": [f"Batch_{b}" for b in batch_labels],
            }
        )

    def _create_var_metadata(self, n_features: int) -> pl.DataFrame:
        """Create feature metadata DataFrame."""
        return pl.DataFrame(
            {
                "protein_id": [f"Prot_{i:05d}" for i in range(n_features)],
                "gene_name": [f"Gene_{i:05d}" for i in range(n_features)],
            }
        )

    def _log_provenance(
        self,
        container: ScpContainer,
        n_samples: int,
        n_features: int,
        missing_rate: float,
        missing_pattern: str,
        distribution: str,
        with_batch_effect: bool,
        n_batches: int,
        n_clusters: int,
    ) -> None:
        """Log generation parameters to container history."""
        container.log_operation(
            action="benchmark_data_generation",
            params={
                "n_samples": n_samples,
                "n_features": n_features,
                "missing_rate": missing_rate,
                "missing_pattern": missing_pattern,
                "distribution": distribution,
                "with_batch_effect": with_batch_effect,
                "n_batches": n_batches,
                "n_clusters": n_clusters,
                "seed": self.seed,
            },
            description=f"Generated benchmark data: {n_samples} samples x {n_features} features, "
            f"{missing_pattern.upper()} missing at {missing_rate:.1%} rate",
        )


# =============================================================================
# Utility Functions
# =============================================================================


def get_actual_missing_rate(
    container: ScpContainer, assay_name: str = "proteins", layer_name: str = "raw"
) -> float:
    """Calculate actual missing rate in a container.

    Parameters
    ----------
    container : ScpContainer
        Container to analyze.
    assay_name : str, default="proteins"
        Assay name.
    layer_name : str, default="raw"
        Layer name.

    Returns
    -------
    float
        Actual missing rate.
    """
    matrix = container.assays[assay_name].layers[layer_name]
    X = matrix.X

    if isinstance(X, np.ndarray):
        return np.isnan(X).mean()
    else:
        # Sparse matrix - use mask
        M = matrix.M
        if M is not None:
            return (M != MaskCode.VALID).mean()
        return 0.0


def summarize_container(container: ScpContainer, assay_name: str = "proteins") -> dict[str, Any]:
    """Summarize container statistics.

    Parameters
    ----------
    container : ScpContainer
        Container to summarize.
    assay_name : str, default="proteins"
        Assay name.

    Returns
    -------
    dict
        Summary statistics.
    """
    assay = container.assays[assay_name]
    X = assay.layers["raw"].X

    # Handle sparse/dense
    if isinstance(X, np.ndarray):
        valid_mask = ~np.isnan(X)
        valid_values = X[valid_mask]
    else:
        valid_values = X.data
        valid_mask = np.ones(X.nnz, dtype=bool)

    return {
        "n_samples": container.n_samples,
        "n_features": assay.n_features,
        "missing_rate": get_actual_missing_rate(container, assay_name),
        "mean_expression": float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0,
        "std_expression": float(np.std(valid_values)) if len(valid_values) > 0 else 0.0,
        "n_clusters": container.obs["true_cluster"].n_unique()
        if "true_cluster" in container.obs.columns
        else 0,
        "n_batches": container.obs["batch"].n_unique() if "batch" in container.obs.columns else 0,
    }


# =============================================================================
# Demo / Main
# =============================================================================


def _run_demo() -> None:
    """Run demonstration of the benchmark data generator."""
    print("=" * 70)
    print("ScpTensor Benchmark Data Generator Demo")
    print("=" * 70)

    gen = BenchmarkDataGenerator(seed=42)

    # Demo 1: Quick scenarios
    print("\n1. Quick Predefined Scenarios")
    print("-" * 40)
    for scenario_name in ["small_mcar", "medium_mar", "large_mnar_high"]:
        container = gen.quick(scenario_name)
        stats = summarize_container(container)
        print(f"\n{scenario_name}:")
        print(f"  Shape: {stats['n_samples']} samples x {stats['n_features']} features")
        print(f"  Missing rate: {stats['missing_rate']:.1%}")
        print(f"  Clusters: {stats['n_clusters']}, Batches: {stats['n_batches']}")

    # Demo 2: Custom generation
    print("\n\n2. Custom Generation")
    print("-" * 40)
    container = gen.generate(
        n_samples=100,
        n_features=200,
        missing_rate=0.35,
        missing_pattern="mar",
        distribution="log_normal",
        with_batch_effect=True,
        n_batches=3,
        n_clusters=4,
    )
    stats = summarize_container(container)
    print("\nCustom container:")
    print(f"  Shape: {stats['n_samples']} samples x {stats['n_features']} features")
    print(f"  Missing rate: {stats['missing_rate']:.1%}")
    print(f"  Mean expression: {stats['mean_expression']:.2f}")
    print(f"  Std expression: {stats['std_expression']:.2f}")

    # Demo 3: Distribution comparison
    print("\n\n3. Distribution Comparison")
    print("-" * 40)
    for dist in ["normal", "log_normal", "multimodal"]:
        container = gen.generate(
            n_samples=100,
            n_features=200,
            missing_rate=0.2,
            missing_pattern="mcar",
            distribution=dist,  # type: ignore
        )
        stats = summarize_container(container)
        print(f"\n{dist}:")
        print(f"  Mean: {stats['mean_expression']:.2f}, Std: {stats['std_expression']:.2f}")

    # Demo 4: Missing pattern comparison
    print("\n\n4. Missing Pattern Comparison")
    print("-" * 40)
    for pattern in ["mcar", "mar", "mnar"]:
        container = gen.generate(
            n_samples=100,
            n_features=200,
            missing_rate=0.3,
            missing_pattern=pattern,  # type: ignore
            distribution="log_normal",
        )
        actual_rate = get_actual_missing_rate(container)
        print(f"\n{pattern.upper()}:")
        print(f"  Target rate: 30%, Actual rate: {actual_rate:.1%}")

    # Demo 5: Generate multiple scenarios
    print("\n\n5. Generate Multiple Scenarios")
    print("-" * 40)
    scenarios = gen.generate_scenarios(
        [
            {"scale": "small", "missing_rate": "low", "missing_pattern": "mcar"},
            {"scale": "medium", "missing_rate": "medium", "missing_pattern": "mar"},
            {"scale": "large", "missing_rate": "high", "missing_pattern": "mnar"},
        ]
    )
    for name, container in scenarios:
        stats = summarize_container(container)
        print(f"\n{name}:")
        print(
            f"  {stats['n_samples']} x {stats['n_features']}, missing: {stats['missing_rate']:.1%}"
        )

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    _run_demo()
