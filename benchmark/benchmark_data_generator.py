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
from typing import TYPE_CHECKING, Any, Literal, cast

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

ScaleType = Literal["small", "medium", "large", "xlarge"]
MissingRateType = Literal["low", "medium", "high"]
MissingPatternType = Literal["mcar", "mar", "mnar"]
DistributionType = Literal["normal", "log_normal", "multimodal", "heavy_tailed"]


# =============================================================================
# Constants
# =============================================================================

# Scale configurations
SCALE_CONFIG: dict[ScaleType, tuple[int, int]] = {
    "small": (50, 100),  # 50 samples x 100 features
    "medium": (200, 500),  # 200 samples x 500 features
    "large": (500, 1000),  # 500 samples x 1000 features
    "xlarge": (500, 10000),  # 500 samples x 10,000 features
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
_EPS = 1e-12


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
    "medium_heavy_tailed": {
        "scale": "medium",
        "missing_rate": "medium",
        "missing_pattern": "mar",
        "distribution": "heavy_tailed",
    },
    "xlarge_mar_batch": {
        "scale": "xlarge",
        "missing_rate": "medium",
        "missing_pattern": "mar",
        "distribution": "log_normal",
        "with_batch_effect": True,
        "n_batches": 4,
        "batch_effect_strength": 0.9,
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
        batch_effect_strength: float = 0.6,
        batch_feature_fraction: float = 0.35,
        confounded_batches: bool = False,
        batch_confounding_strength: float = 0.6,
        cluster_separation: float = 1.5,
        noise_scale: float = 1.0,
        mar_strength: float = 1.0,
        mnar_slope: float = 2.0,
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
        batch_effect_strength : float, default=0.6
            Strength of additive/multiplicative batch effects.
        batch_feature_fraction : float, default=0.35
            Fraction of proteins receiving strong feature-specific batch effects.
        confounded_batches : bool, default=False
            If True, induce biological group/batch confounding in batch assignment.
        batch_confounding_strength : float, default=0.6
            Degree of group/batch confounding in [0, 1].
        cluster_separation : float, default=1.5
            Strength of biological cluster separation.
        noise_scale : float, default=1.0
            Multiplicative noise scale for signal generation.
        mar_strength : float, default=1.0
            Strength of MAR sampling weights.
        mnar_slope : float, default=2.0
            Logistic slope for MNAR intensity dependence.
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
            If parameters are outside allowed ranges.
        """
        # Validate parameters
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        if n_features < 2:
            raise ValueError(f"n_features must be >= 2, got {n_features}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if n_batches < 1:
            raise ValueError(f"n_batches must be >= 1, got {n_batches}")
        if n_clusters > n_samples:
            raise ValueError(
                f"n_clusters must be <= n_samples, got n_clusters={n_clusters}, n_samples={n_samples}"
            )
        if not 0.0 <= missing_rate <= 0.9:
            raise ValueError(f"missing_rate must be in [0, 0.9], got {missing_rate}")
        if missing_pattern not in ("mcar", "mar", "mnar"):
            raise ValueError(f"Invalid missing_pattern: {missing_pattern}")
        if distribution not in ("normal", "log_normal", "multimodal", "heavy_tailed"):
            raise ValueError(f"Invalid distribution: {distribution}")
        if not 0.0 <= batch_feature_fraction <= 1.0:
            raise ValueError(
                f"batch_feature_fraction must be in [0, 1], got {batch_feature_fraction}"
            )
        if not 0.0 <= batch_confounding_strength <= 1.0:
            raise ValueError(
                f"batch_confounding_strength must be in [0, 1], got {batch_confounding_strength}"
            )
        if batch_effect_strength < 0.0:
            raise ValueError(f"batch_effect_strength must be >= 0, got {batch_effect_strength}")
        if cluster_separation < 0.0:
            raise ValueError(f"cluster_separation must be >= 0, got {cluster_separation}")
        if noise_scale <= 0.0:
            raise ValueError(f"noise_scale must be > 0, got {noise_scale}")
        if mar_strength < 0.0:
            raise ValueError(f"mar_strength must be >= 0, got {mar_strength}")
        if mnar_slope <= 0.0:
            raise ValueError(f"mnar_slope must be > 0, got {mnar_slope}")

        # Generate cluster and batch assignments
        cluster_labels = self._assign_clusters(n_samples, n_clusters)
        batch_labels = self._assign_batches(
            n_samples=n_samples,
            n_batches=n_batches,
            cluster_labels=cluster_labels,
            confounded=confounded_batches,
            confounding_strength=batch_confounding_strength,
        )

        # Generate base expression matrix
        X = self._generate_expression(
            n_samples=n_samples,
            n_features=n_features,
            distribution=distribution,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            cluster_separation=cluster_separation,
            noise_scale=noise_scale,
        )

        # Add batch effects if requested
        if with_batch_effect and n_batches > 1 and batch_effect_strength > 0:
            X = self._add_batch_effects(
                X=X,
                batch_labels=batch_labels,
                n_batches=n_batches,
                batch_effect_strength=batch_effect_strength,
                batch_feature_fraction=batch_feature_fraction,
            )

        # Generate missing mask based on pattern
        M = self._generate_missing_mask(
            X=X,
            missing_rate=missing_rate,
            missing_pattern=missing_pattern,
            batch_labels=batch_labels,
            mar_strength=mar_strength,
            mnar_slope=mnar_slope,
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
            batch_effect_strength=batch_effect_strength,
            confounded_batches=confounded_batches,
            batch_confounding_strength=batch_confounding_strength,
            cluster_separation=cluster_separation,
            noise_scale=noise_scale,
            mar_strength=mar_strength,
            mnar_slope=mnar_slope,
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
            - "medium_normal", "medium_multimodal", "medium_heavy_tailed"
            - "xlarge_mar_batch"

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
            - scale: {"small", "medium", "large", "xlarge"} (optional)
            - n_samples: int (optional, overrides scale)
            - n_features: int (optional, overrides scale)
            - missing_rate: {"low", "medium", "high"} or float or [min, max]
            - missing_pattern: {"mcar", "mar", "mnar"}
            - distribution: {"normal", "log_normal", "multimodal", "heavy_tailed"}
            - with_batch_effect: bool (optional)
            - n_batches: int (optional)
            - n_clusters: int (optional)
            - batch_effect_strength: float (optional)
            - batch_feature_fraction: float (optional)
            - confounded_batches: bool (optional)
            - batch_confounding_strength: float (optional)
            - cluster_separation: float (optional)
            - noise_scale: float (optional)
            - mar_strength: float (optional)
            - mnar_slope: float (optional)

        Returns
        -------
        ScpContainer
            Generated container.
        """
        # Resolve shape from scale, then allow explicit overrides.
        scale = config.get("scale", "medium")
        if scale not in SCALE_CONFIG:
            raise KeyError(f"Unknown scale '{scale}', expected one of {list(SCALE_CONFIG)}")
        n_samples, n_features = SCALE_CONFIG[scale]
        n_samples = int(config.get("n_samples", n_samples))
        n_features = int(config.get("n_features", n_features))

        # Resolve missing rate
        mr_config = config.get("missing_rate", "medium")
        if isinstance(mr_config, str):
            if mr_config not in MISSING_RATE_CONFIG:
                raise KeyError(
                    f"Unknown missing_rate label '{mr_config}', "
                    f"expected one of {list(MISSING_RATE_CONFIG)}"
                )
            mr_label = cast(MissingRateType, mr_config)
            mr_min, mr_max = MISSING_RATE_CONFIG[mr_label]
            missing_rate = self.rng.uniform(mr_min, mr_max)
        elif isinstance(mr_config, list | tuple) and len(mr_config) == 2:
            mr_min = float(mr_config[0])
            mr_max = float(mr_config[1])
            missing_rate = self.rng.uniform(min(mr_min, mr_max), max(mr_min, mr_max))
        else:
            missing_rate = float(mr_config)

        # Get other parameters
        missing_pattern: MissingPatternType = config.get("missing_pattern", "mar")
        distribution: DistributionType = config.get("distribution", "log_normal")
        with_batch_effect: bool = config.get("with_batch_effect", False)
        n_batches: int = config.get("n_batches", _DEFAULT_N_BATCHES)
        n_clusters: int = config.get("n_clusters", _DEFAULT_N_CLUSTERS)
        batch_effect_strength: float = float(config.get("batch_effect_strength", 0.6))
        batch_feature_fraction: float = float(config.get("batch_feature_fraction", 0.35))
        confounded_batches: bool = bool(config.get("confounded_batches", False))
        batch_confounding_strength: float = float(config.get("batch_confounding_strength", 0.6))
        cluster_separation: float = float(config.get("cluster_separation", 1.5))
        noise_scale: float = float(config.get("noise_scale", 1.0))
        mar_strength: float = float(config.get("mar_strength", 1.0))
        mnar_slope: float = float(config.get("mnar_slope", 2.0))

        return self.generate(
            n_samples=n_samples,
            n_features=n_features,
            missing_rate=missing_rate,
            missing_pattern=missing_pattern,
            distribution=distribution,
            with_batch_effect=with_batch_effect,
            n_batches=n_batches,
            n_clusters=n_clusters,
            batch_effect_strength=batch_effect_strength,
            batch_feature_fraction=batch_feature_fraction,
            confounded_batches=confounded_batches,
            batch_confounding_strength=batch_confounding_strength,
            cluster_separation=cluster_separation,
            noise_scale=noise_scale,
            mar_strength=mar_strength,
            mnar_slope=mnar_slope,
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

    def _assign_batches(
        self,
        n_samples: int,
        n_batches: int,
        cluster_labels: NDArray[np.int64],
        confounded: bool = False,
        confounding_strength: float = 0.6,
    ) -> NDArray[np.int64]:
        """Assign samples to batches with optional biological confounding."""
        if n_batches <= 1:
            return np.zeros(n_samples, dtype=np.int64)

        if not confounded:
            labels = np.arange(n_samples) % n_batches
            self.rng.shuffle(labels)
            return labels.astype(np.int64, copy=False)

        batch_labels = np.empty(n_samples, dtype=np.int64)
        strength = float(np.clip(confounding_strength, 0.0, 1.0))
        base_prob = np.full(n_batches, (1.0 - strength) / n_batches, dtype=float)
        cluster_ids = np.unique(cluster_labels)
        preferred_offset = int(self.rng.integers(0, max(1, n_batches)))

        for cluster_id in cluster_ids:
            mask = cluster_labels == cluster_id
            n_group = int(np.sum(mask))
            if n_group == 0:
                continue

            preferred_batch = int((int(cluster_id) + preferred_offset) % n_batches)
            probs = base_prob.copy()
            probs[preferred_batch] += strength
            probs /= np.sum(probs)
            batch_labels[mask] = self.rng.choice(n_batches, size=n_group, replace=True, p=probs)

        return batch_labels

    def _zscore(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score with safe fallback for degenerate vectors."""
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        out = np.zeros_like(arr, dtype=float)
        if not np.any(finite):
            return out

        mean = float(np.mean(arr[finite]))
        std = float(np.std(arr[finite], ddof=0))
        if std < _EPS:
            return out

        out[finite] = (arr[finite] - mean) / std
        return out

    def _mean_ignore_nan(
        self,
        X: NDArray[np.float64],  # noqa: N803
        axis: int,
        keepdims: bool = False,
    ) -> NDArray[np.float64]:
        """Compute mean without NumPy empty-slice warnings."""
        finite = np.isfinite(X)
        counts = np.sum(finite, axis=axis, keepdims=keepdims)
        sums = np.sum(np.where(finite, X, 0.0), axis=axis, keepdims=keepdims)
        return np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts > 0,
        )

    def _std_ignore_nan(
        self,
        X: NDArray[np.float64],  # noqa: N803
        axis: int,
        keepdims: bool = False,
    ) -> NDArray[np.float64]:
        """Compute std without NumPy empty-slice warnings."""
        mean = self._mean_ignore_nan(X, axis=axis, keepdims=True)
        finite = np.isfinite(X)
        centered_sq = np.where(finite, (X - mean) ** 2, 0.0)
        counts = np.sum(finite, axis=axis, keepdims=keepdims)
        var = np.divide(
            np.sum(centered_sq, axis=axis, keepdims=keepdims),
            counts,
            out=np.zeros_like(np.sum(centered_sq, axis=axis, keepdims=keepdims), dtype=float),
            where=counts > 0,
        )
        return np.sqrt(var)

    def _draw_weighted_indices(
        self,
        weights: NDArray[np.float64],
        n_select: int,
    ) -> NDArray[np.int64]:
        """Draw unique indices using non-negative weights."""
        flat = np.asarray(weights, dtype=float).ravel()
        total = flat.size
        target = int(min(max(n_select, 0), total))
        if target == 0:
            return np.array([], dtype=np.int64)

        flat = flat.copy()
        flat[~np.isfinite(flat)] = 0.0
        flat = np.clip(flat, 0.0, None)

        positive = np.flatnonzero(flat > 0.0)
        if positive.size == 0:
            return self.rng.choice(total, size=target, replace=False).astype(np.int64)

        if positive.size <= target:
            if positive.size == target:
                return positive.astype(np.int64, copy=False)
            remain = target - positive.size
            zero_idx = np.flatnonzero(flat <= 0.0)
            extra = self.rng.choice(zero_idx, size=remain, replace=False)
            return np.concatenate([positive, extra]).astype(np.int64, copy=False)

        probs = flat[positive]
        probs /= np.sum(probs)
        return self.rng.choice(positive, size=target, replace=False, p=probs).astype(np.int64)

    def _generate_expression(
        self,
        n_samples: int,
        n_features: int,
        distribution: DistributionType,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
        cluster_separation: float = 1.5,
        noise_scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate base expression matrix with specified distribution."""
        if distribution == "normal":
            X = self._generate_normal(
                n_samples,
                n_features,
                cluster_labels,
                n_clusters,
                cluster_separation=cluster_separation,
                noise_scale=noise_scale,
            )
        elif distribution == "log_normal":
            X = self._generate_log_normal(
                n_samples,
                n_features,
                cluster_labels,
                n_clusters,
                cluster_separation=cluster_separation,
                noise_scale=noise_scale,
            )
        elif distribution == "multimodal":
            X = self._generate_multimodal(
                n_samples,
                n_features,
                cluster_labels,
                n_clusters,
                cluster_separation=cluster_separation,
                noise_scale=noise_scale,
            )
        elif distribution == "heavy_tailed":
            X = self._generate_heavy_tailed(
                n_samples,
                n_features,
                cluster_labels,
                n_clusters,
                cluster_separation=cluster_separation,
                noise_scale=noise_scale,
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return X

    def _apply_cluster_effects(
        self,
        X: NDArray[np.float64],  # noqa: N803
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
        cluster_separation: float,
        marker_fraction: float = 0.2,
    ) -> NDArray[np.float64]:
        """Inject cluster-specific marker patterns."""
        n_features = X.shape[1]
        marker_count = min(n_features, max(8, int(n_features * marker_fraction)))
        center = (n_clusters - 1) / 2.0
        out = X.copy()

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if not np.any(mask):
                continue

            centroid_shift = (cluster_id - center) * 0.15 * cluster_separation
            out[mask, :] += centroid_shift

            markers = self.rng.choice(n_features, size=marker_count, replace=False)
            marker_shift = self.rng.normal(
                loc=cluster_separation,
                scale=max(0.2, 0.25 * cluster_separation),
                size=marker_count,
            )
            out[np.ix_(mask, markers)] += marker_shift

        return out

    def _generate_normal(
        self,
        n_samples: int,
        n_features: int,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
        cluster_separation: float = 1.5,
        noise_scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate normally distributed data."""
        base = self.rng.normal(0.0, 1.0 * noise_scale, size=(n_samples, n_features))
        return self._apply_cluster_effects(
            base,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            cluster_separation=cluster_separation,
            marker_fraction=0.18,
        )

    def _generate_log_normal(
        self,
        n_samples: int,
        n_features: int,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
        cluster_separation: float = 1.5,
        noise_scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate log-normal distributed data (typical for proteomics)."""
        base = self.rng.normal(15.0, 1.4 * noise_scale, size=(n_samples, n_features))
        return self._apply_cluster_effects(
            base,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            cluster_separation=cluster_separation,
            marker_fraction=0.22,
        )

    def _generate_multimodal(
        self,
        n_samples: int,
        n_features: int,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
        cluster_separation: float = 1.5,
        noise_scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate multimodal distribution (simulating cell subpopulations)."""
        X = self.rng.normal(11.0, 1.2 * noise_scale, size=(n_samples, n_features))
        features_per_cluster = max(1, n_features // max(1, n_clusters))

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if not np.any(mask):
                continue

            start = cluster_id * features_per_cluster
            end = (
                n_features
                if cluster_id == n_clusters - 1
                else min(n_features, (cluster_id + 1) * features_per_cluster)
            )
            mode_center = 12.5 + cluster_separation * (0.8 + 0.2 * cluster_id)
            X[np.ix_(mask, np.arange(start, end))] = self.rng.normal(
                mode_center,
                0.9 * noise_scale,
                size=(int(np.sum(mask)), end - start),
            )

        return self._apply_cluster_effects(
            X,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            cluster_separation=0.6 * cluster_separation,
            marker_fraction=0.12,
        )

    def _generate_heavy_tailed(
        self,
        n_samples: int,
        n_features: int,
        cluster_labels: NDArray[np.int64],
        n_clusters: int,
        cluster_separation: float = 1.5,
        noise_scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate heavy-tailed protein-level matrix."""
        base = 14.0 + self.rng.standard_t(df=3, size=(n_samples, n_features)) * (1.1 * noise_scale)
        return self._apply_cluster_effects(
            base,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            cluster_separation=cluster_separation,
            marker_fraction=0.20,
        )

    def _add_batch_effects(
        self,
        X: NDArray[np.float64],
        batch_labels: NDArray[np.int64],
        n_batches: int,
        batch_effect_strength: float = 0.6,
        batch_feature_fraction: float = 0.35,
    ) -> NDArray[np.float64]:
        """Add batch effects to the expression matrix."""
        n_features = X.shape[1]
        n_batch_features = min(n_features, max(1, int(n_features * batch_feature_fraction)))
        X_batched = X.copy()

        for b in range(n_batches):
            mask = batch_labels == b
            if not np.any(mask):
                continue
            global_shift = self.rng.normal(0.0, 0.35 * batch_effect_strength)
            scale = float(np.exp(self.rng.normal(0.0, 0.08 * batch_effect_strength)))

            feature_idx = self.rng.choice(n_features, size=n_batch_features, replace=False)
            feature_shift = np.zeros(n_features, dtype=float)
            feature_shift[feature_idx] = self.rng.normal(
                0.0,
                0.8 * batch_effect_strength,
                size=n_batch_features,
            )

            X_batched[mask, :] = X_batched[mask, :] * scale + global_shift + feature_shift

        return X_batched

    def _generate_missing_mask(
        self,
        X: NDArray[np.float64],
        missing_rate: float,
        missing_pattern: MissingPatternType,
        batch_labels: NDArray[np.int64],
        mar_strength: float = 1.0,
        mnar_slope: float = 2.0,
    ) -> NDArray[np.int8]:
        """Generate missing value mask based on specified pattern."""
        n_samples, n_features = X.shape
        M = np.zeros((n_samples, n_features), dtype=np.int8)

        if missing_rate == 0:
            return M

        total_elements = n_samples * n_features
        target_missing = int(np.round(total_elements * missing_rate))
        target_missing = min(max(target_missing, 0), total_elements)
        if target_missing == 0:
            return M
        if target_missing == total_elements:
            M[:, :] = MaskCode.LOD
            return M

        if missing_pattern == "mcar":
            missing_indices = self._apply_mcar(M, target_missing)
        elif missing_pattern == "mar":
            missing_indices = self._apply_mar(
                X,
                M,
                target_missing,
                batch_labels,
                mar_strength=mar_strength,
            )
        elif missing_pattern == "mnar":
            missing_indices = self._apply_mnar(
                X,
                M,
                target_missing,
                mnar_slope=mnar_slope,
            )
        else:
            missing_indices = np.array([], dtype=np.int64)

        rows, cols = np.unravel_index(missing_indices, (n_samples, n_features))
        M[rows, cols] = MaskCode.LOD

        return M

    def _apply_mcar(
        self,
        M: NDArray[np.int8],
        target_missing: int,
    ) -> NDArray[np.int64]:
        """Apply Missing Completely At Random pattern."""
        n_samples, n_features = M.shape
        total_elements = n_samples * n_features

        return self.rng.choice(total_elements, size=target_missing, replace=False).astype(np.int64)

    def _apply_mar(
        self,
        X: NDArray[np.float64],
        M: NDArray[np.int8],
        target_missing: int,
        batch_labels: NDArray[np.int64],
        mar_strength: float = 1.0,
    ) -> NDArray[np.int64]:
        """Apply Missing At Random pattern.

        Missing probability depends on observed values (e.g., samples with
        lower overall intensity have more missing values).
        """
        del M
        sample_means = self._mean_ignore_nan(X, axis=1, keepdims=False)
        feature_stds = self._std_ignore_nan(X, axis=0, keepdims=False)

        sample_component = -self._zscore(sample_means)[:, None]
        feature_component = self._zscore(feature_stds)[None, :]

        batch_component = np.zeros_like(sample_component)
        unique_batches = np.unique(batch_labels)
        if unique_batches.size > 1:
            batch_signal = np.zeros(sample_means.shape[0], dtype=float)
            for batch_id in unique_batches:
                mask = batch_labels == batch_id
                if np.any(mask):
                    batch_signal[mask] = float(np.mean(sample_means[mask]))
            batch_component = self._zscore(batch_signal)[:, None]

        logits = mar_strength * (
            0.85 * sample_component + 0.35 * feature_component + 0.25 * batch_component
        )
        probabilities = expit(logits)
        return self._draw_weighted_indices(probabilities, target_missing)

    def _apply_mnar(
        self,
        X: NDArray[np.float64],
        M: NDArray[np.int8],
        target_missing: int,
        mnar_slope: float = 2.0,
    ) -> NDArray[np.int64]:
        """Apply Missing Not At Random pattern.

        Missing probability depends on the (unobserved) true value.
        Lower abundance proteins are more likely to be missing.
        """
        del M
        feature_means = self._mean_ignore_nan(X, axis=0, keepdims=False)
        feature_component = -self._zscore(feature_means)[None, :]
        intensity_component = -self._zscore(X)

        logits = mnar_slope * (0.75 * intensity_component + 0.25 * feature_component)
        probabilities = expit(logits)
        return self._draw_weighted_indices(probabilities, target_missing)

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
        batch_effect_strength: float,
        confounded_batches: bool,
        batch_confounding_strength: float,
        cluster_separation: float,
        noise_scale: float,
        mar_strength: float,
        mnar_slope: float,
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
                "batch_effect_strength": batch_effect_strength,
                "confounded_batches": confounded_batches,
                "batch_confounding_strength": batch_confounding_strength,
                "cluster_separation": cluster_separation,
                "noise_scale": noise_scale,
                "mar_strength": mar_strength,
                "mnar_slope": mnar_slope,
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
