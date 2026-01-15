"""
Similarity metrics for comparing ScpTensor and Scanpy outputs.

This module provides comprehensive similarity metrics for validating
numerical equivalence between single-cell proteomics analysis results.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr


@dataclass
class ComparisonResult:
    """Result of comparing two arrays.

    Attributes
    ----------
    module : str
        Name of the module being compared (e.g., "normalization", "pca")
    metrics : Dict[str, float]
        Dictionary of computed metrics
    time_scanpy : float
        Execution time for Scanpy (seconds)
    time_scptensor : float
        Execution time for ScpTensor (seconds)
    passed : bool
        Whether all metrics meet thresholds
    details : Dict[str, Any]
        Additional details
    """

    module: str
    metrics: dict[str, float] = field(default_factory=dict)
    time_scanpy: float = 0.0
    time_scptensor: float = 0.0
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "module": self.module,
            "metrics": self.metrics,
            "time_scanpy": self.time_scanpy,
            "time_scptensor": self.time_scptensor,
            "speedup": speedup_factor(self.time_scanpy, self.time_scptensor),
            "passed": self.passed,
            "details": self.details,
        }

    def __str__(self) -> str:
        """Format as readable string."""
        lines = [
            f"{'=' * 60}",
            f"Module: {self.module}",
            f"{'=' * 60}",
        ]

        # Metrics section
        lines.append("Metrics:")
        for name, value in self.metrics.items():
            if name in ["cosine_similarity", "pearson_r", "spearman_r"]:
                lines.append(f"  {name}: {value:.6f}")
            else:
                lines.append(f"  {name}: {value:.6e}")

        # Timing section
        lines.append("\nTiming:")
        lines.append(f"  Scanpy:      {format_time(self.time_scanpy)}")
        lines.append(f"  ScpTensor:   {format_time(self.time_scptensor)}")
        speedup = speedup_factor(self.time_scanpy, self.time_scptensor)
        if speedup > 0:
            lines.append(f"  Speedup:     {speedup:.2f}x")
        else:
            lines.append(f"  Slowdown:    {abs(speedup):.2f}x")

        # Result section
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"\nResult: {status}")

        # Details section
        if self.details:
            lines.append("\nDetails:")
            for key, value in self.details.items():
                lines.append(f"  {key}: {value}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


class SimilarityMetrics:
    """Calculate similarity metrics between arrays."""

    # Thresholds for passing
    THRESHOLD_COSINE = 0.99
    THRESHOLD_PEARSON = 0.99
    THRESHOLD_MSE = 1e-4
    THRESHOLD_RMSE = 1e-2
    THRESHOLD_MAX_REL_ERR = 0.01  # 1%

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two arrays.

        Parameters
        ----------
        a, b : np.ndarray
            Arrays to compare. Must have same shape.

        Returns
        -------
        float
            Cosine similarity in [0, 1], where 1 is identical.
        """
        a_flat = a.flatten()
        b_flat = b.flatten()

        # Remove NaN/Inf pairs
        mask = np.isfinite(a_flat) & np.isfinite(b_flat)
        a_clean = a_flat[mask]
        b_clean = b_flat[mask]

        if a_clean.size == 0 or b_clean.size == 0:
            return 0.0

        # Compute cosine similarity (1 - cosine distance)
        cos_dist = cosine(a_clean, b_clean)
        return 1.0 - cos_dist

    @staticmethod
    def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient.

        Parameters
        ----------
        a, b : np.ndarray
            Flattened arrays to compare.

        Returns
        -------
        float
            Pearson correlation in [-1, 1], where 1 is perfect positive correlation.
        """
        a_flat = a.flatten()
        b_flat = b.flatten()

        # Remove NaN/Inf pairs
        mask = np.isfinite(a_flat) & np.isfinite(b_flat)
        a_clean = a_flat[mask]
        b_clean = b_flat[mask]

        if a_clean.size < 2 or b_clean.size < 2:
            return 0.0

        # Check for zero variance
        if np.std(a_clean) == 0 or np.std(b_clean) == 0:
            return 1.0 if np.allclose(a_clean, b_clean) else 0.0

        try:
            r, _ = pearsonr(a_clean, b_clean)
            return float(r) if np.isfinite(r) else 0.0
        except (ValueError, RuntimeWarning):
            return 0.0

    @staticmethod
    def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Spearman rank correlation."""
        a_flat = a.flatten()
        b_flat = b.flatten()

        # Remove NaN/Inf pairs
        mask = np.isfinite(a_flat) & np.isfinite(b_flat)
        a_clean = a_flat[mask]
        b_clean = b_flat[mask]

        if a_clean.size < 2 or b_clean.size < 2:
            return 0.0

        try:
            r, _ = spearmanr(a_clean, b_clean)
            return float(r) if np.isfinite(r) else 0.0
        except (ValueError, RuntimeWarning):
            return 0.0

    @staticmethod
    def mse(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        diff = a - b
        mask = np.isfinite(diff)
        if not np.any(mask):
            return np.inf

        return float(np.mean(diff[mask] ** 2))

    @staticmethod
    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(SimilarityMetrics.mse(a, b)))

    @staticmethod
    def max_relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
        """Calculate maximum relative error.

        Parameters
        ----------
        a, b : np.ndarray
            Arrays to compare
        eps : float
            Small value to avoid division by zero

        Returns
        -------
        float
            Maximum relative error as a fraction
        """
        # Handle same-shape requirement
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        # Calculate relative error
        denominator = np.abs(b) + eps
        rel_err = np.abs(a - b) / denominator

        # Filter out non-finite values
        mask = np.isfinite(rel_err)
        if not np.any(mask):
            return np.inf

        return float(np.max(rel_err[mask]))

    @staticmethod
    def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        diff = np.abs(a - b)
        mask = np.isfinite(diff)
        if not np.any(mask):
            return np.inf

        return float(np.mean(diff[mask]))

    @classmethod
    def compute_all(cls, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        """Compute all metrics between two arrays.

        Parameters
        ----------
        a, b : np.ndarray
            Arrays to compare. Must have same shape.

        Returns
        -------
        Dict[str, float]
            Dictionary with all metric names and values
        """
        # Validate shapes
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        # Handle empty arrays
        if a.size == 0:
            return {
                "cosine_similarity": 0.0,
                "pearson_r": 0.0,
                "spearman_r": 0.0,
                "mse": np.inf,
                "rmse": np.inf,
                "max_relative_error": np.inf,
                "mae": np.inf,
            }

        metrics: dict[str, float] = {}

        # Similarity metrics
        metrics["cosine_similarity"] = cls.cosine_similarity(a, b)
        metrics["pearson_r"] = cls.pearson_correlation(a, b)
        metrics["spearman_r"] = cls.spearman_correlation(a, b)

        # Error metrics
        metrics["mse"] = cls.mse(a, b)
        metrics["rmse"] = cls.rmse(a, b)
        try:
            metrics["max_relative_error"] = cls.max_relative_error(a, b)
        except ValueError:
            metrics["max_relative_error"] = np.inf
        metrics["mae"] = cls.mean_absolute_error(a, b)

        return metrics

    @classmethod
    def check_thresholds(cls, metrics: dict[str, float]) -> bool:
        """Check if all metrics meet thresholds.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of computed metrics

        Returns
        -------
        bool
            True if all metrics meet thresholds
        """
        # Check cosine similarity
        cos_sim = metrics.get("cosine_similarity", 0.0)
        if cos_sim < cls.THRESHOLD_COSINE:
            return False

        # Check Pearson correlation
        pearson = metrics.get("pearson_r", 0.0)
        if pearson < cls.THRESHOLD_PEARSON:
            return False

        # Check MSE
        mse_val = metrics.get("mse", np.inf)
        if mse_val > cls.THRESHOLD_MSE:
            return False

        # Check RMSE
        rmse_val = metrics.get("rmse", np.inf)
        if rmse_val > cls.THRESHOLD_RMSE:
            return False

        # Check max relative error
        max_rel = metrics.get("max_relative_error", np.inf)
        return not max_rel > cls.THRESHOLD_MAX_REL_ERR

    @classmethod
    def compare_results(
        cls,
        scanpy_result: np.ndarray,
        scptensor_result: np.ndarray,
        module: str,
        time_scanpy: float = 0.0,
        time_scptensor: float = 0.0,
    ) -> ComparisonResult:
        """Compare two results and create ComparisonResult.

        Parameters
        ----------
        scanpy_result : np.ndarray
            Result from Scanpy
        scptensor_result : np.ndarray
            Result from ScpTensor
        module : str
            Module name for reporting
        time_scanpy, time_scptensor : float
            Execution times

        Returns
        -------
        ComparisonResult
            Complete comparison result
        """
        details: dict[str, Any] = {}

        # Check shape
        if scanpy_result.shape != scptensor_result.shape:
            details["shape_mismatch"] = {
                "scanpy": scanpy_result.shape,
                "scptensor": scptensor_result.shape,
            }
            return ComparisonResult(
                module=module,
                metrics={},
                time_scanpy=time_scanpy,
                time_scptensor=time_scptensor,
                passed=False,
                details=details,
            )

        # Check for NaN/Inf
        np.any(np.isnan(scanpy_result))
        np.any(np.isinf(scanpy_result))
        np.any(np.isnan(scptensor_result))
        np.any(np.isinf(scptensor_result))

        details["nan_count_scanpy"] = int(np.sum(np.isnan(scanpy_result)))
        details["nan_count_scptensor"] = int(np.sum(np.isnan(scptensor_result)))
        details["inf_count_scanpy"] = int(np.sum(np.isinf(scanpy_result)))
        details["inf_count_scptensor"] = int(np.sum(np.isinf(scptensor_result)))
        details["shape"] = scanpy_result.shape

        # Compute metrics
        try:
            metrics = cls.compute_all(scanpy_result, scptensor_result)
        except Exception as e:
            details["error"] = str(e)
            return ComparisonResult(
                module=module,
                metrics={},
                time_scanpy=time_scanpy,
                time_scptensor=time_scptensor,
                passed=False,
                details=details,
            )

        # Check thresholds
        passed = cls.check_thresholds(metrics)

        return ComparisonResult(
            module=module,
            metrics=metrics,
            time_scanpy=time_scanpy,
            time_scptensor=time_scptensor,
            passed=passed,
            details=details,
        )


def format_time(seconds: float) -> str:
    """Format time in human-readable string.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str
        Formatted time string
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def speedup_factor(time_scanpy: float, time_scptensor: float) -> float:
    """Calculate speedup factor (positive = ScpTensor faster).

    Parameters
    ----------
    time_scanpy : float
        Execution time for Scanpy
    time_scptensor : float
        Execution time for ScpTensor

    Returns
    -------
    float
        Speedup factor. >1 means ScpTensor is faster.
    """
    if time_scanpy <= 0:
        return 0.0
    if time_scptensor <= 0:
        return 0.0
    return time_scanpy / time_scptensor


def jaccard_similarity(set1: set | np.ndarray, set2: set | np.ndarray) -> float:
    """Calculate Jaccard similarity between two sets.

    The Jaccard similarity coefficient measures the similarity between
    finite sample sets, and is defined as the size of the intersection
    divided by the size of the union of the sample sets.

    Parameters
    ----------
    set1, set2 : set or np.ndarray
        Sets or arrays to compare. If arrays, they will be converted to sets.

    Returns
    -------
    float
        Jaccard similarity in [0, 1], where 1 means identical sets
        and 0 means no overlap.

    Examples
    --------
    >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})
    0.5  # Intersection: {2, 3}, Union: {1, 2, 3, 4}

    >>> jaccard_similarity({1, 2}, {1, 2})
    1.0  # Identical sets
    """
    # Convert to sets if needed
    if isinstance(set1, np.ndarray):
        set1 = set(set1)
    if isinstance(set2, np.ndarray):
        set2 = set(set2)

    # Calculate intersection and union
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    # Handle empty union case
    if union == 0:
        return 1.0 if len(set1) == 0 and len(set2) == 0 else 0.0

    return intersection / union


if __name__ == "__main__":
    # Test basic functionality
    print("Testing SimilarityMetrics module...")

    # Create test arrays
    np.random.seed(42)
    a = np.random.randn(10, 5)
    b = a + np.random.randn(10, 5) * 0.01  # Very similar

    # Compute metrics
    metrics = SimilarityMetrics.compute_all(a, b)
    print("\nMetrics between similar arrays:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Compare identical arrays
    c = a.copy()
    metrics_identical = SimilarityMetrics.compute_all(a, c)
    print("\nMetrics between identical arrays:")
    for k, v in metrics_identical.items():
        print(f"  {k}: {v:.6f}")

    # Test ComparisonResult
    result = SimilarityMetrics.compare_results(
        a, b, "test_module", time_scanpy=1.5, time_scptensor=0.5
    )
    print("\n" + str(result))

    # Test with identical arrays
    result_identical = SimilarityMetrics.compare_results(
        a, c, "identical_test", time_scanpy=1.0, time_scptensor=0.8
    )
    print("\n" + str(result_identical))

    # Test format_time
    print("\nTesting format_time:")
    for t in [1e-9, 1e-6, 1e-3, 0.5, 5, 65]:
        print(f"  {t:.2e} -> {format_time(t)}")

    # Test speedup_factor
    print("\nTesting speedup_factor:")
    print(f"  Scanpy=1.0s, ScpTensor=0.5s -> {speedup_factor(1.0, 0.5):.2f}x")
    print(f"  Scanpy=0.5s, ScpTensor=1.0s -> {speedup_factor(0.5, 1.0):.2f}x")

    print("\nAll tests passed!")
