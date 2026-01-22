"""Result dataclasses for storing and comparing benchmark results.

This module provides structured dataclasses for storing preprocessing method
benchmark results between ScpTensor and Scanpy frameworks.

The result classes support:
- Method specification metadata
- Performance metrics (runtime, memory, throughput)
- Accuracy metrics (MSE, MAE, correlation, etc.)
- Biological metrics (kBET, iLISI, cLISI, ASW, etc.)
- Method comparison with derived ratios
- Collection of all benchmark results by category

Examples
--------
Create a method specification:

>>> from scptensor.benchmark.core.result import MethodSpec, MethodCategory
>>> spec = MethodSpec(
...     name="log_normalize",
...     display_name="Log Normalize",
...     category=MethodCategory.NORMALIZATION,
...     layer=ComparisonLayer.SHARED,
...     framework="scptensor",
...     description="Logarithmic transformation with offset",
...     parameters={"base": 2.0, "offset": 1.0}
... )

Create and compare benchmark results:

>>> result_a = BenchmarkResult(
...     method_spec=spec,
...     performance=PerformanceMetrics(runtime_seconds=0.5, memory_mb=100.0, throughput=1000.0),
...     accuracy=AccuracyMetrics(mse=0.01, mae=0.05, correlation=0.99, spearman_correlation=0.98, cosine_similarity=0.99),
...     biological=BiologicalMetrics(kbet_score=0.8, ilisi_score=0.7, clisi_score=0.9, asw_score=0.85, variance_preserved=0.95),
...     error=None,
...     timestamp="2026-01-19T10:00:00"
... )
>>> result_b = BenchmarkResult(...)
>>> comparison = ComparisonResult(method_a=spec, method_b=spec, result_a=result_a, result_b=result_b)
>>> print(f"Speedup: {comparison.speedup_ratio:.2f}x")
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# =============================================================================
# Enums
# =============================================================================


class MethodCategory(Enum):
    """Categories of preprocessing methods for single-cell proteomics analysis.

    Attributes
    ----------
    NORMALIZATION
        Data normalization and scaling methods.
    IMPUTATION
        Missing value imputation methods.
    INTEGRATION
        Batch correction and data integration methods.
    QC
        Quality control filtering methods.
    DIMENSIONALITY_REDUCTION
        Dimensionality reduction methods (PCA, UMAP, etc.).
    FEATURE_SELECTION
        Feature selection methods (HVG, VST, etc.).
    PIPELINE
        End-to-end pipeline workflows.
    """

    NORMALIZATION = "normalization"
    IMPUTATION = "imputation"
    INTEGRATION = "integration"
    QC = "qc"
    DIMENSIONALITY_REDUCTION = "dim_reduction"
    FEATURE_SELECTION = "feature_selection"
    PIPELINE = "pipeline"


class ComparisonLayer(Enum):
    """Layers of comparison strategy between ScpTensor and Scanpy.

    Attributes
    ----------
    SHARED
        Methods available in both frameworks for direct comparison.
    SCPTENSOR_INTERNAL
        Methods only available in ScpTensor.
    SCANPY_INTERNAL
        Methods only available in Scanpy.
    """

    SHARED = "shared"
    SCPTENSOR_INTERNAL = "scptensor_internal"
    SCANPY_INTERNAL = "scanpy_internal"


# =============================================================================
# Method Specification
# =============================================================================


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Specification for a preprocessing method.

    Attributes
    ----------
    name : str
        Unique method identifier (e.g., "log_normalize", "knn_impute").
    display_name : str
        Human-readable name for display in reports and plots.
    category : MethodCategory
        Category of the method (normalization, imputation, etc.).
    layer : ComparisonLayer
        Comparison layer indicating framework availability.
    framework : str
        Framework name ("scptensor", "scanpy", or "both").
    description : str
        Brief description of the method.
    parameters : dict[str, Any]
        Default parameters used for the method.
    """

    name: str
    display_name: str
    category: MethodCategory
    layer: ComparisonLayer
    framework: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the method specification.
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category.value,
            "layer": self.layer.value,
            "framework": self.framework,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MethodSpec:
        """Create MethodSpec from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing method specification data.

        Returns
        -------
        MethodSpec
            Reconstructed method specification.
        """
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            category=MethodCategory(data["category"]),
            layer=ComparisonLayer(data["layer"]),
            framework=data["framework"],
            description=data["description"],
            parameters=data.get("parameters", {}),
        )


# =============================================================================
# Metrics Dataclasses
# =============================================================================


class _Serializable:
    """Base mixin for serialization of dataclasses with enum conversion."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]):
        """Create instance from dictionary."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary using field names."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class PerformanceMetrics(_Serializable):
    """Computational performance metrics for a method run.

    Attributes
    ----------
    runtime_seconds : float
        Total execution time in seconds.
    memory_mb : float
        Peak memory usage in megabytes.
    throughput : float
        Throughput measure (samples per second).
    """

    runtime_seconds: float
    memory_mb: float
    throughput: float

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> PerformanceMetrics:
        """Create PerformanceMetrics from dictionary.

        Parameters
        ----------
        data : dict[str, float]
            Dictionary containing performance metrics.

        Returns
        -------
        PerformanceMetrics
            Reconstructed performance metrics.
        """
        return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass(frozen=True, slots=True)
class AccuracyMetrics(_Serializable):
    """Accuracy metrics for comparing processed results against ground truth.

    Attributes
    ----------
    mse : float
        Mean squared error between predicted and true values.
    mae : float
        Mean absolute error between predicted and true values.
    correlation : float
        Pearson correlation coefficient.
    spearman_correlation : float
        Spearman rank correlation coefficient.
    cosine_similarity : float
        Cosine similarity between vectors.
    """

    mse: float
    mae: float
    correlation: float
    spearman_correlation: float
    cosine_similarity: float

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> AccuracyMetrics:
        """Create AccuracyMetrics from dictionary.

        Parameters
        ----------
        data : dict[str, float]
            Dictionary containing accuracy metrics.

        Returns
        -------
        AccuracyMetrics
            Reconstructed accuracy metrics.
        """
        return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass(frozen=True, slots=True)
class BiologicalMetrics(_Serializable):
    """Biological fidelity metrics for benchmark evaluation.

    Attributes
    ----------
    group_separation : float
        Separation between biological groups (e.g., silhouette score).
    biological_signal_preservation : float
        Preservation of true biological signals.
    clustering_consistency : float
        Consistency of clustering with ground truth.
    biological_variance_explained : float
        Percentage of variance explained by biological factors.
    differential_expression_concordance : float | None
        DE analysis consistency, if available.
    """

    group_separation: float
    biological_signal_preservation: float
    clustering_consistency: float
    biological_variance_explained: float
    differential_expression_concordance: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> BiologicalMetrics:
        """Create BiologicalMetrics from dictionary.

        Parameters
        ----------
        data : dict[str, float]
            Dictionary containing biological metrics.

        Returns
        -------
        BiologicalMetrics
            Reconstructed biological metrics.
        """
        return cls(
            group_separation=data["group_separation"],
            biological_signal_preservation=data["biological_signal_preservation"],
            clustering_consistency=data["clustering_consistency"],
            biological_variance_explained=data["biological_variance_explained"],
            differential_expression_concordance=data.get("differential_expression_concordance"),
        )


# =============================================================================
# Single Result
# =============================================================================


@dataclass(slots=True)
class BenchmarkResult:
    """Result from running a single preprocessing method benchmark.

    Attributes
    ----------
    method_spec : MethodSpec
        Specification of the method that was run.
    performance : PerformanceMetrics
        Computational performance metrics.
    accuracy : AccuracyMetrics | None
        Accuracy metrics against ground truth, if available.
    biological : BiologicalMetrics | None
        Biological preservation metrics, if available.
    error : str | None
        Error message if the method failed, None otherwise.
    timestamp : str
        ISO timestamp of when the benchmark was run.
    dataset_name : str | None
        Name of the dataset used for the benchmark.
    n_samples : int | None
        Number of samples in the dataset.
    n_features : int | None
        Number of features in the dataset.
    """

    method_spec: MethodSpec
    performance: PerformanceMetrics
    accuracy: AccuracyMetrics | None = None
    biological: BiologicalMetrics | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_name: str | None = None
    n_samples: int | None = None
    n_features: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the benchmark result.
        """
        return {
            "method_spec": self.method_spec.to_dict(),
            "performance": self.performance.to_dict(),
            "accuracy": self.accuracy.to_dict() if self.accuracy else None,
            "biological": self.biological.to_dict() if self.biological else None,
            "error": self.error,
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create BenchmarkResult from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing benchmark result data.

        Returns
        -------
        BenchmarkResult
            Reconstructed benchmark result.
        """
        return cls(
            method_spec=MethodSpec.from_dict(data["method_spec"]),
            performance=PerformanceMetrics.from_dict(data["performance"]),
            accuracy=AccuracyMetrics.from_dict(data["accuracy"]) if data.get("accuracy") else None,
            biological=BiologicalMetrics.from_dict(data["biological"])
            if data.get("biological")
            else None,
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            dataset_name=data.get("dataset_name"),
            n_samples=data.get("n_samples"),
            n_features=data.get("n_features"),
        )

    @property
    def is_success(self) -> bool:
        """Check if the benchmark run was successful.

        Returns
        -------
        bool
            True if no error occurred, False otherwise.
        """
        return self.error is None


# =============================================================================
# Comparison Result
# =============================================================================


@dataclass(slots=True)
class ComparisonResult:
    """Result of comparing two method runs.

    Provides computed ratios and deltas for easy comparison between
    ScpTensor and Scanpy implementations of the same method.

    Attributes
    ----------
    method_a : MethodSpec
        Specification of method A (typically ScpTensor).
    method_b : MethodSpec
        Specification of method B (typically Scanpy).
    result_a : BenchmarkResult
        Benchmark result for method A.
    result_b : BenchmarkResult
        Benchmark result for method B.
    comparison_name : str | None
        Optional name for this comparison.
    """

    method_a: MethodSpec
    method_b: MethodSpec
    result_a: BenchmarkResult
    result_b: BenchmarkResult
    comparison_name: str | None = None

    @property
    def speedup_ratio(self) -> float | None:
        """Compute speedup ratio of A over B.

        Returns
        -------
        float | None
            Runtime_B / Runtime_A. Values > 1 mean A is faster.
            None if either result failed.
        """
        if not (self.result_a.is_success and self.result_b.is_success):
            return None
        rt_a = self.result_a.performance.runtime_seconds
        return self.result_b.performance.runtime_seconds / rt_a if rt_a > 0 else None

    @property
    def memory_ratio(self) -> float | None:
        """Compute memory ratio of A to B.

        Returns
        -------
        float | None
            Memory_A / Memory_B. Values < 1 mean A uses less memory.
            None if either result failed.
        """
        if not (self.result_a.is_success and self.result_b.is_success):
            return None
        mem_b = self.result_b.performance.memory_mb
        return self.result_a.performance.memory_mb / mem_b if mem_b > 0 else None

    @property
    def accuracy_delta(self) -> dict[str, float] | None:
        """Compute accuracy difference between A and B.

        Returns
        -------
        dict[str, float] | None
            Dictionary mapping metric names to (A - B) differences.
            Positive values mean A performed better. None if either
            result failed or has no accuracy metrics.
        """
        if not (self.result_a.is_success and self.result_b.is_success):
            return None
        acc_a, acc_b = self.result_a.accuracy, self.result_b.accuracy
        if acc_a is None or acc_b is None:
            return None
        return {
            "mse": acc_b.mse - acc_a.mse,
            "mae": acc_b.mae - acc_a.mae,
            "correlation": acc_a.correlation - acc_b.correlation,
            "spearman_correlation": acc_a.spearman_correlation - acc_b.spearman_correlation,
            "cosine_similarity": acc_a.cosine_similarity - acc_b.cosine_similarity,
        }

    @property
    def biological_delta(self) -> dict[str, float] | None:
        """Compute biological metrics difference between A and B.

        Returns
        -------
        dict[str, float] | None
            Dictionary mapping metric names to (A - B) differences.
            Positive values mean A performed better. None if either
            result failed or has no biological metrics.
        """
        if not (self.result_a.is_success and self.result_b.is_success):
            return None
        bio_a, bio_b = self.result_a.biological, self.result_b.biological
        if bio_a is None or bio_b is None:
            return None
        return {
            "group_separation": bio_a.group_separation - bio_b.group_separation,
            "biological_signal_preservation": bio_a.biological_signal_preservation
            - bio_b.biological_signal_preservation,
            "clustering_consistency": bio_a.clustering_consistency - bio_b.clustering_consistency,
            "biological_variance_explained": bio_a.biological_variance_explained
            - bio_b.biological_variance_explained,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the comparison result.
        """
        return {
            "method_a": self.method_a.to_dict(),
            "method_b": self.method_b.to_dict(),
            "result_a": self.result_a.to_dict(),
            "result_b": self.result_b.to_dict(),
            "comparison_name": self.comparison_name,
            "speedup_ratio": self.speedup_ratio,
            "memory_ratio": self.memory_ratio,
            "accuracy_delta": self.accuracy_delta,
            "biological_delta": self.biological_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonResult:
        """Create ComparisonResult from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing comparison result data.

        Returns
        -------
        ComparisonResult
            Reconstructed comparison result.
        """
        return cls(
            method_a=MethodSpec.from_dict(data["method_a"]),
            method_b=MethodSpec.from_dict(data["method_b"]),
            result_a=BenchmarkResult.from_dict(data["result_a"]),
            result_b=BenchmarkResult.from_dict(data["result_b"]),
            comparison_name=data.get("comparison_name"),
        )


# =============================================================================
# Results Container
# =============================================================================


@dataclass(slots=True)
class BenchmarkResults:
    """Container for all benchmark results across method categories.

    Provides methods for adding, retrieving, and exporting benchmark results.
    Results are organized by category for easy access and comparison.

    Attributes
    ----------
    normalization : dict[str, BenchmarkResult]
        Results for normalization methods.
    imputation : dict[str, BenchmarkResult]
        Results for imputation methods.
    integration : dict[str, BenchmarkResult]
        Results for batch correction/integration methods.
    qc : dict[str, BenchmarkResult]
        Results for quality control methods.
    dim_reduction : dict[str, BenchmarkResult]
        Results for dimensionality reduction methods.
    feature_selection : dict[str, BenchmarkResult]
        Results for feature selection methods.
    pipeline : dict[str, BenchmarkResult]
        Results for end-to-end pipelines.
    comparisons : list[ComparisonResult]
        Direct comparison results between methods.
    metadata : dict[str, Any]
        Additional metadata about the benchmark run.
    """

    normalization: dict[str, BenchmarkResult] = field(default_factory=dict)
    imputation: dict[str, BenchmarkResult] = field(default_factory=dict)
    integration: dict[str, BenchmarkResult] = field(default_factory=dict)
    qc: dict[str, BenchmarkResult] = field(default_factory=dict)
    dim_reduction: dict[str, BenchmarkResult] = field(default_factory=dict)
    feature_selection: dict[str, BenchmarkResult] = field(default_factory=dict)
    pipeline: dict[str, BenchmarkResult] = field(default_factory=dict)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    _CATEGORY_FIELDS: tuple[str, ...] = (
        "normalization",
        "imputation",
        "integration",
        "qc",
        "dim_reduction",
        "feature_selection",
        "pipeline",
    )

    def _get_category_dict(self, category: MethodCategory) -> dict[str, BenchmarkResult]:
        """Get the results dictionary for a category.

        Parameters
        ----------
        category : MethodCategory
            The category to retrieve.

        Returns
        -------
        dict[str, BenchmarkResult]
            The results dictionary for the category.
        """
        return getattr(self, category.value, {})

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the appropriate category.

        Parameters
        ----------
        result : BenchmarkResult
            The benchmark result to add.
        """
        self._get_category_dict(result.method_spec.category)[result.method_spec.name] = result

    def add_comparison(self, comparison: ComparisonResult) -> None:
        """Add a comparison result.

        Parameters
        ----------
        comparison : ComparisonResult
            The comparison result to add.
        """
        self.comparisons.append(comparison)

    def get_result(self, method_name: str, category: MethodCategory) -> BenchmarkResult | None:
        """Get a benchmark result by method name and category.

        Parameters
        ----------
        method_name : str
            Name of the method.
        category : MethodCategory
            Category of the method.

        Returns
        -------
        BenchmarkResult | None
            The benchmark result if found, None otherwise.
        """
        return self._get_category_dict(category).get(method_name)

    def get_results_by_category(self, category: MethodCategory) -> dict[str, BenchmarkResult]:
        """Get all results for a specific category.

        Parameters
        ----------
        category : MethodCategory
            The category to retrieve results for.

        Returns
        -------
        dict[str, BenchmarkResult]
            Dictionary mapping method names to results.
        """
        return self._get_category_dict(category).copy()

    def get_successful_results(self, category: MethodCategory) -> dict[str, BenchmarkResult]:
        """Get all successful results for a category.

        Parameters
        ----------
        category : MethodCategory
            The category to filter.

        Returns
        -------
        dict[str, BenchmarkResult]
            Dictionary of successful benchmark results.
        """
        return {
            name: result
            for name, result in self._get_category_dict(category).items()
            if result.is_success
        }

    def iter_all_results(self) -> Iterator[tuple[MethodCategory, str, BenchmarkResult]]:
        """Iterate over all results across all categories.

        Yields
        ------
        tuple[MethodCategory, str, BenchmarkResult]
            Tuples of (category, method_name, result).
        """
        for category in MethodCategory:
            for name, result in self._get_category_dict(category).items():
                yield category, name, result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of all benchmark results.
        """
        return {
            cat: {name: r.to_dict() for name, r in getattr(self, cat).items()}
            for cat in self._CATEGORY_FIELDS
        } | {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResults:
        """Create BenchmarkResults from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing benchmark results data.

        Returns
        -------
        BenchmarkResults
            Reconstructed benchmark results container.
        """
        results = cls()
        for cat in cls._CATEGORY_FIELDS:
            for name, result_data in data.get(cat, {}).items():
                getattr(results, cat)[name] = BenchmarkResult.from_dict(result_data)
        results.comparisons = [ComparisonResult.from_dict(c) for c in data.get("comparisons", [])]
        results.metadata = data.get("metadata", {})
        return results

    def save_json(self, filepath: str) -> None:
        """Save results to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output JSON file.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> BenchmarkResults:
        """Load results from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the input JSON file.

        Returns
        -------
        BenchmarkResults
            Loaded benchmark results.
        """
        with open(filepath) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> dict[str, Any]:
        """Generate a summary of all benchmark results.

        Returns
        -------
        dict[str, Any]
            Summary statistics including counts and averages.
        """
        total_results = successful = failed = 0
        by_category: dict[str, dict[str, int]] = {}

        for category in MethodCategory:
            category_dict = self._get_category_dict(category)
            total = len(category_dict)
            successful_count = sum(1 for r in category_dict.values() if r.is_success)

            by_category[category.value] = {
                "total": total,
                "successful": successful_count,
                "failed": total - successful_count,
            }
            total_results += total
            successful += successful_count
            failed += total - successful_count

        return {
            "total_results": total_results,
            "successful": successful,
            "failed": failed,
            "by_category": by_category,
            "num_comparisons": len(self.comparisons),
        }


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "MethodCategory",
    "ComparisonLayer",
    "MethodSpec",
    "PerformanceMetrics",
    "AccuracyMetrics",
    "BiologicalMetrics",
    "BenchmarkResult",
    "ComparisonResult",
    "BenchmarkResults",
]
