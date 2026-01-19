"""Method registry for ScpTensor vs Scanpy comparison benchmarks.

This module defines all comparable methods and their metadata.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from collections.abc import Iterable


# =============================================================================
# Method Categories
# =============================================================================


class MethodCategory(Enum):
    """Categories of analysis methods."""

    NORMALIZATION = "normalization"
    IMPUTATION = "imputation"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    CLUSTERING = "clustering"
    FEATURE_SELECTION = "feature_selection"
    BATCH_CORRECTION = "batch_correction"


class ComparisonLayer(Enum):
    """Layers of comparison strategy."""

    SHARED = "shared"  # Direct ScpTensor vs Scanpy comparison
    SCPTENSOR_INTERNAL = "scptensor_internal"  # ScpTensor-only methods
    SCANPY_INTERNAL = "scanpy_internal"  # Scanpy-only methods


# =============================================================================
# Method Metadata
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MethodInfo:
    """Metadata for a comparison method.

    Attributes
    ----------
    name : str
        Unique method identifier.
    display_name : str
        Human-readable name.
    category : MethodCategory
        Method category.
    layer : ComparisonLayer
        Comparison layer.
    framework : str
        Framework name ("scptensor" or "scanpy").
    parameters : dict
        Default parameters for the method.
    description : str
        Brief description.
    """

    name: str
    display_name: str
    category: MethodCategory
    layer: ComparisonLayer
    framework: str
    parameters: dict
    description: str


# =============================================================================
# Method Registry
# =============================================================================


# Shared methods (Layer 1: Direct comparison)
SHARED_METHODS: tuple[MethodInfo, ...] = (
    # Normalization
    MethodInfo(
        name="log_normalize",
        display_name="Log Normalize",
        category=MethodCategory.NORMALIZATION,
        layer=ComparisonLayer.SHARED,
        framework="both",
        parameters={"base": 2.0, "offset": 1.0},
        description="Logarithmic transformation with offset",
    ),
    MethodInfo(
        name="z_score_normalize",
        display_name="Z-Score Normalize",
        category=MethodCategory.NORMALIZATION,
        layer=ComparisonLayer.SHARED,
        framework="both",
        parameters={},
        description="Z-score standardization",
    ),
    # Imputation
    MethodInfo(
        name="knn_impute",
        display_name="KNN Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SHARED,
        framework="both",
        parameters={"k": 15},
        description="K-nearest neighbors imputation",
    ),
    # Dimensionality Reduction
    MethodInfo(
        name="pca",
        display_name="PCA",
        category=MethodCategory.DIMENSIONALITY_REDUCTION,
        layer=ComparisonLayer.SHARED,
        framework="both",
        parameters={"n_components": 50},
        description="Principal Component Analysis",
    ),
    MethodInfo(
        name="umap",
        display_name="UMAP",
        category=MethodCategory.DIMENSIONALITY_REDUCTION,
        layer=ComparisonLayer.SHARED,
        framework="both",
        parameters={"n_components": 2, "n_neighbors": 15},
        description="UMAP nonlinear embedding",
    ),
    # Clustering
    MethodInfo(
        name="kmeans",
        display_name="K-means",
        category=MethodCategory.CLUSTERING,
        layer=ComparisonLayer.SHARED,
        framework="both",
        parameters={"n_clusters": 5, "random_state": 42},
        description="K-means clustering",
    ),
    # Feature Selection
    MethodInfo(
        name="hvg",
        display_name="Highly Variable Genes",
        category=MethodCategory.FEATURE_SELECTION,
        layer=ComparisonLayer.SHARED,
        framework="both",
        parameters={"n_top_genes": 2000},
        description="Highly variable feature selection",
    ),
)


# ScpTensor internal methods (Layer 2)
SCPTENSOR_INTERNAL_METHODS: tuple[MethodInfo, ...] = (
    # Advanced Imputation
    MethodInfo(
        name="svd_impute",
        display_name="SVD Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={"rank": 50},
        description="SVD-based imputation",
    ),
    MethodInfo(
        name="bpca_impute",
        display_name="BPCA Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="Bayesian PCA imputation",
    ),
    MethodInfo(
        name="missforest_impute",
        display_name="MissForest Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="Random forest-based imputation",
    ),
    MethodInfo(
        name="lls_impute",
        display_name="LLS Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="Local least squares imputation",
    ),
    MethodInfo(
        name="minprob_impute",
        display_name="MinProb Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="Minimum probability imputation",
    ),
    MethodInfo(
        name="qrilc_impute",
        display_name="QRILC Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="QRILC imputation for LOD values",
    ),
    # Batch Correction
    MethodInfo(
        name="combat",
        display_name="ComBat",
        category=MethodCategory.BATCH_CORRECTION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="ComBat batch correction",
    ),
    MethodInfo(
        name="harmony",
        display_name="Harmony",
        category=MethodCategory.BATCH_CORRECTION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="Harmony batch correction",
    ),
    MethodInfo(
        name="mnn_correct",
        display_name="MNN Correct",
        category=MethodCategory.BATCH_CORRECTION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="Mutual Nearest Neighbors correction",
    ),
    MethodInfo(
        name="scanorama",
        display_name="Scanorama",
        category=MethodCategory.BATCH_CORRECTION,
        layer=ComparisonLayer.SCPTENSOR_INTERNAL,
        framework="scptensor",
        parameters={},
        description="Scanorama integration",
    ),
)


# Scanpy internal methods (Layer 3)
SCANPY_INTERNAL_METHODS: tuple[MethodInfo, ...] = (
    MethodInfo(
        name="magic_impute",
        display_name="Magic Impute",
        category=MethodCategory.IMPUTATION,
        layer=ComparisonLayer.SCANPY_INTERNAL,
        framework="scanpy",
        parameters={},
        description="Magic diffusion-based imputation",
    ),
    MethodInfo(
        name="diffusion_maps",
        display_name="Diffusion Maps",
        category=MethodCategory.DIMENSIONALITY_REDUCTION,
        layer=ComparisonLayer.SCANPY_INTERNAL,
        framework="scanpy",
        parameters={},
        description="Diffusion maps dimensionality reduction",
    ),
    MethodInfo(
        name="tsne",
        display_name="t-SNE",
        category=MethodCategory.DIMENSIONALITY_REDUCTION,
        layer=ComparisonLayer.SCANPY_INTERNAL,
        framework="scanpy",
        parameters={"n_components": 2},
        description="t-SNE nonlinear embedding",
    ),
    MethodInfo(
        name="paga",
        display_name="PAGA",
        category=MethodCategory.CLUSTERING,
        layer=ComparisonLayer.SCANPY_INTERNAL,
        framework="scanpy",
        parameters={},
        description="PAGA trajectory inference",
    ),
)


# =============================================================================
# Method Registry Class
# =============================================================================


class MethodRegistry:
    """Registry for comparison methods."""

    def __init__(self) -> None:
        """Initialize the method registry."""
        self._methods: dict[str, MethodInfo] = {}

        # Register all methods
        for method in SHARED_METHODS:
            self._methods[method.name] = method
        for method in SCPTENSOR_INTERNAL_METHODS:
            self._methods[method.name] = method
        for method in SCANPY_INTERNAL_METHODS:
            self._methods[method.name] = method

    def get(self, name: str) -> MethodInfo | None:
        """Get method info by name.

        Parameters
        ----------
        name : str
            Method name.

        Returns
        -------
        MethodInfo | None
            Method info or None if not found.
        """
        return self._methods.get(name)

    def list_shared(self) -> tuple[MethodInfo, ...]:
        """List shared methods (direct comparison).

        Returns
        -------
        tuple[MethodInfo, ...]
            Shared methods.
        """
        return tuple(m for m in self._methods.values() if m.layer == ComparisonLayer.SHARED)

    def list_by_category(self, category: MethodCategory) -> tuple[MethodInfo, ...]:
        """List methods by category.

        Parameters
        ----------
        category : MethodCategory
            Method category.

        Returns
        -------
        tuple[MethodInfo, ...]
            Methods in the category.
        """
        return tuple(m for m in self._methods.values() if m.category == category)

    def list_by_framework(self, framework: str) -> tuple[MethodInfo, ...]:
        """List methods by framework.

        Parameters
        ----------
        framework : str
            Framework name ("scptensor" or "scanpy").

        Returns
        -------
        tuple[MethodInfo, ...]
            Methods for the framework.
        """
        return tuple(m for m in self._methods.values() if m.framework == framework or m.framework == "both")

    def list_scptensor_internal(self) -> tuple[MethodInfo, ...]:
        """List ScpTensor internal methods.

        Returns
        -------
        tuple[MethodInfo, ...]
            ScpTensor-only methods.
        """
        return tuple(m for m in self._methods.values() if m.layer == ComparisonLayer.SCPTENSOR_INTERNAL)

    def list_scanpy_internal(self) -> tuple[MethodInfo, ...]:
        """List Scanpy internal methods.

        Returns
        -------
        tuple[MethodInfo, ...]
            Scanpy-only methods.
        """
        return tuple(m for m in self._methods.values() if m.layer == ComparisonLayer.SCANPY_INTERNAL)


# Global registry instance
_registry: MethodRegistry | None = None


def get_registry() -> MethodRegistry:
    """Get the global method registry.

    Returns
    -------
    MethodRegistry
        Global method registry.
    """
    global _registry
    if _registry is None:
        _registry = MethodRegistry()
    return _registry


# =============================================================================
# Comparison Pairs
# =============================================================================


def get_comparison_pairs() -> dict[str, tuple[str, str]]:
    """Get method pairs for direct comparison.

    Returns
    -------
    dict[str, tuple[str, str]]
        Mapping of comparison name to (scptensor_method, scanpy_method).
    """
    return {
        "log_normalize": ("log_normalize", "log_normalize"),
        "z_score_normalize": ("z_score_normalize", "z_score_normalize"),
        "knn_impute": ("knn_impute", "knn_impute"),
        "pca": ("pca", "pca"),
        "umap": ("umap", "umap"),
        "kmeans": ("kmeans", "kmeans"),
        "hvg": ("hvg", "highly_variable_genes"),
    }


def get_internal_baseline_pairs() -> dict[str, str]:
    """Get baseline methods for internal comparison.

    Returns
    -------
    dict[str, str]
        Mapping of method to its baseline for comparison.
    """
    return {
        # Imputation baselines
        "svd_impute": "knn_impute",
        "bpca_impute": "knn_impute",
        "missforest_impute": "knn_impute",
        "lls_impute": "knn_impute",
        "minprob_impute": "knn_impute",
        "qrilc_impute": "knn_impute",
        # Batch correction baselines
        "combat": "no_correction",
        "harmony": "combat",
        "mnn_correct": "combat",
        "scanorama": "combat",
    }
