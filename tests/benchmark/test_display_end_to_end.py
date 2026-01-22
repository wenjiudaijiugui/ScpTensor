"""Tests for end-to-end display module.

Tests cover:
- scptensor.benchmark.display.end_to_end.EndToEndDisplay: Pipeline comparison visualization
- scptensor.benchmark.display.end_to_end.PipelineResult: Pipeline result dataclass
- scptensor.benchmark.display.end_to_end.PipelineStep: Pipeline step dataclass
- scptensor.benchmark.display.end_to_end.ClusteringMetrics: Clustering metrics dataclass
- scptensor.benchmark.display.end_to_end.IntermediateResults: Intermediate results dataclass
- scptensor.benchmark.display.end_to_end.compute_jaccard_index: Jaccard index computation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from scptensor.benchmark.display.end_to_end import (
    ClusteringMetrics,
    EndToEndDisplay,
    IntermediateResults,
    PipelineResult,
    PipelineStep,
    compute_jaccard_index,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for display outputs.

    Returns
    -------
    Path
        Path to temporary output directory.
    """
    return tmp_path / "benchmark_results"


@pytest.fixture
def sample_pipeline_steps() -> list[PipelineStep]:
    """Create sample pipeline steps.

    Returns
    -------
    list[PipelineStep]
        List of sample pipeline steps.
    """
    return [
        PipelineStep(
            name="qc",
            display_name="Quality Control",
            method_name="basic_qc",
            parameters={"min_cells": 3, "min_features": 1},
            runtime_seconds=0.5,
            memory_mb=50.0,
        ),
        PipelineStep(
            name="normalization",
            display_name="Normalization",
            method_name="log_normalize",
            parameters={"base": 2.0, "offset": 1.0},
            runtime_seconds=0.3,
            memory_mb=30.0,
        ),
        PipelineStep(
            name="imputation",
            display_name="Imputation",
            method_name="knn_impute",
            parameters={"n_neighbors": 5},
            runtime_seconds=1.2,
            memory_mb=80.0,
        ),
        PipelineStep(
            name="integration",
            display_name="Batch Integration",
            method_name="combat",
            parameters={},
            runtime_seconds=0.8,
            memory_mb=60.0,
        ),
    ]


@pytest.fixture
def sample_clustering_metrics() -> ClusteringMetrics:
    """Create sample clustering metrics.

    Returns
    -------
    ClusteringMetrics
        Sample clustering metrics with various scores.
    """
    return ClusteringMetrics(
        silhouette_score=0.5,
        davies_bouldin_score=0.8,
        calinski_harabasz_score=120.0,
        ari_score=0.7,
        nmi_score=0.75,
        n_clusters=3,
    )


@pytest.fixture
def sample_intermediate_results() -> list[IntermediateResults]:
    """Create sample intermediate results.

    Returns
    -------
    list[IntermediateResults]
        List of intermediate results after each step.
    """
    return [
        IntermediateResults(
            step_name="qc",
            n_cells=95,
            n_features=480,
            sparsity=0.45,
            total_runtime=0.5,
        ),
        IntermediateResults(
            step_name="normalization",
            n_cells=95,
            n_features=480,
            sparsity=0.45,
            total_runtime=0.8,
        ),
        IntermediateResults(
            step_name="imputation",
            n_cells=95,
            n_features=480,
            sparsity=0.0,
            total_runtime=2.0,
        ),
        IntermediateResults(
            step_name="integration",
            n_cells=95,
            n_features=480,
            sparsity=0.0,
            total_runtime=2.8,
        ),
    ]


@pytest.fixture
def sample_pipeline_result(
    sample_pipeline_steps: list[PipelineStep],
    sample_clustering_metrics: ClusteringMetrics,
    sample_intermediate_results: list[IntermediateResults],
) -> PipelineResult:
    """Create a sample pipeline result.

    Returns
    -------
    PipelineResult
        Sample pipeline result with all fields populated.
    """
    np.random.seed(42)
    return PipelineResult(
        framework="scptensor",
        pipeline_steps=sample_pipeline_steps,
        umap_embedding=np.random.randn(100, 2),
        cluster_labels=np.random.randint(0, 3, 100),
        clustering_metrics=sample_clustering_metrics,
        intermediate_results=sample_intermediate_results,
        total_runtime=2.8,
        total_memory_mb=220.0,
        dataset_name="test_dataset",
    )


# ============================================================================
# Tests for compute_jaccard_index
# ============================================================================


class TestComputeJaccardIndex:
    """Tests for compute_jaccard_index function."""

    def test_identical_labels(self) -> None:
        """Test Jaccard index with identical labelings."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = compute_jaccard_index(labels, labels)
        assert result == 1.0

    def test_completely_different_labels(self) -> None:
        """Test Jaccard index with completely different labelings."""
        labels_a = np.array([0, 0, 1, 1])
        labels_b = np.array([0, 1, 0, 1])
        result = compute_jaccard_index(labels_a, labels_b)
        # Each sample is in a different cluster between the two
        assert 0.0 <= result <= 1.0

    def test_permuted_labels(self) -> None:
        """Test Jaccard index with permuted cluster IDs."""
        labels_a = np.array([0, 0, 1, 1, 2, 2])
        labels_b = np.array([2, 2, 0, 0, 1, 1])
        result = compute_jaccard_index(labels_a, labels_b)
        # Same clusters, different IDs
        assert result == 1.0

    def test_single_element(self) -> None:
        """Test Jaccard index with single element arrays."""
        labels_a = np.array([0])
        labels_b = np.array([0])
        result = compute_jaccard_index(labels_a, labels_b)
        assert result == 1.0

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test Jaccard index raises ValueError for mismatched lengths."""
        labels_a = np.array([0, 0, 1, 1])
        labels_b = np.array([0, 1, 1])

        with pytest.raises(ValueError, match="must have same length"):
            compute_jaccard_index(labels_a, labels_b)

    @pytest.mark.parametrize(
        ("labels_a", "labels_b", "expected_range"),
        [
            (np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 1, 1, 1, 1]), (0.0, 1.0)),
            (np.array([0, 1, 2, 0, 1, 2]), np.array([0, 1, 2, 0, 1, 2]), (1.0, 1.0)),
        ],
    )
    def test_jaccard_index_various_labelings(
        self,
        labels_a: npt.NDArray[np.int_],
        labels_b: npt.NDArray[np.int_],
        expected_range: tuple[float, float],
    ) -> None:
        """Test compute_jaccard_index with various cluster labelings."""
        result = compute_jaccard_index(labels_a, labels_b)
        assert expected_range[0] <= result <= expected_range[1]


# ============================================================================
# Tests for PipelineStep dataclass
# ============================================================================


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_create_pipeline_step(self, sample_pipeline_steps: list[PipelineStep]) -> None:
        """Test creating a PipelineStep with all fields."""
        step = sample_pipeline_steps[0]
        assert step.name == "qc"
        assert step.display_name == "Quality Control"
        assert step.method_name == "basic_qc"
        assert step.parameters == {"min_cells": 3, "min_features": 1}
        assert step.runtime_seconds == 0.5
        assert step.memory_mb == 50.0

    def test_pipeline_step_with_optional_fields(self) -> None:
        """Test PipelineStep with None optional fields."""
        step = PipelineStep(
            name="test",
            display_name="Test Step",
            method_name=None,
            parameters=None,
            runtime_seconds=None,
            memory_mb=None,
        )
        assert step.method_name is None
        assert step.parameters is None
        assert step.runtime_seconds is None
        assert step.memory_mb is None

    def test_dataclass_is_frozen(self, sample_pipeline_steps: list[PipelineStep]) -> None:
        """Test PipelineStep is frozen (immutable)."""
        step = sample_pipeline_steps[0]
        with pytest.raises(Exception):  # FrozenInstanceError
            step.name = "new_name"


# ============================================================================
# Tests for ClusteringMetrics dataclass
# ============================================================================


class TestClusteringMetrics:
    """Tests for ClusteringMetrics dataclass."""

    def test_create_clustering_metrics(self, sample_clustering_metrics: ClusteringMetrics) -> None:
        """Test creating ClusteringMetrics with all fields."""
        assert sample_clustering_metrics.silhouette_score == 0.5
        assert sample_clustering_metrics.davies_bouldin_score == 0.8
        assert sample_clustering_metrics.calinski_harabasz_score == 120.0
        assert sample_clustering_metrics.ari_score == 0.7
        assert sample_clustering_metrics.nmi_score == 0.75
        assert sample_clustering_metrics.n_clusters == 3

    def test_clustering_metrics_with_none_values(self) -> None:
        """Test ClusteringMetrics with None optional values."""
        metrics = ClusteringMetrics(
            silhouette_score=None,
            davies_bouldin_score=None,
            calinski_harabasz_score=None,
            ari_score=None,
            nmi_score=None,
            n_clusters=0,
        )
        assert metrics.silhouette_score is None
        assert metrics.davies_bouldin_score is None
        assert metrics.n_clusters == 0


# ============================================================================
# Tests for IntermediateResults dataclass
# ============================================================================


class TestIntermediateResults:
    """Tests for IntermediateResults dataclass."""

    def test_create_intermediate_results(
        self, sample_intermediate_results: list[IntermediateResults]
    ) -> None:
        """Test creating IntermediateResults with all fields."""
        result = sample_intermediate_results[0]
        assert result.step_name == "qc"
        assert result.n_cells == 95
        assert result.n_features == 480
        assert result.sparsity == 0.45
        assert result.total_runtime == 0.5

    def test_dataclass_is_frozen(
        self, sample_intermediate_results: list[IntermediateResults]
    ) -> None:
        """Test IntermediateResults is frozen (immutable)."""
        result = sample_intermediate_results[0]
        with pytest.raises(Exception):  # FrozenInstanceError
            result.step_name = "new_name"


# ============================================================================
# Tests for PipelineResult dataclass
# ============================================================================


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_create_pipeline_result(self, sample_pipeline_result: PipelineResult) -> None:
        """Test creating PipelineResult with all fields."""
        assert sample_pipeline_result.framework == "scptensor"
        assert len(sample_pipeline_result.pipeline_steps) == 4
        assert sample_pipeline_result.umap_embedding is not None
        assert sample_pipeline_result.cluster_labels is not None
        assert sample_pipeline_result.clustering_metrics is not None
        assert len(sample_pipeline_result.intermediate_results) == 4
        assert sample_pipeline_result.total_runtime == 2.8
        assert sample_pipeline_result.dataset_name == "test_dataset"

    def test_pipeline_result_with_optional_none_fields(self) -> None:
        """Test PipelineResult with None optional fields."""
        result = PipelineResult(
            framework="test",
            pipeline_steps=[],
            umap_embedding=None,
            cluster_labels=None,
            clustering_metrics=None,
            intermediate_results=[],
            total_runtime=0.0,
            total_memory_mb=0.0,
        )
        assert result.umap_embedding is None
        assert result.cluster_labels is None
        assert result.clustering_metrics is None

    def test_umap_embedding_shape(self, sample_pipeline_result: PipelineResult) -> None:
        """Test UMAP embedding has correct shape."""
        assert sample_pipeline_result.umap_embedding is not None
        assert sample_pipeline_result.umap_embedding.shape == (100, 2)

    def test_cluster_labels_shape(self, sample_pipeline_result: PipelineResult) -> None:
        """Test cluster labels match embedding samples."""
        assert sample_pipeline_result.cluster_labels is not None
        assert len(sample_pipeline_result.cluster_labels) == 100

    def test_dataclass_is_frozen(self, sample_pipeline_result: PipelineResult) -> None:
        """Test PipelineResult is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_pipeline_result.framework = "new_framework"


# ============================================================================
# Tests for EndToEndDisplay class (without matplotlib mocking)
# ============================================================================


class TestEndToEndDisplay:
    """Tests for EndToEndDisplay class."""

    def test_initialization(self, temp_output_dir: Path) -> None:
        """Test EndToEndDisplay initialization."""
        display = EndToEndDisplay(output_dir=temp_output_dir)

        assert display.output_dir == temp_output_dir
        assert display._figures_dir == temp_output_dir / "figures" / "end_to_end"
        assert display._figures_dir.exists()

    def test_initialization_with_string_path(self, temp_output_dir: Path) -> None:
        """Test EndToEndDisplay with string output_dir."""
        display = EndToEndDisplay(output_dir=str(temp_output_dir))

        assert display.output_dir == temp_output_dir
        assert isinstance(display.output_dir, Path)

    def test_framework_display_names(self) -> None:
        """Test framework display name mapping."""
        assert EndToEndDisplay.FRAMEWORK_DISPLAY_NAMES["scptensor"] == "ScpTensor"
        assert EndToEndDisplay.FRAMEWORK_DISPLAY_NAMES["scanpy"] == "Scanpy"

    def test_module_colors(self) -> None:
        """Test module colors are accessible via common module."""
        from scptensor.benchmark.display.common import get_module_colors

        colors = get_module_colors("end_to_end")
        assert colors.primary == "#00bcd4"  # Cyan
        assert colors.secondary == "#ff00ff"  # Magenta


# ============================================================================
# Parametrized tests
# ============================================================================


class TestEndToEndParametrized:
    """Parametrized tests for EndToEndDisplay."""

    @pytest.mark.parametrize(
        ("framework", "expected_display_name"),
        [
            ("scptensor", "ScpTensor"),
            ("scanpy", "Scanpy"),
            ("unknown", "Unknown"),
        ],
    )
    def test_framework_display_names_mapping(
        self, framework: str, expected_display_name: str
    ) -> None:
        """Test framework display name mapping."""
        display_name = EndToEndDisplay.FRAMEWORK_DISPLAY_NAMES.get(framework, framework.title())
        assert display_name == expected_display_name

    @pytest.mark.parametrize(
        ("labels_a", "labels_b", "min_jaccard", "max_jaccard"),
        [
            (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 1.0, 1.0),
            (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), 1.0, 1.0),
            (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), 1.0, 1.0),
            (np.array([0, 0, 0, 1]), np.array([0, 1, 0, 1]), 0.0, 1.0),
        ],
    )
    def test_compute_jaccard_index_various_labels(
        self,
        labels_a: npt.NDArray[np.int_],
        labels_b: npt.NDArray[np.int_],
        min_jaccard: float,
        max_jaccard: float,
    ) -> None:
        """Test compute_jaccard_index with various label patterns."""
        result = compute_jaccard_index(labels_a, labels_b)
        assert min_jaccard <= result <= max_jaccard

    @pytest.mark.parametrize(
        ("n_samples", "n_features"),
        [(50, 100), (100, 200), (200, 500)],
    )
    def test_pipeline_result_various_sizes(self, n_samples: int, n_features: int) -> None:
        """Test PipelineResult with various data sizes."""
        np.random.seed(42)
        result = PipelineResult(
            framework="test",
            pipeline_steps=[],
            umap_embedding=np.random.randn(n_samples, 2),
            cluster_labels=np.random.randint(0, 3, n_samples),
            intermediate_results=[
                IntermediateResults(
                    step_name="qc",
                    n_cells=n_samples,
                    n_features=n_features,
                    sparsity=0.3,
                    total_runtime=1.0,
                )
            ],
        )
        assert result.umap_embedding is not None
        assert result.umap_embedding.shape == (n_samples, 2)


# ============================================================================
# Edge case tests
# ============================================================================


class TestEndToEndEdgeCases:
    """Edge case tests for EndToEndDisplay."""

    def test_compute_jaccard_index_empty_arrays(self) -> None:
        """Test compute_jaccard_index with empty arrays."""
        labels_a = np.array([], dtype=int)
        labels_b = np.array([], dtype=int)
        result = compute_jaccard_index(labels_a, labels_b)
        # Empty arrays should return 1.0 (perfect match of nothing)
        assert result == 1.0

    def test_display_with_nested_output_dir(self, temp_output_dir: Path) -> None:
        """Test display with deeply nested output directory."""
        nested_dir = temp_output_dir / "deeply" / "nested" / "path"

        display = EndToEndDisplay(output_dir=nested_dir)

        assert display._figures_dir.exists()
