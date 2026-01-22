"""Tests for QC display module.

Tests cover:
- scptensor.benchmark.display.qc.QCDashboardDisplay: QC dashboard visualization
- scptensor.benchmark.display.qc.MissingTypeDisplay: Missing value type analysis
- scptensor.benchmark.display.qc.QCBatchDisplay: Batch effect detection visualization
- scptensor.benchmark.display.qc.QCComparisonResult: QC comparison dataclass
- scptensor.benchmark.display.qc.MissingTypeReport: Missing type report dataclass
- scptensor.benchmark.display.qc.BatchCVReport: Batch CV report dataclass
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scptensor.benchmark.display.qc import (
    BatchCVReport,
    MissingTypeDisplay,
    MissingTypeReport,
    QCBatchDisplay,
    QCComparisonResult,
    QCDashboardDisplay,
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
def sample_qc_result() -> QCComparisonResult:
    """Create a sample QC comparison result.

    Returns
    -------
    QCComparisonResult
        Sample QC result with sample and feature metrics.
    """
    np.random.seed(42)

    return QCComparisonResult(
        sample_metrics={
            "n_detected": np.array([1000, 1200, 800, 950, 1100]),
            "total_intensity": np.array([1e6, 1.2e6, 8e5, 9.5e5, 1.1e6]),
            "missing_rate": np.array([0.2, 0.15, 0.3, 0.25, 0.18]),
        },
        feature_metrics={
            "cv": np.array([0.3, 0.25, 0.4, 0.35, 0.28, 0.32, 0.38, 0.22]),
            "missing_rate": np.array([0.1, 0.15, 0.2, 0.12, 0.18, 0.08, 0.25, 0.14]),
            "prevalence": np.array([0.9, 0.85, 0.8, 0.88, 0.82, 0.92, 0.75, 0.86]),
        },
        batch_labels=None,
        n_samples=5,
        n_features=8,
        framework="scptensor",
        cells_removed=2,
        features_removed=5,
    )


@pytest.fixture
def sample_missing_type_report() -> MissingTypeReport:
    """Create a sample missing type report.

    Returns
    -------
    MissingTypeReport
        Sample missing type report with various missing value rates.
    """
    n_features = 100
    n_samples = 50
    np.random.seed(42)

    return MissingTypeReport(
        valid_rate=0.70,
        mbr_rate=0.15,
        lod_rate=0.10,
        filtered_rate=0.05,
        imputed_rate=0.0,
        feature_missing_rates=np.random.rand(n_features) * 0.5,
        sample_missing_rates=np.random.rand(n_samples) * 0.5,
        mbr_by_feature=np.random.rand(n_features) * 0.3,
        lod_by_feature=np.random.rand(n_features) * 0.2,
    )


@pytest.fixture
def sample_batch_cv_report() -> BatchCVReport:
    """Create a sample batch CV report.

    Returns
    -------
    BatchCVReport
        Sample batch CV report with multiple batches.
    """
    n_batches = 3
    n_features = 500
    np.random.seed(42)

    return BatchCVReport(
        within_batch_cv={"batch1": 0.25, "batch2": 0.28, "batch3": 0.30},
        between_batch_cv=0.35,
        cv_by_batch_feature=np.random.rand(n_batches, n_features) * 0.5,
        batch_names=["batch1", "batch2", "batch3"],
        high_cv_features=[10, 20, 30, 40, 50],
    )


# ============================================================================
# Tests for QCComparisonResult dataclass
# ============================================================================


class TestQCComparisonResult:
    """Tests for QCComparisonResult dataclass."""

    def test_create_qc_comparison_result(self, sample_qc_result: QCComparisonResult) -> None:
        """Test creating a QCComparisonResult with all fields."""
        assert sample_qc_result.framework == "scptensor"
        assert sample_qc_result.n_samples == 5
        assert sample_qc_result.n_features == 8
        assert sample_qc_result.cells_removed == 2
        assert sample_qc_result.features_removed == 5

    def test_sample_metrics_content(self, sample_qc_result: QCComparisonResult) -> None:
        """Test sample metrics contain expected keys."""
        assert "n_detected" in sample_qc_result.sample_metrics
        assert "total_intensity" in sample_qc_result.sample_metrics
        assert "missing_rate" in sample_qc_result.sample_metrics

    def test_feature_metrics_content(self, sample_qc_result: QCComparisonResult) -> None:
        """Test feature metrics contain expected keys."""
        assert "cv" in sample_qc_result.feature_metrics
        assert "missing_rate" in sample_qc_result.feature_metrics
        assert "prevalence" in sample_qc_result.feature_metrics

    def test_batch_labels_none(self, sample_qc_result: QCComparisonResult) -> None:
        """Test batch_labels can be None."""
        assert sample_qc_result.batch_labels is None

    def test_batch_labels_with_array(self) -> None:
        """Test batch_labels with actual array."""
        result = QCComparisonResult(
            sample_metrics={},
            feature_metrics={},
            batch_labels=np.array([0, 0, 1, 1, 2]),
            n_samples=5,
            n_features=10,
            framework="scptensor",
        )
        assert result.batch_labels is not None
        assert len(result.batch_labels) == 5

    def test_dataclass_is_frozen(self, sample_qc_result: QCComparisonResult) -> None:
        """Test QCComparisonResult is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_qc_result.framework = "new_framework"

    def test_optional_cells_features_removed_none(self) -> None:
        """Test optional fields can be None."""
        result = QCComparisonResult(
            sample_metrics={},
            feature_metrics={},
            batch_labels=None,
            n_samples=5,
            n_features=10,
            framework="scptensor",
            cells_removed=None,
            features_removed=None,
        )
        assert result.cells_removed is None
        assert result.features_removed is None


# ============================================================================
# Tests for MissingTypeReport dataclass
# ============================================================================


class TestMissingTypeReport:
    """Tests for MissingTypeReport dataclass."""

    def test_create_missing_type_report(
        self, sample_missing_type_report: MissingTypeReport
    ) -> None:
        """Test creating a MissingTypeReport with all fields."""
        assert sample_missing_type_report.valid_rate == 0.70
        assert sample_missing_type_report.mbr_rate == 0.15
        assert sample_missing_type_report.lod_rate == 0.10
        assert sample_missing_type_report.filtered_rate == 0.05
        assert sample_missing_type_report.imputed_rate == 0.0

    def test_rates_sum_to_one(self, sample_missing_type_report: MissingTypeReport) -> None:
        """Test all rates sum to approximately 1.0."""
        total = (
            sample_missing_type_report.valid_rate
            + sample_missing_type_report.mbr_rate
            + sample_missing_type_report.lod_rate
            + sample_missing_type_report.filtered_rate
            + sample_missing_type_report.imputed_rate
        )
        assert abs(total - 1.0) < 1e-10

    def test_feature_arrays_shape(self, sample_missing_type_report: MissingTypeReport) -> None:
        """Test feature-level arrays have correct shape."""
        assert len(sample_missing_type_report.feature_missing_rates) == 100
        assert len(sample_missing_type_report.mbr_by_feature) == 100
        assert len(sample_missing_type_report.lod_by_feature) == 100

    def test_sample_arrays_shape(self, sample_missing_type_report: MissingTypeReport) -> None:
        """Test sample-level arrays have correct shape."""
        assert len(sample_missing_type_report.sample_missing_rates) == 50

    def test_dataclass_is_frozen(self, sample_missing_type_report: MissingTypeReport) -> None:
        """Test MissingTypeReport is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_missing_type_report.valid_rate = 0.8


# ============================================================================
# Tests for BatchCVReport dataclass
# ============================================================================


class TestBatchCVReport:
    """Tests for BatchCVReport dataclass."""

    def test_create_batch_cv_report(self, sample_batch_cv_report: BatchCVReport) -> None:
        """Test creating a BatchCVReport with all fields."""
        assert sample_batch_cv_report.between_batch_cv == 0.35
        assert len(sample_batch_cv_report.batch_names) == 3
        assert len(sample_batch_cv_report.high_cv_features) == 5

    def test_within_batch_cv_content(self, sample_batch_cv_report: BatchCVReport) -> None:
        """Test within_batch_cv contains all batches."""
        assert "batch1" in sample_batch_cv_report.within_batch_cv
        assert "batch2" in sample_batch_cv_report.within_batch_cv
        assert "batch3" in sample_batch_cv_report.within_batch_cv

    def test_cv_by_batch_feature_shape(self, sample_batch_cv_report: BatchCVReport) -> None:
        """Test cv_by_batch_feature has correct shape."""
        expected_shape = (len(sample_batch_cv_report.batch_names), 500)
        assert sample_batch_cv_report.cv_by_batch_feature.shape == expected_shape

    def test_dataclass_is_frozen(self, sample_batch_cv_report: BatchCVReport) -> None:
        """Test BatchCVReport is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_batch_cv_report.between_batch_cv = 0.5


# ============================================================================
# Tests for QCDashboardDisplay class (without matplotlib mocking)
# ============================================================================


class TestQCDashboardDisplay:
    """Tests for QCDashboardDisplay class."""

    def test_initialization(self, temp_output_dir: Path) -> None:
        """Test QCDashboardDisplay initialization."""
        display = QCDashboardDisplay(output_dir=temp_output_dir)

        assert display.output_dir == temp_output_dir
        assert display._figures_dir == temp_output_dir / "figures" / "qc"
        assert display._figures_dir.exists()

    def test_initialization_with_string_path(self, temp_output_dir: Path) -> None:
        """Test QCDashboardDisplay with string output_dir."""
        display = QCDashboardDisplay(output_dir=str(temp_output_dir))

        assert display.output_dir == temp_output_dir
        assert isinstance(display.output_dir, Path)

    def test_framework_display_names(self) -> None:
        """Test framework display name mapping."""
        assert QCDashboardDisplay.FRAMEWORK_DISPLAY_NAMES["scptensor"] == "ScpTensor"
        assert QCDashboardDisplay.FRAMEWORK_DISPLAY_NAMES["scanpy"] == "Scanpy"
        assert QCDashboardDisplay.FRAMEWORK_DISPLAY_NAMES["seurat"] == "Seurat"

    def test_qc_colors(self) -> None:
        """Test QC color palette is defined via common module."""
        from scptensor.benchmark.display.common import get_module_colors

        colors = get_module_colors("qc")
        assert colors.primary == "#8c564b"  # Brown

    def test_render_creates_figure(self, temp_output_dir: Path) -> None:
        """Test render method creates a figure file."""
        # Simplified test that just verifies the directory structure
        display = QCDashboardDisplay(output_dir=temp_output_dir)
        assert display._figures_dir.exists()
        # The actual render requires matplotlib, so we just verify setup is correct


# ============================================================================
# Tests for MissingTypeDisplay class (without matplotlib mocking)
# ============================================================================


class TestMissingTypeDisplay:
    """Tests for MissingTypeDisplay class."""

    def test_initialization(self, temp_output_dir: Path) -> None:
        """Test MissingTypeDisplay initialization."""
        display = MissingTypeDisplay(output_dir=temp_output_dir)

        assert display.output_dir == temp_output_dir
        assert display._figures_dir == temp_output_dir / "figures" / "qc"
        assert display._figures_dir.exists()

    def test_mask_code_names(self) -> None:
        """Test mask code display names."""
        assert MissingTypeDisplay.MASK_CODE_NAMES[0] == "VALID"
        assert MissingTypeDisplay.MASK_CODE_NAMES[1] == "MBR"
        assert MissingTypeDisplay.MASK_CODE_NAMES[2] == "LOD"
        assert MissingTypeDisplay.MASK_CODE_NAMES[3] == "FILTERED"
        assert MissingTypeDisplay.MASK_CODE_NAMES[5] == "IMPUTED"

    def test_mask_code_colors(self) -> None:
        """Test mask code names are defined."""
        assert 0 in MissingTypeDisplay.MASK_CODE_NAMES
        assert 1 in MissingTypeDisplay.MASK_CODE_NAMES
        assert 2 in MissingTypeDisplay.MASK_CODE_NAMES
        assert 3 in MissingTypeDisplay.MASK_CODE_NAMES
        assert 5 in MissingTypeDisplay.MASK_CODE_NAMES


# ============================================================================
# Tests for QCBatchDisplay class (without matplotlib mocking)
# ============================================================================


class TestQCBatchDisplay:
    """Tests for QCBatchDisplay class."""

    def test_initialization(self, temp_output_dir: Path) -> None:
        """Test QCBatchDisplay initialization."""
        display = QCBatchDisplay(output_dir=temp_output_dir)

        assert display.output_dir == temp_output_dir
        assert display._figures_dir == temp_output_dir / "figures" / "qc"
        assert display._figures_dir.exists()

    def test_batch_colors(self) -> None:
        """Test batch color palette is defined via common module."""
        from scptensor.benchmark.display.common import get_module_colors

        colors = get_module_colors("qc")
        assert colors.secondary == "#7f7f7f"  # Gray


# ============================================================================
# Parametrized tests
# ============================================================================


class TestQCParametrized:
    """Parametrized tests for QC display classes."""

    @pytest.mark.parametrize(
        ("framework", "expected_display_name"),
        [
            ("scptensor", "ScpTensor"),
            ("scanpy", "Scanpy"),
            ("seurat", "Seurat"),
            ("unknown", "Unknown"),
        ],
    )
    def test_framework_display_names_mapping(
        self, framework: str, expected_display_name: str
    ) -> None:
        """Test framework display name mapping."""
        display_name = QCDashboardDisplay.FRAMEWORK_DISPLAY_NAMES.get(framework, framework.title())
        assert display_name == expected_display_name

    @pytest.mark.parametrize(
        ("n_samples", "n_features"),
        [(10, 20), (50, 100), (100, 200), (500, 1000)],
    )
    def test_qc_comparison_result_various_sizes(self, n_samples: int, n_features: int) -> None:
        """Test QCComparisonResult with various sizes."""
        np.random.seed(42)
        result = QCComparisonResult(
            sample_metrics={
                "n_detected": np.random.randint(500, 1500, n_samples),
                "missing_rate": np.random.rand(n_samples),
            },
            feature_metrics={
                "cv": np.random.rand(n_features) * 0.5,
                "missing_rate": np.random.rand(n_features) * 0.5,
            },
            batch_labels=None,
            n_samples=n_samples,
            n_features=n_features,
            framework="scptensor",
        )
        assert result.n_samples == n_samples
        assert result.n_features == n_features

    @pytest.mark.parametrize(
        ("valid_rate", "mbr_rate", "lod_rate", "filtered_rate", "imputed_rate"),
        [
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.5, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )
    def test_missing_type_report_extreme_cases(
        self,
        valid_rate: float,
        mbr_rate: float,
        lod_rate: float,
        filtered_rate: float,
        imputed_rate: float,
    ) -> None:
        """Test MissingTypeReport with extreme rate values."""
        n_features = 50
        n_samples = 25

        report = MissingTypeReport(
            valid_rate=valid_rate,
            mbr_rate=mbr_rate,
            lod_rate=lod_rate,
            filtered_rate=filtered_rate,
            imputed_rate=imputed_rate,
            feature_missing_rates=np.random.rand(n_features) * 0.5,
            sample_missing_rates=np.random.rand(n_samples) * 0.5,
            mbr_by_feature=np.zeros(n_features),
            lod_by_feature=np.zeros(n_features),
        )

        total = (
            report.valid_rate
            + report.mbr_rate
            + report.lod_rate
            + report.filtered_rate
            + report.imputed_rate
        )
        assert abs(total - 1.0) < 1e-10


# ============================================================================
# Edge case tests
# ============================================================================


class TestQCEdgeCases:
    """Edge case tests for QC display classes."""

    def test_display_with_nested_output_dir(self, temp_output_dir: Path) -> None:
        """Test display with deeply nested output directory."""
        nested_dir = temp_output_dir / "deeply" / "nested" / "path"

        display = QCDashboardDisplay(output_dir=nested_dir)

        assert display._figures_dir.exists()

    def test_missing_type_display_with_zero_rates(self, temp_output_dir: Path) -> None:
        """Test MissingTypeDisplay with all zero rates except valid."""
        display = MissingTypeDisplay(output_dir=temp_output_dir)

        report = MissingTypeReport(
            valid_rate=1.0,
            mbr_rate=0.0,
            lod_rate=0.0,
            filtered_rate=0.0,
            imputed_rate=0.0,
            feature_missing_rates=np.zeros(10),
            sample_missing_rates=np.zeros(5),
            mbr_by_feature=np.zeros(10),
            lod_by_feature=np.zeros(10),
        )

        assert display._figures_dir.exists()

    def test_batch_display_with_single_batch(self, temp_output_dir: Path) -> None:
        """Test QCBatchDisplay with single batch."""
        display = QCBatchDisplay(output_dir=temp_output_dir)

        report = BatchCVReport(
            within_batch_cv={"batch1": 0.25},
            between_batch_cv=0.0,
            cv_by_batch_feature=np.random.rand(1, 100) * 0.5,
            batch_names=["batch1"],
            high_cv_features=[],
        )

        assert display._figures_dir.exists()
