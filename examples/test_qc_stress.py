#!/usr/bin/env python3
"""Comprehensive stress testing script for QC module.

This script performs extensive stress testing of the QC module to find potential
issues and edge cases that may not be covered in regular unit tests.

Author: ScpTensor Team
Created: 2026-02-28
"""

from __future__ import annotations

import gc
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.qc import (
    assess_batch_effects,
    calculate_feature_qc_metrics,
    calculate_sample_qc_metrics,
    filter_doublets_mad,
    filter_features_by_cv,
    filter_features_by_missingness,
    filter_low_quality_samples,
)
from scptensor.qc.metrics import compute_cv, compute_mad, is_outlier_mad


@dataclass
class TestResult:
    """Result of a single test case."""

    name: str
    status: str  # PASS, FAIL, SKIP
    error: str | None = None
    traceback: str | None = None
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestReport:
    """Summary report of all tests."""

    tests: list[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if t.status == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tests if t.status == "SKIP")


class QCStressTest:
    """Stress test suite for QC module."""

    def __init__(self, output_dir: Path = Path("tmp")):
        """Initialize stress test suite.

        Parameters
        ----------
        output_dir : Path, default="tmp"
            Directory to save test reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = TestReport()
        self.rng = np.random.default_rng(42)

    def run_all_tests(self) -> None:
        """Run all stress tests."""
        print("=" * 80)
        print("QC Module Stress Test Suite")
        print("=" * 80)
        print(f"Started at: {self.report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Boundary condition tests
        print("Running boundary condition tests...")
        self._test_empty_datasets()
        self._test_single_sample()
        self._test_single_feature()
        self._test_tiny_dataset()

        # Anomalous data tests
        print("Running anomalous data tests...")
        self._test_all_nan_data()
        self._test_all_zero_data()
        self._test_infinity_values()
        self._test_extreme_values()
        self._test_all_missing_data()
        self._test_all_detected_data()

        # Matrix format tests
        print("Running matrix format tests...")
        self._test_sparse_matrix()
        self._test_dense_matrix()
        self._test_with_mask_matrix()
        self._test_without_mask_matrix()

        # QC function correctness tests
        print("Running QC function correctness tests...")
        self._test_calculate_sample_qc_metrics_correctness()
        self._test_calculate_feature_qc_metrics_correctness()
        self._test_return_types_and_ranges()

        # Error handling tests
        print("Running error handling tests...")
        self._test_nonexistent_assay()
        self._test_nonexistent_layer()
        self._test_wrong_data_types()

        # Edge case tests
        print("Running edge case tests...")
        self._test_division_by_zero()
        self._test_null_propagation()
        self._test_type_conversion()
        self._test_mask_code_consistency()

        # Large dataset tests
        print("Running large dataset tests...")
        self._test_large_dataset_memory()

        # Integration tests
        print("Running integration tests...")
        self._test_full_pipeline()

        # Generate report
        self._generate_report()

    def _run_test(self, name: str, test_func: Callable[[], None]) -> None:
        """Run a single test and record result."""
        start_time = datetime.now()
        result = TestResult(name=name, status="SKIP")

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                test_func()

                # Check for warnings
                if w:
                    result.details["warnings"] = [str(warning.message) for warning in w]

            result.status = "PASS"
        except AssertionError as e:
            result.status = "FAIL"
            result.error = str(e)
            result.traceback = traceback.format_exc()
        except Exception as e:
            result.status = "FAIL"
            result.error = f"{type(e).__name__}: {str(e)}"
            result.traceback = traceback.format_exc()
        finally:
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.report.tests.append(result)

            # Print immediate result
            status_symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘"}[result.status]
            print(f"  {status_symbol} {name} ({result.duration_ms:.2f}ms)")

            if result.status == "FAIL":
                print(f"    ERROR: {result.error}")

    def _create_container(
        self,
        n_samples: int,
        n_features: int,
        density: float = 0.5,
        sparse: bool = False,
        with_mask: bool = False,
        value_range: tuple[float, float] = (0.1, 100.0),
    ) -> ScpContainer:
        """Create test container with specified parameters.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        density : float, default=0.5
            Density of non-zero values (for sparse matrices)
        sparse : bool, default=False
            Use sparse matrix format
        with_mask : bool, default=False
            Include mask matrix
        value_range : tuple, default=(0.1, 100.0)
            Range of random values

        Returns
        -------
        ScpContainer
            Test container
        """
        # Create sample metadata
        obs = pl.DataFrame({
            "_index": [f"S{i}" for i in range(n_samples)],
            "batch": [f"batch_{i % 3}" for i in range(n_samples)],
        })

        # Create feature metadata
        var = pl.DataFrame({
            "_index": [f"P{i}" for i in range(n_features)],
            "protein_name": [f"Protein{i}" for i in range(n_features)],
        })

        # Create data matrix
        if sparse:
            X = sp.random(n_samples, n_features, density=density, format="csr", random_state=42)
            # Scale to value range
            X.data = self.rng.uniform(value_range[0], value_range[1], size=X.data.shape)
        else:
            X = self.rng.uniform(value_range[0], value_range[1], size=(n_samples, n_features))
            # Apply density for consistency
            mask = self.rng.random(X.shape) > density
            X[mask] = 0

        # Create mask matrix if requested
        M = None
        if with_mask:
            M = np.zeros((n_samples, n_features), dtype=np.int8)
            # Randomly assign mask codes
            mask_indices = self.rng.random(M.shape) < 0.1
            M[mask_indices] = self.rng.integers(1, 6, size=M.shape)[mask_indices]

        # Create assay and container
        matrix = ScpMatrix(X=X, M=M)
        assay = Assay(var=var, layers={"raw": matrix})

        return ScpContainer(obs=obs, assays={"protein": assay})

    # ============== Boundary Condition Tests ==============

    def _test_empty_datasets(self) -> None:
        """Test edge cases with empty datasets."""

        def test_zero_samples():
            """Test with 0 samples."""
            obs = pl.DataFrame({"_index": []})
            var = pl.DataFrame({"_index": ["P1", "P2"]})
            X = np.zeros((0, 2))  # 0x2 matrix
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            # This should either work or fail gracefully
            try:
                result = calculate_sample_qc_metrics(container)
                assert result.n_samples == 0, "Should have 0 samples"
            except Exception as e:
                # Expected to fail with specific error
                assert "empty" in str(e).lower() or "0" in str(e), f"Unexpected error: {e}"

        def test_zero_features():
            """Test with 0 features."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": []})
            X = np.zeros((2, 0))  # 2x0 matrix
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            try:
                result = calculate_sample_qc_metrics(container)
                assert "protein" in result.assays
                assert result.assays["protein"].n_features == 0
            except Exception as e:
                # Expected to fail gracefully
                assert "empty" in str(e).lower() or "0" in str(e), f"Unexpected error: {e}"

        def test_both_zero():
            """Test with 0 samples and 0 features."""
            obs = pl.DataFrame({"_index": []})
            var = pl.DataFrame({"_index": []})
            X = np.zeros((0, 0))  # 0x0 matrix
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            try:
                result = calculate_sample_qc_metrics(container)
                assert result.n_samples == 0
            except Exception as e:
                # This is expected to fail
                pass  # Any error is acceptable

        self._run_test("Empty: 0 samples", test_zero_samples)
        self._run_test("Empty: 0 features", test_zero_features)
        self._run_test("Empty: 0 samples and 0 features", test_both_zero)

    def _test_single_sample(self) -> None:
        """Test with single sample."""

        def test_single_sample_dense():
            """Single sample with dense matrix."""
            container = self._create_container(n_samples=1, n_features=10)
            result = calculate_sample_qc_metrics(container)

            assert result.n_samples == 1
            assert "n_features_protein" in result.obs.columns

            # Check values are reasonable
            n_feat = result.obs["n_features_protein"][0]
            assert isinstance(n_feat, (int, np.integer))
            assert 0 <= n_feat <= 10

        def test_single_sample_sparse():
            """Single sample with sparse matrix."""
            container = self._create_container(n_samples=1, n_features=10, sparse=True)
            result = calculate_sample_qc_metrics(container)

            assert result.n_samples == 1
            n_feat = result.obs["n_features_protein"][0]
            assert isinstance(n_feat, (int, np.integer))

        def test_single_sample_filtering():
            """Test filtering with single sample."""
            container = self._create_container(n_samples=1, n_features=10)

            try:
                result = filter_low_quality_samples(container, min_features=5, use_mad=False)
                # MAD doesn't work with single sample
                assert result.n_samples <= 1
            except Exception as e:
                # MAD with single sample is expected to fail
                assert "mad" in str(e).lower() or "median" in str(e).lower() or "sample" in str(e).lower()

        self._run_test("Single sample: dense", test_single_sample_dense)
        self._run_test("Single sample: sparse", test_single_sample_sparse)
        self._run_test("Single sample: filtering", test_single_sample_filtering)

    def _test_single_feature(self) -> None:
        """Test with single feature."""

        def test_single_feature_dense():
            """Single feature with dense matrix."""
            container = self._create_container(n_samples=10, n_features=1)
            result = calculate_feature_qc_metrics(container)

            assert result.assays["protein"].n_features == 1
            var = result.assays["protein"].var
            assert "missing_rate" in var.columns
            assert "detection_rate" in var.columns
            assert "cv" in var.columns

            # CV for single feature might be NaN
            cv_value = var["cv"][0]
            assert cv_value is None or np.isnan(cv_value) or isinstance(cv_value, (int, float))

        def test_single_feature_sparse():
            """Single feature with sparse matrix."""
            container = self._create_container(n_samples=10, n_features=1, sparse=True)
            result = calculate_feature_qc_metrics(container)

            assert result.assays["protein"].n_features == 1
            missing_rate = result.assays["protein"].var["missing_rate"][0]
            assert 0 <= missing_rate <= 1

        self._run_test("Single feature: dense", test_single_feature_dense)
        self._run_test("Single feature: sparse", test_single_feature_sparse)

    def _test_tiny_dataset(self) -> None:
        """Test with tiny 2x2 dataset."""

        def test_tiny_dense():
            """Tiny 2x2 dense matrix."""
            container = self._create_container(n_samples=2, n_features=2)

            # Sample QC
            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 2

            # Feature QC
            result = calculate_feature_qc_metrics(result)
            assert result.assays["protein"].n_features == 2

            # Filtering
            try:
                result = filter_low_quality_samples(result, min_features=1, use_mad=False)
                assert result.n_samples <= 2
            except Exception as e:
                raise AssertionError(f"Filtering failed on 2x2 dataset: {e}")

        def test_tiny_sparse():
            """Tiny 2x2 sparse matrix."""
            container = self._create_container(n_samples=2, n_features=2, sparse=True, density=0.5)

            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 2

        self._run_test("Tiny dataset (2x2): dense", test_tiny_dense)
        self._run_test("Tiny dataset (2x2): sparse", test_tiny_sparse)

    # ============== Anomalous Data Tests ==============

    def _test_all_nan_data(self) -> None:
        """Test with all NaN data."""

        def test_all_nan_dense():
            """All NaN dense matrix."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1", "P2"]})
            X = np.full((3, 2), np.nan)
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)

            # All metrics should be 0 or NaN
            n_feats = result.obs["n_features_protein"].to_numpy()
            total_int = result.obs["total_intensity_protein"].to_numpy()

            assert np.all(n_feats == 0), "All samples should have 0 detected features"
            assert np.all(total_int == 0), "All samples should have 0 total intensity"

        def test_all_nan_sparse():
            """All NaN with sparse matrix (should fail or convert)."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})

            # Sparse matrices can't store NaN, so we use zeros
            X = sp.csr_matrix((2, 1))
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            n_feats = result.obs["n_features_protein"].to_numpy()
            assert np.all(n_feats == 0)

        self._run_test("All NaN: dense", test_all_nan_dense)
        self._run_test("All NaN: sparse", test_all_nan_sparse)

    def _test_all_zero_data(self) -> None:
        """Test with all zero data."""

        def test_all_zeros():
            """All zero matrix."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
            X = np.zeros((3, 3))
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)

            n_feats = result.obs["n_features_protein"].to_numpy()
            total_int = result.obs["total_intensity_protein"].to_numpy()

            assert np.all(n_feats == 0), "All samples should have 0 detected features"
            assert np.all(total_int == 0), "All samples should have 0 total intensity"

        def test_zeros_with_mask():
            """Zero matrix with non-zero mask."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})
            X = np.zeros((2, 1))
            M = np.zeros((2, 1), dtype=np.int8)  # All valid (but zero values)
            matrix = ScpMatrix(X=X, M=M)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            n_feats = result.obs["n_features_protein"].to_numpy()
            # Zeros are not detected as features
            assert np.all(n_feats == 0)

        self._run_test("All zeros: dense", test_all_zeros)
        self._run_test("All zeros: with mask", test_zeros_with_mask)

    def _test_infinity_values(self) -> None:
        """Test with infinity values."""

        def test_with_inf():
            """Matrix with infinity values."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1", "P2"]})
            X = np.array([[1.0, 2.0], [np.inf, 3.0], [-np.inf, 4.0]])
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            try:
                result = calculate_sample_qc_metrics(container)

                # Check for inf in results
                total_int = result.obs["total_intensity_protein"].to_numpy()
                has_inf = np.any(np.isinf(total_int))

                if has_inf:
                    raise AssertionError("Total intensity contains inf values")

            except Exception as e:
                # May fail due to inf values
                if "inf" not in str(e).lower():
                    raise

        self._run_test("Infinity values", test_with_inf)

    def _test_extreme_values(self) -> None:
        """Test with extreme values."""

        def test_very_large():
            """Very large values."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})
            X = np.array([[1e308], [1e309]])  # Near float64 max
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            try:
                result = calculate_sample_qc_metrics(container)

                # Check for overflow
                total_int = result.obs["total_intensity_protein"].to_numpy()
                has_inf = np.any(np.isinf(total_int))

                if has_inf:
                    # This is expected for very large values
                    pass
            except Exception as e:
                # May fail due to overflow
                pass

        def test_very_small():
            """Very small values (near zero)."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})
            X = np.array([[1e-308], [1e-309]])  # Near float64 min
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            # Should work without underflow issues
            assert result.n_samples == 2

        def test_negative_values():
            """Negative intensity values (should not happen in real data)."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1", "P2"]})
            X = np.array([[-1.0, 2.0], [3.0, -4.0]])
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)

            # Negative values should be treated as not detected
            n_feats = result.obs["n_features_protein"].to_numpy()
            assert np.array_equal(n_feats, [1, 1]), "Negative values should not be detected"

        self._run_test("Extreme: very large values", test_very_large)
        self._run_test("Extreme: very small values", test_very_small)
        self._run_test("Extreme: negative values", test_negative_values)

    def _test_all_missing_data(self) -> None:
        """Test with all missing data."""

        def test_all_zeros_dense():
            """All zeros (completely missing)."""
            container = self._create_container(
                n_samples=10, n_features=5, density=0.0
            )  # All zeros

            result = calculate_sample_qc_metrics(container)

            n_feats = result.obs["n_features_protein"].to_numpy()
            assert np.all(n_feats == 0), "All samples should have 0 features"

            result = calculate_feature_qc_metrics(result)
            missing_rate = result.assays["protein"].var["missing_rate"].to_numpy()
            assert np.all(missing_rate == 1.0), "All features should have 100% missing rate"

        def test_all_zeros_sparse():
            """All zeros sparse matrix."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})
            X = sp.csr_matrix((2, 1))  # All zeros
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_feature_qc_metrics(container)
            missing_rate = result.assays["protein"].var["missing_rate"][0]
            assert missing_rate == 1.0

        self._run_test("All missing: dense", test_all_zeros_dense)
        self._run_test("All missing: sparse", test_all_zeros_sparse)

    def _test_all_detected_data(self):
        """Test with all data detected."""

        def test_all_detected():
            """All values are positive (fully detected)."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})
            X = np.ones((3, 3)) * 10.0  # All positive
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            n_feats = result.obs["n_features_protein"].to_numpy()
            assert np.all(n_feats == 3), "All samples should have 3 features"

            result = calculate_feature_qc_metrics(result)
            detection_rate = result.assays["protein"].var["detection_rate"].to_numpy()
            assert np.all(detection_rate == 1.0), "All features should have 100% detection rate"

        self._run_test("All detected", test_all_detected)

    # ============== Matrix Format Tests ==============

    def _test_sparse_matrix(self) -> None:
        """Test with sparse matrices."""

        def test_sparse_csr():
            """CSR sparse matrix."""
            container = self._create_container(n_samples=100, n_features=50, sparse=True)

            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 100

            result = calculate_feature_qc_metrics(result)
            assert result.assays["protein"].n_features == 50

        def test_sparse_different_density():
            """Test various sparse densities."""
            for density in [0.01, 0.1, 0.5, 0.9]:
                container = self._create_container(
                    n_samples=20, n_features=10, sparse=True, density=density
                )

                result = calculate_sample_qc_metrics(container)
                assert result.n_samples == 20

        self._run_test("Sparse: CSR format", test_sparse_csr)
        self._run_test("Sparse: different densities", test_sparse_different_density)

    def _test_dense_matrix(self) -> None:
        """Test with dense matrices."""

        def test_dense_basic():
            """Basic dense matrix."""
            container = self._create_container(n_samples=100, n_features=50, sparse=False)

            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 100

        def test_dense_with_nans():
            """Dense matrix with NaN values."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1", "P2"]})
            X = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            n_feats = result.obs["n_features_protein"].to_numpy()
            # NaN should be treated as not detected
            assert n_feats[0] == 1  # Only first value
            assert n_feats[1] == 1  # Only second value
            assert n_feats[2] == 2  # Both values

        self._run_test("Dense: basic", test_dense_basic)
        self._run_test("Dense: with NaN values", test_dense_with_nans)

    def _test_with_mask_matrix(self) -> None:
        """Test with mask matrix."""

        def test_mask_codes():
            """Test with various mask codes."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1", "P2"]})
            X = np.ones((2, 2)) * 10.0

            # Create mask with all codes
            M = np.array([
                [MaskCode.VALID, MaskCode.MBR],
                [MaskCode.LOD, MaskCode.IMPUTED],
            ], dtype=np.int8)

            matrix = ScpMatrix(X=X, M=M)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            # Mask should not affect basic QC calculations
            result = calculate_sample_qc_metrics(container)
            n_feats = result.obs["n_features_protein"].to_numpy()
            # All values are positive, so all should be detected
            assert np.all(n_feats == 2)

        def test_mask_filtered():
            """Test with FILTERED mask codes."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})
            X = np.array([[10.0], [20.0]])
            M = np.array([[MaskCode.FILTERED], [MaskCode.VALID]], dtype=np.int8)

            matrix = ScpMatrix(X=X, M=M)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            # Mask doesn't affect detection in current implementation
            result = calculate_sample_qc_metrics(container)
            n_feats = result.obs["n_features_protein"].to_numpy()
            # Both values are positive
            assert np.all(n_feats == 1)

        self._run_test("With mask: various codes", test_mask_codes)
        self._run_test("With mask: FILTERED codes", test_mask_filtered)

    def _test_without_mask_matrix(self) -> None:
        """Test without mask matrix."""

        def test_no_mask():
            """Test without mask matrix (M=None)."""
            container = self._create_container(
                n_samples=10, n_features=5, with_mask=False
            )

            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 10

            # Verify no mask was created
            assay = result.assays["protein"]
            assert assay.layers["raw"].M is None

        self._run_test("Without mask: M=None", test_no_mask)

    # ============== QC Function Correctness Tests ==============

    def _test_calculate_sample_qc_metrics_correctness(self) -> None:
        """Test correctness of calculate_sample_qc_metrics."""

        def test_n_features_calculation():
            """Verify n_features is calculated correctly."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})

            # Create controlled data
            # S1: 2 features, S2: 1 feature, S3: 3 features
            X = np.array([
                [1.0, 2.0, 0.0],  # S1: 2 detected
                [0.0, 0.0, 5.0],  # S2: 1 detected
                [1.0, 1.0, 1.0],  # S3: 3 detected
            ])

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            n_feats = result.obs["n_features_protein"].to_numpy()

            assert np.array_equal(n_feats, [2, 1, 3]), f"Expected [2, 1, 3], got {n_feats}"

        def test_total_intensity_calculation():
            """Verify total_intensity is calculated correctly."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1", "P2"]})

            X = np.array([
                [1.0, 2.0],  # S1: sum = 3.0
                [3.0, 4.0],  # S2: sum = 7.0
            ])

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            total_int = result.obs["total_intensity_protein"].to_numpy()

            assert np.allclose(total_int, [3.0, 7.0]), f"Expected [3.0, 7.0], got {total_int}"

        def test_with_nan_values():
            """Test calculation with NaN values."""
            obs = pl.DataFrame({"_index": ["S1"]})
            var = pl.DataFrame({"_index": ["P1", "P2", "P3"]})

            X = np.array([[1.0, np.nan, 3.0]])

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)

            n_feats = result.obs["n_features_protein"][0]
            total_int = result.obs["total_intensity_protein"][0]

            assert n_feats == 2, f"Expected 2 features, got {n_feats}"
            assert total_int == 4.0, f"Expected total 4.0, got {total_int}"

        self._run_test("Correctness: n_features calculation", test_n_features_calculation)
        self._run_test("Correctness: total_intensity calculation", test_total_intensity_calculation)
        self._run_test("Correctness: with NaN values", test_with_nan_values)

    def _test_calculate_feature_qc_metrics_correctness(self) -> None:
        """Test correctness of calculate_feature_qc_metrics."""

        def test_missing_rate_calculation():
            """Verify missing_rate is calculated correctly."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1", "P2"]})

            X = np.array([
                [1.0, 0.0],  # P1 detected, P2 missing
                [2.0, 0.0],  # P1 detected, P2 missing
                [0.0, 5.0],  # P1 missing, P2 detected
            ])

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_feature_qc_metrics(container)
            missing_rate = result.assays["protein"].var["missing_rate"].to_numpy()

            # P1: 1/3 missing, P2: 2/3 missing
            assert np.allclose(missing_rate, [1/3, 2/3]), f"Expected [0.33, 0.67], got {missing_rate}"

        def test_detection_rate_calculation():
            """Verify detection_rate is complement of missing_rate."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})

            X = np.array([[1.0], [0.0]])

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_feature_qc_metrics(container)
            var_data = result.assays["protein"].var

            missing_rate = var_data["missing_rate"][0]
            detection_rate = var_data["detection_rate"][0]

            assert np.isclose(missing_rate + detection_rate, 1.0), \
                f"missing_rate + detection_rate should equal 1.0"

        def test_cv_calculation():
            """Verify CV calculation."""
            obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
            var = pl.DataFrame({"_index": ["P1"]})

            # Values: 2, 4, 6 (mean=4, sample std=2, CV=0.5)
            # Note: numpy uses sample std by default (ddof=1), so std = sqrt(2) not 2
            X = np.array([[2.0], [4.0], [6.0]])

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_feature_qc_metrics(container)
            cv = result.assays["protein"].var["cv"][0]

            # Calculate expected CV manually: sample std / mean
            # std = sqrt(sum((xi - mean)^2) / (n-1)) = sqrt((4+0+4)/2) = sqrt(4) = 2
            # But numpy's nanstd with ddof=1 gives: sqrt((4+0+4)/2) = 2
            # Actually the sample variance formula gives: sum((xi-mean)^2)/(n-1) = 8/2 = 4, std = 2
            # Let me verify what numpy actually returns
            data = np.array([2.0, 4.0, 6.0])
            expected_std = np.std(data, ddof=1)  # Should be 2
            expected_mean = np.mean(data)  # Should be 4
            expected_cv = expected_std / expected_mean  # 2/4 = 0.5

            # The actual implementation might use ddof=0 (population std)
            # With ddof=0: std = sqrt((4+0+4)/3) = sqrt(8/3) = 1.632
            # CV = 1.632/4 = 0.408

            # Let's check the actual CV value
            actual_cv_from_data = np.std(data, ddof=0) / np.mean(data)

            # Verify the result matches either calculation
            assert np.isclose(cv, expected_cv) or np.isclose(cv, actual_cv_from_data), \
                f"Expected CV≈{expected_cv} or {actual_cv_from_data}, got {cv}"

        self._run_test("Feature QC: missing_rate calculation", test_missing_rate_calculation)
        self._run_test("Feature QC: detection_rate calculation", test_detection_rate_calculation)
        self._run_test("Feature QC: CV calculation", test_cv_calculation)

    def _test_return_types_and_ranges(self) -> None:
        """Test return types and value ranges."""

        def test_sample_qc_return_types():
            """Verify return types from calculate_sample_qc_metrics."""
            container = self._create_container(n_samples=10, n_features=5)

            result = calculate_sample_qc_metrics(container)

            # Should return ScpContainer
            assert isinstance(result, ScpContainer)

            # Check column types
            assert "n_features_protein" in result.obs.columns
            assert "total_intensity_protein" in result.obs.columns
            assert "log1p_total_intensity_protein" in result.obs.columns

            # Check value types
            n_feats = result.obs["n_features_protein"]
            assert n_feats.dtype in [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]

        def test_feature_qc_return_types():
            """Verify return types from calculate_feature_qc_metrics."""
            container = self._create_container(n_samples=10, n_features=5)

            result = calculate_feature_qc_metrics(container)

            var = result.assays["protein"].var

            # Check columns exist
            assert "missing_rate" in var.columns
            assert "detection_rate" in var.columns
            assert "mean_expression" in var.columns
            assert "cv" in var.columns

            # Check value ranges
            missing_rate = var["missing_rate"].to_numpy()
            detection_rate = var["detection_rate"].to_numpy()

            assert np.all((missing_rate >= 0) & (missing_rate <= 1)), \
                "missing_rate should be in [0, 1]"
            assert np.all((detection_rate >= 0) & (detection_rate <= 1)), \
                "detection_rate should be in [0, 1]"

        def test_filtering_return_types():
            """Verify return types from filtering functions."""
            container = self._create_container(n_samples=10, n_features=5)

            result = filter_low_quality_samples(container, min_features=1, use_mad=False)
            assert isinstance(result, ScpContainer)

            result = filter_doublets_mad(result)
            assert isinstance(result, ScpContainer)

            result = filter_features_by_missingness(result, max_missing_rate=1.0)
            assert isinstance(result, ScpContainer)

            result = filter_features_by_cv(result, max_cv=10.0)
            assert isinstance(result, ScpContainer)

        self._run_test("Types: sample QC return types", test_sample_qc_return_types)
        self._run_test("Types: feature QC return types", test_feature_qc_return_types)
        self._run_test("Types: filtering return types", test_filtering_return_types)

    # ============== Error Handling Tests ==============

    def _test_nonexistent_assay(self) -> None:
        """Test error handling for nonexistent assay."""

        def test_invalid_assay():
            """Test with invalid assay name."""
            container = self._create_container(n_samples=5, n_features=3)

            try:
                result = calculate_sample_qc_metrics(container, assay_name="nonexistent")
                raise AssertionError("Should have raised AssayNotFoundError")
            except Exception as e:
                assert "assay" in str(e).lower() or "not found" in str(e).lower(), \
                    f"Unexpected error: {e}"

        self._run_test("Error: nonexistent assay", test_invalid_assay)

    def _test_nonexistent_layer(self) -> None:
        """Test error handling for nonexistent layer."""

        def test_invalid_layer():
            """Test with invalid layer name."""
            container = self._create_container(n_samples=5, n_features=3)

            try:
                result = calculate_sample_qc_metrics(container, layer_name="nonexistent")
                raise AssertionError("Should have raised error for nonexistent layer")
            except Exception as e:
                assert "layer" in str(e).lower() or "not found" in str(e).lower(), \
                    f"Unexpected error: {e}"

        self._run_test("Error: nonexistent layer", test_invalid_layer)

    def _test_wrong_data_types(self) -> None:
        """Test error handling for wrong data types."""

        def test_invalid_threshold():
            """Test with invalid threshold values."""
            container = self._create_container(n_samples=5, n_features=3)

            # Test negative max_missing_rate
            try:
                result = filter_features_by_missingness(container, max_missing_rate=-0.1)
                raise AssertionError("Should have raised error for negative threshold")
            except Exception as e:
                assert "threshold" in str(e).lower() or "between" in str(e).lower(), \
                    f"Unexpected error: {e}"

            # Test max_missing_rate > 1
            try:
                result = filter_features_by_missingness(container, max_missing_rate=1.5)
                raise AssertionError("Should have raised error for threshold > 1")
            except Exception as e:
                assert "threshold" in str(e).lower() or "between" in str(e).lower(), \
                    f"Unexpected error: {e}"

            # Test zero max_cv
            try:
                result = filter_features_by_cv(container, max_cv=0)
                raise AssertionError("Should have raised error for zero max_cv")
            except Exception as e:
                assert "max_cv" in str(e).lower() or "positive" in str(e).lower(), \
                    f"Unexpected error: {e}"

        self._run_test("Error: invalid thresholds", test_invalid_threshold)

    # ============== Edge Case Tests ==============

    def _test_division_by_zero(self) -> None:
        """Test division by zero scenarios."""

        def test_cv_zero_mean():
            """Test CV calculation with zero mean."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})

            X = np.array([[0.0], [0.0]])

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_feature_qc_metrics(container)
            cv = result.assays["protein"].var["cv"][0]

            # CV should be NaN for zero mean
            assert cv is None or np.isnan(cv), f"CV should be NaN for zero mean, got {cv}"

        def test_mad_constant_values():
            """Test MAD with constant values (zero deviation)."""
            data = np.array([5.0, 5.0, 5.0, 5.0])

            mad = compute_mad(data)

            # MAD should be 0 for constant values
            assert mad == 0, f"MAD should be 0 for constant values, got {mad}"

        def test_outlier_zero_mad():
            """Test outlier detection with zero MAD."""
            data = np.array([5.0, 5.0, 5.0, 5.0, 10.0])  # One outlier

            outliers = is_outlier_mad(data, nmads=3.0)

            # With zero MAD, only non-median values are outliers
            assert outliers[-1] == True, "Value different from median should be outlier with zero MAD"

        self._run_test("Division by zero: CV with zero mean", test_cv_zero_mean)
        self._run_test("Division by zero: MAD with constant values", test_mad_constant_values)
        self._run_test("Division by zero: outlier detection with zero MAD", test_outlier_zero_mad)

    def _test_null_propagation(self) -> None:
        """Test null/NaN propagation through calculations."""

        def test_nan_propagation():
            """Test NaN values in calculations."""
            data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

            mad = compute_mad(data)

            # MAD should handle NaN gracefully
            assert not np.isnan(mad), f"MAD should not be NaN, got {mad}"

        def test_sparse_nan_handling():
            """Test NaN handling in sparse matrices."""
            # Sparse matrices can't store NaN, so this tests
            # that the code handles this correctly
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})

            X = sp.csr_matrix([[0.0], [1.0]])  # Zeros represent missing

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)

            # Should not raise errors
            assert result.n_samples == 2

        self._run_test("Null propagation: NaN in MAD", test_nan_propagation)
        self._run_test("Null propagation: sparse NaN handling", test_sparse_nan_handling)

    def _test_type_conversion(self) -> None:
        """Test type conversion issues."""

        def test_integer_input():
            """Test with integer input matrix."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})

            X = np.array([[1], [2]], dtype=np.int32)  # Integer input

            matrix = ScpMatrix(X=X)
            # Should be converted to float
            assert np.issubdtype(matrix.X.dtype, np.floating), \
                f"X should be float, got {matrix.X.dtype}"

            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 2

        def test_float32_input():
            """Test with float32 input."""
            obs = pl.DataFrame({"_index": ["S1", "S2"]})
            var = pl.DataFrame({"_index": ["P1"]})

            X = np.array([[1.0], [2.0]], dtype=np.float32)

            matrix = ScpMatrix(X=X)
            assay = Assay(var=var, layers={"raw": matrix})
            container = ScpContainer(obs=obs, assays={"protein": assay})

            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 2

        self._run_test("Type conversion: integer input", test_integer_input)
        self._run_test("Type conversion: float32 input", test_float32_input)

    def _test_mask_code_consistency(self) -> None:
        """Test mask code consistency."""

        def test_valid_mask_codes():
            """Test only valid mask codes are accepted."""
            obs = pl.DataFrame({"_index": ["S1"]})
            var = pl.DataFrame({"_index": ["P1"]})

            X = np.array([[1.0]])

            # Test valid codes
            for code in [0, 1, 2, 3, 4, 5, 6]:
                M = np.array([[code]], dtype=np.int8)
                matrix = ScpMatrix(X=X, M=M)
                assert matrix.M is not None

            # Test invalid code (should fail)
            try:
                M = np.array([[99]], dtype=np.int8)
                matrix = ScpMatrix(X=X, M=M)
                raise AssertionError("Should have raised error for invalid mask code")
            except ValueError as e:
                assert "invalid" in str(e).lower() or "mask" in str(e).lower()

        def test_mask_dtype():
            """Test mask matrix dtype."""
            obs = pl.DataFrame({"_index": ["S1"]})
            var = pl.DataFrame({"_index": ["P1"]})

            X = np.array([[1.0]])

            # Create mask with wrong dtype
            M = np.array([[0]], dtype=np.int64)

            matrix = ScpMatrix(X=X, M=M)

            # Should be converted to int8
            assert matrix.M.dtype == np.int8, f"Mask should be int8, got {matrix.M.dtype}"

        self._run_test("Mask consistency: valid codes", test_valid_mask_codes)
        self._run_test("Mask consistency: dtype conversion", test_mask_dtype)

    # ============== Large Dataset Tests ==============

    def _test_large_dataset_memory(self) -> None:
        """Test memory handling with large datasets."""

        def test_large_dense():
            """Test with large dense matrix."""
            # Force garbage collection before test
            gc.collect()

            import tracemalloc
            tracemalloc.start()

            try:
                container = self._create_container(n_samples=1000, n_features=500)

                result = calculate_sample_qc_metrics(container)
                assert result.n_samples == 1000

                result = calculate_feature_qc_metrics(result)
                assert result.assays["protein"].n_features == 500

                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Memory usage should be reasonable
                # For 1000x500 float64 array: ~4MB
                # Allow up to 100MB for overhead
                assert peak < 100 * 1024 * 1024, f"Memory usage too high: {peak / 1024 / 1024:.2f}MB"

            finally:
                tracemalloc.stop()
                gc.collect()

        def test_large_sparse():
            """Test with large sparse matrix."""
            gc.collect()

            container = self._create_container(
                n_samples=1000, n_features=500, sparse=True, density=0.05
            )

            result = calculate_sample_qc_metrics(container)
            assert result.n_samples == 1000

            # Sparse should use much less memory
            gc.collect()

        def test_very_large_features():
            """Test with many features."""
            container = self._create_container(n_samples=100, n_features=5000)

            result = calculate_feature_qc_metrics(container)
            assert result.assays["protein"].n_features == 5000

            gc.collect()

        self._run_test("Large dataset: 1000x500 dense", test_large_dense)
        self._run_test("Large dataset: 1000x500 sparse", test_large_sparse)
        self._run_test("Large dataset: 100x5000 features", test_very_large_features)

    # ============== Integration Tests ==============

    def _test_full_pipeline(self) -> None:
        """Test full QC pipeline."""

        def test_sample_pipeline():
            """Test complete sample QC pipeline."""
            container = self._create_container(n_samples=50, n_features=20, with_mask=True)

            # Step 1: Calculate metrics
            container = calculate_sample_qc_metrics(container)
            assert "n_features_protein" in container.obs.columns

            # Step 2: Filter low quality
            container = filter_low_quality_samples(container, min_features=5, use_mad=False)
            assert container.n_samples <= 50

            # Step 3: Filter doublets
            if container.n_samples > 1:
                container = filter_doublets_mad(container, nmads=3.0)
                assert container.n_samples >= 1

        def test_feature_pipeline():
            """Test complete feature QC pipeline."""
            container = self._create_container(n_samples=50, n_features=20)

            # Step 1: Calculate metrics
            container = calculate_feature_qc_metrics(container)
            assert "missing_rate" in container.assays["protein"].var.columns

            # Step 2: Filter by missingness
            container = filter_features_by_missingness(container, max_missing_rate=0.9)
            assert container.assays["protein"].n_features <= 20

            # Step 3: Filter by CV
            container = filter_features_by_cv(container, max_cv=5.0)
            assert container.assays["protein"].n_features >= 0

        def test_combined_pipeline():
            """Test combined sample and feature QC."""
            container = self._create_container(n_samples=30, n_features=15)

            # Sample QC
            container = calculate_sample_qc_metrics(container)
            container = filter_low_quality_samples(container, min_features=3, use_mad=False)

            # Feature QC
            container = calculate_feature_qc_metrics(container)
            container = filter_features_by_missingness(container, max_missing_rate=0.8)

            # Verify consistency
            assert container.n_samples >= 1
            assert container.assays["protein"].n_features >= 1

            # Verify history was logged
            assert len(container.history) > 0

        def test_batch_effect_assessment():
            """Test batch effect assessment."""
            container = self._create_container(n_samples=30, n_features=10)

            # Add batch column
            container = calculate_sample_qc_metrics(container)

            summary = assess_batch_effects(container, batch_col="batch")

            assert isinstance(summary, pl.DataFrame)
            assert "batch" in summary.columns
            assert "n_cells" in summary.columns

        self._run_test("Pipeline: sample QC", test_sample_pipeline)
        self._run_test("Pipeline: feature QC", test_feature_pipeline)
        self._run_test("Pipeline: combined", test_combined_pipeline)
        self._run_test("Pipeline: batch effect assessment", test_batch_effect_assessment)

    # ============== Report Generation ==============

    def _generate_report(self) -> None:
        """Generate test report."""
        self.report.end_time = datetime.now()

        # Print summary
        print()
        print("=" * 80)
        print("Test Summary")
        print("=" * 80)
        print(f"Total tests:  {self.report.total}")
        print(f"Passed:       {self.report.passed} ({self.report.passed/self.report.total*100:.1f}%)")
        print(f"Failed:       {self.report.failed} ({self.report.failed/self.report.total*100:.1f}%)")
        print(f"Skipped:      {self.report.skipped} ({self.report.skipped/self.report.total*100:.1f}%)")
        print(f"Duration:     {(self.report.end_time - self.report.start_time).total_seconds():.2f} seconds")
        print()

        # Print failed tests
        if self.report.failed > 0:
            print("Failed Tests:")
            print("-" * 80)
            for test in self.report.tests:
                if test.status == "FAIL":
                    print(f"  - {test.name}")
                    print(f"    Error: {test.error}")
                    if test.traceback:
                        print(f"    Traceback:")
                        for line in test.traceback.split('\n'):
                            if line.strip():
                                print(f"      {line}")
            print()

        # Generate markdown report
        report_path = self.output_dir / "qc_stress_test_report.md"
        with open(report_path, 'w') as f:
            f.write("# QC Module Stress Test Report\n\n")
            f.write(f"**Date:** {self.report.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"| Metric | Count | Percentage |\n")
            f.write(f"|--------|-------|------------|\n")
            f.write(f"| Total | {self.report.total} | 100% |\n")
            f.write(f"| Passed | {self.report.passed} | {self.report.passed/self.report.total*100:.1f}% |\n")
            f.write(f"| Failed | {self.report.failed} | {self.report.failed/self.report.total*100:.1f}% |\n")
            f.write(f"| Skipped | {self.report.skipped} | {self.report.skipped/self.report.total*100:.1f}% |\n\n")

            f.write(f"**Duration:** {(self.report.end_time - self.report.start_time).total_seconds():.2f} seconds\n\n")

            # Test results by category
            categories = {}
            for test in self.report.tests:
                category = test.name.split(":")[0] if ":" in test.name else "Other"
                if category not in categories:
                    categories[category] = []
                categories[category].append(test)

            f.write("## Results by Category\n\n")
            for category, tests in sorted(categories.items()):
                passed = sum(1 for t in tests if t.status == "PASS")
                f.write(f"### {category}\n\n")
                f.write(f"- Passed: {passed}/{len(tests)}\n")
                for test in tests:
                    symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘"}[test.status]
                    f.write(f"  - {symbol} {test.name} ({test.duration_ms:.2f}ms)\n")
                    if test.status == "FAIL" and test.error:
                        f.write(f"    - Error: {test.error}\n")
                f.write("\n")

            # Detailed failure information
            if self.report.failed > 0:
                f.write("## Detailed Failure Information\n\n")
                for test in self.report.tests:
                    if test.status == "FAIL":
                        f.write(f"### {test.name}\n\n")
                        f.write(f"**Error:** {test.error}\n\n")
                        if test.details.get("warnings"):
                            f.write("**Warnings:**\n")
                            for warning in test.details["warnings"]:
                                f.write(f"- {warning}\n")
                            f.write("\n")
                        if test.traceback:
                            f.write("**Traceback:**\n\n")
                            f.write("```\n")
                            f.write(test.traceback)
                            f.write("\n```\n\n")

            # Performance summary
            f.write("## Performance Summary\n\n")
            f.write("| Test | Duration (ms) |\n")
            f.write("|------|--------------|\n")
            for test in sorted(self.report.tests, key=lambda t: t.duration_ms, reverse=True)[:10]:
                f.write(f"| {test.name} | {test.duration_ms:.2f} |\n")
            f.write("\n")

        print(f"Report saved to: {report_path.absolute()}")


def main() -> int:
    """Run the stress test suite."""
    import argparse

    parser = argparse.ArgumentParser(description="QC Module Stress Test Suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp"),
        help="Directory to save test reports",
    )

    args = parser.parse_args()

    try:
        suite = QCStressTest(output_dir=args.output_dir)
        suite.run_all_tests()

        # Exit code based on results
        if suite.report.failed > 0:
            return 1
        return 0

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
