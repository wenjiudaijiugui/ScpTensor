#!/usr/bin/env python3
"""Comprehensive comparison of ScpTensor vs Scanpy for Normalization and QC.

This script benchmarks and compares:
1. Normalization methods:
   - Log normalization
   - Z-score normalization
   - Median centering (ScpTensor-only)
   - Quantile normalization (ScpTensor-only)

2. Quality Control metrics:
   - Sample QC metrics (total counts, features detected)
   - Feature QC metrics (missing rate, CV)
   - Outlier detection performance

Usage:
    python run_qc_norm_comparison.py [--quick] [--full] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import anndata as ad
    import scanpy as sc

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("Warning: Scanpy not available. Only ScpTensor methods will be tested.")

from scptensor.benchmark.synthetic_data import SyntheticDataset
from scptensor.core.structures import ScpContainer
from scptensor.normalization import norm_log, norm_median, norm_quantile
from scptensor.qc import (
    calculate_feature_qc_metrics,
    calculate_sample_qc_metrics,
)
from scptensor.qc.metrics import is_outlier_mad
from scptensor.standardization import zscore

# =============================================================================
# Data Generation
# =============================================================================


def create_benchmark_datasets(quick: bool = False) -> list[ScpContainer]:
    """Create benchmark datasets with varying characteristics."""
    if quick:
        return [
            SyntheticDataset(
                n_samples=100,
                n_features=500,
                n_groups=2,
                n_batches=2,
                missing_rate=0.2,
                random_seed=42,
            ).generate()
        ]

    configs = [
        {
            "n_samples": 100,
            "n_features": 500,
            "n_groups": 2,
            "n_batches": 2,
            "missing_rate": 0.15,
            "random_seed": 42,
        },
        {
            "n_samples": 500,
            "n_features": 1000,
            "n_groups": 3,
            "n_batches": 3,
            "missing_rate": 0.25,
            "random_seed": 123,
        },
        {
            "n_samples": 1000,
            "n_features": 2000,
            "n_groups": 4,
            "n_batches": 4,
            "missing_rate": 0.35,
            "random_seed": 456,
        },
    ]

    datasets = []
    for config in configs:
        datasets.append(SyntheticDataset(**config).generate())
    return datasets


# =============================================================================
# Normalization Comparison
# =============================================================================


def run_normalization_comparison(
    datasets: list[ScpContainer],
    output_dir: Path,
) -> dict[str, Any]:
    """Compare normalization methods between ScpTensor and Scanpy."""
    print("\n" + "=" * 70)
    print("NORMALIZATION COMPARISON")
    print("=" * 70)

    results = {
        "log_normalize": [],
        "z_score_normalize": [],
        "scptensor_exclusives": {
            "median_center": [],
            "quantile_normalize": [],
        },
    }

    for idx, dataset in enumerate(datasets, 1):
        n_samples = dataset.n_samples
        n_features = dataset.assays["protein"].n_features
        missing_rate = np.mean(dataset.assays["protein"].layers["raw"].M > 0)

        print(
            f"\n[{idx}/{len(datasets)}] Dataset: {n_samples}×{n_features}, {missing_rate:.1%} missing"
        )

        X = dataset.assays["protein"].layers["raw"].X.copy()
        M = dataset.assays["protein"].layers["raw"].M.copy()

        # Test 1: Log normalization
        print("  Testing log_normalize...")
        norm_results = test_log_normalize(dataset, X, M)
        results["log_normalize"].append(norm_results)
        print_norm_result("log_normalize", norm_results)

        # Test 2: Z-score normalization
        print("  Testing z_score_normalize...")
        norm_results = test_z_score_normalize(dataset, X, M)
        results["z_score_normalize"].append(norm_results)
        print_norm_result("z_score_normalize", norm_results)

        # Test 3: Median centering (ScpTensor-only)
        print("  Testing median_center (ScpTensor-only)...")
        norm_results = test_median_center(dataset, X, M)
        results["scptensor_exclusives"]["median_center"].append(norm_results)
        print_norm_result("median_center", norm_results)

        # Test 4: Quantile normalization (ScpTensor-only)
        print("  Testing quantile_normalize (ScpTensor-only)...")
        norm_results = test_quantile_normalize(dataset, X, M)
        results["scptensor_exclusives"]["quantile_normalize"].append(norm_results)
        print_norm_result("quantile_normalize", norm_results)

    # Save results
    results_file = output_dir / "normalization_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nNormalization results saved to: {results_file}")

    return results


def test_log_normalize(
    dataset: ScpContainer,
    X: np.ndarray,
    M: np.ndarray,
) -> dict[str, Any]:
    """Test log normalization."""
    result = {}

    # ScpTensor
    start = time.time()
    try:
        container_st = norm_log(dataset.copy(), "protein", "raw", "log", base=2.0, offset=1.0)
        X_st = container_st.assays["protein"].layers["log"].X
        time_st = time.time() - start
        success_st = True
    except Exception as e:
        X_st = None
        time_st = 0
        success_st = False
        print(f"    ScpTensor error: {e}")

    result["scptensor"] = {
        "time": time_st,
        "success": success_st,
    }

    # Scanpy
    if SCANPY_AVAILABLE:
        start = time.time()
        try:
            X_nan = X.copy().astype(float)
            X_nan[M > 0] = np.nan
            adata = ad.AnnData(X=X_nan)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata, base=np.log(2))
            X_sp = np.nan_to_num(adata.X, nan=0.0)
            time_sp = time.time() - start
            success_sp = True

            # Compute accuracy
            if success_st:
                correlation = np.corrcoef(X_st.flatten(), X_sp.flatten())[0, 1]
                mse = np.mean((X_st - X_sp) ** 2)
            else:
                correlation = None
                mse = None
        except Exception as e:
            X_sp = None
            time_sp = 0
            success_sp = False
            correlation = None
            mse = None
            print(f"    Scanpy error: {e}")

        result["scanpy"] = {
            "time": time_sp,
            "success": success_sp,
        }
        result["comparison"] = {
            "speedup": time_sp / time_st if time_st > 0 and time_sp > 0 else None,
            "correlation": correlation,
            "mse": mse,
        }
    else:
        result["scanpy"] = None
        result["comparison"] = None

    return result


def test_z_score_normalize(
    dataset: ScpContainer,
    X: np.ndarray,
    M: np.ndarray,
) -> dict[str, Any]:
    """Test z-score normalization."""
    result = {}

    # ScpTensor
    start = time.time()
    try:
        container_st = zscore(dataset.copy(), "protein", "raw", "zscore", axis=0, ddof=1)
        X_st = container_st.assays["protein"].layers["zscore"].X
        time_st = time.time() - start
        success_st = True
    except Exception as e:
        X_st = None
        time_st = 0
        success_st = False
        print(f"    ScpTensor error: {e}")

    result["scptensor"] = {
        "time": time_st,
        "success": success_st,
    }

    # Scanpy
    if SCANPY_AVAILABLE:
        start = time.time()
        try:
            X_nan = X.copy().astype(float)
            X_nan[M > 0] = np.nan
            adata = ad.AnnData(X=X_nan)
            sc.pp.scale(adata, zero_center=True, max_value=None)
            X_sp = np.nan_to_num(adata.X, nan=0.0)
            time_sp = time.time() - start
            success_sp = True

            # Compute accuracy
            if success_st:
                correlation = np.corrcoef(X_st.flatten(), X_sp.flatten())[0, 1]
                mse = np.mean((X_st - X_sp) ** 2)
            else:
                correlation = None
                mse = None
        except Exception as e:
            X_sp = None
            time_sp = 0
            success_sp = False
            correlation = None
            mse = None
            print(f"    Scanpy error: {e}")

        result["scanpy"] = {
            "time": time_sp,
            "success": success_sp,
        }
        result["comparison"] = {
            "speedup": time_sp / time_st if time_st > 0 and time_sp > 0 else None,
            "correlation": correlation,
            "mse": mse,
        }
    else:
        result["scanpy"] = None
        result["comparison"] = None

    return result


def test_median_center(
    dataset: ScpContainer,
    X: np.ndarray,
    M: np.ndarray,
) -> dict[str, Any]:
    """Test median centering (ScpTensor-only)."""
    result = {}

    start = time.time()
    try:
        container_st = norm_median(
            dataset.copy(), "protein", "raw", "median_centered", add_global_median=False
        )
        X_st = container_st.assays["protein"].layers["median_centered"].X
        time_st = time.time() - start
        success_st = True

        # Verify median is ~0
        medians = np.nanmedian(X_st, axis=1)
        result["verification"] = {
            "mean_median": float(np.mean(np.abs(medians))),
            "max_median": float(np.max(np.abs(medians))),
        }
    except Exception as e:
        X_st = None
        time_st = 0
        success_st = False
        result["verification"] = None
        print(f"    ScpTensor error: {e}")

    result["scptensor"] = {
        "time": time_st,
        "success": success_st,
    }

    return result


def test_quantile_normalize(
    dataset: ScpContainer,
    X: np.ndarray,
    M: np.ndarray,
) -> dict[str, Any]:
    """Test quantile normalization (ScpTensor-only)."""
    result = {}

    start = time.time()
    try:
        container_st = norm_quantile(dataset.copy(), "protein", "raw", "quantile")
        X_st = container_st.assays["protein"].layers["quantile"].X
        time_st = time.time() - start
        success_st = True

        # Verify distributions are similar
        if success_st:
            col_means = np.mean(X_st, axis=0)
            cv_col_means = np.std(col_means) / np.mean(col_means) if np.mean(col_means) > 0 else 0
            result["verification"] = {
                "cv_column_means": float(cv_col_means),
            }
    except Exception as e:
        X_st = None
        time_st = 0
        success_st = False
        result["verification"] = None
        print(f"    ScpTensor error: {e}")

    result["scptensor"] = {
        "time": time_st,
        "success": success_st,
    }

    return result


def print_norm_result(method: str, result: dict[str, Any]) -> None:
    """Print normalization result summary."""
    st = result.get("scptensor", {})
    sp = result.get("scanpy")
    comp = result.get("comparison")

    if st and st["success"]:
        time_str = f"{st['time'] * 1000:.1f}ms"
        if sp and sp["success"]:
            speedup = comp.get("speedup", 0)
            corr = comp.get("correlation", 0)
            print(
                f"    ScpTensor: {time_str}, Scanpy: {sp['time'] * 1000:.1f}ms, Speedup: {speedup:.2f}x, Corr: {corr:.3f}"
            )
        else:
            print(f"    ScpTensor: {time_str}")
    else:
        print("    ScpTensor: Failed")


# =============================================================================
# QC Comparison
# =============================================================================


def run_qc_comparison(
    datasets: list[ScpContainer],
    output_dir: Path,
) -> dict[str, Any]:
    """Compare QC metrics between ScpTensor and Scanpy."""
    print("\n" + "=" * 70)
    print("QUALITY CONTROL COMPARISON")
    print("=" * 70)

    results = {
        "sample_qc_metrics": [],
        "feature_qc_metrics": [],
        "outlier_detection": [],
    }

    for idx, dataset in enumerate(datasets, 1):
        n_samples = dataset.n_samples
        n_features = dataset.assays["protein"].n_features
        print(f"\n[{idx}/{len(datasets)}] Dataset: {n_samples}×{n_features}")

        X = dataset.assays["protein"].layers["raw"].X.copy()
        M = dataset.assays["protein"].layers["raw"].M.copy()

        # Test 1: Sample QC metrics
        print("  Testing sample QC metrics...")
        qc_result = test_sample_qc_metrics(dataset, X, M)
        results["sample_qc_metrics"].append(qc_result)
        print_qc_result("sample_qc", qc_result)

        # Test 2: Feature QC metrics
        print("  Testing feature QC metrics...")
        qc_result = test_feature_qc_metrics(dataset, X, M)
        results["feature_qc_metrics"].append(qc_result)
        print_qc_result("feature_qc", qc_result)

        # Test 3: Outlier detection
        print("  Testing outlier detection...")
        qc_result = test_outlier_detection(dataset, X, M)
        results["outlier_detection"].append(qc_result)
        print_qc_result("outlier_detection", qc_result)

    # Save results
    results_file = output_dir / "qc_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nQC results saved to: {results_file}")

    return results


def test_sample_qc_metrics(
    dataset: ScpContainer,
    X: np.ndarray,
    M: np.ndarray,
) -> dict[str, Any]:
    """Test sample QC metrics computation."""
    result = {}

    # ScpTensor
    start = time.time()
    try:
        container_st = calculate_sample_qc_metrics(dataset.copy(), "protein", "raw")
        obs_st = container_st.obs

        total_intensity = obs_st["total_intensity_protein"].to_numpy()
        n_features_detected = obs_st["n_features_protein"].to_numpy()
        missing_rate = 1 - n_features_detected / dataset.assays["protein"].n_features

        time_st = time.time() - start
        success_st = True

        result["scptensor"] = {
            "time": time_st,
            "success": success_st,
            "metrics": {
                "mean_total_intensity": float(np.mean(total_intensity)),
                "mean_n_features": float(np.mean(n_features_detected)),
                "mean_missing_rate": float(np.mean(missing_rate)),
            },
        }
    except Exception as e:
        time_st = 0
        success_st = False
        result["scptensor"] = {"time": time_st, "success": success_st}
        print(f"    ScpTensor error: {e}")

    # Scanpy
    if SCANPY_AVAILABLE:
        start = time.time()
        try:
            X_nan = X.copy().astype(float)
            X_nan[M > 0] = np.nan
            adata = ad.AnnData(X=X_nan)

            adata.obs["n_counts"] = np.nansum(X_nan, axis=1)
            adata.obs["n_genes"] = np.sum(~np.isnan(X_nan), axis=1)
            adata.obs["pct_counts"] = adata.obs["n_counts"] / np.nansum(X_nan) * 100

            time_sp = time.time() - start
            success_sp = True

            result["scanpy"] = {
                "time": time_sp,
                "success": success_sp,
                "metrics": {
                    "mean_total_intensity": float(np.mean(adata.obs["n_counts"])),
                    "mean_n_features": float(np.mean(adata.obs["n_genes"])),
                },
            }

            if success_st:
                speedup = time_sp / time_st if time_st > 0 and time_sp > 0 else None
                result["comparison"] = {"speedup": speedup}
        except Exception as e:
            time_sp = 0
            success_sp = False
            result["scanpy"] = {"time": time_sp, "success": success_sp}
            result["comparison"] = None
            print(f"    Scanpy error: {e}")
    else:
        result["scanpy"] = None
        result["comparison"] = None

    return result


def test_feature_qc_metrics(
    dataset: ScpContainer,
    X: np.ndarray,
    M: np.ndarray,
) -> dict[str, Any]:
    """Test feature QC metrics computation."""
    result = {}

    # ScpTensor
    start = time.time()
    try:
        container_st = calculate_feature_qc_metrics(dataset.copy(), "protein", "raw")
        var_st = container_st.assays["protein"].var

        missing_rate = var_st["missing_rate"].to_numpy()
        detection_rate = var_st["detection_rate"].to_numpy()
        cv = var_st["cv"].to_numpy()

        time_st = time.time() - start
        success_st = True

        result["scptensor"] = {
            "time": time_st,
            "success": success_st,
            "metrics": {
                "mean_missing_rate": float(np.mean(missing_rate)),
                "mean_detection_rate": float(np.mean(detection_rate)),
                "mean_cv": float(np.mean(cv[~np.isnan(cv)])) if np.any(~np.isnan(cv)) else None,
            },
        }
    except Exception as e:
        time_st = 0
        success_st = False
        result["scptensor"] = {"time": time_st, "success": success_st}
        print(f"    ScpTensor error: {e}")

    # Scanpy
    if SCANPY_AVAILABLE:
        start = time.time()
        try:
            X_nan = X.copy().astype(float)
            X_nan[M > 0] = np.nan
            adata = ad.AnnData(X=X_nan)

            adata.var["n_cells"] = np.sum(~np.isnan(X_nan), axis=0)
            adata.var["mean_counts"] = np.nanmean(X_nan, axis=0)
            adata.var["pct_dropout"] = (1 - adata.var["n_cells"] / X_nan.shape[0]) * 100

            time_sp = time.time() - start
            success_sp = True

            result["scanpy"] = {
                "time": time_sp,
                "success": success_sp,
                "metrics": {
                    "mean_missing_rate": float(np.mean(adata.var["pct_dropout"]) / 100),
                    "mean_n_cells": float(np.mean(adata.var["n_cells"])),
                },
            }

            if success_st:
                speedup = time_sp / time_st if time_st > 0 and time_sp > 0 else None
                result["comparison"] = {"speedup": speedup}
        except Exception as e:
            time_sp = 0
            success_sp = False
            result["scanpy"] = {"time": time_sp, "success": success_sp}
            result["comparison"] = None
            print(f"    Scanpy error: {e}")
    else:
        result["scanpy"] = None
        result["comparison"] = None

    return result


def test_outlier_detection(
    dataset: ScpContainer,
    X: np.ndarray,
    M: np.ndarray,
) -> dict[str, Any]:
    """Test outlier detection performance."""
    result = {}

    # ScpTensor MAD-based outlier detection
    start = time.time()
    try:
        container_st = calculate_sample_qc_metrics(dataset.copy(), "protein", "raw")
        total_intensity = container_st.obs["total_intensity_protein"].to_numpy()

        outliers = is_outlier_mad(total_intensity, nmads=3.0, direction="lower")
        n_outliers = np.sum(outliers)

        time_st = time.time() - start
        success_st = True

        result["scptensor"] = {
            "time": time_st,
            "success": success_st,
            "n_outliers": int(n_outliers),
            "outlier_rate": float(n_outliers / len(total_intensity)),
        }
    except Exception as e:
        time_st = 0
        success_st = False
        result["scptensor"] = {"time": time_st, "success": success_st}
        print(f"    ScpTensor error: {e}")

    result["scanpy"] = None
    result["comparison"] = None

    return result


def print_qc_result(test_type: str, result: dict[str, Any]) -> None:
    """Print QC result summary."""
    st = result.get("scptensor", {})
    sp = result.get("scanpy")
    comp = result.get("comparison")

    if st and st.get("success"):
        time_str = f"{st['time'] * 1000:.1f}ms"
        if sp and sp.get("success"):
            speedup = comp.get("speedup", 0) if comp else 0
            print(
                f"    ScpTensor: {time_str}, Scanpy: {sp['time'] * 1000:.1f}ms, Speedup: {speedup:.2f}x"
            )
        else:
            print(f"    ScpTensor: {time_str}")

        if "metrics" in st:
            metrics = st["metrics"]
            if test_type == "sample_qc":
                print(f"      Mean total intensity: {metrics['mean_total_intensity']:.1f}")
                print(f"      Mean n features: {metrics['mean_n_features']:.0f}")
            elif test_type == "feature_qc":
                print(f"      Mean missing rate: {metrics['mean_missing_rate']:.1%}")
                print(f"      Mean detection rate: {metrics['mean_detection_rate']:.1%}")
        if "n_outliers" in st:
            print(f"      Outliers detected: {st['n_outliers']} ({st['outlier_rate']:.1%})")
    else:
        print("    ScpTensor: Failed")


# =============================================================================
# Summary and Reporting
# =============================================================================


def generate_summary_report(
    norm_results: dict[str, Any],
    qc_results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate comprehensive summary report."""
    report_lines = []
    report_lines.append("# ScpTensor vs Scanpy: Normalization & QC Comparison Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("## Normalization Methods\n")

    # Log normalize
    if norm_results.get("log_normalize"):
        report_lines.append("### Log Normalize\n")
        report_lines.append("| Dataset | ScpTensor | Scanpy |")
        report_lines.append("|---------|-----------|--------|")
        for idx, result in enumerate(norm_results["log_normalize"], 1):
            st = result["scptensor"]
            sp = result.get("scanpy", {})
            comp = result.get("comparison", {})
            if st["success"] and sp:
                speedup = comp.get("speedup", "N/A")
                corr = comp.get("correlation", "N/A")
                report_lines.append(
                    f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | {sp['time'] * 1000:.2f}ms (Speedup: {speedup:.2f}x, Corr: {corr:.3f}) |"
                )
            elif st["success"]:
                report_lines.append(f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | N/A |")
        report_lines.append("")

    # Z-score normalize
    if norm_results.get("z_score_normalize"):
        report_lines.append("### Z-Score Normalize\n")
        report_lines.append("| Dataset | ScpTensor | Scanpy |")
        report_lines.append("|---------|-----------|--------|")
        for idx, result in enumerate(norm_results["z_score_normalize"], 1):
            st = result["scptensor"]
            sp = result.get("scanpy", {})
            comp = result.get("comparison", {})
            if st["success"] and sp:
                speedup = comp.get("speedup", "N/A")
                corr = comp.get("correlation", "N/A")
                report_lines.append(
                    f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | {sp['time'] * 1000:.2f}ms (Speedup: {speedup:.2f}x, Corr: {corr:.3f}) |"
                )
            elif st["success"]:
                report_lines.append(f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | N/A |")
        report_lines.append("")

    # ScpTensor exclusives
    st_exclusives = norm_results.get("scptensor_exclusives", {})
    if st_exclusives.get("median_center"):
        report_lines.append("### Median Centering (ScpTensor-exclusive)\n")
        report_lines.append("| Dataset | Time | Verification |")
        report_lines.append("|---------|------|--------------|")
        for idx, result in enumerate(st_exclusives["median_center"], 1):
            st = result["scptensor"]
            if st["success"]:
                verif = result.get("verification", {})
                mean_med = verif.get("mean_median", "N/A") if verif else "N/A"
                report_lines.append(
                    f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | Mean |median|: {mean_med:.6f} |"
                )
        report_lines.append("")

    if st_exclusives.get("quantile_normalize"):
        report_lines.append("### Quantile Normalization (ScpTensor-exclusive)\n")
        report_lines.append("| Dataset | Time | CV of Column Means |")
        report_lines.append("|---------|------|-------------------|")
        for idx, result in enumerate(st_exclusives["quantile_normalize"], 1):
            st = result["scptensor"]
            if st["success"]:
                verif = result.get("verification", {})
                cv = verif.get("cv_column_means", "N/A") if verif else "N/A"
                report_lines.append(f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | {cv:.6f} |")
        report_lines.append("")

    # QC summary
    report_lines.append("## Quality Control Metrics\n")

    if qc_results.get("sample_qc_metrics"):
        report_lines.append("### Sample QC Metrics\n")
        report_lines.append("| Dataset | ScpTensor Time | Scanpy Time | Speedup |")
        report_lines.append("|---------|----------------|-------------|---------|")
        for idx, result in enumerate(qc_results["sample_qc_metrics"], 1):
            st = result["scptensor"]
            sp = result.get("scanpy", {})
            comp = result.get("comparison", {})
            if st["success"] and sp:
                speedup = comp.get("speedup", "N/A")
                report_lines.append(
                    f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | {sp['time'] * 1000:.2f}ms | {speedup:.2f}x |"
                )
            elif st["success"]:
                report_lines.append(f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | N/A | N/A |")
        report_lines.append("")

    if qc_results.get("feature_qc_metrics"):
        report_lines.append("### Feature QC Metrics\n")
        report_lines.append("| Dataset | ScpTensor Time | Scanpy Time | Speedup |")
        report_lines.append("|---------|----------------|-------------|---------|")
        for idx, result in enumerate(qc_results["feature_qc_metrics"], 1):
            st = result["scptensor"]
            sp = result.get("scanpy", {})
            comp = result.get("comparison", {})
            if st["success"] and sp:
                speedup = comp.get("speedup", "N/A")
                report_lines.append(
                    f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | {sp['time'] * 1000:.2f}ms | {speedup:.2f}x |"
                )
            elif st["success"]:
                report_lines.append(f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | N/A | N/A |")
        report_lines.append("")

    if qc_results.get("outlier_detection"):
        report_lines.append("### Outlier Detection (ScpTensor-exclusive)\n")
        report_lines.append("| Dataset | Time | Outliers | Rate |")
        report_lines.append("|---------|------|----------|------|")
        for idx, result in enumerate(qc_results["outlier_detection"], 1):
            st = result["scptensor"]
            if st["success"]:
                n_out = st.get("n_outliers", "N/A")
                rate = st.get("outlier_rate", "N/A")
                report_lines.append(
                    f"| Dataset {idx} | {st['time'] * 1000:.2f}ms | {n_out} | {rate:.1%} |"
                )
        report_lines.append("")

    # Write report
    report_file = output_dir / "comparison_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {report_file}")


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare ScpTensor vs Scanpy for Normalization and QC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with 1 small dataset"
    )
    parser.add_argument("--full", action="store_true", help="Run full benchmark with 3 datasets")
    parser.add_argument(
        "--output-dir", type=str, default="qc_norm_comparison_results", help="Output directory"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("ScpTensor vs Scanpy: Normalization & QC Comparison")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Scanpy available: {SCANPY_AVAILABLE}")
    print(f"Mode: {'Quick' if args.quick else 'Full' if args.full else 'Default (1 dataset)'}")

    # Create datasets
    print("\nGenerating benchmark datasets...")
    datasets = create_benchmark_datasets(quick=args.quick)
    print(f"Created {len(datasets)} dataset(s)")

    # Run normalization comparison
    norm_results = run_normalization_comparison(datasets, output_dir)

    # Run QC comparison
    qc_results = run_qc_comparison(datasets, output_dir)

    # Generate summary report
    generate_summary_report(norm_results, qc_results, output_dir)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
