"""Generate test figures for all benchmark display modules.

This script creates sample data and generates figures for each display class
in the benchmark display module. Useful for testing visualization output
and creating example figures.

Run as: python -m scptensor.benchmark.display.generate_test_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt


def generate_mock_data(
    n_samples: int = 200,
    n_features: int = 200,
    missing_rate: float = 0.25,
    n_batches: int = 3,
    n_clusters: int = 4,
    random_seed: int = 42,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int]:
    """Generate mock data for testing display modules.

    Parameters
    ----------
    n_samples : int, default=200
        Number of samples to generate.
    n_features : int, default=200
        Number of features to generate.
    missing_rate : float, default=0.25
        Proportion of missing values (0.0 to 1.0).
    n_batches : int, default=3
        Number of batches to simulate.
    n_clusters : int, default=4
        Number of clusters to simulate.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int]
        Dictionary containing generated data arrays.
    """
    rng = np.random.default_rng(random_seed)

    # Generate base data with log-normal distribution (typical for SCP data)
    data = rng.lognormal(mean=2.0, sigma=0.8, size=(n_samples, n_features))

    # Add cluster structure
    cluster_means = rng.lognormal(mean=2.5, sigma=0.5, size=(n_clusters, n_features))
    cluster_labels = rng.integers(0, n_clusters, size=n_samples)
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        data[cluster_mask] += cluster_means[i] * 0.3

    # Add batch effects
    batch_labels = rng.integers(0, n_batches, size=n_samples)
    batch_effects = rng.lognormal(mean=0.0, sigma=0.15, size=(n_batches, n_features))
    for i in range(n_batches):
        batch_mask = batch_labels == i
        data[batch_mask] *= 1 + batch_effects[i] * 0.2

    # Create missing value mask
    mask = rng.random(data.shape) < missing_rate
    data_masked = data.copy()
    data_masked[mask] = np.nan

    # Generate UMAP-like 2D embedding (simulated with scaled PCA-like transform)
    # In real data this would be actual UMAP output
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    data_filled = np.nan_to_num(data_masked, nan=data.mean())
    umap_coords = pca.fit_transform(data_filled)

    # Generate PCA results
    pca_full = PCA(n_components=min(50, n_features, n_samples))
    pca_full.fit(data_filled)
    pca_components = pca_full.transform(data_filled)[:, :10]

    return {
        "data": data,
        "data_masked": data_masked,
        "data_filled": data_filled,
        "cluster_labels": cluster_labels,
        "batch_labels": batch_labels,
        "umap_coords": umap_coords,
        "pca_components": pca_components,
        "explained_variance": pca_full.explained_variance_,
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_batches": n_batches,
        "n_clusters": n_clusters,
        "rng": rng,
    }


def generate_normalization_figures(
    mock_data: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int],
    output_dir: Path,
) -> list[Path]:
    """Generate test figures for normalization display modules.

    Parameters
    ----------
    mock_data : dict
        Dictionary containing mock data arrays.
    output_dir : Path
        Output directory for figures.

    Returns
    -------
    list[Path]
        List of paths to generated figures.
    """
    from scptensor.benchmark.display.normalization import (
        LogNormalizeDisplay,
        NormalizationComparisonResult,
        ZScoreDisplay,
        ZScoreVerificationResult,
    )

    generated_paths: list[Path] = []

    # LogNormalizeDisplay
    print("Generating LogNormalizeDisplay figures...")
    log_display = LogNormalizeDisplay(output_dir=output_dir)

    # Create mock normalization result
    raw_data = mock_data["data"]
    log_normalized = np.log2(raw_data + 1)
    scanpy_normalized = log_normalized * 1.0  # Simulated slight difference

    log_result = NormalizationComparisonResult(
        raw_data=raw_data,
        scptensor_normalized=log_normalized,
        scanpy_normalized=scanpy_normalized,
        method_name="log_normalize",
        base=2.0,
        offset=1.0,
    )

    path = log_display.render_distribution_flow(log_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = log_display.render_agreement(log_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    # ZScoreDisplay
    print("Generating ZScoreDisplay figures...")
    zscore_display = ZScoreDisplay(output_dir=output_dir)

    # Create mock z-score result
    before_data = log_normalized
    after_data = (before_data - before_data.mean(axis=0, keepdims=True)) / (
        before_data.std(axis=0, ddof=1, keepdims=True) + 1e-8
    )

    zscore_result = ZScoreVerificationResult(
        before_data=before_data,
        after_data=after_data,
        has_missing=True,
        axis=0,
        ddof=1,
        mean=float(after_data.mean()),
        std=float(after_data.std()),
    )

    path = zscore_display.render_verification(zscore_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    return generated_paths


def generate_imputation_figures(
    mock_data: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int],
    output_dir: Path,
) -> list[Path]:
    """Generate test figures for imputation display modules.

    Parameters
    ----------
    mock_data : dict
        Dictionary containing mock data arrays.
    output_dir : Path
        Output directory for figures.

    Returns
    -------
    list[Path]
        List of paths to generated figures.
    """
    from scptensor.benchmark.display.imputation import (
        ExclusiveImputationResults,
        ExclusiveImputeDisplay,
        ImputationComparisonResult,
        KNNImputeDisplay,
    )

    generated_paths: list[Path] = []
    rng = mock_data["rng"]

    # KNNImputeDisplay
    print("Generating KNNImputeDisplay figures...")
    knn_display = KNNImputeDisplay(output_dir=output_dir)

    # Create mock imputation results for different missing rates
    missing_rates = [0.1, 0.2, 0.3, 0.4]
    knn_results = []

    for mr in missing_rates:
        ground_truth = mock_data["data"]
        # Simulate imputed data with noise
        imputed_scptensor = ground_truth + rng.normal(
            0, 0.05 * ground_truth.std(), ground_truth.shape
        )
        imputed_scanpy = ground_truth + rng.normal(0, 0.07 * ground_truth.std(), ground_truth.shape)

        # Add some missing values back
        mask = rng.random(ground_truth.shape) < mr
        imputed_scptensor[mask] = ground_truth[mask] + rng.normal(0, 0.1, size=np.sum(mask))
        imputed_scanpy[mask] = ground_truth[mask] + rng.normal(0, 0.12, size=np.sum(mask))

        # Calculate metrics
        mse_scptensor = float(np.mean((ground_truth - imputed_scptensor) ** 2))
        mae_scptensor = float(np.mean(np.abs(ground_truth - imputed_scptensor)))
        corr_scptensor = float(np.corrcoef(ground_truth.ravel(), imputed_scptensor.ravel())[0, 1])

        mse_scanpy = float(np.mean((ground_truth - imputed_scanpy) ** 2))
        mae_scanpy = float(np.mean(np.abs(ground_truth - imputed_scanpy)))
        corr_scanpy = float(np.corrcoef(ground_truth.ravel(), imputed_scanpy.ravel())[0, 1])

        result = ImputationComparisonResult(
            method_name="knn",
            missing_rate=mr,
            ground_truth=ground_truth,
            scptensor_imputed=imputed_scptensor,
            competitor_imputed=imputed_scanpy,
            framework="scanpy",
            mse=mse_scptensor,
            mae=mae_scptensor,
            correlation=corr_scptensor,
            runtime_seconds=1.5 + mr * 2,
            competitor_mse=mse_scanpy,
            competitor_mae=mae_scanpy,
            competitor_correlation=corr_scanpy,
            competitor_runtime_seconds=2.0 + mr * 2.5,
        )
        knn_results.append(result)

    path = knn_display.render_accuracy_table(knn_results)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = knn_display.render_performance_comparison(knn_results)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    # ExclusiveImputeDisplay
    print("Generating ExclusiveImputeDisplay figures...")
    exclusive_display = ExclusiveImputeDisplay(output_dir=output_dir)

    methods = ["ppca", "svd", "bpca", "mf"]
    n_methods = len(methods)
    n_rates = len(missing_rates)

    # Generate mock performance matrices
    mse_matrix = rng.uniform(0.02, 0.15, size=(n_methods, n_rates))
    mae_matrix = rng.uniform(0.1, 0.3, size=(n_methods, n_rates))
    corr_matrix = 1.0 - rng.uniform(0.01, 0.15, size=(n_methods, n_rates))
    runtime_matrix = rng.uniform(1.0, 10.0, size=(n_methods, n_rates))

    exclusive_results = ExclusiveImputationResults(
        methods=methods,
        missing_rates=missing_rates,
        mse_matrix=mse_matrix,
        mae_matrix=mae_matrix,
        correlation_matrix=corr_matrix,
        runtime_matrix=runtime_matrix,
        baseline_method="knn",
        baseline_mse=[0.08, 0.10, 0.12, 0.15],
    )

    path = exclusive_display.render_mse_heatmap(exclusive_results, metric="mse")
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = exclusive_display.render_performance_advantage(exclusive_results)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = exclusive_display.render_missing_rate_response(exclusive_results, metric="mse")
    generated_paths.append(path)
    print(f"  Generated: {path}")

    return generated_paths


def generate_integration_figures(
    mock_data: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int],
    output_dir: Path,
) -> list[Path]:
    """Generate test figures for integration display modules.

    Parameters
    ----------
    mock_data : dict
        Dictionary containing mock data arrays.
    output_dir : Path
        Output directory for figures.

    Returns
    -------
    list[Path]
        List of paths to generated figures.
    """
    from scptensor.benchmark.display.integration import (
        IntegrationComparisonResult,
        IntegrationDisplay,
        IntegrationMetricsSummary,
    )

    generated_paths: list[Path] = []

    print("Generating IntegrationDisplay figures...")
    integration_display = IntegrationDisplay(output_dir=output_dir)

    batch_labels = mock_data["batch_labels"]
    umap_before = mock_data["umap_coords"]

    # Simulate batch-corrected UMAP (more mixed batches)
    rng = mock_data["rng"]
    umap_after = umap_before + rng.normal(0, 0.5, size=umap_before.shape)

    # Generate results for different methods
    methods = ["combat", "harmony", "mnn", "scanorama"]
    integration_results = []

    for method in methods:
        # Add method-specific variation to UMAP
        method_umap = umap_after + rng.normal(0, 0.3, size=umap_after.shape)

        result = IntegrationComparisonResult(
            method_name=method,
            framework="scptensor",
            batch_labels=batch_labels,
            umap_before=umap_before,
            umap_after=method_umap,
            kbet_score=float(rng.uniform(0.6, 0.9)),
            ilisi_score=float(rng.uniform(0.5, 0.8)),
            clisi_score=float(rng.uniform(0.7, 0.95)),
            asw_score=float(rng.uniform(0.1, 0.4)),
            runtime_seconds=float(rng.uniform(1.0, 5.0)),
            n_batches=mock_data["n_batches"],
            n_clusters=mock_data["n_clusters"],
        )
        integration_results.append(result)

    # Generate batch effect removal figure for first method
    path = integration_display.render_batch_effect_removal(integration_results[0])
    generated_paths.append(path)
    print(f"  Generated: {path}")

    # Generate method comparison grid
    # Also add scanpy results
    for method in methods:
        scanpy_umap = umap_after + rng.normal(0, 0.35, size=umap_after.shape)
        result = IntegrationComparisonResult(
            method_name=method,
            framework="scanpy",
            batch_labels=batch_labels,
            umap_before=umap_before,
            umap_after=scanpy_umap,
            kbet_score=float(rng.uniform(0.55, 0.85)),
            ilisi_score=float(rng.uniform(0.45, 0.75)),
            clisi_score=float(rng.uniform(0.65, 0.9)),
            asw_score=float(rng.uniform(0.15, 0.45)),
            runtime_seconds=float(rng.uniform(1.5, 6.0)),
        )
        integration_results.append(result)

    path = integration_display.render_method_comparison(
        integration_results, methods=("combat", "harmony")
    )
    generated_paths.append(path)
    print(f"  Generated: {path}")

    # Generate metrics summary
    metrics_summary = IntegrationMetricsSummary(
        methods=["ComBat", "Harmony", "MNN", "Scanorama"],
        frameworks=["scptensor"] * 4,
        kbet_scores=np.array([0.85, 0.82, 0.75, 0.78]),
        ilisi_scores=np.array([0.72, 0.68, 0.65, 0.70]),
        clisi_scores=np.array([0.88, 0.85, 0.80, 0.82]),
        asw_scores=np.array([0.15, 0.18, 0.25, 0.20]),
        runtimes=np.array([2.5, 3.8, 4.2, 3.5]),
    )

    path = integration_display.render_biological_metrics_radar(metrics_summary)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = integration_display.render_conservatism_analysis(integration_results[:4])
    generated_paths.append(path)
    print(f"  Generated: {path}")

    return generated_paths


def generate_dim_reduction_figures(
    mock_data: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int],
    output_dir: Path,
) -> list[Path]:
    """Generate test figures for dimensionality reduction display modules.

    Parameters
    ----------
    mock_data : dict
        Dictionary containing mock data arrays.
    output_dir : Path
        Output directory for figures.

    Returns
    -------
    list[Path]
        List of paths to generated figures.
    """
    from scptensor.benchmark.display.dim_reduction import (
        PCADisplay,
        PCAResult,
        UMAPDisplay,
        UMAPResult,
    )

    generated_paths: list[Path] = []
    rng = mock_data["rng"]

    # PCADisplay
    print("Generating PCADisplay figures...")
    pca_display = PCADisplay(output_dir=output_dir)

    # Create PCA result for ScpTensor
    scptensor_pca = PCAResult(
        framework="scptensor",
        components=mock_data["pca_components"],
        explained_variance=mock_data["explained_variance"][:10],
        explained_variance_ratio=mock_data["explained_variance_ratio"][:10],
        n_components=10,
        loadings=rng.normal(0, 0.3, size=(mock_data["n_features"], 10)),
        feature_names=[f"Protein_{i}" for i in range(mock_data["n_features"])],
        runtime_seconds=0.5,
    )

    path = pca_display.render_variance_explained(scptensor_pca)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = pca_display.render_per_component_variance(scptensor_pca)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    # UMAPDisplay
    print("Generating UMAPDisplay figures...")
    umap_display = UMAPDisplay(output_dir=output_dir)

    cluster_labels = mock_data["cluster_labels"]

    # Create UMAP results for both frameworks
    scptensor_umap = UMAPResult(
        framework="scptensor",
        embedding=mock_data["umap_coords"],
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        runtime_seconds=1.2,
    )

    # Simulated scanpy UMAP (slightly different)
    scanpy_umap_coords = mock_data["umap_coords"] + rng.normal(
        0, 0.3, size=mock_data["umap_coords"].shape
    )
    scanpy_umap = UMAPResult(
        framework="scanpy",
        embedding=scanpy_umap_coords,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        runtime_seconds=1.5,
    )

    path = umap_display.render_embedding_comparison(
        scptensor_umap, scanpy_umap, cluster_labels=cluster_labels
    )
    generated_paths.append(path)
    print(f"  Generated: {path}")

    return generated_paths


def generate_qc_figures(
    mock_data: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int],
    output_dir: Path,
) -> list[Path]:
    """Generate test figures for QC display modules.

    Parameters
    ----------
    mock_data : dict
        Dictionary containing mock data arrays.
    output_dir : Path
        Output directory for figures.

    Returns
    -------
    list[Path]
        List of paths to generated figures.
    """
    from scptensor.benchmark.display.qc import (
        BatchCVReport,
        MissingTypeDisplay,
        MissingTypeReport,
        QCBatchDisplay,
        QCComparisonResult,
        QCDashboardDisplay,
    )

    generated_paths: list[Path] = []
    rng = mock_data["rng"]
    n_samples = mock_data["n_samples"]
    n_features = mock_data["n_features"]

    # QCDashboardDisplay
    print("Generating QCDashboardDisplay figures...")
    qc_display = QCDashboardDisplay(output_dir=output_dir)

    sample_metrics = {
        "n_detected": rng.integers(100, n_features, size=n_samples).astype(float),
        "total_intensity": rng.lognormal(14, 1, size=n_samples),
        "missing_rate": rng.uniform(0.1, 0.5, size=n_samples),
    }

    feature_metrics = {
        "cv": rng.uniform(0.1, 0.8, size=n_features),
        "missing_rate": rng.uniform(0.05, 0.9, size=n_features),
        "prevalence": rng.uniform(0.1, 1.0, size=n_features),
    }

    qc_result = QCComparisonResult(
        sample_metrics=sample_metrics,
        feature_metrics=feature_metrics,
        batch_labels=mock_data["batch_labels"],
        n_samples=n_samples,
        n_features=n_features,
        framework="scptensor",
        cells_removed=5,
        features_removed=10,
    )

    path = qc_display.render_dashboard(qc_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = qc_display.render_sample_feature_heatmap(qc_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    # MissingTypeDisplay
    print("Generating MissingTypeDisplay figures...")
    missing_type_display = MissingTypeDisplay(output_dir=output_dir)

    missing_report = MissingTypeReport(
        valid_rate=0.70,
        mbr_rate=0.15,
        lod_rate=0.10,
        filtered_rate=0.05,
        imputed_rate=0.0,
        feature_missing_rates=rng.uniform(0.05, 0.8, size=n_features),
        sample_missing_rates=rng.uniform(0.1, 0.5, size=n_samples),
        mbr_by_feature=rng.uniform(0.02, 0.3, size=n_features),
        lod_by_feature=rng.uniform(0.01, 0.2, size=n_features),
    )

    path = missing_type_display.render_missing_type_distribution(missing_report)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = missing_type_display.render_missing_rate_by_type(missing_report)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = missing_type_display.render_feature_missing_scatter(missing_report)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    # QCBatchDisplay
    print("Generating QCBatchDisplay figures...")
    batch_display = QCBatchDisplay(output_dir=output_dir)

    batch_cv_report = BatchCVReport(
        within_batch_cv={"batch1": 0.25, "batch2": 0.28, "batch3": 0.30},
        between_batch_cv=0.35,
        cv_by_batch_feature=rng.uniform(0.1, 0.6, size=(3, n_features)),
        batch_names=["batch1", "batch2", "batch3"],
        high_cv_features=list(range(20)),
    )

    path = batch_display.render_batch_pca(mock_data["umap_coords"], mock_data["batch_labels"])
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = batch_display.render_batch_cv_comparison(batch_cv_report)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = batch_display.render_batch_cv_heatmap(batch_cv_report)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    return generated_paths


def generate_end_to_end_figures(
    mock_data: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int_] | int],
    output_dir: Path,
) -> list[Path]:
    """Generate test figures for end-to-end display modules.

    Parameters
    ----------
    mock_data : dict
        Dictionary containing mock data arrays.
    output_dir : Path
        Output directory for figures.

    Returns
    -------
    list[Path]
        List of paths to generated figures.
    """
    from scptensor.benchmark.display.end_to_end import (
        ClusteringMetrics,
        EndToEndDisplay,
        IntermediateResults,
        PipelineResult,
        PipelineStep,
    )

    generated_paths: list[Path] = []
    rng = mock_data["rng"]

    print("Generating EndToEndDisplay figures...")
    end_to_end_display = EndToEndDisplay(output_dir=output_dir)

    # Create pipeline steps
    pipeline_steps = [
        PipelineStep(
            name="qc",
            display_name="Quality Control",
            method_name="basic_qc",
            parameters={"min_features": 50, "max_missing_rate": 0.8},
            runtime_seconds=0.5,
        ),
        PipelineStep(
            name="normalization",
            display_name="Normalization",
            method_name="log_normalize",
            parameters={"base": 2.0, "offset": 1.0},
            runtime_seconds=0.3,
        ),
        PipelineStep(
            name="imputation",
            display_name="Imputation",
            method_name="knn_impute",
            parameters={"n_neighbors": 5},
            runtime_seconds=2.5,
        ),
        PipelineStep(
            name="integration",
            display_name="Batch Correction",
            method_name="combat",
            parameters={"batch_column": "batch"},
            runtime_seconds=1.8,
        ),
        PipelineStep(
            name="dim_reduction",
            display_name="Dimensionality Reduction",
            method_name="pca",
            parameters={"n_components": 10},
            runtime_seconds=0.8,
        ),
        PipelineStep(
            name="clustering",
            display_name="Clustering",
            method_name="kmeans",
            parameters={"n_clusters": 4},
            runtime_seconds=1.2,
        ),
    ]

    # Create intermediate results
    intermediate_results = [
        IntermediateResults(
            step_name="qc",
            n_cells=195,
            n_features=190,
            sparsity=0.28,
            total_runtime=0.5,
        ),
        IntermediateResults(
            step_name="normalization",
            n_cells=195,
            n_features=190,
            sparsity=0.28,
            total_runtime=0.8,
        ),
        IntermediateResults(
            step_name="imputation",
            n_cells=195,
            n_features=190,
            sparsity=0.0,
            total_runtime=3.3,
        ),
        IntermediateResults(
            step_name="integration",
            n_cells=195,
            n_features=190,
            sparsity=0.0,
            total_runtime=5.1,
        ),
        IntermediateResults(
            step_name="clustering",
            n_cells=195,
            n_features=190,
            sparsity=0.0,
            total_runtime=7.1,
        ),
    ]

    # Create clustering metrics
    clustering_metrics = ClusteringMetrics(
        silhouette_score=float(rng.uniform(0.4, 0.7)),
        davies_bouldin_score=float(rng.uniform(0.5, 1.2)),
        calinski_harabasz_score=float(rng.uniform(100, 500)),
        ari_score=float(rng.uniform(0.6, 0.9)),
        nmi_score=float(rng.uniform(0.7, 0.95)),
        n_clusters=4,
    )

    # Create ScpTensor pipeline result
    scptensor_result = PipelineResult(
        framework="scptensor",
        pipeline_steps=pipeline_steps,
        umap_embedding=mock_data["umap_coords"],
        cluster_labels=mock_data["cluster_labels"],
        clustering_metrics=clustering_metrics,
        intermediate_results=intermediate_results,
        total_runtime=7.1,
        total_memory_mb=250.0,
        dataset_name="test_data",
    )

    # Create competitor pipeline result (similar but slightly different)
    competitor_umap = mock_data["umap_coords"] + rng.normal(
        0, 0.5, size=mock_data["umap_coords"].shape
    )
    competitor_cluster_labels = mock_data["cluster_labels"]
    # Shuffle some cluster labels to create difference
    shuffle_indices = rng.choice(len(competitor_cluster_labels), size=20, replace=False)
    competitor_cluster_labels[shuffle_indices] = rng.integers(0, 4, size=20)

    competitor_clustering_metrics = ClusteringMetrics(
        silhouette_score=float(rng.uniform(0.35, 0.65)),
        davies_bouldin_score=float(rng.uniform(0.6, 1.4)),
        calinski_harabasz_score=float(rng.uniform(80, 450)),
        ari_score=float(rng.uniform(0.5, 0.85)),
        nmi_score=float(rng.uniform(0.6, 0.9)),
        n_clusters=4,
    )

    competitor_result = PipelineResult(
        framework="scanpy",
        pipeline_steps=pipeline_steps,
        umap_embedding=competitor_umap,
        cluster_labels=competitor_cluster_labels,
        clustering_metrics=competitor_clustering_metrics,
        intermediate_results=intermediate_results,
        total_runtime=8.5,
        total_memory_mb=300.0,
        dataset_name="test_data",
    )

    # Generate figures
    path = end_to_end_display.render_pipeline_comparison(scptensor_result, competitor_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = end_to_end_display.render_intermediate_comparison(scptensor_result, competitor_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = end_to_end_display.render_clustering_comparison(scptensor_result, competitor_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = end_to_end_display.render_cluster_overlap_analysis(scptensor_result, competitor_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    path = end_to_end_display.render_quality_metrics_comparison(scptensor_result, competitor_result)
    generated_paths.append(path)
    print(f"  Generated: {path}")

    return generated_paths


def generate_all_figures(
    output_dir: str | Path = "benchmark_results",
    modules: list[str] | None = None,
    n_samples: int = 200,
    n_features: int = 200,
    missing_rate: float = 0.25,
    n_batches: int = 3,
    n_clusters: int = 4,
    dpi: int = 300,
) -> dict[str, list[Path]]:
    """Generate test figures for all display modules.

    Parameters
    ----------
    output_dir : str | Path, default="benchmark_results"
        Output directory for figures.
    modules : list[str] | None, default=None
        List of modules to generate figures for. If None, generates all.
        Options: "normalization", "imputation", "integration", "dim_reduction",
        "qc", "end_to_end".
    n_samples : int, default=200
        Number of samples to generate.
    n_features : int, default=200
        Number of features to generate.
    missing_rate : float, default=0.25
        Proportion of missing values (0.0 to 1.0).
    n_batches : int, default=3
        Number of batches to simulate.
    n_clusters : int, default=4
        Number of clusters to simulate.
    dpi : int, default=300
        DPI for saved figures.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary mapping module names to lists of generated figure paths.
    """
    output_path = Path(output_dir)
    # The display classes append 'figures/' to the output_dir, so we should
    # pass the parent directory directly
    if output_path.name == "figures":
        output_path = output_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup plot style
    try:
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["axes.unicode_minus"] = False

    print("Generating mock data...")
    mock_data = generate_mock_data(
        n_samples=n_samples,
        n_features=n_features,
        missing_rate=missing_rate,
        n_batches=n_batches,
        n_clusters=n_clusters,
    )
    print(f"  Data shape: {mock_data['n_samples']} samples x {mock_data['n_features']} features")
    print(f"  Batches: {mock_data['n_batches']}, Clusters: {mock_data['n_clusters']}")

    all_modules = {
        "normalization": generate_normalization_figures,
        "imputation": generate_imputation_figures,
        "integration": generate_integration_figures,
        "dim_reduction": generate_dim_reduction_figures,
        "qc": generate_qc_figures,
        "end_to_end": generate_end_to_end_figures,
    }

    if modules is None:
        modules = list(all_modules.keys())

    results: dict[str, list[Path]] = {}

    for module_name in modules:
        if module_name not in all_modules:
            print(f"Warning: Unknown module '{module_name}', skipping...")
            continue

        print(f"\n{'=' * 60}")
        print(f"Generating figures for: {module_name}")
        print("=" * 60)

        try:
            paths = all_modules[module_name](mock_data, output_path)
            results[module_name] = paths
            print(f"Generated {len(paths)} figures for {module_name}")
        except Exception as e:
            print(f"Error generating figures for {module_name}: {e}")
            results[module_name] = []

    return results


def main() -> None:
    """Main entry point for the test figure generation script."""
    parser = argparse.ArgumentParser(
        description="Generate test figures for benchmark display modules."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        choices=[
            "normalization",
            "imputation",
            "integration",
            "dim_reduction",
            "qc",
            "end_to_end",
            "all",
        ],
        default=["all"],
        help="Modules to generate figures for (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of samples to generate (default: 200)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=200,
        help="Number of features to generate (default: 200)",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.25,
        help="Proportion of missing values (default: 0.25)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=3,
        help="Number of batches (default: 3)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of clusters (default: 4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all generated figure paths after completion",
    )

    args = parser.parse_args()

    # Handle "all" option
    if "all" in args.modules:
        modules = None
    else:
        modules = args.modules

    print("=" * 60)
    print("ScpTensor Benchmark Display - Test Figure Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Modules: {', '.join(modules) if modules else 'all'}")
    print(f"Samples: {args.n_samples}, Features: {args.n_features}")
    print(f"Missing rate: {args.missing_rate:.1%}")
    print(f"Batches: {args.n_batches}, Clusters: {args.n_clusters}")
    print("=" * 60)

    results = generate_all_figures(
        output_dir=args.output_dir,
        modules=modules,
        n_samples=args.n_samples,
        n_features=args.n_features,
        missing_rate=args.missing_rate,
        n_batches=args.n_batches,
        n_clusters=args.n_clusters,
        dpi=args.dpi,
    )

    # Print summary
    total_figures = sum(len(paths) for paths in results.values())
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total figures generated: {total_figures}")
    for module_name, paths in results.items():
        print(f"  {module_name}: {len(paths)} figures")

    if args.list:
        print("\n" + "=" * 60)
        print("GENERATED FIGURES")
        print("=" * 60)
        for module_name, paths in results.items():
            print(f"\n{module_name}:")
            for path in paths:
                print(f"  {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
