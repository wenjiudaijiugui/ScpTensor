"""Generate all comparison visualizations from benchmark results.

Run with: python -m scptensor.benchmark.generate_all_plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scptensor.benchmark.comparison_engine import ComparisonResult, MethodResult
from scptensor.benchmark.comparison_viz import ComparisonVisualizer, PlotStyle, get_visualizer
from scptensor.benchmark.data_provider import COMPARISON_DATASETS, get_provider
from scptensor.benchmark.comparison_engine import get_engine


def generate_all_plots(
    results_dir: str | Path = "benchmark_results",
    style: PlotStyle = PlotStyle.SCIENCE,
) -> None:
    results_dir = Path(results_dir)
    results_file = results_dir / "comparison_results.json"

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run benchmark first:")
        print("  python -m scptensor.benchmark.run_scanpy_comparison")
        return

    with open(results_file) as f:
        data = json.load(f)

    results_raw = data.get("results", {})
    results = {}

    for method_name, result_list_raw in results_raw.items():
        converted_list = []
        for r_dict in result_list_raw:
            st_dict = r_dict.get("scptensor_result")
            sp_dict = r_dict.get("scanpy_result")

            converted = ComparisonResult(
                comparison_name=r_dict.get("comparison_name", method_name),
                dataset_name=r_dict.get("dataset_name", ""),
                scptensor_result=MethodResult(**st_dict) if st_dict else None,
                scanpy_result=MethodResult(**sp_dict) if sp_dict else None,
                comparison_metrics=r_dict.get("comparison_metrics", {}),
            )
            converted_list.append(converted)
        results[method_name] = converted_list

    viz = get_visualizer(output_dir=str(results_dir / "figures"), style=style)
    fig_dir = results_dir / "figures"

    print("=== Generating Comparison Plots ===\n")

    print("1. Performance Charts")
    try:
        viz.plot_runtime_comparison(results)
        print("  Runtime comparison")
    except Exception:
        pass

    try:
        viz.plot_speedup_heatmap(results)
        print("  Speedup heatmap")
    except Exception:
        pass

    try:
        _plot_memory_comparison(results, fig_dir / "01_performance")
        print("  Memory comparison")
    except Exception:
        pass

    print("\n2. Normalization Charts")
    for method in ("log_normalize", "z_score_normalize"):
        if method not in results or not results[method]:
            continue

        result = results[method][0]
        if result.scptensor_result and result.scptensor_result.output and result.scanpy_result and result.scanpy_result.output:
            st_out = np.array(result.scptensor_result.output)
            sp_out = np.array(result.scanpy_result.output)

            if st_out.shape == sp_out.shape:
                try:
                    viz.plot_accuracy_scatter(st_out, sp_out, method)
                    print(f"  {method} scatter")
                except Exception:
                    pass

                try:
                    _plot_bland_altman(st_out, sp_out, method, fig_dir / "02_normalization")
                    print(f"  {method} Bland-Altman")
                except Exception:
                    pass

    print("\n3. Imputation Charts")
    if "knn_impute" in results and results["knn_impute"]:
        result = results["knn_impute"][min(1, len(results["knn_impute"]) - 1)]

        if result.scptensor_result and result.scptensor_result.output:
            provider = get_provider()
            dataset = COMPARISON_DATASETS[1]
            X, M, _, _ = provider.get_dataset(dataset)

            if M is None or M.sum() == 0:
                print("  knn_impute analysis: no missing values in test data")
            else:
                try:
                    _plot_imputation_analysis(X, M, np.array(result.scptensor_result.output), fig_dir / "03_imputation")
                    print("  knn_impute analysis")
                except Exception:
                    pass

    print("\n4. Dimensionality Reduction Charts")
    if "pca" in results and results["pca"]:
        result = results["pca"][0]
        if result.scptensor_result and result.scanpy_result:
            st_m = result.scptensor_result.metrics or {}
            sp_m = result.scanpy_result.metrics or {}
            if "variance_ratio" in st_m and "variance_ratio" in sp_m:
                try:
                    viz.plot_pca_variance(st_m["variance_ratio"], sp_m["variance_ratio"])
                    print("  PCA variance comparison")
                except Exception:
                    pass

    if "umap" in results and results["umap"]:
        result = results["umap"][0]
        if result.scptensor_result and result.scanpy_result:
            st_umap = result.scptensor_result.output
            sp_umap = result.scanpy_result.output
            if isinstance(st_umap, np.ndarray) and isinstance(sp_umap, np.ndarray) and st_umap.shape == sp_umap.shape and st_umap.shape[1] == 2:
                try:
                    provider = get_provider()
                    dataset = COMPARISON_DATASETS[1]
                    _, _, _, groups = provider.get_dataset(dataset)
                    group_map = {g: i for i, g in enumerate(np.unique(groups))}
                    labels = np.array([group_map[g] for g in groups])

                    _plot_umap_comparison(st_umap, sp_umap, labels, fig_dir / "04_dim_reduction")
                    print("  UMAP embedding comparison")
                except Exception:
                    pass

    print("\n5. Clustering Charts")
    if "kmeans" in results and results["kmeans"]:
        result = results["kmeans"][0]
        if result.scptensor_result and result.scanpy_result:
            try:
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

                st_labels = result.scptensor_result.output
                sp_labels = result.scanpy_result.output

                ari = adjusted_rand_score(st_labels, sp_labels)
                nmi = normalized_mutual_info_score(st_labels, sp_labels)

                viz.plot_clustering_consistency(st_labels, sp_labels, ari, nmi)
                print("  Kmeans consistency heatmap")
            except Exception:
                pass

    print("\n6. Feature Selection Charts")
    if "hvg" in results and results["hvg"]:
        result = results["hvg"][0]
        if result.scptensor_result and result.scanpy_result:
            try:
                provider = get_provider()
                dataset = COMPARISON_DATASETS[1]
                X, _, _, _ = provider.get_dataset(dataset)

                st_hvg = set(result.scptensor_result.output)
                sp_hvg = set(result.scanpy_result.output)

                _plot_hvg_venn(st_hvg, sp_hvg, X.shape[1], fig_dir / "06_feature_selection")
                print("  HVG Venn diagram")
            except Exception:
                pass

    print("\n7. Summary Charts")
    try:
        _plot_comprehensive_summary(results, fig_dir / "summary")
        print("  Comprehensive summary")
    except Exception:
        pass

    try:
        _plot_summary_radar(results, fig_dir / "summary")
        print("  Summary radar chart")
    except Exception:
        pass

    print("\n=== Plot Generation Complete ===")

    print("\nGenerated files:")
    for png_file in sorted(results_dir.rglob("*.png")):
        print(f"  {png_file.relative_to(results_dir)}")


def _plot_memory_comparison(results: dict, output_dir: Path) -> None:
    methods, scptensor_mem, scanpy_mem = [], [], []

    for method, result_list in results.items():
        if not result_list:
            continue

        st_results = [r.scptensor_result for r in result_list if r.scptensor_result and r.scptensor_result.success]
        sp_results = [r.scanpy_result for r in result_list if r.scanpy_result and r.scanpy_result.success]

        st_mem = float(np.mean([r.memory_mb for r in st_results])) if st_results else 0
        sp_mem = float(np.mean([r.memory_mb for r in sp_results])) if sp_results else 0

        if st_mem > 0 or sp_mem > 0:
            methods.append(method.replace("_", " ").title())
            scptensor_mem.append(st_mem)
            scanpy_mem.append(sp_mem)

    if methods:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width / 2, scptensor_mem, width, label="ScpTensor", color="#2E86AB")
        ax.bar(x + width / 2, scanpy_mem, width, label="Scanpy", color="#A23B72")

        ax.set_xlabel("Method")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("Memory Usage Comparison: ScpTensor vs Scanpy")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "memory_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def _plot_bland_altman(scptensor_output: np.ndarray, scanpy_output: np.ndarray, method_name: str, output_dir: Path) -> None:
    st_flat = scptensor_output.ravel()
    sp_flat = scanpy_output.ravel()

    if len(st_flat) > 10000:
        idx = np.random.choice(len(st_flat), 10000, replace=False)
        st_flat, sp_flat = st_flat[idx], sp_flat[idx]

    mean = (st_flat + sp_flat) / 2
    diff = st_flat - sp_flat

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean, diff, alpha=0.5, s=10, color="#2E86AB")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)

    bias, std_diff = np.mean(diff), np.std(diff)
    ax.axhline(y=bias + 1.96 * std_diff, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=bias - 1.96 * std_diff, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Mean of ScpTensor and Scanpy")
    ax.set_ylabel("Difference (ScpTensor - Scanpy)")
    ax.set_title(f"{method_name.replace('_', ' ').title()}: Bland-Altman Plot")
    ax.grid(alpha=0.3)

    ax.text(0.05, 0.95, f"Bias: {bias:.4f}\nLoA: [{bias + 1.96 * std_diff:.4f}, {bias - 1.96 * std_diff:.4f}]",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / f"{method_name}_bland_altman.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_imputation_analysis(X: np.ndarray, M: np.ndarray, imputed: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_flat = X.ravel()
    imp_flat = imputed.ravel()
    valid_mask = M.ravel() == 0
    x_flat, imp_flat = x_flat[valid_mask], imp_flat[valid_mask]

    if len(x_flat) > 10000:
        idx = np.random.choice(len(x_flat), 10000, replace=False)
        x_flat, imp_flat = x_flat[idx], imp_flat[idx]

    axes[0].scatter(x_flat, imp_flat, alpha=0.3, s=5, color="#2E86AB")
    min_val, max_val = min(x_flat.min(), imp_flat.min()), max(x_flat.max(), imp_flat.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
    axes[0].set_xlabel("Original Value")
    axes[0].set_ylabel("Imputed Value")
    axes[0].set_title("KNN Impute: Original vs Imputed")
    axes[0].grid(alpha=0.3)

    diff = imp_flat - x_flat
    axes[1].hist(diff, bins=50, color="#A23B72", alpha=0.7, edgecolor="black")
    axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Imputed - Original")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Imputation Error Distribution")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "knn_impute_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_umap_comparison(scptensor_umap: np.ndarray, scanpy_umap: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    color_map = {l: colors[i] for i, l in enumerate(unique_labels)}
    point_colors = [color_map[l] for l in labels]

    axes[0].scatter(scptensor_umap[:, 0], scptensor_umap[:, 1], c=point_colors, s=15, alpha=0.6, edgecolors="none")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title("ScpTensor UMAP")

    axes[1].scatter(scanpy_umap[:, 0], scanpy_umap[:, 1], c=point_colors, s=15, alpha=0.6, edgecolors="none")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title("Scanpy UMAP")

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "umap_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_hvg_venn(scptensor_hvg: set, scanpy_hvg: set, total_features: int, output_dir: Path) -> None:
    try:
        from matplotlib_venn import venn2
    except ImportError:
        overlap = len(scptensor_hvg & scanpy_hvg)
        only_scptensor = len(scptensor_hvg - scanpy_hvg)
        only_scanpy = len(scanpy_hvg - scptensor_hvg)

        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["Only ScpTensor", "Only Scanpy", "Overlap"]
        values = [only_scptensor, only_scanpy, overlap]
        ax.bar(categories, values, color=["#2E86AB", "#A23B72", "#6B8E23"])
        ax.set_ylabel("Number of Features")
        ax.set_title("Highly Variable Genes Overlap")
        ax.grid(axis="y", alpha=0.3)

        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

        plt.tight_layout()
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "hvg_venn_alternative.png", dpi=300, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    venn2([set(scptensor_hvg), set(scanpy_hvg)], set_labels=("ScpTensor", "Scanpy"), ax=ax)
    ax.set_title(f"Highly Variable Genes Overlap (Total: {total_features})")

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "hvg_venn.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_comprehensive_summary(results: dict, output_dir: Path) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    methods, valid_methods = list(results.keys()), []
    st_times, sp_times, speedups, corrs, st_mem, sp_mem, maes = [], [], [], [], [], [], []

    for m in methods:
        st_t = np.mean([r.scptensor_result.runtime_seconds for r in results[m] if r.scptensor_result and r.scptensor_result.success]) if any(
            r.scptensor_result and r.scptensor_result.success for r in results[m]) else 0
        sp_t = np.mean([r.scanpy_result.runtime_seconds for r in results[m] if r.scanpy_result and r.scanpy_result.success]) if any(
            r.scanpy_result and r.scanpy_result.success for r in results[m]) else 0

        s = np.mean([r.comparison_metrics.get("speedup", 1.0) for r in results[m] if "speedup" in r.comparison_metrics])
        c = np.mean([r.comparison_metrics.get("correlation", 0) for r in results[m] if "correlation" in r.comparison_metrics])

        sm = np.mean([r.scptensor_result.memory_mb for r in results[m] if r.scptensor_result and r.scptensor_result.success]) if any(
            r.scptensor_result and r.scptensor_result.success for r in results[m]) else 0
        pm = np.mean([r.scanpy_result.memory_mb for r in results[m] if r.scanpy_result and r.scanpy_result.success]) if any(
            r.scanpy_result and r.scanpy_result.success for r in results[m]) else 0

        mse = np.mean([r.comparison_metrics.get("mse", 0) for r in results[m] if "mse" in r.comparison_metrics])
        mae = np.sqrt(mse) if mse > 0 else 0

        valid_methods.append(m)
        st_times.append(st_t)
        sp_times.append(sp_t)
        speedups.append(s)
        corrs.append(max(c, 0))
        st_mem.append(sm)
        sp_mem.append(pm)
        maes.append(mae)

    if not valid_methods:
        return

    x = np.arange(len(valid_methods))
    width = 0.35

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - width / 2, st_times, width, label="ScpTensor", color="#2E86AB")
    ax1.bar(x + width / 2, sp_times, width, label="Scanpy", color="#A23B72")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", "\n") for m in valid_methods], fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    colors_s = ["#6B8E23" if s > 1 else "#E76F51" for s in speedups]
    ax2.bar(range(len(speedups)), speedups, color=colors_s)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Speedup (ScpTensor/Scanpy)")
    ax2.set_title("Performance Speedup")
    ax2.set_xticks(range(len(valid_methods)))
    ax2.set_xticklabels([m.replace("_", "\n") for m in valid_methods], fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(len(corrs)), corrs, color="#2E86AB")
    ax3.set_ylabel("Correlation")
    ax3.set_title("Output Correlation")
    ax3.set_xticks(range(len(valid_methods)))
    ax3.set_xticklabels([m.replace("_", "\n") for m in valid_methods], fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x - width / 2, st_mem, width, label="ScpTensor", color="#2E86AB")
    ax4.bar(x + width / 2, sp_mem, width, label="Scanpy", color="#A23B72")
    ax4.set_ylabel("Memory (MB)")
    ax4.set_title("Memory Usage")
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace("_", "\n") for m in valid_methods], fontsize=8)
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(range(len(maes)), maes, color="#6B8E23")
    ax5.set_ylabel("RMSE")
    ax5.set_title("Output Difference (RMSE)")
    ax5.set_xticks(range(len(valid_methods)))
    ax5.set_xticklabels([m.replace("_", "\n") for m in valid_methods], fontsize=8)
    ax5.grid(axis="y", alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    avg_speedup = np.mean([s for s in speedups if s > 0])
    avg_corr = np.mean([c for c in corrs if c > 0])
    faster = sum(1 for s in speedups if s > 1.1)
    high_corr = sum(1 for c in corrs if c > 0.95)

    summary_text = (
        f"Summary Statistics\n\n"
        f"Methods compared: {len(valid_methods)}\n"
        f"Avg speedup: {avg_speedup:.2f}x\n"
        f"Avg correlation: {avg_corr:.4f}\n\n"
        f"Performance:\n"
        f"  ScpTensor faster: {faster}/{len(speedups)}\n\n"
        f"Accuracy:\n"
        f"  High corr (>0.95): {high_corr}/{len(corrs)}"
    )

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.suptitle("Comprehensive Benchmark Summary", fontsize=14, fontweight="bold")

    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "comprehensive_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_summary_radar(results: dict, output_dir: Path) -> None:
    categories = {"Performance": "speedup", "Accuracy": "correlation", "Efficiency": "memory_ratio"}
    method_scores = {}

    for method, result_list in results.items():
        scores = []
        for cat, key in categories.items():
            val = 1.0
            for r in result_list:
                if key == "speedup":
                    val = r.comparison_metrics.get("speedup", 1.0)
                elif key == "correlation":
                    val = r.comparison_metrics.get("correlation", 1.0)
                elif key == "memory_ratio":
                    if r.scptensor_result and r.scanpy_result:
                        if r.scptensor_result.memory_mb > 0 and r.scanpy_result.memory_mb > 0:
                            val = r.scanpy_result.memory_mb / r.scptensor_result.memory_mb

            if cat == "Performance":
                scores.append(min(val / 2.0, 1.0) if val > 0 else 0)
            elif cat == "Accuracy":
                scores.append(max(val, 0))
            else:
                scores.append(1.0 / max(val, 0.1) if val > 0 else 1.0)
        method_scores[method] = scores

    if not method_scores:
        return

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#2E86AB", "#A23B72", "#6B8E23", "#F4A261", "#E9C46A"]
    for i, (method, scores) in enumerate(method_scores.items()):
        if len(scores) == len(categories):
            values = scores + [scores[0]]
            ax.plot(angles, values, "o-", linewidth=2, label=method.replace("_", " ").title(), color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories.keys())
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Summary Radar Chart\n(normalized scores)", pad=20)
    ax.grid(True)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "summary_radar.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate all comparison visualizations")
    parser.add_argument("--results-dir", default="benchmark_results", help="Directory containing benchmark results")
    parser.add_argument("--style", choices=["science", "ieee", "nature", "default"], default="science", help="Plot style")

    args = parser.parse_args()

    style_map = {"science": PlotStyle.SCIENCE, "ieee": PlotStyle.IEEE, "nature": PlotStyle.NATURE, "default": PlotStyle.DEFAULT}

    generate_all_plots(results_dir=args.results_dir, style=style_map[args.style])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
