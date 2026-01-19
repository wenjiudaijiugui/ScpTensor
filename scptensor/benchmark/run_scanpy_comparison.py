"""Main entry point for ScpTensor vs Scanpy comparison benchmarks.

Run with: python -m scptensor.benchmark.run_scanpy_comparison
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from scptensor.benchmark.comparison_engine import get_engine
from scptensor.benchmark.comparison_viz import PlotStyle, get_visualizer
from scptensor.benchmark.data_provider import ComparisonDataset, COMPARISON_DATASETS
from scptensor.benchmark.report_generator import get_report_generator
from scptensor.benchmark.scanpy_adapter import SCANPY_AVAILABLE

from pathlib import Path


QUICK_CONFIG = {
    "datasets": ["synthetic_small"],
    "methods": ["log_normalize", "pca", "kmeans"],
    "output_dir": "benchmark_results",
}

FULL_CONFIG = {
    "datasets": ["synthetic_small", "synthetic_medium", "synthetic_large"],
    "methods": ["log_normalize", "z_score_normalize", "knn_impute", "pca", "umap", "kmeans", "hvg"],
    "output_dir": "benchmark_results",
}

DEFAULT_METHODS = ["log_normalize", "z_score_normalize", "knn_impute", "pca", "umap", "kmeans", "hvg"]


class BenchmarkRunner:
    """Runner for ScpTensor vs Scanpy comparisons."""

    def __init__(self, output_dir: str = "benchmark_results", plot_style: PlotStyle = PlotStyle.SCIENCE) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self._engine = get_engine(output_dir)
        self._visualizer = get_visualizer(output_dir=str(self._output_dir / "figures"), style=plot_style)
        self._reporter = get_report_generator(output_dir)

    def run(
        self,
        dataset_names: list[str] | None = None,
        methods: list[str] | None = None,
        generate_plots: bool = True,
        generate_report: bool = True,
    ) -> None:
        datasets = [d for d in COMPARISON_DATASETS if dataset_names is None or d.name in dataset_names]
        if not datasets:
            print("Error: No valid datasets specified")
            return

        methods = methods or DEFAULT_METHODS
        print(f"=== ScpTensor vs Scanpy Comparison Benchmark ===")
        print(f"Output directory: {self._output_dir}")
        print(f"Datasets: {[d.name for d in datasets]}")
        print(f"Methods: {methods}\n")

        if not SCANPY_AVAILABLE:
            print("Warning: Scanpy is not installed. Only ScpTensor methods will be run.\n")

        results: dict = {}
        total = len(datasets) * len(methods)

        for idx_d, dataset in enumerate(datasets, 1):
            print(f"--- Dataset: {dataset.name} ({dataset.n_samples}x{dataset.n_features}) ---")

            for idx_m, method in enumerate(methods, 1):
                print(f"[{idx_d * idx_m}/{total}] Running: {method}...", end=" ", flush=True)

                try:
                    result = self._engine.run_shared_comparison(method, dataset)
                    sr = result.scptensor_result
                    sp_res = result.scanpy_result

                    if sr and sr.success:
                        print(f"ScpTensor: {sr.runtime_seconds:.4f}s", end=" ")
                        if sp_res and sp_res.success:
                            speedup = result.comparison_metrics.get("speedup", 1.0)
                            print(f"Scanpy: {sp_res.runtime_seconds:.4f}s (speedup: {speedup:.2f}x)")
                        else:
                            print()
                    else:
                        print("Failed")

                    results.setdefault(method, []).append(result)
                except Exception as e:
                    print(f"Error: {e}")

        print("\n=== Benchmark Complete ===")

        json_path = self._engine.export_results()
        print(f"Results exported to: {json_path}")

        if generate_plots:
            print("\nGenerating plots...")
            self._generate_plots(results)

        if generate_report:
            print("\nGenerating report...")
            report_path = self._reporter.generate_report(self._engine)
            print(f"Report generated: {report_path}")

        print("\n=== Summary ===")
        for method, metrics in self._engine._compute_summary().items():
            print(f"  {method}: {metrics.get('avg_speedup', 1.0):.2f}x speedup")

    def _generate_plots(self, results: dict) -> None:
        viz = self._visualizer
        try:
            viz.plot_runtime_comparison(results)
            print("  - Runtime comparison: figures/01_performance/runtime_comparison.png")

            viz.plot_speedup_heatmap(results)
            print("  - Speedup heatmap: figures/01_performance/speedup_heatmap.png")

            for method, result_list in results.items():
                if not result_list:
                    continue

                result = result_list[0]
                if not result.scptensor_result or not result.scptensor_result.success:
                    continue

                st_out = result.scptensor_result.output
                if st_out is None:
                    continue

                if method == "pca" and result.scanpy_result and result.scanpy_result.success:
                    st_m = result.scptensor_result.metrics or {}
                    sp_m = result.scanpy_result.metrics or {}
                    if "variance_ratio" in st_m and "variance_ratio" in sp_m:
                        try:
                            viz.plot_pca_variance(st_m["variance_ratio"], sp_m["variance_ratio"])
                            print(f"  - PCA variance: figures/04_dim_reduction/pca_variance.png")
                        except Exception:
                            pass

                elif method in ("log_normalize", "z_score_normalize"):
                    if result.scanpy_result and result.scanpy_result.success:
                        sp_out = result.scanpy_result.output
                        if isinstance(st_out, np.ndarray) and isinstance(sp_out, np.ndarray) and st_out.shape == sp_out.shape:
                            try:
                                viz.plot_accuracy_scatter(st_out, sp_out, method)
                                print(f"  - {method} scatter: figures/02_normalization/{method}_comparison.png")
                            except Exception:
                                pass

                elif method == "kmeans" and result.scanpy_result and result.scanpy_result.success:
                    sp_out = result.scanpy_result.output
                    if isinstance(st_out, np.ndarray) and isinstance(sp_out, np.ndarray):
                        try:
                            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                            ari = adjusted_rand_score(st_out, sp_out)
                            nmi = normalized_mutual_info_score(st_out, sp_out)
                            viz.plot_clustering_consistency(st_out, sp_out, ari, nmi)
                            print(f"  - Kmeans consistency: figures/05_clustering/kmeans_consistency.png")
                        except Exception:
                            pass

                elif method == "umap" and result.scanpy_result and result.scanpy_result.success:
                    sp_out = result.scanpy_result.output
                    if isinstance(st_out, np.ndarray) and isinstance(sp_out, np.ndarray) and st_out.shape == sp_out.shape and st_out.shape[1] == 2:
                        try:
                            viz.plot_umap_comparison(st_out, sp_out, None)
                            print(f"  - UMAP comparison: figures/04_dim_reduction/umap_comparison.png")
                        except Exception:
                            pass
        except Exception as e:
            print(f"  Warning: Some plots failed to generate: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ScpTensor vs Scanpy comparison benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scptensor.benchmark.run_scanpy_comparison --quick
  python -m scptensor.benchmark.run_scanpy_comparison --full
  python -m scptensor.benchmark.run_scanpy_comparison --datasets synthetic_medium --methods pca umap
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    parser.add_argument("--datasets", nargs="+", choices=[d.name for d in COMPARISON_DATASETS], help="Datasets to use")
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, help="Methods to compare")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    parser.add_argument("--style", choices=["science", "ieee", "nature", "default"], default="science", help="Plot style")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.quick:
        dataset_names, methods = QUICK_CONFIG["datasets"], QUICK_CONFIG["methods"]
    elif args.full:
        dataset_names, methods = FULL_CONFIG["datasets"], FULL_CONFIG["methods"]
    else:
        dataset_names, methods = args.datasets, args.methods

    style_map = {"science": PlotStyle.SCIENCE, "ieee": PlotStyle.IEEE, "nature": PlotStyle.NATURE, "default": PlotStyle.DEFAULT}

    runner = BenchmarkRunner(output_dir=args.output_dir, plot_style=style_map[args.style])

    try:
        runner.run(dataset_names=dataset_names, methods=methods, generate_plots=not args.no_plots, generate_report=not args.no_report)
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
