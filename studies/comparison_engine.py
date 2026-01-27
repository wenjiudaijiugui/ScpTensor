"""简化的比较引擎模块"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .metrics import calculate_all_metrics, measure_memory_usage


def run_comparison(
    methods: dict[str, Callable],
    datasets: dict[str, Any],
    metrics_list: list[str] = None,
    n_repeats: int = 1,
) -> dict:
    """运行方法比较"""
    if metrics_list is None:
        metrics_list = ["kbet", "ilisi", "clisi", "asw"]

    results = {}

    for method_name, method_func in methods.items():
        results[method_name] = {}

        for dataset_name, dataset in datasets.items():
            print(f"Running {method_name} on {dataset_name}...")

            all_scores = []
            all_runtimes = []
            all_memory = []

            for _ in range(n_repeats):
                # 运行方法
                start_time = time.time()
                start_memory = measure_memory_usage()

                processed = method_func(dataset.copy())

                end_time = time.time()
                end_memory = measure_memory_usage()

                # 计算指标
                scores = calculate_all_metrics(
                    processed.assays["proteins"].layers["raw"].X,
                    processed.obs["batch"].to_numpy(),
                    processed.obs["cell_type"].to_numpy(),
                )

                all_scores.append(scores)
                all_runtimes.append(end_time - start_time)
                all_memory.append(end_memory - start_memory)

            # 汇总结果
            results[method_name][dataset_name] = {
                "scores": _average_scores(all_scores),
                "runtime": np.mean(all_runtimes),
                "memory": np.mean(all_memory),
            }

    return results


def compare_pipelines(
    pipelines: dict[str, Callable], datasets: dict[str, Any], metrics_list: list[str] = None
) -> dict:
    """比较多个分析管道"""
    if metrics_list is None:
        metrics_list = ["kbet", "ilisi", "clisi", "asw"]

    results = {}

    for pipeline_name, pipeline_func in pipelines.items():
        print(f"\nEvaluating pipeline: {pipeline_name}")

        pipeline_results = {}

        for dataset_name, dataset in datasets.items():
            print(f"  Dataset: {dataset_name}")

            # 运行管道
            start_time = time.time()
            result = pipeline_func(dataset)
            elapsed = time.time() - start_time

            # 计算指标
            scores = calculate_all_metrics(
                result.assays["proteins"].layers["raw"].X,
                result.obs["batch"].to_numpy(),
                result.obs["cell_type"].to_numpy(),
            )

            pipeline_results[dataset_name] = {"scores": scores, "runtime": elapsed}

        results[pipeline_name] = pipeline_results

    return results


def rank_methods(results: dict, metric: str = "kbet") -> list:
    """根据指标排序方法"""
    scores = {}

    for method, method_results in results.items():
        method_scores = []
        for dataset_result in method_results.values():
            if metric in dataset_result["scores"]:
                method_scores.append(dataset_result["scores"][metric])
        scores[method] = np.mean(method_scores)

    # 降序排列（分数越高越好）
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked


def generate_comparison_report(results: dict, output_path: str) -> None:
    """生成比较报告"""
    report_lines = []
    report_lines.append("# 方法比较报告\n")
    report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 方法排名
    report_lines.append("## 方法排名\n")
    for metric in ["kbet", "ilisi", "clisi", "asw"]:
        report_lines.append(f"\n### {metric.upper()}\n")
        ranked = rank_methods(results, metric)
        for i, (method, score) in enumerate(ranked, 1):
            report_lines.append(f"{i}. {method}: {score:.4f}")

    # 详细结果
    report_lines.append("\n## 详细结果\n")

    for method, method_results in results.items():
        report_lines.append(f"\n### {method}\n")

        for dataset, dataset_results in method_results.items():
            report_lines.append(f"\n#### {dataset}\n")
            report_lines.append(f"- 运行时间: {dataset_results['runtime']:.2f}s\n")
            report_lines.append("- 分数:\n")

            for metric, score in dataset_results["scores"].items():
                report_lines.append(f"  - {metric}: {score:.4f}")

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"报告已保存到: {output_path}")


def _average_scores(score_list: list) -> dict:
    """平均多次运行的分数"""
    if not score_list:
        return {}

    avg_scores = {}
    for key in score_list[0]:
        values = [s[key] for s in score_list if key in s]
        if values:
            avg_scores[key] = np.mean(values)

    return avg_scores
