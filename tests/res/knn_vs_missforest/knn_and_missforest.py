import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scptensor.core.structures import ScpMatrix
from scptensor.impute.knn import knn
from scptensor.impute.missforest import missforest
from scptensor.utils.data_genetator import ScpDataGenerator

# Set plotting style
try:
    plt.style.use(["science", "no-latex"])
except:
    # Fallback if scienceplots is not installed, though user requested it
    # We assume it is installed as per user rules.
    pass

# Ensure no Chinese characters in plots
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Calculate evaluation metrics for imputed values.

    Args:
        y_true: The complete ground truth matrix.
        y_pred: The imputed matrix.
        mask: Boolean mask where True indicates missing values that were imputed.

    Returns:
        Dictionary containing NRMSE and MAE.
    """
    # Only evaluate on missing values
    y_true_missing = y_true[mask]
    y_pred_missing = y_pred[mask]

    if len(y_true_missing) == 0:
        return {"NRMSE": 0.0, "MAE": 0.0}

    mse = mean_squared_error(y_true_missing, y_pred_missing)
    rmse = np.sqrt(mse)
    std_dev = np.std(y_true_missing)

    nrmse = rmse / std_dev if std_dev > 0 else 0.0
    mae = mean_absolute_error(y_true_missing, y_pred_missing)

    return {"NRMSE": nrmse, "MAE": mae}


def run_benchmark():
    # Parameters for data generation
    # Further reduce scale for immediate results in demo
    n_samples: int = 50
    n_features: int = 200
    missing_rates: list[float] = [0.1, 0.3]
    lod_ratios: list[float] = [0.3, 0.6]
    n_batches: int = 2
    random_seed: int = 42

    results: list[dict[str, Any]] = []

    print(
        f"{'Missing Rate':<15} {'LOD Ratio':<15} {'Method':<15} {'NRMSE':<10} {'MAE':<10} {'Time(s)':<10}"
    )
    print("-" * 75)

    for mr in missing_rates:
        for lr in lod_ratios:
            # Generate Data
            gen = ScpDataGenerator(
                n_samples=n_samples,
                n_features=n_features,
                missing_rate=mr,
                lod_ratio=lr,
                n_batches=n_batches,
                random_seed=random_seed,
            )
            container = gen.generate()
            assay = container.assays["proteins"]
            # The generator returns X without NaNs, but M has the mask.
            # We need to create a version with NaNs for input.
            original_layer = assay.layers["raw"]
            X_complete = original_layer.X.copy()  # Ground Truth
            M_mask = original_layer.M > 0  # Missing Mask

            # Create Input Matrix with NaNs
            X_missing = X_complete.copy()
            X_missing[M_mask] = np.nan

            # Update the 'raw' layer to have NaNs, so imputation functions work correctly
            assay.layers["raw"] = ScpMatrix(X=X_missing, M=original_layer.M.copy())

            # --- Run KNN ---
            start_time = time.time()
            container = knn(
                container=container,
                assay_name="proteins",
                source_layer="raw",
                new_layer_name="imputed_knn",
                k=5,
            )
            knn_time = time.time() - start_time

            X_knn = assay.layers["imputed_knn"].X
            knn_metrics = calculate_metrics(X_complete, X_knn, M_mask)

            results.append(
                {
                    "Missing Rate": mr,
                    "LOD Ratio": lr,
                    "Method": "KNN",
                    "NRMSE": knn_metrics["NRMSE"],
                    "MAE": knn_metrics["MAE"],
                    "Time": knn_time,
                }
            )

            print(
                f"{mr:<15.2f} {lr:<15.2f} {'KNN':<15} {knn_metrics['NRMSE']:<10.4f} {knn_metrics['MAE']:<10.4f} {knn_time:<10.4f}"
            )

            # --- Run MissForest ---
            # Note: MissForest can be slow, using fewer trees for demo/speed if needed
            start_time = time.time()
            container = missforest(
                container=container,
                assay_name="proteins",
                source_layer="raw",
                new_layer_name="imputed_mf",
                n_estimators=10,  # Reduced for speed in test
                max_iter=2,
            )
            mf_time = time.time() - start_time

            X_mf = assay.layers["imputed_mf"].X
            mf_metrics = calculate_metrics(X_complete, X_mf, M_mask)

            results.append(
                {
                    "Missing Rate": mr,
                    "LOD Ratio": lr,
                    "Method": "MissForest",
                    "NRMSE": mf_metrics["NRMSE"],
                    "MAE": mf_metrics["MAE"],
                    "Time": mf_time,
                }
            )

            print(
                f"{mr:<15.2f} {lr:<15.2f} {'MissForest':<15} {mf_metrics['NRMSE']:<10.4f} {mf_metrics['MAE']:<10.4f} {mf_time:<10.4f}"
            )

    # --- Visualization ---
    plot_results(results)


def plot_results(results: list[dict[str, Any]]):
    """
    Visualize the benchmarking results.
    """
    # Convert to Polars DataFrame for easier handling if needed, or just use lists
    # Here we use matplotlib/seaborn directly

    # Prepare data structures
    methods = ["KNN", "MissForest"]
    missing_rates = sorted(list(set(r["Missing Rate"] for r in results)))
    lod_ratios = sorted(list(set(r["LOD Ratio"] for r in results)))

    # 1. NRMSE Comparison (Heatmap-like or Grouped Bar)
    # Let's use FacetGrid-like approach: X-axis=Missing Rate, Y-axis=NRMSE, Hue=Method, Col=LOD Ratio

    fig, axes = plt.subplots(1, len(lod_ratios), figsize=(5 * len(lod_ratios), 5), sharey=True)
    if len(lod_ratios) == 1:
        axes = [axes]

    fig.suptitle("Imputation Performance (NRMSE) by LOD Ratio")

    for i, lr in enumerate(lod_ratios):
        ax = axes[i]
        # Filter data
        subset = [r for r in results if r["LOD Ratio"] == lr]

        # Prepare for bar plot
        mrs = [r["Missing Rate"] for r in subset]
        nrmses = [r["NRMSE"] for r in subset]
        meths = [r["Method"] for r in subset]

        sns.barplot(x=mrs, y=nrmses, hue=meths, ax=ax, palette="muted")

        ax.set_title(f"LOD Ratio = {lr}")
        ax.set_xlabel("Missing Rate")
        if i == 0:
            ax.set_ylabel("NRMSE")
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("nrmse_comparison.png", dpi=300)
    print("\nSaved NRMSE comparison plot to 'nrmse_comparison.png'")

    # 2. MAE Comparison
    fig, axes = plt.subplots(1, len(lod_ratios), figsize=(5 * len(lod_ratios), 5), sharey=True)
    if len(lod_ratios) == 1:
        axes = [axes]

    fig.suptitle("Imputation Performance (MAE) by LOD Ratio")

    for i, lr in enumerate(lod_ratios):
        ax = axes[i]
        subset = [r for r in results if r["LOD Ratio"] == lr]

        mrs = [r["Missing Rate"] for r in subset]
        maes = [r["MAE"] for r in subset]
        meths = [r["Method"] for r in subset]

        sns.barplot(x=mrs, y=maes, hue=meths, ax=ax, palette="muted")

        ax.set_title(f"LOD Ratio = {lr}")
        ax.set_xlabel("Missing Rate")
        if i == 0:
            ax.set_ylabel("MAE")
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("mae_comparison.png", dpi=300)
    print("Saved MAE comparison plot to 'mae_comparison.png'")

    # 3. Time Comparison (Average across LOD ratios for each Missing Rate)
    plt.figure(figsize=(8, 6))

    # Aggregate time by Missing Rate and Method
    time_data = {}
    for r in results:
        key = (r["Missing Rate"], r["Method"])
        if key not in time_data:
            time_data[key] = []
        time_data[key].append(r["Time"])

    avg_time_results = []
    for (mr, meth), times in time_data.items():
        avg_time_results.append({"Missing Rate": mr, "Method": meth, "Time": np.mean(times)})

    t_mrs = [r["Missing Rate"] for r in avg_time_results]
    t_times = [r["Time"] for r in avg_time_results]
    t_meths = [r["Method"] for r in avg_time_results]

    sns.lineplot(x=t_mrs, y=t_times, hue=t_meths, marker="o")
    plt.title("Computation Time vs Missing Rate")
    plt.xlabel("Missing Rate")
    plt.ylabel("Time (s)")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("time_comparison.png", dpi=300)
    print("Saved Time comparison plot to 'time_comparison.png'")


if __name__ == "__main__":
    run_benchmark()
