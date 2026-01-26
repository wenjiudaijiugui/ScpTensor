import time

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from scptensor.dim_reduction.umap import reduce_umap as custom_umap
from scptensor.impute.knn import impute_knn as knn
from scptensor.standardization.zscore import zscore
from scptensor.utils.data_generator import ScpDataGenerator

# Set plotting style
plt.style.use(["science", "no-latex"])


def generate_data():
    """Generate synthetic single-cell proteomics data."""
    print("Generating synthetic data...")
    generator = ScpDataGenerator(
        n_samples=500, n_features=5000, missing_rate=0.3, n_batches=3, n_groups=8, random_seed=42
    )
    container = generator.generate()

    # Corrected access based on data_generator.py source code:
    # container.add_assay("proteins", assay)
    # assay.add_layer("raw", matrix)
    # Note: We return the container directly, matrix extraction happens later after preprocessing

    # Get labels for evaluation (Use 'group' instead of 'batch' for meaningful biological evaluation)
    labels = container.obs["group"].to_list()

    return container, labels


def preprocess_data(container):
    """Apply imputation and normalization."""
    print("Preprocessing data (Imputation + Normalization)...")

    # 1. Imputation (KNN)
    # Input: 'proteins' assay, 'raw' layer (contains missing values)
    # Output: 'imputed_knn' layer
    print("  - Running KNN Imputation...")
    container = knn(
        container=container,
        assay_name="proteins",
        source_layer="raw",
        new_layer_name="imputed_knn",
        k=5,
        weights="uniform",
        batch_size=500,
        oversample_factor=3,
    )

    # 2. Normalization (Z-score)
    # Input: 'proteins' assay, 'imputed_knn' layer
    # Output: 'zscore' layer
    print("  - Running Z-score Normalization...")
    container = zscore(
        container=container,
        assay_name="proteins",
        base_layer_name="imputed_knn",
        new_layer_name="zscore",
        axis=0,  # Normalize features (columns) across samples? Or samples across features?
        # Standard Z-score usually normalizes features (mean=0, std=1 per protein)
        # zscore.py default axis=0 implies column-wise (per feature) normalization if data is (samples, features)
        ddof=1,
    )

    return container


def run_custom_umap(container):
    """Run the custom UMAP implementation."""
    print("Running custom UMAP implementation...")
    start_time = time.time()

    # Run UMAP
    # Input: 'proteins' assay, 'zscore' layer (preprocessed data)
    # Note: custom_umap does NOT accept random_state, but transform_seed.
    container = custom_umap(
        container=container,
        assay_name="proteins",
        source_layer="zscore",
        new_layer_name="custom_umap",
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="euclidean",
        transform_seed=42,
    )

    end_time = time.time()

    # Extract embedding
    # The custom UMAP creates a NEW assay named "{assay_name}_{new_layer_name}"
    # So it should be "proteins_custom_umap"
    umap_assay_name = "proteins_custom_umap"

    if umap_assay_name in container.assays:
        embedding_matrix = container.assays[umap_assay_name].layers["custom_umap"]
        embedding = embedding_matrix.X
    else:
        raise ValueError(f"Expected assay '{umap_assay_name}' not found in container.")

    return embedding, end_time - start_time


def run_standard_umap(container):
    """Run the standard umap-learn implementation on the same preprocessed data."""
    print("Running standard umap-learn implementation...")

    # Extract preprocessed matrix from container
    # Ensure we use the exact same input data as custom UMAP
    matrix = container.assays["proteins"].layers["zscore"].X

    start_time = time.time()

    reducer = umap.UMAP(
        n_neighbors=15, n_components=2, min_dist=0.1, metric="euclidean", random_state=42
    )
    embedding = reducer.fit_transform(matrix)

    end_time = time.time()

    return embedding, end_time - start_time


def evaluate_embedding(embedding, labels):
    """Calculate evaluation metrics for the embedding."""

    # Silhouette Score (-1 to 1, higher is better)
    sil_score = silhouette_score(embedding, labels)

    # Calinski-Harabasz Index (higher is better)
    ch_score = calinski_harabasz_score(embedding, labels)

    # Davies-Bouldin Index (lower is better)
    db_score = davies_bouldin_score(embedding, labels)

    return {
        "Silhouette Score": sil_score,
        "Calinski-Harabasz Index": ch_score,
        "Davies-Bouldin Index": db_score,
    }


def plot_embeddings(custom_emb, standard_emb, labels, save_path="umap_comparison.png"):
    """Plot the embeddings side by side."""

    unique_labels = np.unique(labels)
    # Create a color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_color_map = {label: color for label, color in zip(unique_labels, colors, strict=False)}
    point_colors = [label_color_map[label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Custom UMAP
    axes[0].scatter(custom_emb[:, 0], custom_emb[:, 1], c=point_colors, s=5, alpha=0.7)
    axes[0].set_title("Custom UMAP Implementation")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # Standard UMAP
    axes[1].scatter(standard_emb[:, 0], standard_emb[:, 1], c=point_colors, s=5, alpha=0.7)
    axes[1].set_title("Standard umap-learn Implementation")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    # Create legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=l, markersize=8)
        for l, c in label_color_map.items()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(unique_labels), bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {save_path}")


def main():
    # 1. Generate Data
    try:
        container, labels = generate_data()
    except Exception as e:
        print(f"Data generation failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 2. Preprocess Data
    try:
        container = preprocess_data(container)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. Run Custom UMAP
    try:
        custom_emb, custom_time = run_custom_umap(container)
    except Exception as e:
        print(f"Custom UMAP failed: {e}")
        import traceback

        traceback.print_exc()
        # Proceed if possible or exit
        return

    # 4. Run Standard UMAP
    try:
        standard_emb, standard_time = run_standard_umap(container)
    except Exception as e:
        print(f"Standard UMAP failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 5. Evaluate
    print("\n--- Evaluation Results ---")
    print(f"{'Metric':<25} | {'Custom UMAP':<15} | {'Standard UMAP':<15}")
    print("-" * 60)

    custom_metrics = evaluate_embedding(custom_emb, labels)
    standard_metrics = evaluate_embedding(standard_emb, labels)

    for metric in custom_metrics:
        print(
            f"{metric:<25} | {custom_metrics[metric]:.4f}          | {standard_metrics[metric]:.4f}"
        )

    print("-" * 60)
    print(f"{'Execution Time (s)':<25} | {custom_time:.4f}          | {standard_time:.4f}")

    # 5. Visualize
    plot_embeddings(custom_emb, standard_emb, labels)


if __name__ == "__main__":
    main()
