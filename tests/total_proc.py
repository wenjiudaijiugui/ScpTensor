
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import os
import scienceplots

# Apply style
plt.style.use(["science", "no-latex"])

from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
from scptensor.normalization import log_normalize, sample_median_normalization
from scptensor.standardization import zscore_standardization
from scptensor.impute import knn
from scptensor.integration.combat import combat
from scptensor.dim_reduction import pca, umap
from scptensor.cluster import run_kmeans
from scptensor.viz import qc_completeness, qc_matrix_spy, embedding, volcano



def generate_synthetic_data(n_samples=500, n_features=6000):
    """
    Generate synthetic single-cell proteomics data.
    Size: 500 x 6000
    Missing: 50% (30% random, 20% systematic/LOD)
    Groups: 2 (Group A, Group B)
    Batches: 2 (Batch 1, Batch 2)
    """
    np.random.seed(42)

    # 1. Metadata
    # Groups: 250 A, 250 B
    groups = np.array(['GroupA'] * (n_samples // 2) + ['GroupB'] * (n_samples // 2))
    # Batches: Randomly assigned 50/50
    batches = np.random.choice(['Batch1', 'Batch2'], size=n_samples)

    obs = pl.DataFrame({
        'sample_id': [f'S{i+1:03d}' for i in range(n_samples)],
        'group': groups,
        'batch': batches
    })

    # 2. Expression Data (Log-Normal base)
    # Base expression
    X_true = np.random.lognormal(mean=2, sigma=0.5, size=(n_samples, n_features))

    # Add Group Effect (Shift for Group B in first 100 features)
    X_true[groups == 'GroupB', :100] *= 2.0

    # Add Batch Effect (Shift for Batch 2)
    X_true[batches == 'Batch2', :] *= 1.2

    # 3. Introduce Missing Values
    X_observed = X_true.copy()
    M = np.zeros((n_samples, n_features), dtype=int) # 0=Valid

    # 3.1 Systematic Missing (LOD - Low Abundance) - 20%
    # Remove lowest 20% values overall (simplified simulation of LOD)
    threshold = np.percentile(X_true, 20)
    lod_mask = X_true < threshold
    X_observed[lod_mask] = 0 # Or NaN, but usually 0 in raw
    M[lod_mask] = 2 # 2 = LOD

    # 3.2 Random Missing - 30% of remaining valid
    # Remaining valid indices
    valid_mask = (M == 0)
    n_valid = np.sum(valid_mask)
    n_random_missing = int(n_samples * n_features * 0.3)

    # Randomly select indices to set to missing from valid ones
    valid_indices = np.argwhere(valid_mask)
    random_indices_idx = np.random.choice(len(valid_indices), size=n_random_missing, replace=False)
    random_indices = valid_indices[random_indices_idx]

    X_observed[random_indices[:, 0], random_indices[:, 1]] = 0
    M[random_indices[:, 0], random_indices[:, 1]] = 1 # 1 = Random/MBR

    # 4. Create ScpContainer
    var = pl.DataFrame({
        'protein_id': [f'P{i+1:04d}' for i in range(n_features)],
        '_index': [f'P{i+1:04d}' for i in range(n_features)]
    })

    matrix = ScpMatrix(X=X_observed, M=M)
    assay = Assay(var=var, layers={'raw': matrix}, feature_id_col='protein_id')

    container = ScpContainer(
        assays={'protein': assay},
        obs=obs.with_columns(pl.Series(name="_index", values=obs["sample_id"].to_list())),
        sample_id_col='sample_id'
    )

    print(f"Generated data: {n_samples} samples, {n_features} features.")
    print(f"Missing rate: {np.mean(M != 0):.2%}")

    return container

def run_pipeline():
    output_dir = "tests/pipeline_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Data
    data = pl.read_csv("/home/shenshang/projects/ScpTensor/tests/data/PXD061065/20250204_112949_gbm_sc_full_fasta_19.4_Report.tsv", separator="\t")
    columns = data.columns
    quantity_columns = []
    for col in columns:
        if "Quantity" in col:
            quantity_columns.append(col)
    genes_col = "PG.Genes"
    data = data.select(
        [genes_col, *quantity_columns]
    )
    
    # Convert to ScpContainer
    # Extract expression matrix (X)
    # Replace 'NaN' string with actual nulls if necessary, but read_csv usually handles it.
    # However, let's ensure numeric types.
    
    # Pivot/Melt might be needed depending on structure, but here columns are samples (Quantity...) and rows are genes.
    # So X should be (n_samples, n_features).
    # data shape: (n_features, 1 + n_samples)
    
    protein_ids = data[genes_col].to_list()
    # Remove rows where PG.Genes is null or empty
    data = data.filter(pl.col(genes_col).is_not_null())
    protein_ids = data[genes_col].to_list()
    
    # Extract quantity columns as matrix
    # Transpose: (n_features, n_samples) -> (n_samples, n_features)
    X_df = data.select(quantity_columns)
    X = X_df.to_numpy().T # Shape: (n_samples, n_features)
    
    # Handle Duplicate Feature IDs
    # protein_ids might contain duplicates (e.g. isoforms or same gene mapped)
    # Simple strategy: Make unique by appending suffix
    seen = {}
    unique_protein_ids = []
    for pid in protein_ids:
        if pid in seen:
            seen[pid] += 1
            unique_protein_ids.append(f"{pid}_{seen[pid]}")
        else:
            seen[pid] = 0
            unique_protein_ids.append(pid)
    protein_ids = unique_protein_ids
    
    # Handle Missing Values (NaN)
    # In raw data, NaN usually means missing.
    # M matrix: 0 = Valid, 1 = Missing (we assume all missing are same type initially, or infer LOD later)
    M = np.zeros(X.shape, dtype=np.int8)
    is_nan = np.isnan(X)
    M[is_nan] = 1 # Mark as missing
    X[is_nan] = 0 # Fill with 0 or keep NaN? ScpMatrix usually stores values. Let's keep 0 for sparse compatibility or specific value.
    # But log_normalize expects values.
    # Let's set 0 for now.
    
    n_samples, n_features = X.shape
    
    # Create obs (Samples metadata)
    # Sample IDs are the quantity column names
    sample_ids = quantity_columns
    # Extract batch/group info from sample IDs if possible?
    # Example ID: "Quantity.1.1" -> maybe not informative.
    # Let's create dummy group/batch if not available.
    # Or try to parse: "Quantity" usually has "Run" or "Channel" info in other columns?
    # For now, random assignment for demo purposes as per original synthetic logic
    
    np.random.seed(42)
    groups = np.random.choice(['GroupA', 'GroupB'], size=n_samples)
    batches = np.random.choice(['Batch1', 'Batch2'], size=n_samples)
    
    obs = pl.DataFrame({
        'sample_id': sample_ids,
        'group': groups,
        'batch': batches
    })
    
    # Create var (Features metadata)
    var = pl.DataFrame({
        'protein_id': protein_ids,
        '_index': protein_ids
    })
    
    # Create Container
    matrix = ScpMatrix(X=X, M=M)
    assay = Assay(var=var, layers={'raw': matrix}, feature_id_col='protein_id')
    
    container = ScpContainer(
        assays={'protein': assay},
        obs=obs.with_columns(pl.Series(name="_index", values=obs["sample_id"].to_list())),
        sample_id_col='sample_id'
    )
    
    print(f"Loaded data: {n_samples} samples, {n_features} features.")
    print(f"Missing rate: {np.mean(M != 0):.2%}")

    # Viz: QC (Completeness & Spy)
    print("Plotting QC...")
    ax = qc_completeness(container, group_by='batch')
    plt.savefig(f"{output_dir}/01_qc_completeness.png", dpi=300)
    plt.close()

    ax = qc_matrix_spy(container)
    plt.savefig(f"{output_dir}/01_qc_spy.png", dpi=300)
    plt.close()

    # 2. Normalization (Log2)
    print("\n[Step 2] Normalization (Log2)...")
    log_normalize(container, assay_name='protein', base_layer='raw', new_layer_name='log')
    log_layer = container.assays['protein'].layers['log']
    X_log = log_layer.X
    M_log = log_layer.M
    X_log[M_log != 0] = np.nan
    container.assays['protein'].layers['log'] = ScpMatrix(X=X_log, M=M_log)

    # 3. Imputation (KNN)
    print("\n[Step 3] Imputation (KNN)...")
    knn(container, assay_name='protein', base_layer='log', new_layer_name='imputed', k=5)
    imputed_layer = container.assays['protein'].layers['imputed']
    container.assays['protein'].layers['imputed'] = ScpMatrix(
        X=imputed_layer.X,
        M=np.zeros_like(imputed_layer.X, dtype=np.int8)
    )

    # Viz: Imputation result (Check distribution or spy again? Spy won't change M unless updated, but KNN updates M usually?
    # Our KNN implementation should update M to 0 for imputed values in the new layer)
    # Let's visualize the imputed layer's completeness (should be 100%)
    ax = qc_completeness(container, layer='imputed', group_by='batch')
    plt.title("Data Completeness after KNN Imputation")
    plt.savefig(f"{output_dir}/03_post_imputation_completeness.png", dpi=300)
    plt.close()

    # 4. Debatch (ComBat)
    print("\n[Step 4] Batch Correction (ComBat)...")
    # Correct 'imputed' -> 'corrected'
    combat(container, batch_key='batch', assay_name='protein', base_layer='imputed', new_layer_name='corrected')

    # Viz: PCA before and after batch correction to show effect
    print("Running PCA for Batch Effect Check...")
    # PCA on imputed (Before Correction)
    container = pca(container, assay_name='protein', base_layer_name='imputed', new_assay_name='pca_before', n_components=2)
    # PCA on corrected (After Correction)
    container = pca(container, assay_name='protein', base_layer_name='corrected', new_assay_name='pca_after', n_components=2)

    # Plot PCA Before
    ax = embedding(container, basis='pca_before', color='batch')
    plt.title("PCA Before Batch Correction")
    plt.savefig(f"{output_dir}/04_pca_before_correction.png", dpi=300)
    plt.close()

    # Plot PCA After
    ax = embedding(container, basis='pca_after', color='batch')
    plt.title("PCA After Batch Correction")
    plt.savefig(f"{output_dir}/04_pca_after_correction.png", dpi=300)
    plt.close()

    # 5. Dim Reduction (PCA & UMAP on Corrected Data)
    print("\n[Step 5] Dimensionality Reduction...")
    # We already ran PCA for 'pca_after', let's use it for UMAP
    # UMAP needs 'X' layer from 'pca_after' assay
    container = umap(container, assay_name='pca_after', base_layer='scores', new_assay_name='umap', n_neighbors=30)

    # Plot UMAP by Group (Biological Signal)
    ax = embedding(container, basis='umap', color='group')
    plt.title("UMAP by Group")
    plt.savefig(f"{output_dir}/05_umap_group.png", dpi=300)
    plt.close()

    # Plot UMAP by Batch (Should be mixed)
    ax = embedding(container, basis='umap', color='batch')
    plt.title("UMAP by Batch")
    plt.savefig(f"{output_dir}/05_umap_batch.png", dpi=300)
    plt.close()

    # 6. Clustering (K-Means)
    print("\n[Step 6] Clustering (K-Means)...")
    # Cluster on PCA components (usually better than UMAP coordinates)
    # Add cluster labels to obs
    container = run_kmeans(container, assay_name='pca_after', base_layer='scores', n_clusters=2, key_added='kmeans_cluster')

    # Plot UMAP colored by Cluster
    ax = embedding(container, basis='umap', color='kmeans_cluster')
    plt.title("UMAP by K-Means Cluster")
    plt.savefig(f"{output_dir}/06_umap_kmeans.png", dpi=300)
    plt.close()

    print("\nPipeline Completed Successfully!")
    print(f"Results saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    run_pipeline()
