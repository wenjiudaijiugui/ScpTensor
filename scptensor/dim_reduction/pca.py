from typing import Optional, Union, Type
import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.linalg import svds, LinearOperator
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay
import warnings

class _CenteredScaledLinearOperator(LinearOperator):
    """
    LinearOperator that performs implicit centering and scaling.
    Y = (X - mean) / std
    """
    def __init__(self, X: sp.spmatrix, mean: np.ndarray, std: Optional[np.ndarray] = None):
        self.X = X
        self.mean = mean # Shape (n_features,)
        self.std = std   # Shape (n_features,) or None
        self.shape = X.shape
        self.dtype = X.dtype
        
        # Pre-calculate scaled mean to save ops
        if self.std is not None:
            # Handle division by zero or small std outside or assume handled
            self.mean_scaled = self.mean / self.std
        else:
            self.mean_scaled = self.mean

    def _matvec(self, v):
        # v shape (n_features,)
        # y = (X - mu)/sigma * v = X(v/sigma) - mu * (1^T * (v/sigma))
        # If no scale: y = X v - mu (sum(v)) ?? No.
        # y = (X - mu) v = X v - mu (1^T v) ? No.
        # y = (X - mu) v = X v - 1 (mu^T v)
        
        if self.std is not None:
            v_scaled = v / self.std
        else:
            v_scaled = v
            
        # X v'
        Xv = self.X.dot(v_scaled)
        
        # mu^T v' (scalar)
        mu_dot_v = np.dot(self.mean, v_scaled)
        
        # y = Xv - 1 * scalar
        return Xv - mu_dot_v

    def _rmatvec(self, u):
        # u shape (n_samples,)
        # Y^T u = [ (X - 1 mu) diag(1/sigma) ]^T u
        #       = diag(1/sigma) (X^T - mu^T 1^T) u
        #       = diag(1/sigma) (X^T u - mu^T (1^T u))
        
        sum_u = u.sum() # scalar
        
        # X^T u
        XTu = self.X.T.dot(u)
        
        # mu^T sum_u
        mu_sum_u = self.mean * sum_u
        
        res = XTu - mu_sum_u
        
        if self.std is not None:
            res /= self.std
            
        return res

def _flip_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce deterministic sign convention on SVD results.
    For each component (row in Vt), find the element with the largest absolute value.
    If this element is negative, multiply both the component (row in Vt) and 
    the corresponding score (column in U) by -1.
    
    Ref: Bro, R., et al. (2008). Resolving the sign ambiguity in the singular value decomposition.
    """
    for k in range(Vt.shape[0]):
        # Find index of max absolute value in k-th eigenvector (row of Vt)
        idx = np.argmax(np.abs(Vt[k, :]))
        
        # If that element is negative, flip signs
        if Vt[k, idx] < 0:
            Vt[k, :] *= -1
            U[:, k] *= -1
            
    return U, Vt

def pca(
    container: ScpContainer,
    assay_name: str,
    base_layer_name: str,
    new_assay_name: str = "pca",
    n_components: int = 2,
    center: bool = True,
    scale: bool = False,
    random_state: Optional[Union[int, np.random.RandomState]] = 42,
    dtype: Type = np.float64
) -> ScpContainer:
    """
    Perform Principal Component Analysis (PCA) on a specific layer of an assay.

    Args:
        container (ScpContainer): The data container.
        assay_name (str): The name of the assay to use.
        base_layer_name (str): The name of the layer in the assay to use as input.
        new_assay_name (str, optional): The name of the new assay to store PCA results. Defaults to "pca".
        n_components (int, optional): The number of principal components to compute. Defaults to 2.
        center (bool, optional): Whether to center the data before PCA. Defaults to True.
        scale (bool, optional): Whether to scale the data to unit variance before PCA. Defaults to False.
        random_state (int, RandomState, optional): Seed for the random number generator (for ARPACK). 
                                                   Ensures reproducibility. Defaults to 42.
        dtype (Type, optional): The desired data type for computation (e.g. np.float64 or np.float32).
                                Defaults to np.float64.

    Returns:
        ScpContainer: A new container with the PCA results added as a new assay.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
        raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    input_matrix = assay.layers[base_layer_name]
    X = input_matrix.X
    
    # Check for NaN or infinite values
    # Note: np.any implies checking all elements, which can be slow for large matrices.
    # For sparse matrices, checking .data is more efficient.
    if sp.issparse(X):
        if np.any(np.isnan(X.data)) or np.any(np.isinf(X.data)):
             raise ValueError(
                "Input data contains NaN or infinite values. "
                "ScpTensor PCA requires a complete data matrix. "
                "Please use an imputed layer (e.g. run imputation first)."
            )
    else:
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError(
                "Input data contains NaN or infinite values. "
                "ScpTensor PCA requires a complete data matrix. "
                "Please use an imputed layer (e.g. run imputation first)."
            )

    n_samples, n_features = X.shape
    if n_components > min(n_samples, n_features):
        raise ValueError(f"n_components ({n_components}) cannot be greater than min(n_samples, n_features) ({min(n_samples, n_features)}).")

    # 1. Data Preprocessing & SVD
    # We implement a memory-efficient strategy for sparse matrices using LinearOperator.
    
    # Decide on target dtype
    target_dtype = dtype 
    
    # Check for NaNs/Infs in input X (moved earlier, but good to be sure)
    # Already checked above.

    # Prepare mean and std for centering/scaling
    # Note: We need mean and std even if using LinearOperator to construct it.
    # For sparse matrix, we calculate mean and std without densifying.
    
    mean = None
    std = None
    
    if center:
        if sp.issparse(X):
            mean = np.array(X.mean(axis=0)).flatten()
        else:
            mean = np.mean(X, axis=0)
            
    if scale:
        if sp.issparse(X):
            # Calculate std for sparse matrix without densifying
            # var = E[X^2] - (E[X])^2
            # This is a bit expensive but much cheaper than densifying
            
            # E[X] is mean
            # E[X^2]:
            X_sq = X.copy()
            X_sq.data **= 2
            mean_sq = np.array(X_sq.mean(axis=0)).flatten()
            
            # Variance (population)
            var = mean_sq - mean**2
            
            # Adjust for ddof=1
            # var_sample = var * n / (n-1)
            var *= n_samples / (n_samples - 1)
            
            std = np.sqrt(var)
        else:
            std = np.std(X, axis=0, ddof=1)
            
        # Handle near-zero std
        eps = np.finfo(target_dtype).eps
        std[std < eps] = 1.0

    # Construct LinearOperator or Processed Dense Matrix
    # If X is sparse and we need center/scale, use LinearOperator to avoid densification.
    # If X is dense, we can just process it.
    
    use_linear_operator = False
    X_processed = None
    linear_op = None
    
    if sp.issparse(X):
        if center or scale:
            # Use LinearOperator for implicit centering/scaling
            use_linear_operator = True
            # Ensure X is in target dtype for calculations
            if X.dtype != target_dtype:
                X = X.astype(target_dtype)
            
            linear_op = _CenteredScaledLinearOperator(X, mean if center else np.zeros(n_features), std if scale else None)
        else:
            # Sparse, no center, no scale. Just use X (maybe cast dtype)
            if X.dtype != target_dtype:
                X_processed = X.astype(target_dtype)
            else:
                X_processed = X
    else:
        # Dense input
        if X.dtype == target_dtype:
            X_processed = X.copy() # Copy to avoid modifying original
        else:
            X_processed = X.astype(target_dtype, copy=True)
            
        if center:
            X_processed -= mean
        if scale:
            X_processed /= std

    # 2. SVD Calculation
    
    min_dim = min(n_samples, n_features)
    method_used = "truncated_svd" 
    
    # Heuristic: 
    # If using LinearOperator, we MUST use iterative solver (svds or randomized_svd).
    # Since we don't have randomized_svd imported, we use svds (ARPACK).
    # If dense, we can choose between full SVD (LAPACK) and truncated SVD (ARPACK).
    
    use_full_svd = False
    if not use_linear_operator:
        if n_components >= min_dim:
            use_full_svd = True
        elif min_dim < 2000:
            use_full_svd = True
        elif n_components > 0.2 * min_dim:
            use_full_svd = True

    if use_full_svd:
        method_used = "full_svd"
        U, S, Vt = np.linalg.svd(X_processed, full_matrices=False)
        U_reduced = U[:, :n_components]
        S_reduced = S[:n_components]
        Vt_reduced = Vt[:n_components, :]
    else:
        # Use truncated SVD (ARPACK)
        # If use_linear_operator is True, X_processed is None, we use linear_op
        operator = linear_op if use_linear_operator else X_processed
        
        k = min(n_components, min_dim - 1)
        if k < 1: 
             # Fallback to full SVD if k is too small (shouldn't happen with n_components>=1)
             # But if use_linear_operator is True, we can't use np.linalg.svd without densifying!
             if use_linear_operator:
                 # If we are here, it means min_dim is very small (e.g. 1 or 2) or n_components is 0.
                 # If min_dim is small, densifying is fine.
                 X_dense = X.toarray().astype(target_dtype)
                 if center: X_dense -= mean
                 if scale: X_dense /= std
                 method_used = "full_svd (forced densification)"
                 U, S, Vt = np.linalg.svd(X_dense, full_matrices=False)
                 U_reduced = U[:, :n_components]
                 S_reduced = S[:n_components]
                 Vt_reduced = Vt[:n_components, :]
             else:
                 method_used = "full_svd"
                 U, S, Vt = np.linalg.svd(X_processed, full_matrices=False)
                 U_reduced = U[:, :n_components]
                 S_reduced = S[:n_components]
                 Vt_reduced = Vt[:n_components, :]
        else:
            try:
                # svds requires k < min_dim
                from scipy.sparse.linalg import ArpackError
                U, S, Vt = svds(operator, k=k, which='LM', random_state=random_state)
                # svds returns eigenvalues in increasing order, need to reverse
                U_reduced = U[:, ::-1]
                S_reduced = S[::-1]
                Vt_reduced = Vt[::-1, :]
            except (ArpackError, ValueError, RuntimeError) as e:
                # Fallback
                method_used = "full_svd (fallback)"
                warnings.warn(f"ARPACK SVD failed: {e}. Falling back to full SVD (may be slow/memory intensive).")
                
                if use_linear_operator:
                    # Must densify to use full SVD
                     X_dense = X.toarray().astype(target_dtype)
                     if center: X_dense -= mean
                     if scale: X_dense /= std
                     X_fallback = X_dense
                else:
                     X_fallback = X_processed
                     
                U, S, Vt = np.linalg.svd(X_fallback, full_matrices=False)
                U_reduced = U[:, :n_components]
                S_reduced = S[:n_components]
                Vt_reduced = Vt[:n_components, :]

    # 3. Extract Components and Resolve Sign Ambiguity
    U_reduced, Vt_reduced = _flip_signs(U_reduced, Vt_reduced)
    
    # Calculate Principal Components (Scores)
    scores = U_reduced * S_reduced
    
    # Loadings (Eigenvectors)
    loadings = Vt_reduced.T
    
    # Explained Variance
    # Fix 4: Correct Total Variance Calculation when center=False
    if center:
        # Variance is sum of vars of each feature
        if sp.issparse(X) and not use_linear_operator and X_processed is None:
             # Should have calculated variance during std calc if scale=True, or just calculate it now.
             # We can reuse the logic from scale block
             if scale:
                 # If scale=True, we standardized, so variance of each feature is 1 (approx).
                 # Total variance = n_features
                 total_variance = float(n_features)
             else:
                 # Center=True, Scale=False. Need to calc variance.
                 X_sq = X.copy()
                 X_sq.data **= 2
                 mean_sq = np.array(X_sq.mean(axis=0)).flatten()
                 var = mean_sq - mean**2
                 # ddof=1
                 var *= n_samples / (n_samples - 1)
                 total_variance = np.sum(var)
        elif use_linear_operator:
             # If scaled, total variance is n_features
             if scale:
                 total_variance = float(n_features)
             else:
                 # Centered but not scaled.
                 # var calculated above in scale block? No, only if scale=True.
                 # Recalculate var
                 X_sq = X.copy()
                 X_sq.data **= 2
                 mean_sq = np.array(X_sq.mean(axis=0)).flatten()
                 var = mean_sq - mean**2
                 var *= n_samples / (n_samples - 1)
                 total_variance = np.sum(var)
        else:
            # Dense case
            # If X_processed is scaled, then its variance is 1 for each feature (approx)
            # BUT, if we use standard scaler manually (X_processed /= std), it is correct.
            # However, if X_processed is just centered, we compute var.
            
            # If scaled, we should use n_features as total variance ONLY IF we used population std for scaling?
            # Sklearn uses biased estimator for std (ddof=0) for scaling, but unbiased (ddof=1) for variance?
            # Actually sklearn StandardScaler uses biased std (1/N).
            # np.std(ddof=1) is unbiased (1/(N-1)).
            
            # In our code:
            # if scale: std = np.std(X, axis=0, ddof=1)  <-- Unbiased std
            # X_processed = X / std
            # So variance of X_processed is exactly 1.0 for each feature (if center=True).
            # So total variance should be n_features.
            
            if scale:
                 total_variance = float(n_features)
            else:
                 feature_vars = np.var(X_processed, axis=0, ddof=1)
                 total_variance = np.sum(feature_vars)
    else:
        # If not centered, total variance should be based on the second moment (sum(x^2)/(N-1))
        # rather than variance (sum((x-mean)^2)/(N-1)).
        # Since X_processed is not centered, we calculate sum of squares directly.
        
        # Calculating sum of squares for each feature, then summing them up.
        # Equivalent to frobenius norm squared divided by N-1.
        
        if sp.issparse(X):
             # Frobenius norm squared of sparse matrix = sum(data^2)
             sq_sum = np.sum(X.data ** 2)
        else:
             sq_sum = np.sum(np.square(X_processed))
             
        total_variance = sq_sum / (n_samples - 1)

    eigenvalues = S_reduced ** 2 / (n_samples - 1)
    explained_variance_ratio = eigenvalues / total_variance
    
    # Rename explained_variance_ratio if not centered
    variance_ratio_col_name = "explained_variance_ratio"
    if not center:
        variance_ratio_col_name = "explained_inertia_ratio"

    # 4. Create New Assay for PCA Results
    
    # Feature metadata for the new PCA assay
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    
    pca_var = pl.DataFrame({
        "pc_name": pc_names,
        "explained_variance": eigenvalues,
        variance_ratio_col_name: explained_variance_ratio
    })

    scores_M = np.zeros_like(scores, dtype=np.int8)
    scores_matrix = ScpMatrix(X=scores, M=scores_M)
    
    pca_assay = Assay(
        var=pca_var,
        layers={"scores": scores_matrix},
        feature_id_col="pc_name"
    )
    
    # 5. Store Loadings in Original Assay Metadata
    
    # Clone the original assay
    original_assay = container.assays[assay_name]
    
    # Create loadings DataFrame with Prefixed Names to avoid conflict (Issue 1)
    # Naming convention: {new_assay_name}_PC{i}_loading
    loadings_dict = {}
    for i in range(n_components):
        col_name = f"{new_assay_name}_PC{i+1}_loading"
        loadings_dict[col_name] = loadings[:, i]
        
    loadings_df = pl.DataFrame(loadings_dict)
    
    # Verify dimensions (Issue 2)
    if original_assay.var.height != loadings_df.height:
        raise ValueError(
            f"Dimension mismatch: Original assay features ({original_assay.var.height}) "
            f"!= Calculated loadings ({loadings_df.height}). "
            "This indicates a synchronization issue between 'var' and 'X'."
        )

    # Clean up OLD columns related to this specific PCA run if they exist?
    existing_cols = original_assay.var.columns
    
    cols_to_drop = []
    prefix = f"{new_assay_name}_PC"
    for col in existing_cols:
        if col.startswith(prefix) and "_loading" in col:
            cols_to_drop.append(col)
            
    if cols_to_drop:
        original_var_clean = original_assay.var.drop(cols_to_drop)
    else:
        original_var_clean = original_assay.var
        
    # Explicit Alignment Check (Issue 3)
    # Fix 3: Add explicit assertion before concat
    # Polars concat(how="horizontal") does not check alignment.
    if original_var_clean.height != loadings_df.height:
        raise AssertionError(
            f"Critical alignment error: 'original_var_clean' height ({original_var_clean.height}) "
            f"does not match 'loadings_df' height ({loadings_df.height}) before concatenation. "
            "Data integrity cannot be guaranteed."
        )

    new_original_var = pl.concat([original_var_clean, loadings_df], how="horizontal")
    
    # Create new instance of original assay with updated var
    # Fix 5: Mutable Object Reference Leakage (from original code)
    new_original_assay = Assay(
        var=new_original_var,
        layers=original_assay.layers.copy(),
        feature_id_col=original_assay.feature_id_col 
    )
    
    # Create new container
    new_assays = container.assays.copy()
    new_assays[assay_name] = new_original_assay
    new_assays[new_assay_name] = pca_assay
    
    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        history=list(container.history)
    )
    
    new_container.log_operation(
        action="pca",
        params={
            "source_assay": assay_name,
            "source_layer": base_layer_name,
            "n_components": n_components,
            "center": center,
            "scale": scale,
            "method": method_used, # Fix 1: Use actual method
            "precision": "float64"
        },
        description=f"Performed PCA on {assay_name}/{base_layer_name}, created {new_assay_name}. Loadings added to {assay_name}.var with prefix '{new_assay_name}_'."
    )
    
    return new_container
