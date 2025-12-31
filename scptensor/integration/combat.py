from typing import Optional, List, Union
import numpy as np
import polars as pl
from scptensor.core.structures import ScpContainer, ScpMatrix

# 引用: Johnson, W. E., Li, C., & Rabinovic, A. (2007). Biostatistics.
# 引用: Fortin, J.-P., et al. (2017). bioRxiv (neuroCombat implementation).

def combat(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = 'protein',
    base_layer: str = 'raw',
    new_layer_name: Optional[str] = 'combat',
    covariates: Optional[List[str]] = None
) -> ScpContainer:
    
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
         raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")

    X: np.ndarray = assay.layers[base_layer].X.copy() 
    
    obs_df: pl.DataFrame = container.obs
    if batch_key not in obs_df.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in obs.")
    
    batch: Union[List, np.ndarray, pl.Series] = obs_df[batch_key] # type: ignore
    
    # NaN Handling
    if np.isnan(X).any():
        # [Correction]: Use batch-specific mean imputation
        batch_items_unique = batch.unique().to_numpy()
        for b in batch_items_unique:
            batch_idx = np.where(batch.to_numpy() == b)[0]
            batch_data = X[batch_idx]
            batch_mean = np.nanmean(batch_data, axis=0)
            
            # Find NaNs in this batch
            inds = np.where(np.isnan(batch_data))
            if len(inds[0]) > 0:
                # Map local batch indices to global X indices
                global_rows = batch_idx[inds[0]]
                cols = inds[1]
                X[global_rows, cols] = batch_mean[cols]
        
        # Fallback for any remaining NaNs (e.g. if a gene is all-NaN in a batch)
        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            # If global mean is also NaN (all samples NaN), fill with 0
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
    
    dat: np.ndarray = X.T # (n_features, n_samples)
    
    # Design Matrix Construction
    batch_items = batch.unique().to_numpy()
    
    # Use Polars to_dummies
    # pl.Series.to_dummies() returns a DataFrame with dummy columns.
    # We need to ensure we capture all batch levels.
    # Note: batch.to_dummies() column names are {name}_{value}
    batch_info = batch.to_dummies()
    n_batch = batch_info.shape[1]
    n_sample = batch_info.shape[0]
    sample_per_batch = batch_info.sum().to_numpy().flatten()
    
    # Check singleton batches
    if (sample_per_batch < 2).any():
         raise ValueError("ComBat requires at least 2 samples per batch.")

    # Design matrix for biology (Covariates)
    if covariates:
        covar_df = obs_df.select(covariates)
        # Ensure covariates are numeric (get_dummies for categoricals)
        # Polars equivalent: identify string/categorical cols and to_dummies them
        cat_cols = [c for c, t in covar_df.schema.items() if t in (pl.String, pl.Categorical, pl.Object)]
        
        if cat_cols:
             mod = covar_df.to_dummies(columns=cat_cols, drop_first=True)
        else:
             mod = covar_df
        
        # Convert to float and add intercept
        # Note: Polars cast(pl.Float64)
        # We need to convert to pandas-like structure for concatenation or just use numpy directly later?
        # The original code uses pd.concat([batch_info, mod_for_design], axis=1)
        # We will use pl.concat([batch_info, mod], how="horizontal")
        
        # Add intercept
        mod = mod.with_columns(pl.lit(1.0).alias('intercept'))
    else:
        mod = pl.DataFrame({'intercept': np.ones(n_sample)})

    # [Correction 1]: 
    # Standard implementation constructs the design matrix by dropping the intercept 
    # from the biological model to allow inclusion of all batch columns.
    # This avoids rank deficiency caused by Sum(Batch_i) = Intercept.
    
    if 'intercept' in mod.columns:
        mod_for_design = mod.drop('intercept')
    else:
        mod_for_design = mod
        
    design_matrix = pl.concat([batch_info, mod_for_design], how="horizontal")
    X_design = design_matrix.to_numpy().astype(float)
    
    # [Correction 2]: Robust Rank Checking and Coefficients Calculation
    # We rely on lstsq, but we must calculate rank for degrees of freedom correctly.
    # Reference: sva R package logic.
    
    rank_design = np.linalg.matrix_rank(X_design)
    if rank_design < X_design.shape[1]:
        # [Correction]: Raise Error for Rank Deficiency
        # [修正]: 针对秩亏情况抛出错误
        raise ValueError(
            f"Design matrix is rank deficient. Rank: {rank_design}, Cols: {X_design.shape[1]}.\n"
            "This indicates that batch and biological covariates are confounded.\n"
            "ComBat cannot distinguish between batch effects and biological signals in this case.\n"
            "设计矩阵秩亏。这意味着批次效应与生物学协变量存在混淆。\n"
            "在这种情况下，ComBat 无法区分批次效应和生物学信号。"
        )
    
    # B_hat: (n_params, n_features)
    B_hat = np.linalg.lstsq(X_design, dat.T, rcond=None)[0].T
    
    # Identification of indices
    batch_indices = list(range(n_batch))
    covar_indices = list(range(n_batch, design_matrix.shape[1]))
    
    B_batch = B_hat[:, batch_indices]
    B_covar = B_hat[:, covar_indices]
    X_covar = X_design[:, covar_indices]
    
    # Calculate Grand Mean and Variance
    # [Correction]: Use mean of fitted values to ensure physical meaning with covariates.
    # [修正]: 使用拟合值的均值，以确保在包含协变量时的物理意义。
    grand_mean = np.dot((design_matrix.to_numpy().astype(float) @ B_hat.T).T, np.ones(n_sample)) / n_sample
    
    # Residuals
    residuals = dat - (X_design @ B_hat.T).T
    
    # [Correction 3]: Correct Degrees of Freedom for Sigma
    # sigma = sqrt( SSE / (N - rank) )
    sigma = np.sqrt(np.sum(residuals**2, axis=1) / (n_sample - rank_design))
    
    # Handling zero variance (safety)
    sigma[sigma == 0] = 1e-8
    
    # Standardization
    covar_effect = (X_covar @ B_covar.T).T
    Z = (dat - grand_mean[:, None] - covar_effect) / sigma[:, None]
    
    # --- Empirical Bayes Estimation ---
    
    gamma_hat = np.zeros((n_batch, dat.shape[0]))
    delta_hat = np.zeros((n_batch, dat.shape[0]))
    
    for i, b in enumerate(batch_items):
        idx = np.where(batch.to_numpy() == b)[0]
        Z_batch = Z[:, idx]
        gamma_hat[i] = np.mean(Z_batch, axis=1)
        delta_hat[i] = np.var(Z_batch, axis=1, ddof=1)
        
    # Method of Moments for Priors
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    
    # Inverse Gamma priors
    # Reference: Johnson et al. 2007, Supp Info.
    delta_mean_prior = np.mean(delta_hat, axis=1)
    delta_var_prior = np.var(delta_hat, axis=1, ddof=1)
    
    # Safety for division
    delta_var_prior[delta_var_prior == 0] = 1e-8
    
    a_prior = (delta_mean_prior**2 / delta_var_prior) + 2
    b_prior = delta_mean_prior * (a_prior - 1)
    
    # Improved Vectorized Solver
    def solve_eb(g_hat, d_hat, g_bar, t2, a, b, n, conv=1e-4):
        g_old = g_hat.copy()
        d_old = d_hat.copy()
        
        # Broadcast scalars to arrays
        # Ensure inputs are arrays or properly broadcastable scalars
        # In this loop context, g_bar, t2, a, b are scalars (float)
        # We don't need [:, None] for scalars.
        # But if they were arrays (e.g. per-gene priors?), we would.
        # Given the usage below:
        # gamma_bar[i] is a scalar (mean of gamma_hat for batch i)
        # So we just ensure they are treated as such.
        
        # If they are scalars, just use them as is. 
        # If they are arrays, ensure shape matches g_hat
        
        change = 1
        count = 0
        while change > conv and count < 100:
            # Post Mean Gamma
            # g_new = (n * t2 * g_hat + d_old * g_bar) / (n * t2 + d_old)
            g_new = (n * t2 * g_hat + d_old * g_bar) / (n * t2 + d_old)
            
            # Post Mean Delta
            # sum2 = (n - 1) * d_hat + n * (g_hat - g_new)**2
            sum2 = (n - 1) * d_hat + n * (g_hat - g_new)**2
            d_new = (b + 0.5 * sum2) / (a + n/2)
            
            # [Correction]: Use relative error to avoid instability when g_old is close to 0
            # [修正]: 使用相对误差以避免当 g_old 接近 0 时的不稳定性
            change = np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)) + \
                     np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8))
            
            g_old = g_new
            d_old = d_new
            count += 1
            
        return g_new, d_new

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    
    for i, b in enumerate(batch_items):
        idx = np.where(batch.to_numpy() == b)[0]
        n_samples = len(idx)
        
        # Inputs need to be shaped correctly for broadcasting
        # g_hat[i]: (n_features,) -> (n_features, 1)
        g_h = gamma_hat[i][:, None]
        d_h = delta_hat[i][:, None]
        
        # Run solver
        g_s, d_s = solve_eb(
            g_h, d_h, 
            gamma_bar[i], t2[i], 
            a_prior[i], b_prior[i], 
            n_samples
        )
        gamma_star[i] = g_s.flatten()
        delta_star[i] = d_s.flatten()

    # Final Adjustment
    out_data = np.zeros_like(Z)
    for i, b in enumerate(batch_items):
        idx = np.where(batch.to_numpy() == b)[0]
        # (Z - gamma_star) / sqrt(delta_star)
        out_data[:, idx] = (Z[:, idx] - gamma_star[i][:, None]) / np.sqrt(delta_star[i][:, None])
        
    # De-standardize: Add back Grand Mean and Covariates (Biological Signal)
    out_data = out_data * sigma[:, None] + grand_mean[:, None] + covar_effect
    
    X_corrected = out_data.T
    
    new_matrix = ScpMatrix(X=X_corrected, M=assay.layers[base_layer].M.copy())
    container.assays[assay_name].add_layer(new_layer_name, new_matrix) # type: ignore
    
    # Logging (keep original logic)
    container.log_operation(
        action="integration_combat",
        params={"batch_key": batch_key, "covariates": covariates},
        description=f"ComBat batch correction."
    )
    
    return container