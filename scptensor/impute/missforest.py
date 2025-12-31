from typing import Optional
import numpy as np
import sklearn.ensemble
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay

def missforest(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: Optional[str] = 'imputed_missforest',
    max_iter: int = 10,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 0
) -> ScpContainer:
    """
    Impute missing values using MissForest (Random Forest Imputation).
    
    This implementation follows the MissForest algorithm:
    1. Initial imputation using mean/median.
    2. Iteratively impute each variable (feature) by training a Random Forest
       on observed values of other variables.
    3. Stop when the difference between iterations is small or max_iter is reached.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to use.
        base_layer: Name of the layer containing data with missing values.
        new_layer_name: Name for the new layer with imputed data.
        max_iter: Maximum number of iterations.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of the trees.
        n_jobs: Number of jobs to run in parallel.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level.

    Returns:
        ScpContainer: The updated container with the new layer.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")

    input_matrix = assay.layers[base_layer]
    X_original = input_matrix.X
    M_original = input_matrix.M

    # Check if there are missing values
    if not np.isnan(X_original).any():
        if verbose > 0:
            print("No missing values found. Returning original data.")
        # Even if no missing values, we create the new layer as requested
        new_matrix = ScpMatrix(X=X_original.copy(), M=M_original.copy())
        assay.add_layer(new_layer_name, new_matrix)
        return container

    # --- MissForest Implementation ---
    X_in = X_original.copy()
    n_samples, n_features = X_in.shape
    
    # 1. Initialization: Mean imputation
    missing_mask = np.isnan(X_in)
    col_means = np.nanmean(X_in, axis=0)
    # Handle columns with all NaNs (though unlikely in QC-filtered data)
    col_means[np.isnan(col_means)] = 0.0 
    
    # Fill missing values with column means
    for col_idx in range(n_features):
        X_in[missing_mask[:, col_idx], col_idx] = col_means[col_idx]

    # Sort indices by missing rate (ascending) - not strictly required but common optimization
    # In original paper, variables are sorted by amount of missing values
    missing_counts = missing_mask.sum(axis=0)
    sorted_indices = np.argsort(missing_counts)
    # Filter out columns with no missing values from the loop to save time
    # But they are still used as predictors
    sorted_indices = [idx for idx in sorted_indices if missing_counts[idx] > 0]

    X_old = X_in.copy()
    
    # 2. Iteration
    for i in range(max_iter):
        if verbose > 0:
            print(f"MissForest iteration {i+1}/{max_iter}")
            
        difference = 0.0
        
        for col_idx in sorted_indices:
            # Target variable (current column)
            obs_rows = ~missing_mask[:, col_idx]
            mis_rows = missing_mask[:, col_idx]
            
            if not np.any(mis_rows):
                continue
                
            # Predictors (all other columns)
            # We use current X_in which contains updated values from this iteration
            # and values from previous iteration
            X_train = X_in[obs_rows, :]
            X_train = np.delete(X_train, col_idx, axis=1)
            
            y_train = X_in[obs_rows, col_idx]
            
            X_test = X_in[mis_rows, :]
            X_test = np.delete(X_test, col_idx, axis=1)
            
            # Train Random Forest
            rf = sklearn.ensemble.RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=random_state
            )
            rf.fit(X_train, y_train)
            
            # Predict missing values
            y_pred = rf.predict(X_test)
            
            # Update matrix
            X_in[mis_rows, col_idx] = y_pred

        # Check convergence
        # Normalized Root Mean Squared Difference (NRMSD) or similar
        # Here we use sum of squared differences normalized by total variance or simple L2 norm
        # Original paper uses: sum((X_new - X_old)^2) / sum(X_new^2) for continuous
        
        diff = np.sum((X_in[missing_mask] - X_old[missing_mask]) ** 2)
        norm = np.sum(X_in[missing_mask] ** 2)
        gamma = diff / (norm + 1e-9) # Avoid division by zero
        
        if verbose > 0:
            print(f"  Difference (gamma): {gamma:.6f}")

        if gamma < 1e-4: # Threshold could be a parameter
            if verbose > 0:
                print("Converged.")
            break
            
        X_old = X_in.copy()

    # 3. Create Result
    # M matrix update: 0=Valid, 1=MBR, 2=LOD, 3=Filtered. 
    # Imputed values are effectively "Valid" now for downstream tasks, 
    # but we usually keep the original M to track what was imputed.
    # Or we can introduce a new status if needed. For now, keep M as is.
    
    new_matrix = ScpMatrix(X=X_in, M=M_original.copy())
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="impute_missforest",
        params={
            "assay": assay_name, 
            "max_iter": max_iter, 
            "n_estimators": n_estimators,
            "gamma": gamma
        },
        description=f"MissForest imputation on '{base_layer}' -> '{new_layer_name}'."
    )

    return container
