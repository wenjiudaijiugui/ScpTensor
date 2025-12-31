from typing import Optional, Tuple, Union, Callable, Dict, Any
import numpy as np
import polars as pl
import scipy.sparse
import scipy.optimize
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scptensor.core.structures import ScpContainer, ScpMatrix, Assay
import numba
from numba import prange

@numba.njit(fastmath=True, parallel=True)
def optimize_layout_numba(
    embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    n_epochs: int,
    n_vertices: int,
    epochs_per_sample: np.ndarray,
    a: float,
    b: float,
    rng_state: np.ndarray,
    repulsion_strength: float = 1.0,
    initial_alpha: float = 1.0,
    negative_sample_rate: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    
    dim = embedding.shape[1]
    alpha = initial_alpha
    
    n_edges = head.shape[0]
    
    # Initialize epoch_of_next_sample
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        # Decay learning rate
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
        
        for i in prange(n_edges):
            if epoch_of_next_sample[i] <= n:
                # Get nodes
                j = head[i]
                k = tail[i]
                
                current = embedding[j]
                other = embedding[k]
                
                dist_squared = 0.0
                for d in range(dim):
                    dist_squared += (current[d] - other[d]) ** 2
                
                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * (dist_squared ** (b - 1.0))
                    grad_coeff /= (a * (dist_squared ** b) + 1.0)
                else:
                    grad_coeff = 0.0
                
                for d in range(dim):
                    grad_d = grad_coeff * (current[d] - other[d])
                    grad_d = min(4.0, max(-4.0, grad_d))
                    
                    embedding[j, d] += grad_d * alpha
                    # In symmetric graph, we only update head to avoid double counting.
                    # The symmetric edge (k, j) will be processed separately.
                
                epoch_of_next_sample[i] += epochs_per_sample[i]
                
                # Negative sampling
                # Standard UMAP performs negative_sample_rate negative samples for each positive sample
                for p in range(negative_sample_rate):
                    # Generate random negative sample
                    # Simple LCG step or similar for randomness
                    k_neg = (rng_state[0] + rng_state[1] * i + rng_state[2] * p + n) % n_vertices
                    if k_neg < 0:
                        k_neg += n_vertices
                        
                    if j == k_neg:
                        continue
                        
                    other_neg = embedding[k_neg]
                    
                    dist_squared_neg = 0.0
                    for d in range(dim):
                        dist_squared_neg += (current[d] - other_neg[d]) ** 2
                        
                    if dist_squared_neg > 0.0:
                        grad_coeff_neg = 2.0 * repulsion_strength * b
                        grad_coeff_neg /= (0.001 + dist_squared_neg) * (1.0 + a * (dist_squared_neg ** b))
                    else:
                        grad_coeff_neg = 0.0
                        
                    for d in range(dim):
                        grad_d_neg = grad_coeff_neg * (current[d] - other_neg[d])
                        grad_d_neg = min(4.0, max(-4.0, grad_d_neg))
                        
                        embedding[j, d] += grad_d_neg * alpha
                        # Do not update negative sample
                
    return embedding

def find_ab_params(spread: float, min_dist: float) -> Tuple[float, float]:
    """
    Fit a, b parameters for the UMAP curve.
    
    The curve 1 / (1 + a * x^(2b)) should approximate:
    1.0 if x <= min_dist
    exp(-(x - min_dist) / spread) if x > min_dist
    """
    
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    
    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    
    # Create target curve
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    
    params, _ = scipy.optimize.curve_fit(curve, xv, yv)
    return params[0], params[1]

def compute_sigmas_and_rhos(
    knn_dists: np.ndarray,
    k: int,
    local_connectivity: float = 1.0,
    n_iter: int = 64,
    tol: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rho and sigma for each point using binary search.
    
    Args:
        knn_dists: (N, k) array of distances to nearest neighbors.
        k: Number of neighbors.
        local_connectivity: Parameter for rho calculation. Should be integer for this implementation.
        n_iter: Max iterations for binary search.
        tol: Tolerance for binary search.
        
    Returns:
        sigmas: (N,) array.
        rhos: (N,) array.
    """
    n_samples = knn_dists.shape[0]
    target = np.log2(k)
    
    rhos = np.zeros(n_samples, dtype=np.float32)
    sigmas = np.zeros(n_samples, dtype=np.float32)
    
    # Rho is the distance to the first nearest neighbor (or local_connectivity-th)
    # We enforce rho > 0
    
    # In standard UMAP, rho is dist to the nearest neighbor.
    # If local_connectivity > 1, it might interpolate.
    # Here we stick to standard simple implementation:
    # rho_i = dist to the first neighbor that is not self (index 1 if index 0 is self)
    
    # Assuming knn_dists includes self at index 0 (dist 0)
    
    for i in range(n_samples):
        # Find rho
        # We look for the first non-zero distance, or use the local_connectivity rule
        # Standard UMAP: rho_i = distance to the k-th nearest neighbor where k=local_connectivity
        # But typically local_connectivity=1, so it's the 1st NN.
        
        # Let's approximate local_connectivity interpolation if needed, but for now take the floor
        # NOTE: This is an engineering simplification. Theoretical UMAP interpolates distances if local_connectivity is non-integer.
        # Here we simply take the floor, which is exact for the default local_connectivity=1.0.
        lo = int(np.floor(local_connectivity))
        if lo < 1:
            lo = 1
        if lo >= k:
            lo = k - 1
            
        rhos[i] = knn_dists[i, lo] if knn_dists[i, lo] > 0 else 1e-6 # Avoid zero rho if possible
        
        # Binary search for sigma
        lo_val = 0.0
        hi_val = np.inf
        mid_val = 1.0
        
        # Smooth knn distance summation
        # sum_{j=1}^k exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)
        
        for _ in range(n_iter):
            # Compute sum of weights
            # d_ij are in knn_dists[i]
            
            # Optimisation: d - rho
            diffs = knn_dists[i] - rhos[i]
            diffs[diffs < 0] = 0 # max(0, d - rho)
            
            # exp(-diff / sigma)
            # Add epsilon to sigma to avoid division by zero
            weights = np.exp(-diffs / (mid_val + 1e-6))
            psum = np.sum(weights)
            
            if np.abs(psum - target) < tol:
                break
            
            if psum > target:
                # P sum too big -> sigma too big? 
                # exp(-x/sigma). if sigma increases, x/sigma decreases, -x/sigma increases (less negative), exp increases.
                # So if sum > target, sigma is too big? 
                # Wait.
                # x=10. sigma=1 => exp(-10). sigma=10 => exp(-1). exp(-1) > exp(-10).
                # So larger sigma => larger weights => larger sum.
                # If psum > target, we need to decrease sigma.
                hi_val = mid_val
                mid_val = (lo_val + hi_val) / 2.0
            else:
                # psum < target, we need to increase sigma
                lo_val = mid_val
                if hi_val == np.inf:
                    mid_val *= 2.0
                else:
                    mid_val = (lo_val + hi_val) / 2.0
        
        sigmas[i] = mid_val
        
    return sigmas, rhos

def compute_membership_strengths(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
    n_samples: int,
    set_op_mix_ratio: float = 1.0
) -> scipy.sparse.coo_matrix:
    """
    Construct the fuzzy simplicial set (weighted graph).
    """
    n_neighbors = knn_indices.shape[1]
    
    rows = np.zeros(n_samples * n_neighbors, dtype=np.int32)
    cols = np.zeros(n_samples * n_neighbors, dtype=np.int32)
    vals = np.zeros(n_samples * n_neighbors, dtype=np.float32)
    
    for i in range(n_samples):
        for j in range(n_neighbors):
            neighbor_idx = knn_indices[i, j]
            dist = knn_dists[i, j]
            
            if neighbor_idx == -1:
                continue # Should not happen with proper KNN
                
            val = 0.0
            # Theoretical consistency: self-loop should have membership strength 1.0
            # In standard implementations, this might be set to 0 for graph layout, 
            # but mathematically it should be 1.
            val = np.exp(-max(0, dist - rhos[i]) / (sigmas[i] + 1e-6))
            
            idx = i * n_neighbors + j
            rows[idx] = i
            cols[idx] = neighbor_idx
            vals[idx] = val
            
    graph = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    graph.eliminate_zeros()
    
    # Symmetrization
    # P = P + P.T - P * P.T (element-wise)
    # For sparse matrices, A + B is easy. A * B elementwise is multiply.
    
    transpose = graph.transpose()
    
    # P + P.T - P * P.T
    # Note: scipy sparse multiplication is matrix multiplication, not elementwise.
    # Elementwise multiplication for sparse matrices:
    prod = graph.multiply(transpose)
    
    # Union: A + B - A*B
    res_union = graph + transpose - prod
    # Intersection: A * B
    res_intersection = prod
    
    result = res_union * set_op_mix_ratio + res_intersection * (1.0 - set_op_mix_ratio)
    return result.tocoo()

def spectral_layout(
    graph: scipy.sparse.spmatrix,
    dim: int,
    random_state: Any,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Initialize embedding using spectral decomposition of the graph Laplacian.
    
    Uses the Normalized Laplacian (L_sym = I - D^-1/2 A D^-1/2) as recommended
    by Belkin & Niyogi (2003) and standard UMAP implementations.
    """
    n_samples = graph.shape[0]
    
    # Reference: Belkin & Niyogi, "Laplacian Eigenmaps for Dimensionality Reduction", 2003
    # Remove self-loops to strictly follow the definition of the Normalized Laplacian
    graph.setdiag(0)
    graph.eliminate_zeros()
    
    # 1. Compute degrees
    # graph is symmetric or assumed to be adjacency
    degrees = np.array(graph.sum(axis=1)).flatten()
    
    # 2. Compute D^-1/2
    # Handle division by zero
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    
    D_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    
    # 3. Construct Normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
    # A = graph
    # We want eigenvectors of L corresponding to smallest eigenvalues.
    # This is equivalent to eigenvectors of D^-1/2 * A * D^-1/2 corresponding to LARGEST eigenvalues.
    # Because L = I - M, if Mv = \lambda v, then Lv = (1-\lambda)v.
    # Maximize \lambda (close to 1) => Minimize 1-\lambda (close to 0).
    
    # Using M = D^-1/2 * A * D^-1/2 is numerically more stable for finding largest eigenvalues (LM)
    # than finding smallest eigenvalues (SM) of L.
    
    M = D_inv_sqrt @ graph @ D_inv_sqrt
    
    k = dim + 1
    if n_samples <= k:
        # For very small datasets, use dense solver or fallback
        # If n_samples is tiny, spectral layout is trivial or less useful
        # But to avoid error, let's use dense solver if feasible
        try:
            eigenvalues, eigenvectors = scipy.linalg.eigh(
                M.toarray(), 
                subset_by_index=(n_samples - k, n_samples - 1)
            )
        except:
            return random_state.uniform(low=-10, high=10, size=(n_samples, dim)).astype(np.float32)
    else:
        try:
            # 'LM' = Largest Magnitude.
            # Note: The largest eigenvalue of normalized adjacency is 1.
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                M, k=k, which='LM', tol=1e-4, maxiter=n_samples * 5,
                v0=random_state.uniform(size=n_samples)
            )
        except (scipy.sparse.linalg.ArpackNoConvergence, ValueError, TypeError):
            print("Spectral initialization failed (Arpack), falling back to random.")
            return random_state.uniform(low=-10, high=10, size=(n_samples, dim)).astype(np.float32)
            
    # Sort by eigenvalues descending (closest to 1 first)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    
    # The first eigenvector (index 0) corresponds to eigenvalue ~1.
    # It is related to the stationary distribution. We skip it for embedding.
    raw_eigenvectors = eigenvectors[:, 1:dim+1]
    
    # Restore the eigenvectors to the original space: y = D^-1/2 * v
    # This is necessary because we solved the generalized eigenvalue problem Ly = lambda Dy
    # via the symmetric normalized Laplacian L_sym = D^-1/2 L D^-1/2.
    
    # NOTE: Standard UMAP implementation actually skips this step for layout initialization!
    # Multiplying by D^-1/2 tends to contract high density regions too much.
    # We use the eigenvectors of the normalized Laplacian directly.
    # embedding = D_inv_sqrt @ raw_eigenvectors
    embedding = raw_eigenvectors
    
    # Spectral initialization is often scaled.
    # Standard UMAP scaling:
    # 1. Scale to standard deviation
    # 2. Scale to fit spread
    
    # We follow Kobak & Linderman (2021) and use standard deviation-based scaling
    # to be more robust to outliers.
    # embedding = embedding - np.mean(embedding, axis=0)
    # std = np.std(embedding)
    # if std == 0:
    #     std = 1.0
    # embedding = 10.0 * (embedding / std) 
    
    # Correction V7: Use max absolute value scaling to keep within [-10, 10]
    # This prevents the initial layout from being too spread out or too concentrated.
    expansion = 10.0 / np.abs(embedding).max()
    embedding = (embedding * expansion).astype(np.float32)
    
    # Add noise
    embedding += random_state.normal(scale=0.0001, size=embedding.shape)
    
    return embedding.astype(np.float32)

def optimize_layout(
    embedding: np.ndarray,
    graph: scipy.sparse.coo_matrix,
    n_epochs: int,
    learning_rate: float,
    a: float,
    b: float,
    repulsion_strength: float,
    negative_sample_rate: int,
    random_state: Any,
    verbose: bool = False
) -> np.ndarray:
    """
    Perform SGD optimization with Edge Sampling.
    """
    
    # Correction V7: Do not use upper triangular. Use full symmetric graph.
    # graph = scipy.sparse.triu(graph, format='coo')
    # Use the full symmetric graph to iterate over all directed edges (i, j).
    # Standard UMAP iterates over each directed edge in the symmetric graph.
    if not scipy.sparse.isspmatrix_coo(graph):
        graph = graph.tocoo()
    
    # Extract edges
    head = graph.row
    tail = graph.col
    weights = graph.data
    n_vertices = embedding.shape[0]
    
    # Calculate epochs per sample for each edge based on weight
    # max_weight = weights.max()
    # epochs_per_sample = np.full(weights.shape, n_epochs, dtype=np.float32)
    # epochs_per_sample[weights > 0] = n_epochs / weights[weights > 0]
    
    # Standard UMAP logic:
    # if w=1, sample once per epoch. (epochs_per_sample = 1)
    # if w=0.5, sample once every 2 epochs. (epochs_per_sample = 2)
    # We want roughly w * n_epochs samples total.
    # So we step by 1/w each time?
    # umap-learn uses make_epochs_per_sample which returns the number of epochs to wait between samples.
    # So if weight is 1.0, we wait 1.0 epoch. If weight is 0.5, we wait 2.0 epochs.
    
    weights = weights / weights.max() # Normalize weights just in case
    
    # Standard UMAP logic for epochs_per_sample
    # make_epochs_per_sample in umap-learn:
    # result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    # n_samples = n_epochs * (weights / weights.max())
    # result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    # So if weight is max, n_samples = n_epochs. epochs_per_sample = 1.
    
    epochs_per_sample = np.full(weights.shape, -1.0, dtype=np.float32)
    n_samples_per_edge = n_epochs * (weights / weights.max())
    
    # Avoid division by zero
    valid_mask = n_samples_per_edge > 0
    epochs_per_sample[valid_mask] = float(n_epochs) / n_samples_per_edge[valid_mask]
    
    # Numba requires specific types
    rng_state = random_state.randint(0, 10000, 3).astype(np.int64)
    
    embedding = optimize_layout_numba(
        embedding,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        repulsion_strength,
        learning_rate,
        negative_sample_rate,
        verbose=verbose
    )
    
    return embedding

def umap(
    container: ScpContainer,
    assay_name: str,
    base_layer: str,
    new_layer_name: str = 'umap',
    n_neighbors: int = 15,
    n_components: int = 2,
    metric: str = 'euclidean',
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    min_dist: float = 0.1,
    spread: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: int = 1,
    repulsion_strength: float = 1.0,
    negative_sample_rate: int = 5,
    transform_seed: int = 42,
) -> ScpContainer:
    """
    Uniform Manifold Approximation and Projection (UMAP).
    
    Args:
        container: ScpContainer object.
        assay_name: Name of the assay.
        base_layer: Name of the layer to use.
        new_layer_name: Name for the new embedding layer.
        n_neighbors: Number of neighbors for KNN.
        n_components: Dimensions of the embedding.
        metric: Distance metric.
        n_epochs: Number of optimization epochs.
        learning_rate: Initial learning rate.
        min_dist: Effective minimum distance between embedded points.
        spread: The effective scale of embedded points.
        set_op_mix_ratio: Interpolate between fuzzy union (1.0) and intersection (0.0).
        local_connectivity: Number of nearest neighbors connected at local scale.
        repulsion_strength: Weighting applied to negative samples.
        negative_sample_rate: Number of negative samples per positive sample.
        transform_seed: Random seed.
        
    Returns:
        ScpContainer: Updated container with new layer.
    """
    
    # 1. Validation
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")
        
    matrix = assay.layers[base_layer]
    X = matrix.X
    n_samples = X.shape[0]
    
    # 2. Parameters
    if n_epochs is None:
        if n_samples <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
            
    random_state = check_random_state(transform_seed)
    
    # 3. KNN
    # Using sklearn NearestNeighbors
    # Note: X might be sparse or dense.
    # If metric is 'euclidean', sklearn handles it.
    
    knn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm='auto',
        n_jobs=-1
    )
    knn_model.fit(X)
    knn_dists, knn_indices = knn_model.kneighbors(X)
    
    # 4. Fuzzy Simplicial Set
    sigmas, rhos = compute_sigmas_and_rhos(
        knn_dists, 
        n_neighbors, 
        local_connectivity=local_connectivity
    )
    
    graph = compute_membership_strengths(
        knn_indices, 
        knn_dists, 
        sigmas, 
        rhos, 
        n_samples,
        set_op_mix_ratio=set_op_mix_ratio
    )
    
    # 5. Initialization
    embedding = spectral_layout(
        graph, 
        n_components, 
        random_state, 
        metric
    )
    
    # 6. Optimization
    a, b = find_ab_params(spread, min_dist)
    
    embedding = optimize_layout(
        embedding,
        graph,
        n_epochs,
        learning_rate,
        a,
        b,
        repulsion_strength,
        negative_sample_rate,
        random_state
    )
    
    # 7. Update container
    # ScpMatrix constructor: X, M. obs is not an attribute of ScpMatrix.
    # The resulting embedding has shape (n_samples, n_components).
    # We need a corresponding M matrix (status mask). Assuming all valid for now or copy from X's M if applicable?
    # But M shape must match X shape. X is (N, D_new). X's M is (N, D_old).
    # So we create a new M with all zeros (Valid).
    M_new = np.zeros(embedding.shape, dtype=np.int8)
    
    # Create a new Assay for UMAP embedding because feature dimension changed
    # The new assay will have n_components features.
    # We need to construct a 'var' DataFrame for the new assay.
    
    umap_var_df = pl.DataFrame({
        "feature_id": [f"UMAP_{i+1}" for i in range(n_components)]
    })
    
    umap_assay_name = f"{assay_name}_{new_layer_name}"
    umap_assay = Assay(
        var=umap_var_df,
        layers={
            new_layer_name: ScpMatrix(X=embedding, M=M_new)
        }
    )
    
    container.add_assay(umap_assay_name, umap_assay)
    
    return container
