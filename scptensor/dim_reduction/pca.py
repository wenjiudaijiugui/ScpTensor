"""Principal Component Analysis (PCA) with intelligent SVD solver selection.

This module implements PCA with automatic solver selection based on data characteristics
(matrix size, sparsity, centering) following scikit-learn best practices.

Key features:
- Intelligent SVD solver selection (auto, full, arpack, randomized, covariance_eigh)
- Memory-efficient sparse matrix handling with LinearOperator
- Covariance EIG solver for tall-skinny matrices (n_samples >> n_features)
- Randomized SVD for large dense matrices
- Proper sign convention enforcement for deterministic output

References:
    - Jolliffe, I.T., & Cadima, J. (2016). Principal component analysis:
      a review and recent developments. Philosophical Transactions
      of the Royal Society A, 374(2065), 20150202.
    - Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure
      with randomness: Probabilistic algorithms for constructing approximate
      matrix decompositions. SIAM review, 53(2), 217-288.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackError, LinearOperator, svds

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


# =============================================================================
# TYPE ALIASES AND CONSTANTS
# =============================================================================

SolverType = str  # Literal["full", "arpack", "randomized", "covariance_eigh"]
_BYTES_PER_NNZ = 16  # Approximate: 8 bytes float64 + 4 bytes int32 + 4 bytes int32


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def _check_valid_data(X: np.ndarray | sp.spmatrix) -> None:
    """Check that input data has no NaN or infinite values.

    Args:
        X: Input data matrix (dense or sparse)

    Raises:
        ValueError: If data contains NaN or Inf
    """
    if sp.issparse(X):
        data = X.data
    else:
        data = X.ravel() if isinstance(X, np.ndarray) else X

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values. Please use an imputed layer.")


# =============================================================================
# SOLVER SELECTION LOGIC
# =============================================================================


def _estimate_memory_bytes(X: np.ndarray | sp.spmatrix) -> int:
    """Estimate memory footprint of matrix X in bytes.

    Args:
        X: Input matrix (dense or sparse)

    Returns:
        Estimated memory usage in bytes
    """
    if sp.issparse(X):
        return X.nnz * _BYTES_PER_NNZ
    return X.nbytes


def optimal_svd_solver(
    X: np.ndarray | sp.spmatrix,
    n_components: int,
    center: bool = True,
    max_memory_gb: float = 8.0,
) -> SolverType:
    """Intelligently select optimal SVD solver based on data characteristics.

    Selection logic (inspired by scikit-learn's PCA._fit):
        - Sparse + center: arpack or covariance_eigh
        - Sparse + no center: randomized (TruncatedSVD-like)
        - Dense + tall-skinny (n_features <= 1000, n_samples >= 10*n_features):
          covariance_eigh (compute covariance matrix first)
        - Dense + small (max_dim <= 500): full SVD
        - Dense + large + few components (< 80% of min_dim): randomized
        - Dense + large + many components: full SVD

    Parameters
    ----------
    X : ndarray or spmatrix
        Input matrix (dense or sparse)
    n_components : int
        Number of principal components to compute
    center : bool, optional
        Whether data will be centered before SVD. Default is True.
    max_memory_gb : float, optional
        Maximum memory threshold for full SVD in GB. Default is 8.0.

    Returns
    -------
    str
        Solver name: "full", "arpack", "randomized", or "covariance_eigh"

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> X_dense = np.random.rand(1000, 100)
    >>> optimal_svd_solver(X_dense, n_components=10)
    'covariance_eigh'
    >>> X_sparse = sparse.random(10000, 1000, density=0.01)
    >>> optimal_svd_solver(X_sparse, n_components=50, center=True)
    'arpack'
    """
    n_samples, n_features = X.shape
    min_dim = min(n_samples, n_features)
    max_dim = max(n_samples, n_features)

    # Memory guard: estimate if full SVD would exceed memory limit
    if _estimate_memory_bytes(X) / (1024**3) > max_memory_gb:
        return "randomized"

    # Sparse matrix path
    if sp.issparse(X):
        if center:
            return (
                "arpack"
                if n_components < 100
                else ("covariance_eigh" if n_features <= 5000 else "arpack")
            )
        # Non-centered sparse: TruncatedSVD is optimal
        return "randomized"

    # Dense matrix path

    # Tall and skinny: covariance matrix approach is most efficient
    if n_features <= 1000 and n_samples >= 10 * n_features:
        return "covariance_eigh"

    # Small to medium problems: arpack for compatibility
    if max_dim <= 5000:
        return "arpack"

    # Large with few components: randomized SVD
    if 1 <= n_components < 0.8 * min_dim:
        return "randomized"

    return "full"


# =============================================================================
# LINEAR OPERATOR FOR IMPLICIT CENTERING/SCALING
# =============================================================================


class _CenteredScaledLinearOperator(LinearOperator):
    """LinearOperator that performs implicit centering and scaling.

    This avoids densifying sparse matrices while computing:
        Y = (X - mean) / std

    The transformation is applied implicitly during matrix-vector products,
    which is all that iterative SVD solvers (ARPACK) need.

    Attributes:
        X: Sparse input matrix
        mean: Feature means, shape (n_features,)
        std: Feature standard deviations, shape (n_features,) or None

    Examples
    --------
    >>> from scipy import sparse
    >>> X = sparse.random(1000, 100, density=0.1)
    >>> mean = np.array(X.mean(axis=0)).flatten()
    >>> op = _CenteredScaledLinearOperator(X, mean)
    >>> v = np.random.rand(100)
    >>> y = op @ v  # Implicitly centered
    """

    __slots__ = ("X", "mean", "std", "mean_scaled", "_eps", "_std_safe")

    def __init__(
        self,
        X: sp.spmatrix,
        mean: np.ndarray,
        std: np.ndarray | None = None,
    ) -> None:
        """Initialize the linear operator.

        Args:
            X: Sparse input matrix
            mean: Feature means (subtracted from each column)
            std: Feature standard deviations (optional scaling)
        """
        self.X = X
        self.mean = mean
        self.std = std
        self.shape = X.shape
        self.dtype = X.dtype

        # Pre-compute for efficiency
        self._eps = np.finfo(np.float64).eps
        if std is not None:
            self._std_safe = std.copy()
            self._std_safe[self._std_safe < self._eps] = 1.0
            self.mean_scaled = mean / self._std_safe
        else:
            self._std_safe = None
            self.mean_scaled = mean

    def _matvec(self, v: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Compute y = (X - mean) @ v (with optional scaling).

        Using the identity:
            (X - 1*mean) @ v = X @ v - mean @ (1^T @ v)
                         = X @ v - sum(v) * mean

        Args:
            v: Vector of shape (n_features,)

        Returns:
            Result vector of shape (n_samples,)
        """
        if self._std_safe is not None:
            v_scaled = v / self._std_safe
        else:
            v_scaled = v

        Xv = self.X.dot(v_scaled)
        return Xv - np.dot(self.mean, v_scaled)

    def _rmatvec(self, u: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Compute y = (X - mean)^T @ u (with optional scaling).

        Args:
            u: Vector of shape (n_samples,)

        Returns:
            Result vector of shape (n_features,)
        """
        sum_u = u.sum()
        res = self.X.T.dot(u) - self.mean * sum_u

        if self._std_safe is not None:
            res /= self._std_safe

        return res


# =============================================================================
# SIGN CONVENTION
# =============================================================================


def _flip_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enforce deterministic sign convention on SVD results.

    For each component (row in Vt), find the element with the largest absolute value.
    If this element is negative, multiply both the component (row in Vt) and
    the corresponding score (column in U) by -1.

    This ensures reproducible results across different runs and solvers.

    Reference:
        Bro, R., et al. (2008). Resolving the sign ambiguity in the singular
        value decomposition. Journal of Chemometrics: A History of the
        Singular Value Decomposition, 22(7-8), 724-732.

    Parameters
    ----------
    U : ndarray
        Left singular vectors, shape (n_samples, n_components)
    Vt : ndarray
        Right singular vectors (transposed), shape (n_components, n_features)

    Returns
    -------
    tuple of ndarray
        (U, Vt) with flipped signs for deterministic output
    """
    for k in range(Vt.shape[0]):
        idx = np.argmax(np.abs(Vt[k, :]))
        if Vt[k, idx] < 0:
            Vt[k, :] *= -1
            U[:, k] *= -1
    return U, Vt


# =============================================================================
# SVD SOLVER IMPLEMENTATIONS
# =============================================================================


def _full_svd(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full SVD using LAPACK.

    Parameters
    ----------
    X : ndarray
        Dense centered/scaled matrix
    n_components : int
        Number of components to extract

    Returns
    -------
    tuple of ndarray
        (U_reduced, S_reduced, Vt_reduced)
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components], S[:n_components], Vt[:n_components, :]


def _arpack_svd(
    X: np.ndarray | LinearOperator,
    n_components: int,
    random_state: int | np.random.RandomState | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute truncated SVD using ARPACK.

    Parameters
    ----------
    X : ndarray or LinearOperator
        Dense array or LinearOperator
    n_components : int
        Number of components (must be < min(X.shape))
    random_state : int or RandomState or None, optional
        Random seed for initialization

    Returns
    -------
    tuple of ndarray
        (U_reduced, S_reduced, Vt_reduced)

    Raises
    ------
    ValueError
        If ARPACK fails to converge
    """
    min_dim = min(X.shape)
    k = min(n_components, min_dim - 1)

    if k < 1:
        raise ValueError(
            f"n_components={n_components} is too large for ARPACK with matrix shape {X.shape}"
        )

    try:
        rng = np.random.default_rng(random_state)
        v0 = rng.standard_normal(min_dim)
        U, S, Vt = svds(X, k=k, which="LM", v0=v0)

        # svds returns eigenvalues in increasing order - reverse them
        return U[:, ::-1], S[::-1], Vt[::-1, :]

    except ArpackError as e:
        raise ValueError(
            f"ARPACK failed to converge: {e}. "
            f"Try using a different solver (e.g., 'randomized' or 'full')."
        ) from e


def _randomized_svd(
    X: np.ndarray | LinearOperator,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: int = 4,
    random_state: int | np.random.RandomState | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute randomized SVD using Halko et al. algorithm.

    More efficient than full SVD for large matrices with few components.
    Time complexity: O(m*n*log(k)) vs O(m*n*min(m,n)) for full SVD.

    Parameters
    ----------
    X : ndarray or LinearOperator
        Input matrix (dense or LinearOperator)
    n_components : int
        Number of components to compute
    n_oversamples : int, optional
        Additional random vectors for accuracy. Default is 10.
    n_iter : int, optional
        Power iterations. Default is 4 (recommended 2-8).
    random_state : int or RandomState or None, optional
        Random seed

    Returns
    -------
    tuple of ndarray
        (U_reduced, S_reduced, Vt_reduced)
    """
    from sklearn.utils.extmath import randomized_svd

    U, S, Vt = randomized_svd(
        X,
        n_components=n_components,
        n_oversamples=n_oversamples,
        n_iter=n_iter,
        flip_sign=False,
        random_state=random_state,
    )
    return U, S, Vt


def _covariance_eigh_svd(
    X: np.ndarray,
    mean: np.ndarray,
    n_components: int,
) -> tuple[None, np.ndarray, np.ndarray]:
    """Compute SVD via eigenvalue decomposition of covariance matrix.

    For tall-skinny matrices (n_samples >> n_features), this is more efficient:
        1. Compute C = X.T @ X - n * mean @ mean.T (shape: n_features x n_features)
        2. Compute eigenvalues/vectors of C
        3. Reconstruct SVD from eigendecomposition

    Parameters
    ----------
    X : ndarray
        Input matrix (not centered)
    mean : ndarray
        Feature means (for centering the covariance)
    n_components : int
        Number of components to compute

    Returns
    -------
    tuple
        (None, S_reduced, Vt_reduced)
        Note: U is None and should be reconstructed as X @ V / S
    """
    n_samples, _ = X.shape

    # Compute covariance matrix: C = (X - mean)^T @ (X - mean) / (n-1)
    # Use identity: (X - m)^T @ (X - m) = X^T @ X - n * m @ m^T
    C = (X.T @ X - n_samples * np.outer(mean, mean)) / (n_samples - 1)

    # Eigendecomposition of symmetric matrix
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # eigh returns in ascending order - reverse
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Clip negative eigenvalues (numerical errors)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Reconstruct singular values: S = sqrt(lambda * (n-1))
    S = np.sqrt(eigenvalues * (n_samples - 1))
    Vt = eigenvectors.T

    return None, S[:n_components], Vt[:n_components, :]


# =============================================================================
# VARIANCE COMPUTATION
# =============================================================================


def _compute_sparse_variance(X: sp.spmatrix, mean: np.ndarray, n_samples: int) -> np.ndarray:
    """Compute variance for sparse matrix efficiently.

    Uses: var = E[X^2] - (E[X])^2

    Parameters
    ----------
    X : spmatrix
        Sparse input matrix
    mean : ndarray
        Feature means
    n_samples : int
        Number of samples

    Returns
    -------
    ndarray
        Feature variances
    """
    X_sq = X.copy()
    X_sq.data **= 2
    mean_sq = np.array(X_sq.mean(axis=0)).flatten()
    var = mean_sq - mean**2
    return var * n_samples / (n_samples - 1)


# =============================================================================
# MAIN PCA FUNCTION
# =============================================================================


def reduce_pca(
    container: ScpContainer,
    assay_name: str,
    base_layer_name: str,
    new_assay_name: str = "pca",
    n_components: int = 2,
    center: bool = True,
    scale: bool = False,
    svd_solver: SolverType | None = None,
    random_state: int | np.random.RandomState | None = 42,
    dtype: DTypeLike = np.float64,
) -> ScpContainer:
    """Perform Principal Component Analysis (PCA) with intelligent solver selection.

    The function automatically selects the optimal SVD solver based on data
    characteristics (size, sparsity, centering). This follows scikit-learn's
    best practices while maintaining the ScpTensor data structure.

    Solver Selection:
        - svd_solver=None (auto): Intelligent selection based on data
        - "full": Exact SVD via LAPACK (best for small matrices)
        - "arpack": Iterative SVD via ARPACK (good for sparse + center)
        - "randomized": Randomized SVD via Halko algorithm (fast for large)
        - "covariance_eigh": EVD of covariance (best for tall-skinny data)

    Parameters
    ----------
    container : ScpContainer
        The data container
    assay_name : str
        Name of the assay to use
    base_layer_name : str
        Name of the layer to analyze
    new_assay_name : str, optional
        Name for the new PCA assay. Default is "pca".
    n_components : int, optional
        Number of principal components. Default is 2.
    center : bool, optional
        Whether to center the data. Default is True.
    scale : bool, optional
        Whether to scale to unit variance. Default is False.
    svd_solver : str or None, optional
        SVD solver to use. None for auto-selection.
    random_state : int or RandomState or None, optional
        Random seed for reproducibility. Default is 42.
    dtype : dtype or type, optional
        Data type for computation. Default is np.float64.

    Returns
    -------
    ScpContainer
        New container with PCA results added as a new assay.

    Raises
    ------
    ValueError
        If assay/layer not found, or invalid parameters
    RuntimeError
        If SVD computation fails

    Examples
    --------
    >>> from scptensor import create_test_container
    >>> from scptensor.dim_reduction import reduce_pca
    >>> container = create_test_container()
    >>> result = reduce_pca(container, "proteins", "imputed", n_components=10)
    >>> # Use specific solver
    >>> result = reduce_pca(container, "proteins", "imputed", svd_solver="randomized")
    """
    # ===== VALIDATION =====
    assays = container.assays
    if assay_name not in assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = assays[assay_name]
    if base_layer_name not in assay.layers:
        raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    input_matrix = assay.layers[base_layer_name]
    X = input_matrix.X

    # Check for NaN/Inf
    _check_valid_data(X)

    n_samples, n_features = X.shape
    min_dim = min(n_samples, n_features)

    if n_components > min_dim:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed min(n_samples, n_features) ({min_dim})."
        )

    # ===== SOLVER SELECTION =====
    if svd_solver is None:
        svd_solver = optimal_svd_solver(X, n_components, center=center)
    elif svd_solver not in ("full", "arpack", "randomized", "covariance_eigh"):
        raise ValueError(
            f"svd_solver must be one of: None, 'full', 'arpack', "
            f"'randomized', 'covariance_eigh'. Got: {svd_solver!r}"
        )

    # Sparse matrices only support specific solvers
    if sp.issparse(X) and svd_solver == "full":
        raise ValueError(
            "Sparse matrices do not support 'full' solver. "
            "Use 'arpack', 'randomized', or 'covariance_eigh'."
        )

    # ===== PREPROCESSING =====
    target_dtype = np.dtype(dtype)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    # Compute statistics for centering/scaling
    if center:
        mean = np.array(X.mean(axis=0)).flatten() if sp.issparse(X) else np.mean(X, axis=0)

    if scale:
        if sp.issparse(X):
            std = _compute_sparse_variance(
                X, mean if center is not None else np.zeros(n_features), n_samples
            )
            std = np.sqrt(std)
        else:
            std = np.std(X, axis=0, ddof=1)

        # Handle near-zero std
        eps = np.finfo(target_dtype).eps
        std = np.where(std < eps, 1.0, std)

    # ===== PREPARE DATA FOR SVD =====
    use_linear_operator = False
    X_processed: np.ndarray | sp.spmatrix | None = None
    linear_op: LinearOperator | None = None

    if sp.issparse(X):
        if center or scale:
            use_linear_operator = True
            if X.dtype != target_dtype:
                X = X.astype(target_dtype)
            linear_op = _CenteredScaledLinearOperator(
                X,
                mean if center else np.zeros(n_features),
                std if scale else None,
            )
        else:
            X_processed = X.astype(target_dtype) if X.dtype != target_dtype else X
    else:
        # Dense matrix: copy and center/scale
        if X.dtype == target_dtype:
            X_processed = X.copy()
        else:
            X_processed = X.astype(target_dtype, copy=True)

        if center:
            X_processed -= mean  # type: ignore[operator]
        if scale:
            X_processed /= std  # type: ignore[operator]

    # ===== COMPUTE SVD =====
    U_reduced: np.ndarray | None
    S_reduced: np.ndarray
    Vt_reduced: np.ndarray

    try:
        if svd_solver == "full":
            if use_linear_operator:
                # Must densify for full SVD
                X_dense = X.toarray().astype(target_dtype)
                if center:
                    X_dense -= mean  # type: ignore[operator]
                if scale:
                    X_dense /= std  # type: ignore[operator]
                U_reduced, S_reduced, Vt_reduced = _full_svd(X_dense, n_components)
            else:
                U_reduced, S_reduced, Vt_reduced = _full_svd(
                    X_processed,
                    n_components,  # type: ignore[arg-type]
                )

        elif svd_solver == "arpack":
            operator = linear_op if use_linear_operator else X_processed
            U_reduced, S_reduced, Vt_reduced = _arpack_svd(
                operator,
                n_components,
                random_state,  # type: ignore[arg-type]
            )

        elif svd_solver == "randomized":
            operator = linear_op if use_linear_operator else X_processed
            U_reduced, S_reduced, Vt_reduced = _randomized_svd(
                operator,
                n_components,
                random_state=random_state,  # type: ignore[arg-type]
            )

        else:  # covariance_eigh
            if use_linear_operator:
                # Need to densify or compute differently
                X_dense = X.toarray().astype(target_dtype)
                if center:
                    X_dense -= mean  # type: ignore[operator]
                if scale:
                    X_dense /= std  # type: ignore[operator]
                U_reduced, S_reduced, Vt_reduced = _full_svd(X_dense, n_components)
            else:
                U_temp, S_reduced, Vt_reduced = _covariance_eigh_svd(
                    X,
                    mean if center else np.zeros(n_features),
                    n_components,  # type: ignore[arg-type]
                )
                if U_temp is None:
                    V = Vt_reduced.T
                    U_reduced = (X_processed @ V) / S_reduced  # type: ignore[index, call-arg]
                else:
                    U_reduced = U_temp

    except Exception as e:
        raise RuntimeError(f"SVD computation failed with solver '{svd_solver}': {e}") from e

    # Enforce deterministic sign convention
    if U_reduced is not None:
        U_reduced, Vt_reduced = _flip_signs(U_reduced, Vt_reduced)

    # ===== COMPUTE SCORES AND LOADINGS =====
    scores = U_reduced * S_reduced  # type: ignore[operator]
    loadings = Vt_reduced.T

    # ===== EXPLAINED VARIANCE =====
    if center:
        if scale:
            total_variance = float(n_features)
        elif sp.issparse(X) and use_linear_operator:
            total_variance = np.sum(_compute_sparse_variance(X, mean, n_samples))
        elif X_processed is not None and not sp.issparse(X_processed):
            total_variance = np.sum(np.var(X_processed, axis=0, ddof=1))
        else:
            total_variance = np.sum(_compute_sparse_variance(X, mean, n_samples))
    else:
        # For uncentered data, use total sum of squares
        if sp.issparse(X):
            sq_sum = np.sum(X.data**2)
        elif X_processed is not None:
            sq_sum = np.sum(np.square(X_processed))
        else:
            sq_sum = np.sum(X.data**2)
        total_variance = sq_sum / (n_samples - 1)

    # Eigenvalues and explained variance ratio
    eigenvalues = S_reduced**2 / (n_samples - 1)
    explained_variance_ratio = eigenvalues / total_variance

    variance_ratio_col_name = "explained_variance_ratio" if center else "explained_inertia_ratio"

    # ===== CREATE OUTPUT ASSAY =====
    pc_names = [f"PC{i + 1}" for i in range(n_components)]

    pca_var = pl.DataFrame(
        {
            "pc_name": pc_names,
            "explained_variance": eigenvalues,
            variance_ratio_col_name: explained_variance_ratio,
        }
    )

    scores_M = np.zeros_like(scores, dtype=np.int8)
    scores_matrix = ScpMatrix(X=scores, M=scores_M)

    pca_assay = Assay(
        var=pca_var,
        layers={"scores": scores_matrix},
        feature_id_col="pc_name",
    )

    # ===== STORE LOADINGS IN ORIGINAL ASSAY =====
    # Create loadings DataFrame efficiently
    loading_cols = {
        f"{new_assay_name}_PC{i + 1}_loading": loadings[:, i] for i in range(n_components)
    }
    loadings_df = pl.DataFrame(loading_cols)

    # Update original assay var with loadings
    original_assay = assays[assay_name]
    prefix = f"{new_assay_name}_PC"

    # Remove old loadings for this assay
    existing_cols = original_assay.var.columns
    cols_to_drop = [c for c in existing_cols if c.startswith(prefix) and "_loading" in c]

    original_var_clean = (
        original_assay.var.drop(cols_to_drop) if cols_to_drop else original_assay.var
    )

    new_original_var = pl.concat([original_var_clean, loadings_df], how="horizontal")

    new_original_assay = Assay(
        var=new_original_var,
        layers=original_assay.layers.copy(),
        feature_id_col=original_assay.feature_id_col,
    )

    # ===== CREATE NEW CONTAINER =====
    new_assays = {**assays, assay_name: new_original_assay, new_assay_name: pca_assay}

    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        history=list(container.history),
    )

    # Log operation
    new_container.log_operation(
        action="reduce_pca",
        params={
            "source_assay": assay_name,
            "source_layer": base_layer_name,
            "n_components": n_components,
            "center": center,
            "scale": scale,
            "svd_solver": svd_solver,
            "precision": str(target_dtype),
        },
        description=(
            f"Performed PCA on {assay_name}/{base_layer_name} using '{svd_solver}' solver. "
            f"Created {new_assay_name} assay. Loadings stored in {assay_name}.var."
        ),
    )

    return new_container


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_solver_info() -> dict[str, str]:
    """Get information about available SVD solvers.

    Returns
    -------
    dict
        Dictionary mapping solver names to descriptions
    """
    return {
        "auto": "Automatically select optimal solver based on data characteristics",
        "full": "Full SVD via LAPACK (exact, best for small matrices)",
        "arpack": "Iterative SVD via ARPACK (good for sparse + centered data)",
        "randomized": "Randomized SVD via Halko algorithm (fast for large matrices)",
        "covariance_eigh": "Eigen-decomposition of covariance (best for tall-skinny data)",
    }


__all__ = ["reduce_pca", "pca", "get_solver_info", "optimal_svd_solver"]


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================


def pca(
    container: ScpContainer,
    assay_name: str,
    base_layer_name: str,
    new_assay_name: str = "pca",
    n_components: int = 2,
    center: bool = True,
    scale: bool = False,
    svd_solver: SolverType | None = None,
    random_state: int | np.random.RandomState | None = 42,
    dtype: DTypeLike = np.float64,
) -> ScpContainer:
    """Perform Principal Component Analysis (PCA) with intelligent solver selection.

    .. deprecated:: 0.1.0
        Use :func:`reduce_pca` instead. This function will be removed in a future version.

    Parameters
    ----------
    container : ScpContainer
        The data container
    assay_name : str
        Name of the assay to use
    base_layer_name : str
        Name of the layer to analyze
    new_assay_name : str, optional
        Name for the new PCA assay. Default is "pca".
    n_components : int, optional
        Number of principal components. Default is 2.
    center : bool, optional
        Whether to center the data. Default is True.
    scale : bool, optional
        Whether to scale to unit variance. Default is False.
    svd_solver : str or None, optional
        SVD solver to use. None for auto-selection.
    random_state : int or RandomState or None, optional
        Random seed for reproducibility. Default is 42.
    dtype : dtype or type, optional
        Data type for computation. Default is np.float64.

    Returns
    -------
    ScpContainer
        New container with PCA results added as a new assay.
    """
    import warnings

    warnings.warn(
        "The 'pca' function is deprecated. Use 'reduce_pca' instead. "
        "'pca' will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reduce_pca(
        container=container,
        assay_name=assay_name,
        base_layer_name=base_layer_name,
        new_assay_name=new_assay_name,
        n_components=n_components,
        center=center,
        scale=scale,
        svd_solver=svd_solver,
        random_state=random_state,
        dtype=dtype,
    )
