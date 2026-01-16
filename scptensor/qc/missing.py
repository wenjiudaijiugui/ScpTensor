"""Missing value analysis for single-cell proteomics data QC.

This module provides comprehensive missing value analysis based on mask matrix
codes to track different types of missing values throughout the analysis pipeline.

Mask codes reference:
    VALID (0): Valid, detected values
    MBR (1): Match Between Runs missing
    LOD (2): Below Limit of Detection
    FILTERED (3): Filtered out by quality control
    IMPUTED (5): Imputed/filled value

References
----------
Vanderaa, C., & Gatto, L. (2023). Revisiting the Thorny Issue of Missing
Values in Single-Cell Proteomics. arXiv:2304.06654
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.core.structures import MaskCode, ScpContainer, ScpMatrix


@dataclass
class MissingValueReport:
    """Missing value analysis report.

    Attributes
    ----------
    total_missing_rate : float
        Overall proportion of missing values (non-VALID codes).
    valid_rate : float
        Proportion of VALID (0) values.
    mbr_rate : float
        Proportion of MBR (1) missing values.
    lod_rate : float
        Proportion of LOD (2) missing values.
    imputed_rate : float
        Proportion of IMPUTED (5) values.
    feature_missing_rate : np.ndarray
        Missing rate for each feature [n_features].
    structural_missing_features : list[str]
        List of feature IDs that are structurally missing (100% missing).
    sample_missing_rate : np.ndarray
        Missing rate for each sample [n_samples].
    samples_with_high_missing : list[str]
        List of sample IDs with high missing rates (>50%).
    """

    total_missing_rate: float
    valid_rate: float
    mbr_rate: float
    lod_rate: float
    imputed_rate: float
    feature_missing_rate: np.ndarray
    structural_missing_features: list[str]
    sample_missing_rate: np.ndarray
    samples_with_high_missing: list[str]


def _get_layer(assay, layer_name: str | None = None):
    """Get a layer from assay, defaulting to 'raw' or first available layer.

    Parameters
    ----------
    assay : Assay
        The Assay object.
    layer_name : str | None, default None
        Specific layer name to retrieve, or None for default.

    Returns
    -------
    ScpMatrix
        The ScpMatrix layer.

    Raises
    ------
    LayerNotFoundError
        If no layers exist in the assay.
    """
    from scptensor.core.exceptions import LayerNotFoundError

    if layer_name:
        if layer_name not in assay.layers:
            raise LayerNotFoundError(layer_name, "<assay>")
        return assay.layers[layer_name]

    return assay.layers.get("raw") or next(iter(assay.layers.values()), None)


def _get_mask_matrix(matrix: ScpMatrix) -> np.ndarray | sp.spmatrix:
    """Get mask matrix from ScpMatrix, returning zeros if None.

    Parameters
    ----------
    matrix : ScpMatrix
        The matrix object.

    Returns
    -------
    np.ndarray | sp.spmatrix
        Mask matrix with all values assumed VALID (0) if M is None.
    """
    if matrix.M is not None:
        return matrix.M

    if sp.issparse(matrix.X):
        return sp.csr_matrix(matrix.X.shape, dtype=np.int8)
    return np.zeros(matrix.X.shape, dtype=np.int8)


def _count_mask_codes(
    M: np.ndarray | sp.spmatrix,  # noqa: N803 - M is standard notation for mask matrix
    code: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count occurrences of a specific mask code per feature and per sample.

    Parameters
    ----------
    M : np.ndarray | sp.spmatrix
        Mask matrix.
    code : int
        Mask code to count.

    Returns
    -------
    per_feature : np.ndarray
        Count of code per feature [n_features].
    per_sample : np.ndarray
        Count of code per sample [n_samples].
    """
    if sp.issparse(M):
        M = M.tocsr()  # type: ignore[union-attr]  # noqa: N806
        # Create binary mask for this code
        code_mask = M.data == code
        # Get counts per feature (column sum)
        data_mask = (M.data == code).astype(np.int8)
        M_copy = M.copy()  # noqa: N806
        M_copy.data = data_mask
        per_feature = np.array(M_copy.sum(axis=0)).flatten()
        per_sample = np.array(M_copy.sum(axis=1)).flatten()
    else:
        code_mask = code == M
        per_feature = np.sum(code_mask, axis=0)
        per_sample = np.sum(code_mask, axis=1)

    return per_feature, per_sample


def analyze_missing_types(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
) -> ScpContainer:
    """Analyze mask matrix to identify different types of missing values.

    This function analyzes the mask matrix and adds statistics for each
    mask code type to the assay's var DataFrame. This enables detailed
    tracking of missing value provenance throughout the analysis pipeline.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer containing the mask matrix.

    Returns
    -------
    ScpContainer
        New container with missing type statistics added to assay var:
        - 'mask_valid_count': Count of VALID (0) values per feature
        - 'mask_mbr_count': Count of MBR (1) missing per feature
        - 'mask_lod_count': Count of LOD (2) missing per feature
        - 'mask_filtered_count': Count of FILTERED (3) per feature
        - 'mask_imputed_count': Count of IMPUTED (5) per feature
        - 'mask_valid_rate': Proportion of VALID values per feature
        - 'mask_missing_rate': Proportion of non-VALID values per feature

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.

    Examples
    --------
    >>> container = analyze_missing_types(container, assay_name="protein")
    >>> mbr_rates = container.assays['protein'].var['mask_mbr_count'].to_numpy()
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    matrix = assay.layers[layer_name]
    M = _get_mask_matrix(matrix)  # noqa: N806
    n_samples = M.shape[0]

    # Count each mask code type per feature
    valid_count, _ = _count_mask_codes(M, MaskCode.VALID.value)
    mbr_count, _ = _count_mask_codes(M, MaskCode.MBR.value)
    lod_count, _ = _count_mask_codes(M, MaskCode.LOD.value)
    filtered_count, _ = _count_mask_codes(M, MaskCode.FILTERED.value)
    imputed_count, _ = _count_mask_codes(M, MaskCode.IMPUTED.value)

    # Compute rates
    valid_rate = valid_count / n_samples
    total_missing = n_samples - valid_count
    missing_rate = total_missing / n_samples

    # Add results to var
    new_var = assay.var.with_columns(
        pl.Series("mask_valid_count", valid_count),
        pl.Series("mask_mbr_count", mbr_count),
        pl.Series("mask_lod_count", lod_count),
        pl.Series("mask_filtered_count", filtered_count),
        pl.Series("mask_imputed_count", imputed_count),
        pl.Series("mask_valid_rate", valid_rate),
        pl.Series("mask_missing_rate", missing_rate),
    )

    # Create new container with updated var
    new_assay = assay.subset(np.arange(assay.n_features), copy_data=False)
    new_assay.var = new_var

    new_assays = {
        name: new_assay if name == assay_name else a for name, a in container.assays.items()
    }

    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        links=list(container.links),
        history=list(container.history),
        sample_id_col=container.sample_id_col,
    )

    new_container.log_operation(
        action="analyze_missing_types",
        params={"assay": assay_name, "layer_name": layer_name},
        description=f"Analyzed missing types for {assay.n_features} features. "
        f"Overall missing rate: {np.mean(missing_rate):.3f}.",
    )

    return new_container


def compute_missing_stats(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    high_missing_threshold: float = 0.5,
) -> MissingValueReport:
    """Compute comprehensive missing value statistics.

    This function analyzes the mask matrix to compute global, feature-level,
    and sample-level missing value statistics. It identifies structural missing
    (features that are always missing) and samples with high missing rates.

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer containing the mask matrix.
    high_missing_threshold : float, default 0.5
        Threshold for identifying samples with high missing rates.

    Returns
    -------
    MissingValueReport
        Report object containing:
        - Overall missing rates by type (VALID, MBR, LOD, IMPUTED)
        - Per-feature missing rates
        - Structural missing features (100% missing)
        - Per-sample missing rates
        - Samples with high missing rates

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If high_missing_threshold is not in [0, 1].

    Examples
    --------
    >>> report = compute_missing_stats(container, assay_name="protein")
    >>> print(f"Total missing: {report.total_missing_rate:.2%}")
    >>> print(f"MBR missing: {report.mbr_rate:.2%}")
    >>> print(f"Structural missing: {len(report.structural_missing_features)}")
    """
    if high_missing_threshold < 0 or high_missing_threshold > 1:
        raise ScpValueError(
            f"high_missing_threshold must be in [0, 1], got {high_missing_threshold}.",
            parameter="high_missing_threshold",
            value=high_missing_threshold,
        )

    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    matrix = assay.layers[layer_name]
    M = _get_mask_matrix(matrix)  # noqa: N806
    n_samples, n_features = M.shape
    total_values = n_samples * n_features

    # Convert to dense for easier computation if sparse
    if sp.issparse(M):
        M = M.toarray()  # type: ignore[union-attr]  # noqa: N806

    # Count each mask code globally
    n_valid = np.sum(MaskCode.VALID.value == M)
    n_mbr = np.sum(MaskCode.MBR.value == M)
    n_lod = np.sum(MaskCode.LOD.value == M)
    n_imputed = np.sum(MaskCode.IMPUTED.value == M)

    # Compute rates
    valid_rate = n_valid / total_values
    mbr_rate = n_mbr / total_values
    lod_rate = n_lod / total_values
    imputed_rate = n_imputed / total_values
    total_missing_rate = 1.0 - valid_rate

    # Feature-level missing rate (proportion of non-VALID per feature)
    is_valid = MaskCode.VALID.value == M
    feature_missing_rate = 1.0 - (np.sum(is_valid, axis=0) / n_samples)

    # Identify structural missing features (100% missing)
    structural_mask = feature_missing_rate >= 1.0
    structural_feature_indices = np.where(structural_mask)[0]
    structural_missing_features = (
        assay.feature_ids[structural_feature_indices].to_list()
        if len(structural_feature_indices) > 0
        else []
    )

    # Sample-level missing rate (proportion of non-VALID per sample)
    sample_missing_rate = 1.0 - (np.sum(is_valid, axis=1) / n_features)

    # Identify samples with high missing rates
    high_missing_mask = sample_missing_rate > high_missing_threshold
    high_missing_indices = np.where(high_missing_mask)[0]
    samples_with_high_missing = (
        container.sample_ids[high_missing_indices].to_list()
        if len(high_missing_indices) > 0
        else []
    )

    return MissingValueReport(
        total_missing_rate=total_missing_rate,
        valid_rate=valid_rate,
        mbr_rate=mbr_rate,
        lod_rate=lod_rate,
        imputed_rate=imputed_rate,
        feature_missing_rate=feature_missing_rate,
        structural_missing_features=structural_missing_features,
        sample_missing_rate=sample_missing_rate,
        samples_with_high_missing=samples_with_high_missing,
    )


def report_missing_values(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
    by: str | None = None,
    detection_threshold: float = 0.0,
) -> pl.DataFrame:
    """Generate missing value report similar to scp R package.

    This function generates a comprehensive missing value report with optional
    grouping. For each group, it computes:
    - LocalSensitivityMean: Mean number of detected features per sample
    - LocalSensitivitySd: Standard deviation of detected features
    - TotalSensitivity: Total unique features detected in the group
    - Completeness: Data completeness (1 - missing rate)
    - NumberCells: Number of samples in the group

    Parameters
    ----------
    container : ScpContainer
        Input container with data to analyze.
    assay_name : str, default "protein"
        Name of assay containing the layer.
    layer_name : str, default "raw"
        Name of layer containing the data.
    by : str | None, default None
        Column name in obs to group samples by. If None, returns report
        for all samples as a single group.
    detection_threshold : float, default 0.0
        Value threshold for considering a value as detected.

    Returns
    -------
    pl.DataFrame
        Report DataFrame with columns:
        - 'group': Group name (or 'all' if by is None)
        - 'LocalSensitivityMean': Mean detected features per sample
        - 'LocalSensitivitySd': Std dev of detected features
        - 'TotalSensitivity': Total unique features in group
        - 'Completeness': Data completeness (0-1)
        - 'NumberCells': Number of samples in group

    Raises
    ------
    AssayNotFoundError
        If assay_name does not exist.
    LayerNotFoundError
        If layer_name does not exist in the assay.
    ScpValueError
        If `by` column does not exist in obs.

    Examples
    --------
    >>> # Report for all samples
    >>> report = report_missing_values(container, assay_name="protein")
    >>> print(report)
    >>>
    >>> # Report grouped by batch
    >>> report = report_missing_values(container, by="batch")
    >>> print(report)
    """
    if assay_name not in container.assays:
        raise AssayNotFoundError(assay_name)

    assay = container.assays[assay_name]
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers)
        raise LayerNotFoundError(
            layer_name,
            assay_name,
            hint=f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}.",
        )

    if by is not None and by not in container.obs.columns:
        raise ScpValueError(
            f"Column '{by}' not found in obs. Available columns: {list(container.obs.columns)}",
            parameter="by",
            value=by,
        )

    data = assay.layers[layer_name].X
    n_features = data.shape[1]

    # Get detection matrix
    if sp.issparse(data):
        if detection_threshold == 0.0:
            detected = data > 0
        else:
            detected = data.toarray() > detection_threshold  # type: ignore[union-attr]
    else:
        detected = detection_threshold < data

    # Convert to dense if needed
    if sp.issparse(detected):
        detected = detected.toarray()  # type: ignore[union-attr]

    # Compute number of detected features per sample
    n_detected = np.sum(detected, axis=1).astype(np.int32)

    # Prepare data for grouping
    if by is None:
        # Single group report
        group_names = ["all"]
        sample_indices = [np.arange(container.n_samples)]
    else:
        # Group by specified column
        group_values = container.obs[by].to_numpy()
        unique_groups = np.unique(group_values)
        group_names = [str(g) for g in unique_groups]
        sample_indices = [np.where(group_values == g)[0] for g in unique_groups]

    # Compute statistics per group
    results = []
    for group_name, indices in zip(group_names, sample_indices, strict=True):
        group_detected = detected[indices, :]
        group_n_detected = n_detected[indices]

        # Local sensitivity statistics
        local_mean = float(np.mean(group_n_detected))
        local_sd = float(np.std(group_n_detected, ddof=1))

        # Total sensitivity (unique features in group)
        total_sensitivity = int(np.any(group_detected, axis=0).sum())

        # Completeness (proportion of detected values)
        completeness = float(np.sum(group_n_detected) / (len(indices) * n_features))

        results.append(
            {
                "group": group_name,
                "LocalSensitivityMean": local_mean,
                "LocalSensitivitySd": local_sd,
                "TotalSensitivity": total_sensitivity,
                "Completeness": completeness,
                "NumberCells": len(indices),
            }
        )

    return pl.DataFrame(results)


__all__ = [
    "MissingValueReport",
    "analyze_missing_types",
    "compute_missing_stats",
    "report_missing_values",
]
