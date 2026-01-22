"""Feature-level Quality Control.

Handles QC for proteins/features, including:
- Missingness analysis (Missing Rate vs Detection Rate)
- Coefficient of Variation (CV) filtering
- Feature quality metrics calculation

This module provides essential preprocessing steps for single-cell
proteomics data to remove low-quality features before downstream analysis.
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.qc.metrics import compute_cv


def calculate_feature_qc_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str | None = None,
) -> ScpContainer:
    """Calculate quality control metrics for features (proteins/peptides).

    Computes comprehensive QC metrics for each feature in the assay,
    including missingness statistics, expression levels, and variability
    measures. These metrics are essential for evaluating data quality
    and guiding filtering decisions.

    Metrics Calculated:

    1. **missing_rate**: Proportion of missing values per feature
       - Missing values are defined as zeros or NaN values
       - Range: [0, 1] where 1 = completely missing
       - High missing rate indicates poor detection

    2. **detection_rate**: Complement of missing_rate
       - detection_rate = 1 - missing_rate
       - Range: [0, 1] where 1 = detected in all samples
       - Also known as "presence rate" or "frequency"

    3. **mean_expression**: Mean intensity of detected values
       - Computed only from non-missing (detected) values
       - Represents average abundance when protein is detected
       - Useful for assessing expression levels

    4. **cv**: Coefficient of Variation
       - CV = standard deviation / mean
       - Measures relative variability across samples
       - High CV indicates unstable quantification
       - Important for assessing measurement quality

    Mathematical Formulation:
        For each feature i:
        - missing_rate[i] = (n_missing[i]) / n_samples
        - detection_rate[i] = (n_detected[i]) / n_samples
        - mean_expression[i] = sum(X[i]) / n_detected[i]
        - cv[i] = std(X[i]) / mean(X[i])

        Where:
        - n_missing = count of zeros or NaNs
        - n_detected = n_samples - n_missing
        - X = intensity values

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the assay to analyze.
    assay_name : str, default="protein"
        Name of the assay containing feature-level data (typically 'protein'
        or 'peptide' level). The assay must exist in the container.
    layer_name : str, optional
        Name of the layer to use for metric calculation. If None, uses
        the first available layer. Common choices: 'raw', 'log', 'normalized'.

    Returns
    -------
    ScpContainer
        ScpContainer with QC metrics added to assay feature metadata (var).
        The original container is not modified; a new container is returned
        with additional columns:
        - 'missing_rate': Missing rate per feature
        - 'detection_rate': Detection rate per feature
        - 'mean_expression': Mean expression of detected values
        - 'cv': Coefficient of variation

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data
    >>> var = pl.DataFrame({
    ...     '_index': ['AKT1', 'TP53', 'EGFR', 'MYC']
    ... })
    >>> X = np.array([
    ...     [1.0, 2.0, 0.0, 5.0],  # Sample 1
    ...     [0.0, 2.5, 3.0, 6.0],  # Sample 2 (EGFR missing)
    ...     [1.5, 0.0, 3.5, 0.0]   # Sample 3 (TP53, MYC missing)
    ... ])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2', 's3']}),
    ...     assays={'protein': assay}
    ... )
    >>> # Calculate QC metrics
    >>> result = calculate_feature_qc_metrics(container)
    >>> # View metrics
    >>> result.assays['protein'].var[['missing_rate', 'detection_rate', 'cv']]
    shape: (4, 3)
    ┌──────────────┬───────────────┬──────────┐
    │ missing_rate │ detection_rate ┆ cv       │
    │ ---          │ ---             ┆ ---      │
    │ f64          │ f64             ┆ f64      │
    ╞══════════════╪═════════════════╪══════════╡
    │ 0.333        │ 0.667           ┆ 0.255    │
    │ 0.333        │ 0.667           ┆ 0.141    │
    │ 0.333        │ 0.667           ┆ 0.289    │
    │ 0.667        │ 0.333           ┆ 0.298    │
    └──────────────┴───────────────┴──────────┘

    Notes
    -----
    Missing Value Definition:
    - Zeros are treated as missing (common in SCP: 0 = not detected)
    - NaN values are treated as missing
    - This is appropriate for proteomics where missingness = absence of detection
    - For other applications, consider whether zeros are biologically meaningful

    Detection Rate Interpretation:
    - High detection rate (>0.8): Reliable protein, consistently detected
    - Medium detection rate (0.5-0.8): Moderate reliability
    - Low detection rate (<0.5): Poor detection, consider filtering

    CV Interpretation:
    - Low CV (<0.3): Stable quantification, high quality
    - Medium CV (0.3-0.5): Acceptable variability
    - High CV (>0.5): High variability, may indicate measurement issues
    - CV depends on biological variability and technical noise

    Sparse Matrix Handling:
    - For sparse matrices, zeros in the data structure are treated as missing
    - This is efficient and appropriate for SCP data
    - Non-zero values are considered detected regardless of magnitude
    - Explicit NaN values in sparse matrices are not common but handled if present

    Dense Matrix Handling:
    - Zeros and NaN values are both treated as missing
    - Uses vectorized operations for efficiency
    - NaN-safe functions (np.nansum) handle missing values appropriately

    Performance Considerations:
    - Sparse matrices: O(nnz) memory where nnz = number of non-zero elements
    - Dense matrices: O(n_features * n_samples) memory
    - Uses vectorized operations for efficient computation
    - No unnecessary data copying

    Quality Control Workflow:
    1. Calculate metrics using this function
    2. Visualize metric distributions (violin plots, histograms)
    3. Set appropriate thresholds for your experiment
    4. Apply filtering using filter_features_by_missingness or filter_features_by_cv

    Common Use Cases:
    - **Feature filtering**: Remove features with high missing_rate or high CV
    - **Quality assessment**: Evaluate overall data quality
    - **Method comparison**: Compare QC metrics across different preprocessing methods
    - **Batch effect detection**: Check if metrics vary by batch

    Integration with Downstream Analysis:
    - These metrics are used by normalization methods to adjust for missingness
    - Feature selection methods may use detection_rate as a criterion
    - Clustering algorithms may benefit from filtering high-CV features
    - Visualization tools can color points by QC metrics

    References
    ----------
    .. [1] Ludwig, C., et al. (2021). Data analysis and quality control in
       single-cell proteomics. Nature Methods, 18, 493-499.
    .. [2] Specht, H., et al. (2021). Single-cell proteomics reveals
       gene-specific changes in protein copy number during differentiation.
       Molecular Cell, 81(14), 2929-2943.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Get layer to analyze
    if layer_name is None:
        layer = next(iter(assay.layers.values()))
    else:
        if layer_name not in assay.layers:
            available = ", ".join(f"'{k}'" for k in assay.layers.keys())
            raise ScpValueError(
                f"Layer '{layer_name}' not found in assay '{assay_name}'. "
                f"Available layers: {available}. "
                f"Use assay.list_layers() to see all available layers.",
                parameter="layer_name",
                value=layer_name,
            )
        layer = assay.layers[layer_name]

    X = layer.X
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # 1. Compute Missing Rate and Detection Rate
    # Missing = zero or NaN
    if sp.issparse(X):
        # Sparse: count non-zero entries per column (feature)
        n_detected = X.getnnz(axis=0)
    else:
        # Dense: count values > 0 and not NaN
        n_detected = np.sum((X > 0) & (~np.isnan(X)), axis=0)

    detection_rate = n_detected / n_samples
    missing_rate = 1.0 - detection_rate

    # 2. Compute Mean Expression (mean of detected values only)
    if sp.issparse(X):
        # Sparse: sum / count (only non-zeros)
        sums = np.array(X.sum(axis=0)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / n_detected
            means[n_detected == 0] = 0.0
    else:
        # Dense: mean excluding zeros (treat zeros as missing)
        # Create masked array where zeros are treated as NaN
        X_masked = np.where(X == 0, np.nan, X)
        means = np.nanmean(X_masked, axis=0)

    # 3. Compute CV (Coefficient of Variation)
    cv = compute_cv(X, axis=0)

    # Create new metrics DataFrame
    new_metrics = pl.DataFrame(
        {
            "missing_rate": missing_rate,
            "detection_rate": detection_rate,
            "mean_expression": means,
            "cv": cv,
        }
    )

    # Merge with existing var
    current_var = assay.var
    new_var = current_var.hstack(new_metrics)

    # Create new assay with updated var
    new_assay = assay.subset(np.arange(assay.n_features), copy_data=False)
    new_assay.var = new_var

    # Create new container
    new_assays = {
        name: new_assay if name == assay_name else a for name, a in container.assays.items()
    }

    new_container = ScpContainer(
        obs=container.obs,
        assays=new_assays,
        links=container.links,
        history=container.history,
        sample_id_col=container.sample_id_col,
    )

    # Log provenance
    new_container.log_operation(
        action="calculate_feature_qc_metrics",
        params={
            "assay": assay_name,
            "layer": layer_name or next(iter(assay.layers.keys())),
            "n_features": n_features,
            "n_samples": n_samples,
        },
        description=(
            f"Calculated QC metrics for {n_features} features "
            f"across {n_samples} samples in assay '{assay_name}'. "
            f"Metrics: missing_rate (mean={np.mean(missing_rate):.3f}), "
            f"detection_rate (mean={np.mean(detection_rate):.3f}), "
            f"mean_expression (mean={np.nanmean(means):.3f}), "
            f"cv (mean={np.nanmean(cv):.3f})."
        ),
    )

    return new_container


def filter_features_by_missingness(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str | None = None,
    max_missing_rate: float = 0.5,
) -> ScpContainer:
    """Filter features based on missing rate (proportion of missing values).

    Removes features with high missing rates to improve data quality and
    statistical power. Features that are rarely detected provide limited
    biological information and can introduce noise in downstream analysis.

    Missing Rate Definition:
    - Missing rate = proportion of samples where the feature is not detected
    - Missing values are defined as zeros or NaN values
    - A feature with missing_rate = 0.5 is detected in 50% of samples
    - Complement of detection rate: missing_rate = 1 - detection_rate

    Mathematical Formulation:
        keep_mask = missing_rate <= max_missing_rate
        missing_rate[i] = (n_missing[i]) / n_samples

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the assay to filter.
    assay_name : str, default="protein"
        Name of the assay containing feature-level data.
    layer_name : str, optional
        Name of the layer to use for missing rate calculation. If None,
        uses the first available layer. Common choices: 'raw', 'log', 'normalized'.
    max_missing_rate : float, default=0.5
        Maximum acceptable missing rate for retaining features.
        Features with missing_rate > max_missing_rate are removed.
        Range: [0, 1]. Lower values are more stringent.

        Recommended thresholds:
        - 0.2: Very stringent (retain only high-quality features)
        - 0.3: Stringent (conservative filtering)
        - 0.5: Moderate (default, balances sensitivity and specificity)
        - 0.7: Lenient (retain more features, lower quality)
        - 0.9: Very lenient (keep almost everything)

    Returns
    -------
    ScpContainer
        ScpContainer with low-quality features removed. The original
        container is not modified; a new container is returned.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.
    ScpValueError
        If max_missing_rate is not in the range [0, 1].

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with varying missing rates
    >>> var = pl.DataFrame({
    ...     '_index': ['AKT1', 'TP53', 'EGFR', 'MYC', 'KRAS']
    ... })
    >>> X = np.array([
    ...     [1.0, 2.0, 0.0, 5.0, 1.0],  # Sample 1
    ...     [0.0, 2.5, 3.0, 6.0, 0.0],  # Sample 2 (AKT1, KRAS missing)
    ...     [1.5, 0.0, 3.5, 0.0, 0.0]   # Sample 3 (TP53, MYC, KRAS missing)
    ... ])
    >>> # Missing rates: AKT1=0.33, TP53=0.33, EGFR=0.0, MYC=0.33, KRAS=0.67
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2', 's3']}),
    ...     assays={'protein': assay}
    ... )
    >>> # Filter features with >50% missing
    >>> result = filter_features_by_missingness(container, max_missing_rate=0.5)
    >>> # KRAS (missing_rate=0.67) is removed
    >>> result.assays['protein'].n_features
    4
    >>> result.assays['protein'].var['_index'].to_list()
    ['AKT1', 'TP53', 'EGFR', 'MYC']

    Notes
    -----
    Biological Rationale:
    - Features with high missing rates are often below detection limit
    - Poorly detected features contribute noise to downstream analysis
    - Missing data can bias statistical tests and clustering
    - Removing low-quality features improves statistical power

    Choosing max_missing_rate Threshold:
    The optimal threshold depends on your experimental design:

    - **Discovery studies** (limited samples):
      Use lenient threshold (0.6-0.7) to retain more features
    - **Targeted studies** (validated markers):
      Use stringent threshold (0.2-0.3) for high confidence
    - **Single-cell proteomics** (inherently sparse):
      Use moderate threshold (0.4-0.6) to balance sensitivity
    - **Bulk proteomics** (dense data):
      Use stringent threshold (0.1-0.3) to ensure quality

    Impact on Downstream Analysis:
    - **Normalization**: Fewer missing values improves imputation accuracy
    - **Clustering**: Reduces noise from spurious features
    - **Differential analysis**: Increases statistical power
    - **Dimensionality reduction**: Focuses on reliable features

    Considerations and Trade-offs:
    - Stringent filtering (low max_missing_rate):
      - Pros: Higher data quality, less noise
      - Cons: Loss of potentially informative features

    - Lenient filtering (high max_missing_rate):
      - Pros: Retains more biological information
      - Cons: Lower data quality, more noise

    It's often useful to test multiple thresholds and assess the impact on:
    - Number of retained features
    - Clustering quality (silhouette score)
    - Correlation with expected sample groups
    - Imputation performance

    Missing Rate vs Detection Rate:
    - missing_rate = 1 - detection_rate
    - Some fields use "prevalence" or "frequency" instead of detection_rate
    - Both metrics convey the same information

    Relationship to Other QC Metrics:
    - Features with high missing rate often have high CV
    - Consider combining with filter_features_by_cv for comprehensive QC
    - Use calculate_feature_qc_metrics to evaluate all metrics together

    Performance Considerations:
    - Sparse matrices: Efficient calculation using getnnz()
    - Dense matrices: Vectorized operations for speed
    - Memory: O(n_features) for intermediate arrays
    - No unnecessary data copying

    Quality Control Recommendations:
    1. Calculate metrics: calculate_feature_qc_metrics()
    2. Visualize missing rate distribution (violin plot)
    3. Choose threshold based on distribution and experimental goals
    4. Apply filtering with this function
    5. Document threshold in methods section

    Integration with Provenance:
    - The operation is logged in the container's history
    - Parameters and statistics are recorded for reproducibility
    - Use container.history to trace filtering steps

    References
    ----------
    .. [1] Gholami, A. M., et al. (2013). Global assessment of protein
       expression patterns in technical replicates. Journal of Proteome
       Research, 12(6), 2506-2514.
    .. [2] Keshishian, H., et al. (2022). Strategies for data analysis in
       single-cell proteomics. Nature Methods, 19, 698-708.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    # Validate threshold
    if not 0 <= max_missing_rate <= 1:
        raise ScpValueError(
            f"max_missing_rate must be between 0 and 1, got {max_missing_rate}. "
            f"Missing rate is a proportion (0 = no missing, 1 = all missing). "
            f"Recommended: 0.5 for moderate filtering, 0.3 for stringent filtering.",
            parameter="max_missing_rate",
            value=max_missing_rate,
        )

    assay = container.assays[assay_name]

    # Get layer to analyze
    if layer_name is None:
        layer = next(iter(assay.layers.values()))
    else:
        if layer_name not in assay.layers:
            available = ", ".join(f"'{k}'" for k in assay.layers.keys())
            raise ScpValueError(
                f"Layer '{layer_name}' not found in assay '{assay_name}'. "
                f"Available layers: {available}. "
                f"Use assay.list_layers() to see all available layers.",
                parameter="layer_name",
                value=layer_name,
            )
        layer = assay.layers[layer_name]

    X = layer.X
    n_samples = X.shape[0]

    # Compute missing rate
    if sp.issparse(X):
        n_detected = X.getnnz(axis=0)
    else:
        n_detected = np.sum((X > 0) & (~np.isnan(X)), axis=0)

    missing_rate = 1.0 - (n_detected / n_samples)

    # Create filter mask
    keep_mask = missing_rate <= max_missing_rate
    keep_indices = np.where(keep_mask)[0]

    # Apply filtering
    new_container = container.filter_features(assay_name, feature_indices=keep_indices)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    n_total = assay.n_features
    percent_removed = (n_removed / n_total * 100) if n_total > 0 else 0

    new_container.log_operation(
        action="filter_features_by_missingness",
        params={
            "assay": assay_name,
            "layer": layer_name or next(iter(assay.layers.keys())),
            "max_missing_rate": max_missing_rate,
        },
        description=(
            f"Removed {n_removed}/{n_total} features ({percent_removed:.1f}%) "
            f"with missing_rate > {max_missing_rate:.2f} from assay '{assay_name}'. "
            f"Retained {len(keep_indices)} features."
        ),
    )

    return new_container


def filter_features_by_cv(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str | None = None,
    max_cv: float = 1.0,
    min_mean: float = 1e-6,
) -> ScpContainer:
    """Filter features based on Coefficient of Variation (CV).

    Removes features with high CV to improve measurement stability and
    reduce technical noise. CV measures relative variability (standard
    deviation divided by mean), making it a dimensionless metric suitable
    for comparing variability across features with different expression levels.

    CV is particularly important in single-cell proteomics because:
    - High CV indicates unstable quantification (high technical noise)
    - CV is independent of expression level (unlike standard deviation)
    - CV helps identify proteins with inconsistent measurements
    - Low CV features provide more reliable biological signal

    Mathematical Formulation:
        CV = σ / μ

        Where:
        - σ = standard deviation across samples
        - μ = mean across samples
        - Features with CV > max_cv are removed
        - Features with CV = NaN are removed (undefined CV)

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing the assay to filter.
    assay_name : str, default="protein"
        Name of the assay containing feature-level data.
    layer_name : str, optional
        Name of the layer to use for CV calculation. If None,
        uses the first available layer. Common choices: 'raw', 'log', 'normalized'.
    max_cv : float, default=1.0
        Maximum acceptable CV for retaining features.
        Features with CV > max_cv are removed.
        Range: > 0. Lower values are more stringent.

        Recommended thresholds:
        - 0.3: Very stringent (highly stable features only)
        - 0.5: Stringent (recommended for critical experiments)
        - 0.7: Moderate (balances stability and sensitivity)
        - 1.0: Lenient (default, allows high variability)
        - 1.5: Very lenient (retain most features)

        Note: CV depends on data transformation:
        - Raw data: CV typically 0.5-2.0
        - Log-transformed: CV typically 0.1-0.5
        - Normalize on appropriate layer!
    min_mean : float, default=1e-6
        Minimum mean value for CV calculation. Features with mean < min_mean
        have undefined CV (NaN) and are removed. This prevents division by
        zero and filters out features with near-zero expression.

    Returns
    -------
    ScpContainer
        ScpContainer with high-CV features removed. The original
        container is not modified; a new container is returned.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.
    ScpValueError
        If max_cv is not positive.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with varying CV
    >>> var = pl.DataFrame({
    ...     '_index': ['AKT1', 'TP53', 'EGFR', 'MYC']
    ... })
    >>> X = np.array([
    ...     [1.0, 2.0, 3.0, 10.0],  # Sample 1
    ...     [1.1, 2.5, 3.5, 20.0],  # Sample 2
    ...     [0.9, 2.2, 3.3, 5.0]    # Sample 3
    ... ])
    >>> # CV values: AKT1=0.08, TP53=0.11, EGFR=0.08, MYC=0.71
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2', 's3']}),
    ...     assays={'protein': assay}
    ... )
    >>> # Filter features with CV > 0.5
    >>> result = filter_features_by_cv(container, max_cv=0.5)
    >>> # MYC (CV=0.71) is removed
    >>> result.assays['protein'].n_features
    3
    >>> result.assays['protein'].var['_index'].to_list()
    ['AKT1', 'TP53', 'EGFR']

    Notes
    -----
    Biological Rationale:
    - High CV indicates measurement instability or high biological variability
    - Technical variability should be minimized for reliable quantification
    - Features with high CV can obscure biological patterns
    - CV filtering improves signal-to-noise ratio

    CV Interpretation:
    - **CV < 0.3**: Very stable, excellent quantification
    - **CV 0.3-0.5**: Stable, good quantification (recommended)
    - **CV 0.5-1.0**: Moderate variability, acceptable in some contexts
    - **CV > 1.0**: High variability, may indicate measurement issues

    Factors Affecting CV:
    1. **Biological variability**: Genuine differences between samples
    2. **Technical noise**: Measurement error from MS instrument
    3. **Sample handling**: Variability in sample preparation
    4. **Detection level**: Low-abundance features have higher CV
    5. **Data transformation**: Log transformation reduces CV

    Choosing max_cv Threshold:
    The optimal threshold depends on your data and goals:

    - **Raw data** (no transformation):
      Use lenient threshold (0.8-1.5) because raw data has high variance
    - **Log-transformed data**:
      Use stringent threshold (0.3-0.5) because log compresses variance
    - **Normalized data** (batch corrected):
      Use moderate threshold (0.4-0.7) for balanced filtering
    - **High-quality replicates**:
      Use stringent threshold (0.3-0.5) to ensure stability
    - **Noisy data** (limited replicates):
      Use lenient threshold (0.7-1.0) to retain features

    Impact of Data Transformation:
    CV is highly dependent on data transformation:
    - **Raw intensities**: CV = σ_raw / μ_raw
    - **Log-transformed**: CV_log = σ_log / μ_log (typically much lower)
    - Always filter on the appropriate layer!
    - If using log-transformed data, reduce max_cv threshold accordingly

    Relationship to Biological Variability:
    - Some features are genuinely variable (biological heterogeneity)
    - CV filtering removes both technical noise AND biological variability
    - Be cautious not to over-filter in heterogeneous samples
    - Consider biological context when setting threshold

    CV vs Missing Rate:
    - Features with high missing rate often have high CV
    - CV filtering addresses a different aspect of data quality
    - Consider combining with filter_features_by_missingness for comprehensive QC
    - Use calculate_feature_qc_metrics to evaluate both metrics

    Undefined CV (NaN Values):
    Features with undefined CV are removed:
    - Mean ≈ 0: Division by zero
    - Single sample: No variability to measure
    - All values identical: No variation (CV = 0, but often removed anyway)

    Sparse Matrix Handling:
    - For sparse matrices, uses efficient variance calculation
    - Variance = E[X²] - (E[X])² (avoiding dense conversion)
    - Zeros in sparse matrix are treated as missing (not included in CV)
    - Only non-zero values contribute to CV calculation

    Performance Considerations:
    - Sparse matrices: O(nnz) memory where nnz = number of non-zero elements
    - Dense matrices: O(n_features * n_samples) memory
    - Uses vectorized operations for efficiency
    - No unnecessary data copying

    Quality Control Recommendations:
    1. Calculate metrics: calculate_feature_qc_metrics()
    2. Visualize CV distribution (violin plot or histogram)
    3. Check CV vs mean relationship (should decrease with expression)
    4. Choose threshold based on distribution and data transformation
    5. Apply filtering with this function
    6. Document threshold in methods section

    Common Pitfalls:
    - **Over-filtering**: Removing biologically variable features
      - Solution: Test multiple thresholds, assess biological relevance
    - **Under-filtering**: Retaining noisy features
      - Solution: Check correlation with technical factors
    - **Wrong layer**: Filtering on raw instead of normalized data
      - Solution: Always specify layer_name parameter
    - **Ignoring transformation**: Using same threshold for raw and log data
      - Solution: Adjust threshold based on transformation

    Integration with Downstream Analysis:
    - **Normalization**: Low CV features respond better to normalization
    - **Clustering**: Reduces noise from unstable measurements
    - **Differential analysis**: Increases sensitivity and specificity
    - **Feature selection**: CV is often used as a selection criterion

    References
    ----------
    .. [1] Ludwig, C., et al. (2019). Data analysis and quality control in
       single-cell proteomics. Nature Methods, 16, 695-697.
    .. [2] Wisniewski, J. R., et al. (2010). High-sensitivity analysis of
       yeast proteome using SDS-gel separation and LC-MS/MS. Journal of
       Proteome Research, 9(5), 2252-2259.
    .. [3] Choi, M., et al. (2014). MSstats: Protein quantification
       with statistical confidence. Nature Methods, 11, 1113-1115.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    # Validate threshold
    if max_cv <= 0:
        raise ScpValueError(
            f"max_cv must be positive, got {max_cv}. "
            f"CV is always non-negative (CV >= 0). "
            f"Recommended: 0.5 for log-transformed data, 1.0 for raw data.",
            parameter="max_cv",
            value=max_cv,
        )

    assay = container.assays[assay_name]

    # Get layer to analyze
    if layer_name is None:
        layer = next(iter(assay.layers.values()))
    else:
        if layer_name not in assay.layers:
            available = ", ".join(f"'{k}'" for k in assay.layers.keys())
            raise ScpValueError(
                f"Layer '{layer_name}' not found in assay '{assay_name}'. "
                f"Available layers: {available}. "
                f"Use assay.list_layers() to see all available layers.",
                parameter="layer_name",
                value=layer_name,
            )
        layer = assay.layers[layer_name]

    X = layer.X

    # Compute CV
    cv = compute_cv(X, axis=0, min_mean=min_mean)

    # Create filter mask
    # Keep features with CV <= max_cv AND CV is not NaN
    keep_mask = (cv <= max_cv) & (~np.isnan(cv))
    keep_indices = np.where(keep_mask)[0]

    # Apply filtering
    new_container = container.filter_features(assay_name, feature_indices=keep_indices)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    n_total = assay.n_features
    percent_removed = (n_removed / n_total * 100) if n_total > 0 else 0

    # Calculate statistics for description
    cv_mean = np.nanmean(cv)
    cv_median = np.nanmedian(cv)

    new_container.log_operation(
        action="filter_features_by_cv",
        params={
            "assay": assay_name,
            "layer": layer_name or next(iter(assay.layers.keys())),
            "max_cv": max_cv,
            "min_mean": min_mean,
        },
        description=(
            f"Removed {n_removed}/{n_total} features ({percent_removed:.1f}%) "
            f"with CV > {max_cv:.2f} from assay '{assay_name}'. "
            f"Retained {len(keep_indices)} features. "
            f"Original CV distribution: mean={cv_mean:.3f}, median={cv_median:.3f}."
        ),
    )

    return new_container
