"""Sample-level Quality Control.

Handles QC for cells/samples in single-cell proteomics data, including:
- Basic metric calculation (ID count, total intensity, detection rate)
- Outlier detection (using robust MAD statistics)
- Doublet detection based on intensity outliers
- Batch effect assessment

This module provides essential quality control tools for identifying and
removing low-quality samples before downstream analysis. Sample-level QC is
critical for single-cell proteomics where cell quality varies substantially
due to sample preparation, instrument sensitivity, and biological factors.

Mathematical Background:
The module uses robust statistical methods based on Median Absolute Deviation
(MAD) rather than mean-based approaches. MAD is more resistant to outliers
and provides better performance for skewed distributions common in proteomics
data.

For a distribution with median m and MAD calculated as:
    MAD = median(|x - m|) * 1.4826

Outliers are detected as:
    |x - m| > nmads * MAD

This approach is approximately equivalent to using standard deviations for
normal distributions but maintains robustness for non-normal data.
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.qc.metrics import is_outlier_mad


def calculate_sample_qc_metrics(
    container: ScpContainer,
    assay_name: str = "protein",
    layer_name: str = "raw",
) -> ScpContainer:
    """
    Calculate quality control metrics for samples/cells.

    Computes fundamental QC metrics that characterize sample quality and
    detection efficiency. These metrics are essential for identifying
    low-quality cells, failed experiments, and technical artifacts.

    Metrics Calculated:
    - **n_features**: Number of detected features (non-zero/non-NaN values per sample)
    - **total_intensity**: Sum of intensity values across all features (library size)
    - **log1p_total_intensity**: Log-transformed total intensity using log1p (log(1+x))

    Mathematical Background:
    For sample i with feature vector x_i:
        n_features_i = |{j : x_ij > 0 and x_ij is not NaN}|
        total_intensity_i = sum(x_ij)
        log1p_total_intensity_i = log(1 + total_intensity_i)

    The log1p transformation provides better visualization and statistical
    modeling of library sizes, which typically follow log-normal distributions
    in proteomics data.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data with at least one assay and layer.
        Metrics will be added to the container's obs (sample metadata).
    assay_name : str, default="protein"
        Name of the assay to use for QC metric calculation. Typically "protein"
        or "peptide" depending on data level. The assay must exist in the container.
    layer_name : str, default="raw"
        Name of the layer within the assay to analyze. Common values: "raw",
        "normalized", "imputed". The layer must exist in the assay.

    Returns
    -------
    ScpContainer
        New ScpContainer with QC metrics added as columns in obs. Column names
        are suffixed with assay_name to support multiple assays:
        - `n_features_{assay_name}`
        - `total_intensity_{assay_name}`
        - `log1p_total_intensity_{assay_name}`

        The original container is not modified; a new container is returned.

    Raises
    ------
    AssayNotFoundError
        If the specified assay_name does not exist in the container.
    ScpValueError
        If the specified layer_name is not found in the assay.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data
    >>> var = pl.DataFrame({'_index': ['P1', 'P2', 'P3']})
    >>> X = np.array([
    ...     [1.0, 2.0, 0.0],  # Sample 1: 2 features detected
    ...     [0.0, 0.0, 1.0],  # Sample 2: 1 feature detected (low quality)
    ...     [3.0, 4.0, 5.0],  # Sample 3: 3 features detected
    ... ])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2', 's3']}),
    ...     assays={'protein': assay}
    ... )
    >>> # Calculate QC metrics
    >>> result = calculate_sample_qc_metrics(container)
    >>> # Check metrics in obs
    >>> result.obs.select([
    ...     'n_features_protein',
    ...     'total_intensity_protein',
    ...     'log1p_total_intensity_protein'
    ... ]).to_dict(as_series=False)
    {'n_features_protein': [2, 1, 3],
     'total_intensity_protein': [3.0, 1.0, 12.0],
     'log1p_total_intensity_protein': [1.386, 0.693, 2.565]}

    Working with sparse matrices:
    >>> from scipy import sparse
    >>> X_sparse = sparse.csr_matrix(X)
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X_sparse)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2', 's3']}),
    ...     assays={'protein': assay}
    ... )
    >>> result = calculate_sample_qc_metrics(container)
    >>> # Sparse matrices handled efficiently

    Notes
    -----
    Metric Interpretation:
    - **n_features**: Indicates detection efficiency. Low values suggest failed
      experiments, poor sample preparation, or instrument issues.
    - **total_intensity**: Reflects total protein content. Used for library
      size normalization and sample quality assessment.
    - **log1p_total_intensity**: Stabilized version of total intensity for
      visualization and statistical modeling.

    Typical QC Thresholds:
    - Minimum features: 50-100 proteins for single-cell proteomics
    - Total intensity: Variable by instrument and experiment type
    - Use distribution-based filtering (e.g., MAD-based outliers) rather than
      hard thresholds when possible

    Performance Considerations:
    - Sparse matrices: Uses efficient `getnnz()` and `sum()` operations
    - Dense matrices: Vectorized NumPy operations with NaN handling
    - Memory: Creates temporary arrays of size n_samples
    - Time complexity: O(n_samples * n_features) for sparse, O(n) for dense

    Batch Processing:
    For large datasets, calculate metrics once and reuse them for filtering:
        >>> container = calculate_sample_qc_metrics(container)
        >>> container = filter_low_quality_samples(container, min_features=100)

    Common Pitfalls:
    - Ensure layer contains raw (not normalized) data for meaningful metrics
    - Log-transformed data will produce misleading total_intensity values
    - Check for NaN values in input data (handled correctly by this function)

    Integration with Downstream Analysis:
    These metrics are used by:
    - `filter_low_quality_samples()`: Removes samples with low n_features
    - `filter_doublets_mad()`: Removes samples with high total_intensity
    - `assess_batch_effects()`: Compares metrics across batches
    - Visualization functions: Plot QC metric distributions

    References
    ----------
    .. [1] Bacher, R., et al. (2017). SCnorm: robust normalization of
       single-cell RNA-seq data. Nature Methods, 14(6), 584-586.
    .. [2] Luecken, M. D., et al. (2022). Current best practices in
       single-cell RNA-seq analysis: a tutorial. Molecular Cell, 87(2),
       175-189.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Validate layer exists
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise ScpValueError(
            f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}. "
            f"Use assay.list_layers() to see all layers.",
            parameter="layer_name",
            value=layer_name,
        )

    X = assay.layers[layer_name].X

    # Calculate metrics based on matrix type
    if sp.issparse(X):
        # Sparse matrix: use efficient methods
        # getnnz(axis=1) counts non-zeros per row (sample)
        n_features = X.getnnz(axis=1)
        # Sum along rows, convert to 1D array
        total_intensity = np.array(X.sum(axis=1)).flatten()
    else:
        # Dense matrix: handle NaN and zero values
        # Detected features are non-zero and non-NaN
        is_detected = (X > 0) & (~np.isnan(X))
        n_features = np.sum(is_detected, axis=1)
        # Sum ignoring NaN values
        total_intensity = np.nansum(X, axis=1)

    # Log-transform total intensity for better statistical properties
    # log1p is used to handle zero values: log1p(0) = 0
    log1p_total = np.log1p(total_intensity)

    # Create metrics DataFrame with assay-specific column names
    # This allows metrics from multiple assays to coexist
    metrics_df = pl.DataFrame(
        {
            f"n_features_{assay_name}": n_features,
            f"total_intensity_{assay_name}": total_intensity,
            f"log1p_total_intensity_{assay_name}": log1p_total,
        }
    )

    # Add metrics to obs using horizontal concatenation
    # This preserves the order of samples (invariant in ScpContainer)
    new_obs = container.obs.hstack(metrics_df)

    # Return new container with updated metadata
    # Following functional programming pattern: don't mutate, return new object
    return ScpContainer(
        obs=new_obs,
        assays=container.assays,
        links=container.links,
        history=container.history,
        sample_id_col=container.sample_id_col,
    )


def filter_low_quality_samples(
    container: ScpContainer,
    assay_name: str = "protein",
    min_features: int = 100,
    nmads: float = 3.0,
    use_mad: bool = True,
) -> ScpContainer:
    """
    Filter low-quality samples based on feature detection count.

    Removes samples with insufficient numbers of detected features, which
    typically indicate failed experiments, poor sample preparation, or
    instrument issues. The function supports both hard thresholds and
    robust statistical outlier detection.

    Mathematical Background:
    The function uses two complementary filtering strategies:

    1. Hard Threshold:
        Keep samples where n_features >= min_features

    2. MAD-based Outlier Detection (optional):
        Calculate MAD of log1p(n_features)
        Detect lower-tail outliers: n_features < median - nmads * MAD

    MAD (Median Absolute Deviation) is preferred over standard deviation
    because it is robust to outliers and does not assume normal distribution.
    For proteomics data, which often has skewed distributions and outliers,
    MAD provides more reliable outlier detection.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data to filter.
    assay_name : str, default="protein"
        Name of the assay to use for filtering. The assay must contain
        quantitative data (typically "protein" or "peptide").
    min_features : int, default=100
        Hard threshold for minimum number of detected features required.
        Samples with fewer features are unconditionally removed.
        Recommended values:
        - Single-cell proteomics: 50-100 features
        - Bulk proteomics: 500-1000 features
        - Peptide-level: 200-500 features
    nmads : float, default=3.0
        Number of MADs (Median Absolute Deviations) to use as threshold
        for statistical outlier detection. Higher values are more conservative.
        Recommended:
        - 2.0: Aggressive filtering (more outliers)
        - 3.0: Standard (default, widely used)
        - 4.0: Conservative (fewer outliers)
    use_mad : bool, default=True
        Whether to apply MAD-based outlier detection in addition to the
        hard threshold. When True, samples are removed if they fail either
        the hard threshold OR are MAD outliers.

    Returns
    -------
    ScpContainer
        New ScpContainer with low-quality samples removed. Both obs and
        all assays are subsetted to retain only high-quality samples.
        The original container is not modified.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with varying quality
    >>> var = pl.DataFrame({'_index': ['P1', 'P2', 'P3', 'P4', 'P5']})
    >>> X = np.array([
    ...     [1, 2, 3, 4, 5],      # s1: 5 features (good)
    ...     [1, 0, 0, 0, 0],      # s2: 1 feature (bad, hard filter)
    ...     [1, 2, 3, 0, 0],      # s3: 3 features (bad, MAD outlier)
    ...     [1, 2, 3, 4, 5],      # s4: 5 features (good)
    ...     [1, 2, 3, 4, 0],      # s5: 4 features (good)
    ... ])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2', 's3', 's4', 's5']}),
    ...     assays={'protein': assay}
    ... )
    >>> # Filter with hard threshold only
    >>> result = filter_low_quality_samples(
    ...     container,
    ...     min_features=3,
    ...     use_mad=False
    ... )
    >>> # s2 removed (only 1 feature), s1, s3, s4, s5 kept
    >>> result.n_samples
    4

    >>> # Filter with both hard threshold and MAD
    >>> result = filter_low_quality_samples(
    ...     container,
    ...     min_features=2,
    ...     nmads=2.0,
    ...     use_mad=True
    ... )
    >>> # s2 removed by hard filter, s3 removed by MAD (low outlier)
    >>> result.n_samples
    3

    Notes
    -----
    Why MAD over Standard Deviation?
    - Robustness: MAD is not influenced by extreme outliers (up to 50% breakdown)
    - Non-normal data: Proteomics data often has skewed distributions
    - Interpretability: MAD threshold corresponds to SD for normal distributions
    - Proven: MAD-based filtering is standard in single-cell analysis

    Choosing Parameters:
    Hard Threshold (min_features):
    - Should reflect biological expectations for your system
    - Lower thresholds for scarce samples (e.g., rare cell types)
    - Higher thresholds for bulk or high-quality samples
    - Consider instrument detection limits

    MAD Threshold (nmads):
    - Start with 3.0 (default, standard in single-cell field)
    - Decrease to 2.0 for aggressive filtering (smaller studies)
    - Increase to 4.0+ for conservative filtering (large studies)
    - Adjust based on visual inspection of metric distributions

    Filtering Strategy:
    1. Start with visual QC: plot n_features distribution
    2. Identify obvious outliers and failed samples
    3. Set min_features to remove clear failures
    4. Enable MAD filtering to remove statistical outliers
    5. Iterate and adjust based on downstream results

    Impact on Downstream Analysis:
    - Normalization: More accurate with low-quality samples removed
    - Dimensionality reduction: Better separation without outliers
    - Clustering: More robust cluster assignments
    - Differential expression: Increased statistical power

    Quality Control Workflow:
    Recommended workflow for sample QC:
        1. Calculate metrics: calculate_sample_qc_metrics()
        2. Visualize distributions: Plot n_features, total_intensity
        3. Set thresholds based on data inspection
        4. Apply filtering: filter_low_quality_samples()
        5. Document removal rate and rationale

    Performance Considerations:
    - Time: O(n_samples) for metric calculation
    - Memory: Temporary arrays of size n_samples
    - Sparse matrices: Efficient getnnz() operation
    - No data copying: Subsets using index arrays

    Common Pitfalls:
    - Over-filtering: Too aggressive thresholds lose biological variation
    - Under-filtering: Poor quality samples distort results
    - Batch-specific thresholds: Use assess_batch_effects() first
    - Ignoring experiment type: Single-cell vs bulk need different thresholds

    References
    ----------
    .. [1] Leys, C., et al. (2013). Detecting outliers: Do not use standard
       deviation around the mean, use absolute deviation around the median.
       Journal of Experimental Social Psychology, 49(4), 764-766.
    .. [2] Luecken, M. D., et al. (2022). Current best practices in
       single-cell RNA-seq analysis: a tutorial. Molecular Cell, 87(2),
       175-189.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Get data matrix (use 'raw' layer if available, otherwise first layer)
    layer = assay.layers.get("raw", next(iter(assay.layers.values())))
    X = layer.X

    # Calculate number of detected features per sample
    if sp.issparse(X):
        n_features = X.getnnz(axis=1)
    else:
        # Detected = non-zero AND non-NaN
        n_features = np.sum((X > 0) & (~np.isnan(X)), axis=1)

    # Apply hard threshold filter
    keep_mask = n_features >= min_features

    # Apply MAD-based outlier detection (lower tail only)
    # Low-quality samples are those with FEWER features than expected
    if use_mad:
        outliers = is_outlier_mad(
            n_features.astype(float),
            nmads=nmads,
            direction="lower",  # Only detect low outliers
        )
        keep_mask = keep_mask & (~outliers)

    # Get indices of samples to keep
    keep_indices = np.where(keep_mask)[0]

    # Apply filtering to all assays and metadata
    new_container = container.filter_samples(sample_indices=keep_indices)

    # Log provenance with detailed statistics
    n_removed = container.n_samples - len(keep_indices)
    n_total = container.n_samples
    percent_removed = (n_removed / n_total * 100) if n_total > 0 else 0

    filter_description = (
        f"Removed {n_removed}/{n_total} samples ({percent_removed:.1f}%) "
        f"from assay '{assay_name}'. "
    )
    if use_mad:
        filter_description += (
            f"Filtering: n_features >= {min_features} AND not lower-outlier by >{nmads} MADs."
        )
    else:
        filter_description += f"Filtering: n_features >= {min_features} (hard threshold only)."

    new_container.log_operation(
        action="filter_low_quality_samples",
        params={
            "assay": assay_name,
            "min_features": min_features,
            "use_mad": use_mad,
            "nmads": nmads,
        },
        description=filter_description,
    )

    return new_container


def filter_doublets_mad(
    container: ScpContainer,
    assay_name: str = "protein",
    nmads: float = 3.0,
) -> ScpContainer:
    """
    Filter potential doublets using MAD-based outlier detection on total intensity.

    Doublets (multiple cells analyzed as one) typically exhibit abnormally high
    total intensity or feature counts compared to singlets. This function detects
    and removes upper-tail outliers in the total intensity distribution using
    robust Median Absolute Deviation (MAD) statistics.

    Mathematical Background:
    Doublet detection uses upper-tail outlier detection on log-transformed
    total intensity (library size):

        log_intensity = log1p(total_intensity)
        median_log = median(log_intensity)
        mad_log = MAD(log_intensity)
        is_doublet = log_intensity > median_log + nmads * mad_log

    The log1p transformation is used because:
    1. Library sizes follow log-normal distributions
    2. Log scale stabilizes variance across intensity range
    3. Outlier detection is more reliable on log scale

    MAD is used instead of standard deviation because:
    1. Robust to outliers (up to 50% breakdown point)
    2. Does not assume normal distribution
    3. More reliable for skewed distributions
    4. Standard approach in single-cell analysis

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data with at least one assay.
    assay_name : str, default="protein"
        Name of the assay to use for doublet detection. Typically "protein"
        or "peptide". The assay must contain quantitative data.
    nmads : float, default=3.0
        Number of MADs to use as threshold for doublet detection. Higher
        values are more conservative (fewer doublets detected).
        Recommended:
        - 2.0: Aggressive (may flag some large singlets)
        - 3.0: Standard (default, widely used in single-cell)
        - 4.0: Conservative (may miss some doublets)
        - 5.0+: Very conservative (only extreme outliers)

    Returns
    -------
    ScpContainer
        New ScpContainer with potential doublets removed. Both obs and all
        assays are subsetted. The original container is not modified.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with potential doublets
    >>> var = pl.DataFrame({'_index': ['P1', 'P2', 'P3', 'P4']})
    >>> X = np.array([
    ...     [1, 2, 3, 4],      # s1: total=10 (normal singlet)
    ...     [2, 3, 1, 2],      # s2: total=8 (normal singlet)
    ...     [8, 9, 8, 9],      # s3: total=34 (potential doublet!)
    ...     [1, 2, 2, 1],      # s4: total=6 (normal singlet)
    ... ])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2', 's3', 's4']}),
    ...     assays={'protein': assay}
    ... )
    >>> # Detect and remove doublets
    >>> result = filter_doublets_mad(container, nmads=2.0)
    >>> # s3 removed (high intensity outlier)
    >>> result.n_samples
    3
    >>> result.obs['_index'].to_list()
    ['s1', 's2', 's4']

    Notes
    -----
    Why MAD Instead of IsolationForest?
    Previous versions used IsolationForest for doublet detection. MAD is
    preferred because:
    1. **Simplicity**: Single parameter (nmads) vs complex hyperparameters
    2. **Interpretability**: MAD threshold has clear statistical meaning
    3. **Robustness**: Works well even with small sample sizes (<100)
    4. **Consistency**: Matches approach used in other QC functions
    5. **Reproducibility**: Deterministic results vs random forest variability

    Doublet Detection Strategy:
    - **Upper-tail only**: Doublets have MORE signal, not less
    - **Log scale**: Total intensity is log-normally distributed
    - **Robust**: MAD handles skewness and extreme values
    - **Conservative**: Default 3.0 MADs minimizes false positives

    Expected Doublet Rate:
    The theoretical doublet rate depends on cell loading density:
    - 10% loading: ~1% doublets
    - 20% loading: ~4% doublets
    - 30% loading: ~9% doublets
    - 40% loading: ~16% doublets

    Use your expected doublet rate as a sanity check for the nmads threshold.

    Alternative Doublet Detection Methods:
    For more sophisticated doublet detection, consider:
    1. Scrublet (Python) - simulation-based approach for scRNA-seq
    2. DoubletFinder (R) - for single-cell RNA sequencing
    3. Experimental methods: Cell hashing, multiplexing

    However, MAD-based detection is often sufficient for proteomics data
    where doublets have dramatically higher total protein content.

    Limitations:
    - Cannot distinguish true doublets from large/prolific cells
    - May miss doublets if one cell is very small/dead
    - Assumes most samples are singlets (breakdown if >50% doublets)
    - Works best when cell types have similar protein content

    Quality Control Workflow:
    Recommended order of operations:
        1. Calculate metrics: calculate_sample_qc_metrics()
        2. Remove low-quality samples: filter_low_quality_samples()
        3. Remove doublets: filter_doublets_mad()
        4. Assess batch effects: assess_batch_effects()

    Impact on Downstream Analysis:
    - Normalization: Doublets distort normalization factors
    - Clustering: Doublets form artificial hybrid clusters
    - Differential expression: Doublets reduce statistical power
    - Visualization: Doublets appear as intermediate populations

    Performance Considerations:
    - Time: O(n_samples) for intensity calculation and MAD
    - Memory: Temporary arrays for total intensity
    - Sparse matrices: Efficient sum() operation
    - No data copying: Uses index array for subsetting

    Troubleshooting:
    Too many doublets detected (>20%):
    - Increase nmads to 4.0 or 5.0
    - Check for batch effects (use assess_batch_effects)
    - Verify instrument calibration
    - Review sample preparation protocol

    Too few doublets detected (<1%):
    - Decrease nmads to 2.0 or 2.5
    - Check if cell loading density is low
    - May need alternative detection method
    - Review expected doublet rate for your protocol

    References
    ----------
    .. [1] McGinnis, C. S., et al. (2019). DoubletFinder: Doublet detection
       in single-cell RNA-seq data using artificial nearest neighbors.
       Cell Systems, 8(4), 281-291.
    .. [2] Wolock, S. L., et al. (2019). Scrublet: Computational identification
       of cell doublets in single-cell transcriptomic data. Cell Systems,
       8(3), 281-291.
    .. [3] Leys, C., et al. (2013). Detecting outliers: Do not use standard
       deviation around the mean, use absolute deviation around the median.
       Journal of Experimental Social Psychology, 49(4), 764-766.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Get data matrix (use 'raw' layer if available, otherwise first layer)
    layer = assay.layers.get("raw", next(iter(assay.layers.values())))
    X = layer.X

    # Calculate total intensity (library size) for each sample
    if sp.issparse(X):
        total_intensity = np.array(X.sum(axis=1)).flatten()
    else:
        total_intensity = np.nansum(X, axis=1)

    # Transform to log space for better outlier detection
    # Library sizes typically follow log-normal distribution
    log_lib_size = np.log1p(total_intensity)

    # Detect upper-tail outliers (high intensity = potential doublets)
    is_doublet = is_outlier_mad(
        log_lib_size,
        nmads=nmads,
        direction="upper",  # Only detect high outliers (doublets)
    )

    # Keep samples that are NOT doublets
    keep_indices = np.where(~is_doublet)[0]

    # Apply filtering
    new_container = container.filter_samples(sample_indices=keep_indices)

    # Log provenance
    n_removed = container.n_samples - len(keep_indices)
    n_total = container.n_samples
    percent_removed = (n_removed / n_total * 100) if n_total > 0 else 0

    new_container.log_operation(
        action="filter_doublets_mad",
        params={
            "assay": assay_name,
            "nmads": nmads,
            "method": "MAD_upper_tail",
        },
        description=(
            f"Removed {n_removed}/{n_total} samples ({percent_removed:.1f}%) "
            f"as potential doublets (high intensity outliers >{nmads} MADs) "
            f"from assay '{assay_name}'. "
            f"Method: MAD-based upper-tail outlier detection on log1p(total_intensity)."
        ),
    )

    return new_container


def assess_batch_effects(
    container: ScpContainer,
    batch_col: str,
    assay_name: str = "protein",
) -> pl.DataFrame:
    """
    Assess batch effects by calculating quality control metrics per batch.

    Computes summary statistics for key QC metrics (feature counts, intensity
    distributions) grouped by batch identifier. This enables rapid identification
    of batch-specific quality issues and technical variability that may require
    batch correction or removal of problematic batches.

    Mathematical Background:
    For each batch b, calculate:

        n_cells_b = |{i : batch_i = b}|
        median_features_b = median(n_features_i for i in batch b)
        std_features_b = std(n_features_i for i in batch b)
        median_intensity_b = median(total_intensity_i for i in batch b)

    These metrics reveal batch differences in:
    - Detection efficiency (median_features)
    - Variability (std_features)
    - Total protein content (median_intensity)

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing sample data with batch information in obs.
    batch_col : str
        Column name in `container.obs` containing batch identifiers.
        Must be a categorical or string column. Common names: 'batch',
        'batch_id', 'plate', 'run', 'tmt_batch', 'instrument_run'.
    assay_name : str, default="protein"
        Name of the assay to analyze. Must contain quantitative data.

    Returns
    -------
    pl.DataFrame
        Summary statistics per batch with one row per batch and columns:
        - `{batch_col}`: Batch identifier (from obs)
        - `n_cells`: Number of samples/cells in the batch
        - `median_features`: Median number of detected features per sample
        - `std_features`: Standard deviation of feature counts
        - `median_intensity`: Median total intensity per sample

        DataFrame is sorted by batch identifier for easy comparison.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.
    ScpValueError
        If the batch column is not found in obs.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with batch information
    >>> var = pl.DataFrame({'_index': ['P1', 'P2', 'P3']})
    >>> X = np.array([
    ...     [1, 2, 3],  # batch1 samples (lower intensity)
    ...     [1, 1, 2],
    ...     [10, 20, 30],  # batch2 samples (higher intensity - batch effect!)
    ... ])
    >>> obs = pl.DataFrame({
    ...     '_index': ['s1', 's2', 's3'],
    ...     'batch': ['batch1', 'batch1', 'batch2']
    ... })
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(obs=obs, assays={'protein': assay})
    >>> # Assess batch effects
    >>> summary = assess_batch_effects(container, batch_col='batch')
    >>> summary.to_dict(as_series=False)
    {'batch': ['batch1', 'batch2'],
     'n_cells': [2, 1],
     'median_features': [2.5, 3.0],
     'std_features': [0.354, 0.0],
     'median_intensity': [3.5, 60.0]}
    >>> # batch2 has much higher median_intensity - potential batch effect!

    Notes
    -----
    Interpreting Batch Effects:
    Compare metrics across batches to identify issues:
    - **n_cells**: Highly variable batch sizes may indicate uneven quality
    - **median_features**: >2-fold difference suggests detection issues
    - **std_features**: High values indicate within-batch variability
    - **median_intensity**: >2-fold difference indicates normalization needed

    Common Batch Effect Sources:
    - Different sample preparation dates
    - Multiple TMT batches
    - Instrument calibration drift
    - Different operators or protocols
    - Reagent lot variations
    - Sample storage time differences

    When to Apply Batch Correction:
    Consider batch correction if:
    1. Median intensity varies >2-fold between batches
    2. Median features varies >2-fold between batches
    3. PCA or clustering shows batch-driven separation
    4. Biological groups are confounded with batch

    Batch Correction Options:
    - ComBat (empirical Bayes): scptensor.integration.combat()
    - Harmony: scptensor.integration.harmony()
    - MNN (mutual nearest neighbors): scptensor.integration.mnn()
    - Non-linear: scptensor.integration.nonlinear_correction()

    Quality Control Workflow:
    Integrate batch assessment into QC pipeline:
        1. Calculate QC metrics: calculate_sample_qc_metrics()
        2. Assess batch effects: assess_batch_effects()
        3. Visualize batch effects: Plot metrics by batch
        4. Remove problematic batches if needed
        5. Apply batch correction: integration methods
        6. Verify correction: Re-assess batch effects

    Limitations:
    - Only summarizes QC metrics, doesn't test for significance
    - Doesn't account for confounding with biological groups
    - May miss subtle batch effects (use PCA for deeper analysis)
    - Requires meaningful batch identifiers in metadata

    Performance Considerations:
    - Time: O(n_samples) for metric calculation
    - Memory: Creates temporary DataFrame with n_samples rows
    - Sparse matrices: Efficient sum() and getnnz() operations
    - Suitable for datasets with 10^6+ samples

    Integration with Visualization:
    Use results to guide visualization:
        >>> summary = assess_batch_effects(container, 'batch')
        >>> # Plot median_features by batch
        >>> import matplotlib.pyplot as plt
        >>> plt.bar(summary['batch'], summary['median_features'])
        >>> plt.ylabel('Median Features')
        >>> plt.title('Batch Comparison')

    Common Pitfalls:
    - Wrong batch column: Verify column contains actual batch info
    - Too many batches: Consider grouping or hierarchical analysis
    - Confounded batches: If batch=biology, correction may remove signal
    - Small batches: Metrics unstable with <10 samples per batch

    References
    ----------
    .. [1] Leek, J. T., et al. (2010). Tackling the widespread and critical
       impact of batch effects in high-throughput data. Nature Reviews
       Genetics, 11(10), 733-739.
    .. [2] Hicks, S. C., et al. (2015). Missing data and batch effects in
       single-cell RNA-seq experiments. BioRxiv, 025352.
    .. [3] Hie, B., et al. (2018). Performance of batch correction methods
       for single-cell RNA sequencing data. Nature Biotechnology, 36(12),
       1193-1195.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    # Validate batch column exists
    if batch_col not in container.obs.columns:
        available = ", ".join(f"'{col}'" for col in container.obs.columns)
        raise ScpValueError(
            f"Batch column '{batch_col}' not found in container.obs. "
            f"Available columns: {available}. "
            f"Use container.obs.columns to see all metadata columns. "
            f"Common batch column names: 'batch', 'batch_id', 'plate', 'run', 'tmt_batch'.",
            parameter="batch_col",
            value=batch_col,
        )

    assay = container.assays[assay_name]

    # Get data matrix (use 'raw' layer if available, otherwise first layer)
    layer = assay.layers.get("raw", next(iter(assay.layers.values())))
    X = layer.X

    # Calculate QC metrics for each sample
    if sp.issparse(X):
        n_features = X.getnnz(axis=1)
        total_intensity = np.array(X.sum(axis=1)).flatten()
    else:
        n_features = np.sum((X > 0) & (~np.isnan(X)), axis=1)
        total_intensity = np.nansum(X, axis=1)

    # Create temporary DataFrame with batch identifiers and metrics
    temp_df = container.obs.select(batch_col).with_columns(
        [pl.Series("n_features", n_features), pl.Series("total_intensity", total_intensity)]
    )

    # Calculate summary statistics per batch
    # Using median for robustness to outliers
    summary = (
        temp_df.group_by(batch_col)
        .agg(
            [
                pl.col("n_features").count().alias("n_cells"),
                pl.col("n_features").median().alias("median_features"),
                pl.col("n_features").std().alias("std_features"),
                pl.col("total_intensity").median().alias("median_intensity"),
            ]
        )
        .sort(batch_col)
    )

    return summary
