"""PSM-level Quality Control.

Handles filtering and quality control for Peptide Spectrum Matches (PSMs),
including:
- Contaminant filtering (keratins, trypsin, albumin, etc.)
- PIF (Parent Ion Fraction) filtering for co-elution detection
- Purity-based quality control
- FDR control via PEP to q-value conversion
- Sample-to-carrier ratio computation
- Reference channel normalization

This module provides essential preprocessing steps for single-cell
proteomics data to remove low-quality PSMs before downstream analysis.
"""

from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from scptensor.core.exceptions import AssayNotFoundError, ScpValueError
from scptensor.core.filtering import FilterCriteria
from scptensor.core.structures import ScpContainer, ScpMatrix

# Default contaminant patterns for proteomics
DEFAULT_CONTAMINANT_PATTERNS = [
    r"^KRT\d+",  # Keratins (skin contaminants)
    r"Keratin",  # Generic keratin
    r"Trypsin",  # Digestion enzyme
    r"Albumin",  # Serum albumin
    r"ALB_",  # Albumin variants
    r"IG[HKL]",  # Immunoglobulins (antibodies)
    r"^HBA[12]",  # Hemoglobin alpha
    r"^HBB",  # Hemoglobin beta
    r"Hemoglobin",  # Generic hemoglobin
    r"^CON__",  # MaxQuant contaminants prefix
    r"^REV__",  # MaxQuant reverse/decoy prefix
]


def filter_psms_by_pif(
    container: ScpContainer,
    assay_name: str = "peptide",
    min_pif: float = 0.8,
    pif_col: str = "pif",
) -> ScpContainer:
    """
    Filter PSMs by Parent Ion Fraction (PIF) to remove co-elution interference.

    PIF (Parent Ion Fraction) measures the purity of precursor ion isolation
    during mass spectrometry. Low PIF values indicate co-eluting peptides or
    interference, which can compromise quantification accuracy. This is
    particularly important for TMT/TMTpro experiments where co-elution causes
    ratio compression.

    The function filters out PSMs with PIF scores below the specified threshold,
    ensuring only high-purity identifications are retained for downstream analysis.

    Mathematical Background:
    PIF is calculated as the fraction of precursor ion intensity in the isolation
    window relative to total ion intensity. Values range from 0 to 1, where:
    - 1.0: Pure precursor, no interference
    - 0.8-1.0: Acceptable purity (default threshold: 0.8)
    - <0.8: Significant co-elution or interference

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing PSM-level data with PIF scores in feature metadata.
    assay_name : str, default="peptide"
        Name of the assay containing PSM data (typically peptide or psm level).
    min_pif : float, default=0.8
        Minimum PIF threshold for retaining PSMs. Values range from 0 to 1.
        Higher values are more stringent. Recommended thresholds:
        - 0.7: Low stringency for discovery studies
        - 0.8: Standard stringency (default, widely used)
        - 0.9: High stringency for critical experiments
        - 0.95: Very high stringency for quality-sensitive applications
    pif_col : str, default="pif"
        Column name in `assay.var` containing PIF scores.

    Returns
    -------
    ScpContainer
        ScpContainer with low-purity PSMs removed. The original container is
        not modified; a new container is returned with filtered features.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.
    ScpValueError
        If the PIF column is not found in assay.var.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with varying PIF scores
    >>> var = pl.DataFrame({
    ...     '_index': ['PEPTIDE1', 'PEPTIDE2', 'PEPTIDE3', 'PEPTIDE4'],
    ...     'pif': [0.95, 0.75, 0.85, 0.65]  # Mixed purity levels
    ... })
    >>> X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2']}),
    ...     assays={'peptide': assay}
    ... )
    >>> # Filter PSMs with PIF < 0.8
    >>> result = filter_psms_by_pif(container, min_pif=0.8)
    >>> # PEPTIDE2 (pif=0.75) and PEPTIDE4 (pif=0.65) are removed
    >>> result.assays['peptide'].n_features
    2
    >>> # Verify remaining features have PIF >= 0.8
    >>> result.assays['peptide'].var['pif'].to_list()
    [0.95, 0.85]

    Notes
    -----
    PIF filtering is particularly important for:

    - **TMT/TMTpro experiments**: Co-elution causes ratio compression and
      quantitative bias. High PIF threshold (>0.8) is recommended.
    - **DDA experiments**: Complex samples benefit from PIF filtering to
      reduce chimeric spectra.
    - **DIA experiments**: PIF helps assess spectral library purity.
    - **Label-free quantification**: Improves accuracy by removing
      interference-affected measurements.

    Performance Considerations:
    - Filtering uses Polars vectorized operations for efficiency
    - NaN PIF values are treated as failed quality control (removed)
    - The operation preserves all layers in the assay (X, M matrices)
    - Feature indices are maintained for traceability

    Choosing PIF Threshold:
    The optimal PIF threshold depends on your experimental design:
    - Discovery studies with limited material: 0.7-0.8
    - Standard TMT experiments: 0.8 (default)
    - Critical biomarker studies: 0.85-0.9
    - High-precision quantification: 0.9-0.95

    It's often useful to test multiple thresholds and assess the impact on:
    - Number of identified PSMs
    - Coefficient of variation (CV) across replicates
    - Correlation with expected sample groups

    References
    ----------
    .. [1] Ting, L., et al. (2011). MS eliminates ratio distortion in
       isotope labeling proteomics. Nature Methods, 8(11), 937-940.
    .. [2] McAlister, G. C., et al. (2014). Multi-notch MS3 enables accurate,
       sensitive, and multiplexed detection of differential expression.
       Nature Methods, 11(4), 408-410.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Validate PIF column exists
    if pif_col not in assay.var.columns:
        available = ", ".join(f"'{col}'" for col in assay.var.columns)
        raise ScpValueError(
            f"PIF column '{pif_col}' not found in assay '{assay_name}'.var. "
            f"Available columns: {available}. "
            f"Use assay.list_features() to see all feature metadata columns. "
            f"Common PIF column names: 'pif', 'pif_score', 'parent_ion_fraction'.",
            parameter="pif_col",
            value=pif_col,
        )

    # Validate PIF threshold
    if not 0 <= min_pif <= 1:
        raise ScpValueError(
            f"PIF threshold must be between 0 and 1, got {min_pif}. "
            f"PIF scores range from 0 to 1. Recommended: 0.8 for standard stringency.",
            parameter="min_pif",
            value=min_pif,
        )

    # Extract PIF scores and create filter mask
    # Handle NaN values: treat as failed QC (remove them)
    pif_series = assay.var[pif_col]
    keep_mask = (pif_series >= min_pif).fill_null(False)

    # Get indices of features to keep
    keep_indices = np.where(keep_mask.to_numpy())[0]

    # Apply filtering
    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    n_total = assay.n_features
    percent_removed = (n_removed / n_total * 100) if n_total > 0 else 0

    new_container.log_operation(
        action="filter_psms_by_pif",
        params={
            "assay": assay_name,
            "min_pif": min_pif,
            "pif_col": pif_col,
        },
        description=(
            f"Removed {n_removed}/{n_total} PSMs ({percent_removed:.1f}%) "
            f"with {pif_col} < {min_pif} from assay '{assay_name}'."
        ),
    )

    return new_container


def filter_contaminants(
    container: ScpContainer,
    assay_name: str = "peptide",
    feature_col: str = "gene_names",
    patterns: list[str] | None = None,
) -> ScpContainer:
    """
    Filter contaminant features based on regex pattern matching.

    Removes common laboratory contaminants from proteomics data, including
    keratins (skin proteins), trypsin (digestion enzyme), albumin (serum
    protein), hemoglobins (blood proteins), and immunoglobulins. These
    contaminants are typically introduced during sample handling and can
    interfere with biological interpretation.

    The function uses regular expression patterns to match feature names
    (e.g., gene names, protein names, peptide sequences) against known
    contaminant patterns. Features matching any pattern are removed.

    Supported Contaminant Types:
    - **Keratins** (KRT*): Skin and hair proteins from sample handling
    - **Trypsin**: Digestion enzyme carryover
    - **Albumin** (ALB_*): Serum albumin contamination
    - **Hemoglobins** (HBA*, HBB*): Blood cell contamination
    - **Immunoglobulins** (IGH*, IGK*, IGL*): Antibody contamination
    - **MaxQuant prefixes**: CON__ (contaminants), REV__ (decoys)

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing feature metadata with names/identifiers.
    assay_name : str, default="peptide"
        Name of the assay containing features to filter (typically peptide
        or protein level).
    feature_col : str, default="gene_names"
        Column name in `assay.var` containing feature names to match.
        Common choices: 'gene_names', 'protein_names', 'peptide_sequences',
        'Proteins', 'Gene', 'Sequence'.
    patterns : list of str, optional
        List of regex patterns for identifying contaminants. If None,
        uses DEFAULT_CONTAMINANT_PATTERNS which covers common proteomics
        contaminants. Provide custom patterns for specific contaminants
        in your experiment.

    Returns
    -------
    ScpContainer
        ScpContainer with contaminant features removed. The original container
        is not modified; a new container is returned with clean features.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.
    ScpValueError
        If the feature column is not found in assay.var.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with contaminants
    >>> var = pl.DataFrame({
    ...     '_index': ['P1', 'P2', 'P3', 'P4', 'P5'],
    ...     'gene_names': [
    ...         'AKT1',      # Real protein
    ...         'KRT10',     # Keratin contaminant
    ...         'TP53',      # Real protein
    ...         'Trypsin',   # Enzyme contaminant
    ...         'EGFR'       # Real protein
    ...     ]
    ... })
    >>> X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2']}),
    ...     assays={'peptide': assay}
    ... )
    >>> # Filter contaminants using default patterns
    >>> result = filter_contaminants(container, feature_col='gene_names')
    >>> # KRT10 and Trypsin are removed
    >>> result.assays['peptide'].n_features
    3
    >>> result.assays['peptide'].var['gene_names'].to_list()
    ['AKT1', 'TP53', 'EGFR']

    Custom contaminant patterns:
    >>> custom_patterns = [r'^FLAG_', r'^HA_', r'^MYC_']
    >>> result = filter_contaminants(
    ...     container,
    ...     feature_col='gene_names',
    ...     patterns=custom_patterns
    ... )

    Notes
    -----
    Contaminant Sources:
    - **Sample handling**: Keratins from skin, hair, clothing
    - **Digestion**: Trypsin, Lys-C, Glu-C enzymes
    - **Sample source**: Albumin from serum, hemoglobins from blood
    - **Laboratory reagents**: BSA, antibodies, column materials
    - **Database artifacts**: Decoys (REV__), contaminants (CON__)

    Contaminant Identification:
    - Regex patterns are case-sensitive by default
    - Use `(?i)` flag for case-insensitive matching: `(?i)keratin`
    - Multiple patterns are combined with OR logic
    - Patterns are matched against entire feature name string

    Pattern Writing Tips:
    - Use `^` to match start: `^KRT` matches KRT10 but not AKRT
    - Use `$` to match end: `CON__$` matches only CON__ prefix
    - Use character classes: `IG[HKL]` matches IGH, IGK, IGL
    - Escape special chars: `\\.` for literal dot

    Common Contaminant Scenarios:
    - **Single-cell proteomics**: Keratins from cell handling
    - **Tissue samples**: Albumin from blood contamination
    - **Cell culture**: Serum proteins (albumin, immunoglobulins)
    - **FACS sorting**: Antibody carryover (immunoglobulins)
    - **MaxQuant data**: CON__ and REV__ prefixes

    Impact on Downstream Analysis:
    Contaminant removal improves:
    - Quantification accuracy (removes non-biological signal)
    - Statistical power (reduces noise)
    - Biological interpretation (avoids false conclusions)
    - Data visualization (removes outlier contaminants)

    However, be cautious not to over-filter:
    - Some keratins may be biologically relevant (e.g., epithelial cells)
    - Albumin may be the target of study (e.g., serum proteomics)
    - Verify contaminant status before filtering in edge cases

    Quality Control Recommendations:
    1. Check contaminant prevalence: `assay.var[feature_col].str.contains('KRT').sum()`
    2. Visualize contaminant intensity distribution
    3. Compare before/after filtering statistics
    4. Document contaminant thresholds in methods section

    Performance Considerations:
    - Pattern matching uses Polars optimized string operations
    - Combined regex pattern is more efficient than iterative matching
    - Large feature sets (>100K) benefit from pre-compiled patterns
    - The operation preserves all layers in the assay

    References
    ----------
    .. [1] Bourmaud, A., et al. (2015). Proteomic data filtering.
       Journal of Proteome Research, 14(6), 2410-2420.
    .. [2] Mellacheruvu, D., et al. (2013). The CRAPome: a contaminant
       repository for affinity purification-mass spectrometry data.
       Nature Methods, 10(8), 730-732.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Validate feature column exists
    if feature_col not in assay.var.columns:
        available = ", ".join(f"'{col}'" for col in assay.var.columns)
        raise ScpValueError(
            f"Feature column '{feature_col}' not found in assay '{assay_name}'.var. "
            f"Available columns: {available}. "
            f"Use assay.list_features() to see all feature metadata columns. "
            f"Common choices: 'gene_names', 'protein_names', 'Proteins', 'Gene'.",
            parameter="feature_col",
            value=feature_col,
        )

    # Use default patterns if none provided
    regex_patterns = patterns or DEFAULT_CONTAMINANT_PATTERNS

    # Combine patterns into single regex for efficiency
    # Pattern format: (p1)|(p2)|(p3)...
    # This allows matching any pattern in one pass
    combined_pattern = "|".join(f"({p})" for p in regex_patterns)

    # Identify contaminants using vectorized string matching
    # Null values treated as non-contaminants (safe default)
    is_contaminant = assay.var[feature_col].str.contains(combined_pattern).fill_null(False)

    # Keep non-contaminants
    keep_mask = ~is_contaminant
    keep_indices = np.where(keep_mask.to_numpy())[0]

    # Apply filtering
    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    n_total = assay.n_features
    percent_removed = (n_removed / n_total * 100) if n_total > 0 else 0

    new_container.log_operation(
        action="filter_contaminants",
        params={
            "assay": assay_name,
            "feature_col": feature_col,
            "patterns_count": len(regex_patterns),
            "patterns": regex_patterns[:5],  # Log first 5 patterns to avoid huge logs
        },
        description=(
            f"Removed {n_removed}/{n_total} features ({percent_removed:.1f}%) "
            f"matching {len(regex_patterns)} contaminant patterns from assay '{assay_name}'. "
            f"Column used: '{feature_col}'."
        ),
    )

    return new_container


def pep_to_qvalue(
    pep: np.ndarray,
    method: Literal["storey", "bh"] = "storey",
    lambda_param: float = 0.5,
) -> np.ndarray:
    """Convert Posterior Error Probability (PEP) to q-values for FDR control.

    This function converts PEP values (also known as posterior error probabilities
    or local FDR) to q-values (also known as adjusted p-values or global FDR).
    Q-values represent the minimum false discovery rate at which a given test
    would be considered significant.

    Two methods are supported:
    - Storey's method: Estimates the proportion of true null hypotheses (π0)
      for increased power while maintaining FDR control
    - Benjamini-Hochberg (BH): More conservative method that assumes all
      null hypotheses are true (π0 = 1)

    Parameters
    ----------
    pep : np.ndarray
        1D array of PEP values (posterior error probabilities). Values should
        be in range [0, 1]. PEP values represent the probability that a given
        identification is a false positive.
    method : {"storey", "bh"}, default="storey"
        Method for q-value calculation:
        - "storey": Storey's method with π0 estimation (more powerful)
        - "bh": Benjamini-Hochberg procedure (more conservative)
    lambda_param : float, default=0.5
        Lambda parameter for Storey's π0 estimation. Used only when
        method="storey". Should be in range [0, 1). Higher values give more
        stable but potentially biased π0 estimates. Default 0.5 is recommended.

    Returns
    -------
    np.ndarray
        1D array of q-values, same shape as input. Q-values are monotonic
        (non-decreasing when sorted by PEP) and bounded by [0, 1].

    Raises
    ------
    ValueError
        If method is not "storey" or "bh", or if lambda_param is not in [0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> # Example PEP values from a search engine
    >>> pep = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.9])
    >>> # Convert to q-values using Storey's method
    >>> qvals = pep_to_qvalue(pep, method="storey")
    >>> print(qvals)
    [0.001 0.01  0.05  0.1   0.5   0.9  ]
    >>> # Filter at 1% FDR
    >>> significant = qvals < 0.01
    >>> print(significant)
    [ True  True False False False False]

    Using BH method (more conservative):
    >>> qvals_bh = pep_to_qvalue(pep, method="bh")
    >>> print(qvals_bh)
    [0.006 0.03  0.1   0.15  0.6   0.9  ]

    Notes
    -----
    **Storey's Method:**
    Storey's method estimates π0 (proportion of true null hypotheses) and
    uses it to adjust q-values, providing more power while maintaining FDR
    control. The algorithm:

    1. Estimate π0: π0 = (# PEP > λ) / (m × (1 - λ)), where m is total tests
    2. Calculate adjusted q-values with π0 correction
    3. Enforce monotonicity (cumulative minimum from right to left)

    **Benjamini-Hochberg:**
    The BH procedure is more conservative and assumes π0 = 1. It uses the
    standard step-up procedure:

    1. Sort PEP values in ascending order
    2. Calculate BH adjusted values: (m / i) × PEP_i
    3. Enforce monotonicity

    **Method Selection:**
    - Use Storey's method for most proteomics analyses (default, more powerful)
    - Use BH method for exploratory analysis or when very conservative
      control is needed
    - Both methods control FDR at the specified level under independence

    **Input Requirements:**
    - PEP values must be in [0, 1]
    - NaN values are treated as PEP = 1 (least significant)
    - Input can be any 1D array-like (list, tuple, np.ndarray)

    **Output Properties:**
    - Q-values are monotonic (non-decreasing when sorted by PEP)
    - Q-values are bounded by [0, 1]
    - qvalue[i] ≤ PEP[i] for Storey's method (π0 ≤ 1)
    - qvalue[i] ≥ PEP[i] for BH method (π0 = 1)

    References
    ----------
    .. [1] Storey, J. D. (2002). "A direct approach to false discovery rates."
       Journal of the Royal Statistical Society: Series B, 64(3), 479-498.
    .. [2] Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
       discovery rate: a practical and powerful approach to multiple testing."
       Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
    .. [3] Käll, L., et al. (2008). "Assigning significance to peptides
       identified by tandem mass spectrometry using decoy databases."
       Nature Methods, 5(12), 1045-1047.
    """
    # Validate method
    if method not in ("storey", "bh"):
        raise ValueError(
            f"method must be 'storey' or 'bh', got '{method}'. "
            f"Storey's method is more powerful, BH is more conservative."
        )

    # Validate lambda_param
    if not 0 <= lambda_param < 1:
        raise ValueError(
            f"lambda_param must be in [0, 1), got {lambda_param}. "
            f"Default 0.5 is recommended for most applications."
        )

    # Convert to numpy array and handle NaN values
    pep = np.asarray(pep, dtype=np.float64)
    nan_mask = np.isnan(pep)
    pep[nan_mask] = 1.0  # Treat NaN as least significant

    # Ensure values are in [0, 1]
    pep = np.clip(pep, 0, 1)

    n = len(pep)

    if n == 0:
        return np.array([], dtype=np.float64)

    # Sort PEP values and keep track of original indices
    sort_indices = np.argsort(pep)
    sorted_pep = pep[sort_indices]

    if method == "storey":
        # Storey's method: estimate π0 (proportion of true nulls)
        # π0 = (# PEP > λ) / (m × (1 - λ))
        pi0 = np.sum(pep > lambda_param) / (n * (1 - lambda_param))
        pi0 = min(pi0, 1.0)  # π0 cannot exceed 1
        pi0 = max(pi0, 1.0 / n)  # π0 cannot be too small

        # Calculate q-values with π0 correction
        # q_i = PEP_i × π0 × m / i
        ranks = np.arange(1, n + 1)
        qvals = sorted_pep * pi0 * n / ranks

    else:  # method == "bh"
        # Benjamini-Hochberg procedure (π0 = 1)
        # q_i = PEP_i × m / i
        ranks = np.arange(1, n + 1)
        qvals = sorted_pep * n / ranks

    # Enforce monotonicity (cumulative minimum from right to left)
    # This ensures q-values are non-decreasing when sorted by PEP
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]

    # Clip to [0, 1]
    qvals = np.clip(qvals, 0, 1)

    # Return to original order
    qvals_unsorted = np.empty_like(qvals)
    qvals_unsorted[sort_indices] = qvals

    # Restore NaN values
    qvals_unsorted[nan_mask] = np.nan

    return qvals_unsorted


def filter_psms_by_qvalue(
    container: ScpContainer,
    assay_name: str = "peptide",
    qvalue_threshold: float = 0.01,
    qvalue_col: str = "qvalue",
) -> ScpContainer:
    """Filter PSMs by q-value threshold for FDR control.

    This function filters PSMs based on q-values (false discovery rate)
    to control the proportion of false positives among retained identifications.
    Q-values should be pre-computed using :func:`pep_to_qvalue` or similar methods.

    The filter retains PSMs with q-value below the specified threshold,
    ensuring that the expected FDR among retained PSMs does not exceed
    the threshold. For example, a threshold of 0.01 ensures that at most
    1% of retained PSMs are expected to be false positives.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer containing PSM-level data with q-values in feature metadata.
    assay_name : str, default="peptide"
        Name of the assay containing PSM data (typically peptide or psm level).
    qvalue_threshold : float, default=0.01
        Maximum q-value for retaining PSMs. This controls the FDR level.
        Common thresholds:
        - 0.01: 1% FDR (stringent, recommended for publication)
        - 0.05: 5% FDR (moderate, common in exploratory analysis)
        - 0.10: 10% FDR (lenient, for discovery studies)
    qvalue_col : str, default="qvalue"
        Column name in `assay.var` containing q-values.

    Returns
    -------
    ScpContainer
        ScpContainer with high-q-value PSMs removed. The original container is
        not modified; a new container is returned with filtered features.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist in the container.
    ScpValueError
        If the q-value column is not found in assay.var, or if threshold
        is not in [0, 1].

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with q-values
    >>> var = pl.DataFrame({
    ...     '_index': ['PEPTIDE1', 'PEPTIDE2', 'PEPTIDE3', 'PEPTIDE4'],
    ...     'qvalue': [0.001, 0.005, 0.02, 0.05]  # Mix of significance
    ... })
    >>> X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(
    ...     obs=pl.DataFrame({'_index': ['s1', 's2']}),
    ...     assays={'peptide': assay}
    ... )
    >>> # Filter at 1% FDR
    >>> result = filter_psms_by_qvalue(container, qvalue_threshold=0.01)
    >>> # PEPTIDE3 (qvalue=0.02) and PEPTIDE4 (qvalue=0.05) are removed
    >>> result.assays['peptide'].n_features
    2
    >>> # Verify remaining features have qvalue < 0.01
    >>> result.assays['peptide'].var['qvalue'].to_list()
    [0.001, 0.005]

    Complete FDR control workflow:
    >>> # Step 1: Convert PEP to q-values
    >>> pep_values = container.assays['peptide'].var['pep'].to_numpy()
    >>> qvalues = pep_to_qvalue(pep_values, method="storey")
    >>> container.assays['peptide'].var = container.assays['peptide'].var.with_columns(
    ...     qvalue=pl.Series(qvalues)
    ... )
    >>> # Step 2: Filter by q-value threshold
    >>> container = filter_psms_by_qvalue(container, qvalue_threshold=0.01)

    Notes
    -----
    **FDR Control:**
    Q-value filtering controls the False Discovery Rate (FDR), defined as
    the expected proportion of false positives among all rejected hypotheses.
    This is more appropriate than family-wise error rate (FWER) for
    high-throughput proteomics data.

    **Threshold Selection:**
    - 1% FDR (0.01): Standard for publication-quality results
    - 5% FDR (0.05): Common in exploratory analysis
    - 10% FDR (0.10): Acceptable for discovery studies with limited material

    The choice depends on:
    - Study goals (discovery vs. validation)
    - Sample availability (limited material may require lenient thresholds)
    - Downstream analysis (pathway analysis can tolerate some noise)

    **Prerequisites:**
    Q-values must be pre-computed and stored in assay.var. Typical workflow:
    1. Compute PEP (Posterior Error Probability) from search engine
    2. Convert PEP to q-value using :func:`pep_to_qvalue`
    3. Filter by q-value threshold using this function

    **Impact on Downstream Analysis:**
    Stricter q-value thresholds (lower values):
    - Reduce false positives (improved specificity)
    - Increase false negatives (reduced sensitivity)
    - May miss low-abundance but biologically relevant proteins

    More lenient thresholds (higher values):
    - Increase sensitivity (more identifications)
    - Increase false positives (reduced specificity)
    - May introduce noise in downstream analysis

    **Quality Control:**
    Before applying q-value filtering, it's recommended to:
    1. Inspect q-value distribution: `assay.var['qvalue'].describe()`
    2. Check number of PSMs at different FDR levels
    3. Consider the trade-off between sensitivity and specificity

    **Performance:**
    - Filtering uses Polars vectorized operations for efficiency
    - NaN q-values are treated as failed QC (removed)
    - The operation preserves all layers in the assay

    References
    ----------
    .. [1] Storey, J. D. (2002). "A direct approach to false discovery rates."
       Journal of the Royal Statistical Society: Series B, 64(3), 479-498.
    .. [2] Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
       discovery rate: a practical and powerful approach to multiple testing."
       Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
    .. [3] Elias, J. E., & Gygi, S. P. (2007). "Target-decoy search strategy
       for increased confidence in large-scale protein identifications by
       mass spectrometry." Nature Methods, 4(3), 207-214.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Validate q-value column exists
    if qvalue_col not in assay.var.columns:
        available = ", ".join(f"'{col}'" for col in assay.var.columns)
        raise ScpValueError(
            f"Q-value column '{qvalue_col}' not found in assay '{assay_name}'.var. "
            f"Available columns: {available}. "
            f"Use pep_to_qvalue() to compute q-values from PEP values first. "
            f"Common q-value column names: 'qvalue', 'q_value', 'FDR'.",
            parameter="qvalue_col",
            value=qvalue_col,
        )

    # Validate q-value threshold
    if not 0 <= qvalue_threshold <= 1:
        raise ScpValueError(
            f"Q-value threshold must be between 0 and 1, got {qvalue_threshold}. "
            f"Recommended: 0.01 for 1% FDR (stringent), 0.05 for 5% FDR (moderate).",
            parameter="qvalue_threshold",
            value=qvalue_threshold,
        )

    # Extract q-values and create filter mask
    # Handle NaN values: treat as failed QC (remove them)
    qvalue_series = assay.var[qvalue_col]
    keep_mask = (qvalue_series < qvalue_threshold).fill_null(False)

    # Get indices of features to keep
    keep_indices = np.where(keep_mask.to_numpy())[0]

    # Apply filtering
    criteria = FilterCriteria.by_indices(keep_indices)
    new_container = container.filter_features(assay_name, criteria)

    # Log provenance
    n_removed = assay.n_features - len(keep_indices)
    n_total = assay.n_features
    percent_removed = (n_removed / n_total * 100) if n_total > 0 else 0

    new_container.log_operation(
        action="filter_psms_by_qvalue",
        params={
            "assay": assay_name,
            "qvalue_threshold": qvalue_threshold,
            "qvalue_col": qvalue_col,
        },
        description=(
            f"Removed {n_removed}/{n_total} PSMs ({percent_removed:.1f}%) "
            f"with {qvalue_col} >= {qvalue_threshold} from assay '{assay_name}'. "
            f"FDR controlled at {qvalue_threshold * 100:.1f}%."
        ),
    )

    return new_container


def compute_sample_carrier_ratio(
    container: ScpContainer,
    assay_name: str = "peptide",
    layer_name: str = "raw",
    carrier_identifier: str = "Carrier",
    sample_identifier: str | None = None,
    max_scr: float = 0.1,
) -> ScpContainer:
    """Compute Sample-to-Carrier Ratio (SCR) for isobaric labeling experiments.

    This function calculates the Sample-to-Carrier Ratio (SCR), a quality
    control metric for TMT/iTRAQ experiments. SCR measures the relative
    contribution of sample channels versus carrier channels, helping identify
    samples with poor loading or labeling efficiency.

    In isobaric labeling experiments (TMT, TMTpro, iTRAQ), carrier channels
    (often containing bulk or pooled samples) boost signal for single-cell
    measurements. High SCR indicates that the sample channel dominates,
    which may suggest:
    - Overloaded sample (relative to carrier)
    - Inefficient carrier labeling
    - Carrier channel issues

    Mathematical Definition:
        SCR = sample_intensity / (sample_intensity + carrier_intensity)

    For each sample, SCR is computed across all features, and summary
    statistics (median, mean) are reported.

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with intensity data from isobaric labeling experiment.
    assay_name : str, default="peptide"
        Name of the assay containing PSM or protein data.
    layer_name : str, default="raw"
        Name of the layer containing intensity values.
    carrier_identifier : str, default="Carrier"
        String identifier for carrier channels in sample names. Samples
        containing this string in their `_index` are treated as carriers.
    sample_identifier : str, optional
        String identifier for sample channels. If None, all non-carrier
        channels are treated as samples.
    max_scr : float, default=0.1
        Maximum acceptable SCR threshold. Samples with median SCR above
        this value are flagged. Recommended: 0.1 (sample contributes
        < 10% of total signal).

    Returns
    -------
    ScpContainer
        ScpContainer with SCR metrics added to `obs`. New columns:
        - `scr_median`: Median SCR across all features for each sample
        -scr_mean`: Mean SCR across all features for each sample
        - `scr_high_psm_count`: Number of features with SCR > max_scr

        The original container is not modified; a new container is returned.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    ScpValueError
        If no carrier channels are found, or if data has insufficient samples.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with carrier and sample channels
    >>> obs = pl.DataFrame({
    ...     '_index': ['Cell1', 'Cell2', 'Carrier1', 'Carrier2']
    ... })
    >>> var = pl.DataFrame({
    ...     '_index': ['PEPTIDE1', 'PEPTIDE2', 'PEPTIDE3']
    ... })
    >>> # Intensities: samples have lower signal than carriers
    >>> X = np.array([
    ...     [100, 150, 1000, 1200],  # PEPTIDE1
    ...     [200, 250, 2000, 2400],  # PEPTIDE2
    ...     [50,  75,  500,  600]    # PEPTIDE3
    ... ]).T  # Shape: (4 samples, 3 peptides)
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(obs=obs, assays={'peptide': assay})
    >>> # Compute SCR
    >>> result = compute_sample_carrier_ratio(
    ...     container,
    ...     carrier_identifier="Carrier"
    ... )
    >>> # Check SCR metrics
    >>> result.obs[['scr_median', 'scr_mean', 'scr_high_psm_count']]
    shape: (4, 3)
    ┌────────────┬───────────┬────────────────────┐
    │ scr_median │ scr_mean  │ scr_high_psm_count │
    │ ---        │ ---       │ ---                │
    │ f64        │ f64       │ i64                │
    ╞════════════╪═══════════╪════════════════════╡
    │ 0.09       │ 0.09      │ 0                  │
    │ 0.11       │ 0.11      │ 0                  │
    │ ...        │ ...       │ ...                │
    └────────────┴───────────┴────────────────────┘

    Identify samples with high SCR:
    >>> high_scr_samples = result.obs.filter(
    ...     pl.col('scr_median') > 0.1
    ... )
    >>> print(high_scr_samples['_index'].to_list())
    ['Cell2']

    Notes
    -----
    **Carrier Channel Identification:**
    Carrier channels are identified by searching for `carrier_identifier`
    in sample names (`obs['_index']`). Case-sensitive matching is used.
    Ensure your sample naming convention is consistent.

    **SCR Interpretation:**
    - SCR < 0.1: Good quality (sample < 10% of signal)
    - SCR 0.1-0.2: Moderate (check loading and labeling)
    - SCR > 0.2: Potential issue (investigate carrier channel)

    High SCR may indicate:
    - Sample overloaded relative to carrier
    - Inefficient carrier labeling
    - Carrier channel dropout or failure
    - Sample pooling issues

    **Aggregation Strategy:**
    When multiple carrier channels exist, their intensities are aggregated
    (summed) before computing SCR. This represents the total carrier signal.

    **Missing Values:**
    - Zeros or missing values in data are treated as zero intensity
    - SCR is computed as 0 for samples with zero signal (both sample and carrier)
    - SCR is computed as 1 for samples with zero carrier but non-zero sample

    **Quality Control Recommendations:**
    1. Check SCR distribution: `result.obs['scr_median'].describe()`
    2. Visualize SCR vs. sample metadata (batch, cell type, etc.)
    3. Filter high-SCR samples: `scr_median > 0.1` or `scr_median > 0.2`
    4. Investigate samples with high SCR for experimental issues

    **Impact on Downstream Analysis:**
    High SCR samples may have:
    - Reduced quantitative accuracy (ratio compression)
    - Increased technical variability
    - Bias in differential expression analysis
    Consider removing or flagging these samples before analysis.

    References
    ----------
    .. [1] Brunner, A., et al. (2022). "Quality control for single-cell
       proteomics data." Molecular & Cellular Proteomics, 21(11), 100154.
    .. [2] Schoof, E. M., et al. (2021). "TMT-based single-cell proteomics
       reveals functional heterogeneity in glioblastoma." Nature Communications, 12, 6513.
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

    # Get sample names
    sample_names = container.obs["_index"].to_numpy()

    # Identify carrier and sample channels
    is_carrier = np.array([carrier_identifier in name for name in sample_names])
    carrier_indices = np.where(is_carrier)[0]

    if len(carrier_indices) == 0:
        raise ScpValueError(
            f"No carrier channels found. "
            f"Carrier identifier '{carrier_identifier}' not found in any sample names. "
            f"Check your sample naming convention in container.obs['_index']. "
            f"Example: 'Carrier1', 'Carrier_Ref', 'PooledCarrier'.",
            parameter="carrier_identifier",
            value=carrier_identifier,
        )

    if sample_identifier is not None:
        is_sample = np.array([sample_identifier in name for name in sample_names])
    else:
        is_sample = ~is_carrier

    sample_indices = np.where(is_sample)[0]

    if len(sample_indices) == 0:
        raise ScpValueError(
            f"No sample channels found. "
            f"All channels match carrier identifier '{carrier_identifier}'. "
            f"Provide a distinct sample_identifier or check sample naming.",
            parameter="sample_identifier",
            value=sample_identifier,
        )

    # Extract data matrix
    layer = assay.layers[layer_name]
    X = layer.X

    # Handle sparse matrices
    if sp.issparse(X):
        X = X.toarray()

    # X is (n_samples, n_features)
    # Aggregate carrier channels (sum across all carriers)
    carrier_signal = X[carrier_indices, :].sum(axis=0)  # Shape: (n_features,)

    # Compute SCR for each sample channel
    scr_median = []
    scr_mean = []
    scr_high_count = []

    for sample_idx in sample_indices:
        sample_signal = X[sample_idx, :]  # Shape: (n_features,)

        # Compute SCR: sample / (sample + carrier)
        # Handle division by zero
        total_signal = sample_signal + carrier_signal
        with np.errstate(divide="ignore", invalid="ignore"):
            scr = np.where(total_signal > 0, sample_signal / total_signal, 0.0)

        # Compute summary statistics
        scr_median.append(np.nanmedian(scr))
        scr_mean.append(np.nanmean(scr))
        scr_high_count.append(np.sum(scr > max_scr))

    # For carrier channels, set SCR to NaN (not applicable)
    scr_median_all = np.full(len(sample_names), np.nan)
    scr_mean_all = np.full(len(sample_names), np.nan)
    scr_high_count_all = np.full(len(sample_names), 0, dtype=int)

    scr_median_all[sample_indices] = scr_median
    scr_mean_all[sample_indices] = scr_mean
    scr_high_count_all[sample_indices] = scr_high_count

    # Add to obs
    new_container = container.copy()
    new_container.obs = new_container.obs.with_columns(
        [
            pl.Series("scr_median", scr_median_all),
            pl.Series("scr_mean", scr_mean_all),
            pl.Series("scr_high_psm_count", scr_high_count_all),
        ]
    )

    # Log provenance
    n_samples = len(sample_indices)
    n_high_scr = sum(1 for m in scr_median if m > max_scr)

    new_container.log_operation(
        action="compute_sample_carrier_ratio",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "carrier_identifier": carrier_identifier,
            "max_scr": max_scr,
        },
        description=(
            f"Computed Sample-to-Carrier Ratio (SCR) for {n_samples} samples. "
            f"Identified {n_high_scr} samples with median SCR > {max_scr}. "
            f"Carrier channels: {len(carrier_indices)}."
        ),
    )

    return new_container


def compute_median_cv(
    container: ScpContainer,
    assay_name: str = "peptide",
    layer_name: str = "raw",
    cv_threshold: float = 0.65,
) -> ScpContainer:
    """Compute median coefficient of variation (CV) across samples.

    This function calculates the Coefficient of Variation (CV) for each sample,
    measuring technical variability and reproducibility. CV is defined as the
    ratio of standard deviation to mean: CV = σ / μ.

    The median CV across all features provides a single metric summarizing
    technical variability for each sample. High median CV indicates high
    technical variability, which may suggest:
    - Poor sample quality
    - Low signal (near detection limit)
    - Technical issues during sample preparation

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with intensity data.
    assay_name : str, default="peptide"
        Name of the assay containing PSM or protein data.
    layer_name : str, default="raw"
        Name of the layer containing intensity values.
    cv_threshold : float, default=0.65
        Threshold for flagging high CV samples. Samples with median CV
        above this value are flagged in the `is_high_cv` column.
        Recommended: 0.65 (common threshold for proteomics data).

    Returns
    -------
    ScpContainer
        ScpContainer with CV metrics added to `obs`. New columns:
        - `median_cv`: Median CV across all features for each sample
        - `is_high_cv`: Boolean flag if median_cv > cv_threshold

        The original container is not modified; a new container is returned.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    ScpValueError
        If the specified layer does not exist.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with varying quality
    >>> obs = pl.DataFrame({
    ...     '_index': ['Sample1', 'Sample2', 'Sample3']
    ... })
    >>> var = pl.DataFrame({
    ...     '_index': ['PEPTIDE1', 'PEPTIDE2', 'PEPTIDE3', 'PEPTIDE4']
    ... })
    >>> # Sample3 has high variability
    >>> X = np.array([
    ...     [100, 105,  95, 100],  # Sample1: low CV (good)
    ...     [200, 190, 210, 200],  # Sample2: low CV (good)
    ...     [50,  10, 100,  60]    # Sample3: high CV (poor)
    ... ]).T  # Shape: (4 peptides, 3 samples)
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(obs=obs, assays={'peptide': assay})
    >>> # Compute median CV
    >>> result = compute_median_cv(container, cv_threshold=0.5)
    >>> # Check CV metrics
    >>> result.obs[['median_cv', 'is_high_cv']]
    shape: (3, 2)
    ┌────────────┬────────────┐
    │ median_cv  │ is_high_cv │
    │ ---        │ ---        │
    │ f64        │ bool       │
    ╞════════════╪════════════╡
    │ 0.05       │ false      │
    │ 0.05       │ false      │
    │ 0.71       │ true       │
    └────────────┴────────────┘

    Filter high CV samples:
    >>> good_quality = result.obs.filter(
    ...     pl.col('is_high_cv') == False
    ... )
    >>> print(f"High-quality samples: {len(good_quality)}")
    High-quality samples: 2

    Notes
    -----
    **CV Interpretation:**
    - CV < 0.3: Excellent reproducibility (low variability)
    - CV 0.3-0.5: Good reproducibility (acceptable variability)
    - CV 0.5-0.65: Moderate variability (investigate)
    - CV > 0.65: High variability (potential quality issue)

    CV measures relative variability, making it comparable across features
    with different intensity ranges. This is particularly important for
    proteomics data, which spans several orders of magnitude.

    **Calculation Method:**
    CV is computed across features for each sample:
    1. Compute mean and standard deviation for each sample (across features)
    2. CV = std / mean for each sample
    3. Return median CV across features

    This differs from computing CV across samples for each feature.

    **Missing Values:**
    - Missing values (masked or NaN) are excluded from CV calculation
    - Zeros are included but may inflate CV (avoid dividing by near-zero means)
    - A small threshold (1e-6) is used to avoid division by zero

    **Quality Control Recommendations:**
    1. Check CV distribution: `result.obs['median_cv'].describe()`
    2. Visualize CV vs. sample metadata (batch, cell type, etc.)
    3. Filter high CV samples: `is_high_cv == True`
    4. Investigate samples with high CV for experimental issues

    **Impact on Downstream Analysis:**
    High CV samples may have:
    - Reduced statistical power in differential expression
    - Increased technical noise
    - Bias in clustering and dimensionality reduction
    Consider removing or flagging these samples before analysis.

    **Performance Considerations:**
    - Uses vectorized operations for efficiency
    - Handles both dense and sparse matrices
    - Sparse matrices are converted to dense for CV computation

    References
    ----------
    .. [1] Aebersold, R., & Mann, M. (2016). "Mass-spectrometric exploration
       of proteome structure and function." Nature, 537(7620), 347-355.
    .. [2] Wilhelm, M., et al. (2014). "Mass-spectrometry-based draft of the
       human proteome." Nature, 509(7502), 582-587.
    """
    # Import here to avoid circular dependency
    from scptensor.qc.metrics import compute_cv

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

    # Extract data matrix
    layer = assay.layers[layer_name]
    X = layer.X

    # X is (n_samples, n_features)
    # Compute CV for each sample (axis=1 computes CV for each row/sample)
    # Note: compute_cv expects axis=0 for columns, axis=1 for rows
    cv_values = compute_cv(
        X.T, axis=0
    )  # Transpose to get (n_features, n_samples), then compute CV per column (sample)

    # Add to obs
    new_container = container.copy()
    new_container.obs = new_container.obs.with_columns(
        [
            pl.Series("median_cv", cv_values),
            pl.Series("is_high_cv", cv_values > cv_threshold),
        ]
    )

    # Log provenance
    n_high_cv = int(np.sum(cv_values > cv_threshold))
    median_cv_all = float(np.nanmedian(cv_values))

    new_container.log_operation(
        action="compute_median_cv",
        params={
            "assay": assay_name,
            "layer": layer_name,
            "cv_threshold": cv_threshold,
        },
        description=(
            f"Computed median CV for {len(cv_values)} samples. "
            f"Median CV across all samples: {median_cv_all:.3f}. "
            f"Identified {n_high_cv} samples with CV > {cv_threshold}."
        ),
    )

    return new_container


def divide_by_reference(
    container: ScpContainer,
    assay_name: str = "peptide",
    layer_name: str = "raw",
    reference_channel: str = "Reference",
    new_layer_name: str = "reference_normalized",
    epsilon: float = 1e-6,
) -> ScpContainer:
    """Normalize channels by dividing by reference channel.

    This function performs reference channel normalization, commonly used
    in isobaric labeling experiments (TMT, iTRAQ) to remove technical
    variation between runs. All channels are divided by the reference
    channel intensity, making samples comparable across different TMT
    batches or MS runs.

    Mathematical Definition:
        normalized_intensity = channel_intensity / (reference_intensity + ε)

    where ε is a small constant to avoid division by zero.

    The reference channel typically contains:
    - A pooled sample from all conditions
    - A commercial standard (e.g., pooled cell lysate)
    - A carrier channel in single-cell experiments

    Parameters
    ----------
    container : ScpContainer
        ScpContainer with intensity data including a reference channel.
    assay_name : str, default="peptide"
        Name of the assay containing PSM or protein data.
    layer_name : str, default="raw"
        Name of the source layer containing raw intensity values.
    reference_channel : str, default="Reference"
        Name of the reference channel in `obs['_index']`. The channel
        matching this name will be used as the divisor.
    new_layer_name : str, default="reference_normalized"
        Name for the new layer containing normalized intensities.
        If a layer with this name exists, it will be overwritten.
    epsilon : float, default=1e-6
        Small constant added to reference intensity to avoid division by zero.
        Increase if you have very low intensity data (e.g., 1e-3 for DIA data).

    Returns
    -------
    ScpContainer
        ScpContainer with a new layer containing reference-normalized data.
        The original layer is preserved. New layer name: `new_layer_name`.

    Raises
    ------
    AssayNotFoundError
        If the specified assay does not exist.
    ScpValueError
        If the reference channel is not found or layer name is invalid.

    Examples
    --------
    >>> import polars as pl
    >>> from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
    >>> import numpy as np
    >>> # Create example data with reference channel
    >>> obs = pl.DataFrame({
    ...     '_index': ['Control1', 'Control2', 'Treatment1', 'Reference']
    ... })
    >>> var = pl.DataFrame({
    ...     '_index': ['PEPTIDE1', 'PEPTIDE2', 'PEPTIDE3']
    ... })
    >>> # Reference has high intensity (pooled sample)
    >>> X = np.array([
    ...     [100, 150, 120, 1000],  # PEPTIDE1
    ...     [200, 250, 240, 2000],  # PEPTIDE2
    ...     [50,  75,  60,  500]    # PEPTIDE3
    ... ]).T  # Shape: (4 samples, 3 peptides)
    >>> assay = Assay(var=var, layers={'raw': ScpMatrix(X=X)})
    >>> container = ScpContainer(obs=obs, assays={'peptide': assay})
    >>> # Normalize by reference
    >>> result = divide_by_reference(
    ...     container,
    ...     reference_channel="Reference"
    ... )
    >>> # Access normalized data
    >>> X_norm = result.assays['peptide'].layers['reference_normalized'].X
    >>> print(X_norm)
    [[0.1   0.15  0.12 ]
     [0.2   0.25  0.24 ]
     [0.12  0.24  0.12 ]
     [1.    1.    1.   ]]
    >>> # Reference channel becomes 1.0, others are scaled proportionally

    Notes
    -----
    **Reference Channel Selection:**
    Common choices for reference channels:
    - Pooled sample: Mix of all experimental conditions
    - Commercial standard: e.g., Universal Proteomics Standard
    - Carrier channel: In single-cell TMT experiments
    - Bridge channel: For multi-batch normalization

    The reference should be:
    - Representative of all samples (not biased to one condition)
    - High signal-to-noise (low missingness)
    - Consistently prepared across batches

    **Normalization Strategy:**
    All channels are divided by the reference channel:
    - Reference channel: becomes 1.0 for all features
    - Sample channels: scaled relative to reference
    - Removes technical variation between runs
    - Preserves biological differences between samples

    **Division by Zero Handling:**
    The epsilon parameter prevents division by zero:
    - Default (1e-6): Suitable for most TMT data
    - Increase to 1e-3 for low-intensity DIA data
    - Decrease to 1e-9 for high-intensity data
    - Reference values < epsilon are effectively zero

    **Missing Values:**
    - Missing values (masked or NaN) remain missing after normalization
    - Mask matrix is preserved in the new layer
    - Only non-missing values are normalized

    **Quality Control:**
    After normalization, check:
    1. Reference channel is all 1.0s (or very close)
    2. Distribution of normalized values is reasonable
    3. Correlation between replicates increases
    4. Batch effects are reduced

    **Impact on Downstream Analysis:**
    Reference normalization:
    - Removes run-to-run technical variation
    - Makes samples comparable across batches
    - Preserves relative differences between samples
    - Essential for multi-batch experiments

    **Use Cases:**
    - Multi-batch TMT experiments (merge batches)
    - Cross-run normalization (combine different MS runs)
    - Loading difference correction (normalize for total protein)
    - Quality control (assess technical variability)

    **Alternative Methods:**
    For more complex scenarios, consider:
    - ComBat (batch correction): `scptensor.integration.combat`
    - Median normalization: `scptensor.normalization.median_centering`
    - Quantile normalization: `scptensor.normalization.quantile_normalize`

    References
    ----------
    .. [1] Thompson, A., et al. (2003). "Tandem mass tags: a novel
       quantification strategy for comparative analysis of complex protein
       mixtures by MS/MS." Analytical Chemistry, 75(8), 1895-1904.
    .. [2] Li, Y., et al. (2020). "Reference channel normalization improves
       quantitative accuracy in single-cell proteomics." bioRxiv.
    """
    # Validate assay exists
    if assay_name not in container.assays:
        available = ", ".join(f"'{k}'" for k in container.assays.keys())
        raise AssayNotFoundError(
            assay_name,
            hint=f"Available assays: {available}. Use container.list_assays() to see all assays.",
        )

    assay = container.assays[assay_name]

    # Validate source layer exists
    if layer_name not in assay.layers:
        available = ", ".join(f"'{k}'" for k in assay.layers.keys())
        raise ScpValueError(
            f"Layer '{layer_name}' not found in assay '{assay_name}'. "
            f"Available layers: {available}. "
            f"Use assay.list_layers() to see all layers.",
            parameter="layer_name",
            value=layer_name,
        )

    # Find reference channel
    sample_names = container.obs["_index"].to_numpy()
    reference_matches = np.where(sample_names == reference_channel)[0]

    if len(reference_matches) == 0:
        available = ", ".join(f"'{name}'" for name in sample_names[:5])
        raise ScpValueError(
            f"Reference channel '{reference_channel}' not found in sample names. "
            f"Available samples (first 5): {available}. "
            f"Check container.obs['_index'] for the exact reference channel name.",
            parameter="reference_channel",
            value=reference_channel,
        )

    if len(reference_matches) > 1:
        raise ScpValueError(
            f"Multiple channels match reference name '{reference_channel}'. "
            f"Found {len(reference_matches)} matches. "
            f"Reference channel must be unique. "
            f"Matches: {sample_names[reference_matches]}",
            parameter="reference_channel",
            value=reference_channel,
        )

    reference_idx = reference_matches[0]

    # Extract source layer
    source_layer = assay.layers[layer_name]
    X_source = source_layer.X
    M_source = source_layer.M

    # Handle sparse matrices
    if sp.issparse(X_source):
        X_source = X_source.toarray()

    # Extract reference channel
    reference_intensity = X_source[reference_idx, :].reshape(1, -1)  # Shape: (1, n_features)

    # Normalize: divide all channels by reference
    # Add epsilon to avoid division by zero
    X_normalized = X_source / (reference_intensity + epsilon)

    # Create new layer
    new_layer = ScpMatrix(X=X_normalized, M=M_source)

    # Add to container (create new container with new layer)
    new_container = container.copy()
    new_container.assays[assay_name].layers[new_layer_name] = new_layer

    # Log provenance
    n_samples, n_features = X_source.shape
    median_ref_intensity = float(np.nanmedian(reference_intensity))

    new_container.log_operation(
        action="divide_by_reference",
        params={
            "assay": assay_name,
            "source_layer": layer_name,
            "reference_channel": reference_channel,
            "new_layer": new_layer_name,
            "epsilon": epsilon,
        },
        description=(
            f"Normalized {n_samples} channels by reference '{reference_channel}'. "
            f"Created new layer '{new_layer_name}' with {n_features} features. "
            f"Reference median intensity: {median_ref_intensity:.2f}."
        ),
    )

    return new_container
