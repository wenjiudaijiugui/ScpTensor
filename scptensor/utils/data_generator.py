"""Synthetic single-cell proteomics data generator.

This module generates realistic synthetic SCP data using statistical models
that capture the key characteristics of real single-cell proteomics data:

1. MNAR + MAR missing mechanisms: Uses probabilistic dropout (sigmoid-based)
   to simulate intensity-dependent missing values, combined with random missing.
2. Biological correlations: Implements a latent factor model (pathway-based)
   to capture co-expression patterns between proteins.
3. Heteroscedastic noise: Models intensity-dependent variance where
   low-abundance signals have higher coefficients of variation.
4. Batch effects: Simulates technical variation across batches.

References
----------
- Lazar et al. (2016): Probabilistic dropout model for mass spectrometry data
- Vanderaa & Gatto (2021): Review of missing value mechanisms in SCP
- Zappia et al. (2017): Splatter for simulating single-cell RNA data
- Rocke & Lorenzato (1995): Additive + multiplicative noise model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import expit

from scptensor.core.structures import Assay, ScpContainer, ScpMatrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Constants for data generation
_MIN_SAMPLES = 10
_MAX_SAMPLES = 10000
_MIN_FEATURES = 500
_MAX_FEATURES = 10000
_MAX_MISSING_RATE = 0.7
_DEFAULT_PROTEIN_MEAN = 15.0
_DEFAULT_PROTEIN_STD = 2.0
_DEFAULT_BIOLOGICAL_STD = 2.0
_DEFAULT_BATCH_EFFECT_STD = 0.4
_DEFAULT_GROUP_SHIFT_STD = 2.0
_GROUP_SPARSITY_RATIO = 0.5
_PROTEIN_SPARSITY_RATIO = 0.7
_PATHWAY_RATIO = 1.0 / 50.0
_MIN_PATHWAYS = 10
_NOISE_HIGH = 0.8
_NOISE_LOW = 0.3
_DROPOUT_SLOPE = 1.0
_REGULARIZATION_EPS = 1e-8


class ScpDataGenerator:
    """Generator for synthetic single-cell proteomics data in ScpContainer format.

    This class creates realistic synthetic SCP data by modeling:
    - Biological variation through latent factors (pathway-like co-expression)
    - Technical variation (sample efficiency, batch effects)
    - Heteroscedastic noise (intensity-dependent variance)
    - MNAR missing values (probabilistic dropout using sigmoid)
    - MCAR missing values (random missing)
    - Cell groups/clusters with distinct expression patterns

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples (cells). Range: 10-10000.
    n_features : int, default=1000
        Number of features (proteins). Range: 500-10000.
    missing_rate : float, default=0.3
        Total proportion of missing values. Range: 0.0-0.7.
    lod_ratio : float, default=0.6
        Proportion of missing values due to MNAR (probabilistic dropout).
        Remaining missing values are MCAR (random). Range: 0.0-1.0.
    n_batches : int, default=3
        Number of batches for simulating batch effects.
    n_groups : int, default=4
        Number of biological groups (cell types).
    random_seed : int, default=42
        Random seed for reproducibility.
    assay_name : str, default="proteins"
        Name for the assay layer.
    layer_name : str, default="raw"
        Name for the data layer.
    feature_id_col : str, default="protein_id"
        Column name for feature identifiers in var metadata.
    sample_id_col : str, default="sample_id"
        Column name for sample identifiers in obs metadata.
    mask_kind : {"int8", "bool", "none"}, default="int8"
        Type of mask to store: "int8" for integer codes, "bool" for boolean,
        "none" to skip mask creation.
    use_sparse_X : bool, default=False
        Whether to store data matrix X in sparse format.
    use_sparse_M : bool, default=False
        Whether to store mask matrix M in sparse format.

    Attributes
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    missing_rate : float
        Target missing rate.
    lod_ratio : float
        Proportion of MNAR missing values.
    n_batches : int
        Number of batches.
    n_groups : int
        Number of biological groups.

    Examples
    --------
    >>> generator = ScpDataGenerator(
    ...     n_samples=500, n_features=2000, missing_rate=0.4
    ... )
    >>> container = generator.generate()
    >>> print(container)
    ScpContainer(n_samples=500, assays=['proteins'])
    """

    __slots__ = (
        "n_samples",
        "n_features",
        "missing_rate",
        "lod_ratio",
        "n_batches",
        "n_groups",
        "random_seed",
        "assay_name",
        "layer_name",
        "feature_id_col",
        "sample_id_col",
        "mask_kind",
        "use_sparse_X",
        "use_sparse_M",
        "rng",
    )

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 1000,
        missing_rate: float = 0.3,
        lod_ratio: float = 0.6,
        n_batches: int = 3,
        n_groups: int = 4,
        random_seed: int = 42,
        assay_name: str = "proteins",
        layer_name: str = "raw",
        feature_id_col: str = "protein_id",
        sample_id_col: str = "sample_id",
        mask_kind: str = "int8",
        use_sparse_X: bool = False,
        use_sparse_M: bool = False,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.missing_rate = missing_rate
        self.lod_ratio = lod_ratio
        self.n_batches = n_batches
        self.n_groups = n_groups
        self.random_seed = random_seed
        self.assay_name = assay_name
        self.layer_name = layer_name
        self.feature_id_col = feature_id_col
        self.sample_id_col = sample_id_col
        self.mask_kind = mask_kind
        self.use_sparse_X = use_sparse_X
        self.use_sparse_M = use_sparse_M

        self._validate_params()
        self.rng = np.random.default_rng(random_seed)

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if not 0.0 <= self.missing_rate <= _MAX_MISSING_RATE:
            raise ValueError(f"missing_rate must be between 0.0 and {_MAX_MISSING_RATE}")
        if not 0.0 <= self.lod_ratio <= 1.0:
            raise ValueError("lod_ratio must be between 0.0 and 1.0")

    def generate(self) -> ScpContainer:
        """Generate the ScpContainer with synthetic data.

        Returns
        -------
        ScpContainer
            Container with generated synthetic data including:
            - Dense or sparse data matrix X
            - Mask matrix M with missing value codes
            - Feature metadata (protein_id, gene_name, mean_abundance)
            - Sample metadata (sample_id, batch, group, efficiency)
        """
        # Generate complete data matrix
        X_complete = self._generate_complete_matrix()

        # Generate missing mask
        M = self._generate_missing_mask(X_complete)

        # Prepare output matrices
        X_out, M_out = self._prepare_output_matrices(X_complete, M)

        # Create metadata
        var = self._create_feature_metadata()
        obs = self._create_sample_metadata()

        # Create ScpContainer
        matrix = ScpMatrix(X=X_out, M=M_out)
        assay = Assay(var=var, feature_id_col=self.feature_id_col)
        assay.add_layer(self.layer_name, matrix)

        container = ScpContainer(obs=obs, sample_id_col=self.sample_id_col)
        container.add_assay(self.assay_name, assay)
        self._log_provenance(container)

        return container

    def _generate_complete_matrix(self) -> NDArray[np.float64]:
        """Generate the complete data matrix without missing values.

        Returns
        -------
        NDArray[np.float64]
            Complete data matrix of shape (n_samples, n_features).
        """
        # Number of latent pathways
        n_pathways = max(_MIN_PATHWAYS, int(self.n_features * _PATHWAY_RATIO))

        # Assign groups and batches
        group_indices = self._get_group_indices()
        batch_indices = self._get_batch_indices()

        # Generate biological variation (pathway-based co-expression)
        biological_variation = self._generate_biological_variation(n_pathways, group_indices)

        # Generate technical components
        protein_means = self._generate_protein_means()
        sample_efficiencies = self._generate_sample_efficiencies()
        batch_effects = self._generate_batch_effects(batch_indices)

        # Combine components
        X_clean = protein_means + biological_variation + sample_efficiencies + batch_effects

        # Add heteroscedastic noise
        return self._add_heteroscedastic_noise(X_clean)

    def _get_group_indices(self) -> NDArray[np.int64]:
        """Get shuffled group indices for samples."""
        indices = np.array([i % self.n_groups for i in range(self.n_samples)])
        self.rng.shuffle(indices)
        return indices

    def _get_batch_indices(self) -> NDArray[np.int64]:
        """Get batch indices for samples."""
        return np.array([i % self.n_batches for i in range(self.n_samples)])

    def _generate_biological_variation(
        self, n_pathways: int, group_indices: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Generate biological variation using latent factor model.

        Parameters
        ----------
        n_pathways : int
            Number of latent pathways/factors.
        group_indices : NDArray[np.int64]
            Group assignment for each sample.

        Returns
        -------
        NDArray[np.float64]
            Biological variation matrix of shape (n_samples, n_features).
        """
        # Pathway activity per sample
        pathway_activity = self.rng.normal(0.0, 1.0, size=(self.n_samples, n_pathways))

        # Add group-specific shifts to pathway activity
        group_shifts = self._generate_group_shifts(n_pathways)
        pathway_activity += group_shifts[group_indices]

        # Protein loadings for pathways
        protein_loadings = self._generate_protein_loadings(n_pathways)

        # Compute biological variation
        biological = pathway_activity @ protein_loadings

        # Normalize to target standard deviation
        bio_std = biological.std()
        if bio_std > 0:
            biological = biological / bio_std * _DEFAULT_BIOLOGICAL_STD

        return biological

    def _generate_group_shifts(self, n_pathways: int) -> NDArray[np.float64]:
        """Generate group-specific pathway shifts.

        Parameters
        ----------
        n_pathways : int
            Number of pathways.

        Returns
        -------
        NDArray[np.float64]
            Group shifts of shape (n_groups, n_pathways).
        """
        shifts = self.rng.normal(0.0, _DEFAULT_GROUP_SHIFT_STD, size=(self.n_groups, n_pathways))
        sparsity_mask = self.rng.random((self.n_groups, n_pathways)) < _GROUP_SPARSITY_RATIO
        shifts[sparsity_mask] = 0.0
        return shifts

    def _generate_protein_loadings(self, n_pathways: int) -> NDArray[np.float64]:
        """Generate protein loadings for pathways.

        Parameters
        ----------
        n_pathways : int
            Number of pathways.

        Returns
        -------
        NDArray[np.float64]
            Protein loadings of shape (n_pathways, n_features).
        """
        loadings = self.rng.normal(0.0, 1.0, size=(n_pathways, self.n_features))
        sparsity_mask = self.rng.random((n_pathways, self.n_features)) < _PROTEIN_SPARSITY_RATIO
        loadings[sparsity_mask] = 0.0
        return loadings

    def _generate_protein_means(self) -> NDArray[np.float64]:
        """Generate baseline protein abundances.

        Returns
        -------
        NDArray[np.float64]
            Protein means of shape (1, n_features).
        """
        return self.rng.normal(_DEFAULT_PROTEIN_MEAN, _DEFAULT_PROTEIN_STD, (1, self.n_features))

    def _generate_sample_efficiencies(self) -> NDArray[np.float64]:
        """Generate sample-specific efficiency factors.

        Returns
        -------
        NDArray[np.float64]
            Efficiency factors of shape (n_samples, 1).
        """
        return self.rng.normal(0.0, 0.5, (self.n_samples, 1))

    def _generate_batch_effects(self, batch_indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Generate batch effects for samples.

        Parameters
        ----------
        batch_indices : NDArray[np.int64]
            Batch assignment for each sample.

        Returns
        -------
        NDArray[np.float64]
            Batch effects of shape (n_samples, n_features).
        """
        batch_shifts = self.rng.normal(
            0.0, _DEFAULT_BATCH_EFFECT_STD, (self.n_batches, self.n_features)
        )
        return batch_shifts[batch_indices]

    def _add_heteroscedastic_noise(
        self, X_clean: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Add intensity-dependent heteroscedastic noise.

        Parameters
        ----------
        X_clean : NDArray[np.float64]
            Clean signal without noise.

        Returns
        -------
        NDArray[np.float64]
            Noisy data matrix.
        """
        # Normalize intensity to [0, 1]
        min_val, max_val = X_clean.min(), X_clean.max()
        if max_val > min_val:
            norm_intensity = (X_clean - min_val) / (max_val - min_val)
        else:
            norm_intensity = np.zeros_like(X_clean)

        # Noise scale: high intensity -> low noise, low intensity -> high noise
        noise_scale = _NOISE_HIGH - (_NOISE_HIGH - _NOISE_LOW) * norm_intensity

        noise = self.rng.normal(0.0, 1.0, X_clean.shape) * noise_scale
        return X_clean + noise

    def _generate_missing_mask(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        """Generate missing mask with MNAR and MCAR components.

        Parameters
        ----------
        X : NDArray[np.float64]
            Complete data matrix.

        Returns
        -------
        NDArray[np.int8]
            Mask matrix where:
            - 0: Valid (detected)
            - 1: MCAR (random missing)
            - 2: MNAR (probabilistic dropout)
        """
        M = np.zeros((self.n_samples, self.n_features), dtype=np.int8)

        if self.missing_rate == 0:
            return M

        total_elements = self.n_samples * self.n_features
        target_missing = int(total_elements * self.missing_rate)
        target_mnar = int(target_missing * self.lod_ratio)
        target_mcar = target_missing - target_mnar

        # Apply MNAR (probabilistic dropout)
        if target_mnar > 0:
            M = self._apply_probabilistic_dropout(X, M, target_mnar)

        # Apply MCAR (random missing)
        if target_mcar > 0:
            M = self._apply_random_missing(M, target_mcar)

        return M

    def _apply_probabilistic_dropout(
        self, X: NDArray[np.float64], M: NDArray[np.int8], target_mnar: int
    ) -> NDArray[np.int8]:
        """Apply probabilistic dropout (MNAR) using sigmoid function.

        Lower intensity values have higher probability of being missing.

        Parameters
        ----------
        X : NDArray[np.float64]
            Data matrix.
        M : NDArray[np.int8]
            Current mask matrix.
        target_mnar : int
            Target number of MNAR missing values.

        Returns
        -------
        NDArray[np.int8]
            Updated mask matrix.
        """
        total_elements = self.n_samples * self.n_features
        target_rate = target_mnar / total_elements

        # Calibrate bias to achieve target missing rate
        bias_guess = np.percentile(X, target_rate * 100)

        # P_missing = sigmoid(-slope * (X - bias))
        p_missing = expit(-_DROPOUT_SLOPE * (X - bias_guess))

        # Apply dropout
        random_matrix = self.rng.random(X.shape)
        mnar_mask = random_matrix < p_missing
        M[mnar_mask] = 2

        return M

    def _apply_random_missing(
        self, M: NDArray[np.int8], target_mcar: int
    ) -> NDArray[np.int8]:
        """Apply random missing (MCAR) to valid entries.

        Parameters
        ----------
        M : NDArray[np.int8]
            Current mask matrix.
        target_mcar : int
            Target number of MCAR missing values.

        Returns
        -------
        NDArray[np.int8]
            Updated mask matrix.
        """
        valid_mask = M == 0
        valid_indices = np.where(valid_mask.ravel())[0]

        if len(valid_indices) == 0:
            return M

        n_to_mask = min(target_mcar, len(valid_indices))
        selected = self.rng.choice(valid_indices, size=n_to_mask, replace=False)
        rows, cols = np.unravel_index(selected, (self.n_samples, self.n_features))
        M[rows, cols] = 1

        return M

    def _prepare_output_matrices(
        self, X: NDArray[np.float64], M: NDArray[np.int8]
    ) -> tuple[Any, Any]:
        """Prepare output matrices with appropriate sparsity.

        Parameters
        ----------
        X : NDArray[np.float64]
            Data matrix.
        M : NDArray[np.int8]
            Mask matrix.

        Returns
        -------
        tuple
            (X_out, M_out) with appropriate format.
        """
        X_out = X.astype(np.float64)
        if self.use_sparse_X:
            X_out = sp.csr_matrix(X_out)

        if self.mask_kind == "none":
            M_out = None
        elif self.mask_kind == "bool":
            M_out = M != 0
            if self.use_sparse_M:
                M_out = sp.csr_matrix(M_out.astype(np.int8))
        else:  # "int8"
            M_out = M.astype(np.int8)
            if self.use_sparse_M:
                M_out = sp.csr_matrix(M_out)

        return X_out, M_out

    def _create_feature_metadata(self) -> pl.DataFrame:
        """Create feature (protein) metadata.

        Returns
        -------
        pl.DataFrame
            Feature metadata with columns: protein_id, gene_name, mean_abundance.
        """
        protein_means = self._generate_protein_means()

        feature_ids = [f"Prot_{i:05d}" for i in range(self.n_features)]
        var_data = {
            "protein_id": feature_ids,
            "gene_name": [f"Gene_{i:05d}" for i in range(self.n_features)],
            "mean_abundance": protein_means.flatten().tolist(),
        }

        if self.feature_id_col not in var_data:
            var_data[self.feature_id_col] = feature_ids

        return pl.DataFrame(var_data)

    def _create_sample_metadata(self) -> pl.DataFrame:
        """Create sample (cell) metadata.

        Returns
        -------
        pl.DataFrame
            Sample metadata with columns: sample_id, batch, group, efficiency.
        """
        batch_indices = self._get_batch_indices()
        group_indices = self._get_group_indices()
        sample_efficiencies = self._generate_sample_efficiencies()

        sample_ids = [f"Cell_{i:05d}" for i in range(self.n_samples)]
        obs_data = {
            "sample_id": sample_ids,
            "batch": [f"Batch_{i}" for i in batch_indices],
            "group": [f"Group_{i}" for i in group_indices],
            "efficiency": sample_efficiencies.flatten().tolist(),
        }

        if self.sample_id_col not in obs_data:
            obs_data[self.sample_id_col] = sample_ids

        return pl.DataFrame(obs_data)

    def _log_provenance(self, container: ScpContainer) -> None:
        """Log generation parameters to container history.

        Parameters
        ----------
        container : ScpContainer
            Container to log provenance to.
        """
        container.log_operation(
            action="generate_synthetic_data",
            params={
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "missing_rate": self.missing_rate,
                "lod_ratio": self.lod_ratio,
                "n_batches": self.n_batches,
                "seed": self.random_seed,
                "missing_mechanism": "Probabilistic Dropout (Sigmoid) + MCAR",
                "noise_model": "Heteroscedastic (Intensity-dependent)",
                "biological_model": "Latent Factor (Pathway-based Co-expression)",
                "n_groups": self.n_groups,
                "assay_name": self.assay_name,
                "layer_name": self.layer_name,
                "feature_id_col": self.feature_id_col,
                "sample_id_col": self.sample_id_col,
                "mask_kind": self.mask_kind,
                "use_sparse_X": self.use_sparse_X,
                "use_sparse_M": self.use_sparse_M,
            },
            description="Generated synthetic single-cell proteomics data with "
            "biological correlations, heteroscedastic noise, groups, and batch effects.",
        )


if __name__ == "__main__":
    import time

    print("Initializing Generator...")
    gen = ScpDataGenerator(
        n_samples=500, n_features=2000, missing_rate=0.4, lod_ratio=0.6, n_batches=3
    )

    print("Generating Data...")
    start = time.time()
    container = gen.generate()
    elapsed = time.time() - start

    print(f"Generation took {elapsed:.4f} seconds.")
    print(container)

    matrix = container.assays["proteins"].layers["raw"]

    n_total = matrix.X.size
    if matrix.M is not None:
        n_valid = np.sum(matrix.M == 0)
        n_random = np.sum(matrix.M == 1)
        n_lod = np.sum(matrix.M == 2)

        print(f"Total Elements: {n_total}")
        print(f"Valid: {n_valid} ({n_valid / n_total:.2%})")
        print(f"MCAR (M=1): {n_random} ({n_random / n_total:.2%})")
        print(f"MNAR (Probabilistic) (M=2): {n_lod} ({n_lod / n_total:.2%})")
        print(f"Total Missing: {n_random + n_lod} ({(n_random + n_lod) / n_total:.2%})")
    else:
        print("No mask matrix generated (mask_kind='none')")
