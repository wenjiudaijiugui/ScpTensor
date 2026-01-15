"""High-quality synthetic dataset generation for benchmarking."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from scptensor.core.structures import Assay, MaskCode, ScpContainer, ScpMatrix

# Constants
_GROUP_PROTEIN_RATIO = 0.2
_LOD_MISSING_RATIO = 0.6
_EPSILON = 1e-8
_PROTEIN_CLASSES = ["Highly_Variable", "Medium_Variable", "Low_Variable"]
_CLASS_PROBABILITIES = [0.2, 0.6, 0.2]


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    """Configuration for synthetic dataset generation.

    Attributes
    ----------
    n_samples : int
        Number of samples/cells.
    n_features : int
        Number of proteins/features.
    n_groups : int
        Number of biological groups.
    n_batches : int
        Number of experimental batches.
    missing_rate : float
        Overall missing value rate.
    batch_effect_strength : float
        Strength of batch effect (0-1).
    group_effect_strength : float
        Strength of biological group effects (0-1).
    signal_to_noise_ratio : float
        Ratio of biological signal to noise.
    random_seed : int
        Random seed for reproducibility.
    """

    n_samples: int = 200
    n_features: int = 1000
    n_groups: int = 2
    n_batches: int = 2
    missing_rate: float = 0.3
    batch_effect_strength: float = 0.2
    group_effect_strength: float = 0.5
    signal_to_noise_ratio: float = 2.0
    random_seed: int = 42


class SyntheticDataset:
    """Generate high-quality synthetic single-cell proteomics datasets for benchmarking.

    Features:
    - Realistic protein abundance distributions
    - Controlled biological groups and batch effects
    - Configurable missing value patterns
    - Ground truth for validation

    Examples
    --------
    >>> from scptensor.benchmark import SyntheticDataset
    >>> dataset = SyntheticDataset(n_samples=100, n_features=500, random_seed=42)
    >>> container = dataset.generate()
    """

    __slots__ = ("config",)

    def __init__(
        self,
        n_samples: int = 200,
        n_features: int = 1000,
        n_groups: int = 2,
        n_batches: int = 2,
        missing_rate: float = 0.3,
        batch_effect_strength: float = 0.2,
        group_effect_strength: float = 0.5,
        signal_to_noise_ratio: float = 2.0,
        random_seed: int = 42,
    ) -> None:
        """Initialize synthetic dataset parameters.

        Parameters
        ----------
        n_samples : int
            Number of samples/cells.
        n_features : int
            Number of proteins/features.
        n_groups : int
            Number of biological groups.
        n_batches : int
            Number of experimental batches.
        missing_rate : float
            Overall missing value rate.
        batch_effect_strength : float
            Strength of batch effect (0-1).
        group_effect_strength : float
            Strength of biological group effects (0-1).
        signal_to_noise_ratio : float
            Ratio of biological signal to noise.
        random_seed : int
            Random seed for reproducibility.
        """
        self.config = DatasetConfig(
            n_samples=n_samples,
            n_features=n_features,
            n_groups=n_groups,
            n_batches=n_batches,
            missing_rate=missing_rate,
            batch_effect_strength=batch_effect_strength,
            group_effect_strength=group_effect_strength,
            signal_to_noise_ratio=signal_to_noise_ratio,
            random_seed=random_seed,
        )

    def generate(self) -> ScpContainer:
        """Generate a synthetic dataset.

        Returns
        -------
        ScpContainer
            Container with synthetic data and metadata.
        """
        np.random.seed(self.config.random_seed)

        sample_ids = [f"S{i:03d}" for i in range(self.config.n_samples)]
        groups = self._generate_groups()
        batches = self._generate_batches()

        obs = pl.DataFrame(
            {
                "sample_id": sample_ids,
                "group": groups,
                "batch": batches,
                "_index": sample_ids,
            }
        )

        X_true = self._generate_true_expression()
        X_effect = self._add_effects(X_true, groups, batches)
        X_noisy = self._add_noise(X_effect)
        X_observed, M = self._introduce_missing_values(X_noisy)

        var = self._generate_feature_metadata()
        matrix = ScpMatrix(X=X_observed, M=M)
        assay = Assay(var=var, layers={"raw": matrix}, feature_id_col="protein_id")

        return ScpContainer(
            assays={"protein": assay},
            obs=obs,
            sample_id_col="sample_id",
        )

    def _generate_groups(self) -> list[str]:
        """Generate biological group assignments.

        Returns
        -------
        list[str]
            Group labels for each sample.
        """
        samples_per_group = self.config.n_samples // self.config.n_groups
        groups = []

        for i in range(self.config.n_groups):
            groups.extend([f"Group{i + 1}"] * samples_per_group)

        remaining = self.config.n_samples - len(groups)
        if remaining > 0:
            groups.extend([f"Group{self.config.n_groups}"] * remaining)

        return groups

    def _generate_batches(self) -> list[str]:
        """Generate batch assignments.

        Returns
        -------
        list[str]
            Batch labels for each sample.
        """
        samples_per_batch = self.config.n_samples // self.config.n_batches
        batches = []

        for i in range(self.config.n_batches):
            batches.extend([f"Batch{i + 1}"] * samples_per_batch)

        remaining = self.config.n_samples - len(batches)
        if remaining > 0:
            batches.extend([f"Batch{self.config.n_batches}"] * remaining)

        np.random.shuffle(batches)
        return batches

    def _generate_true_expression(self) -> np.ndarray:
        """Generate true protein expression values.

        Returns
        -------
        np.ndarray
            Expression matrix (n_samples x n_features).
        """
        protein_means = np.random.lognormal(mean=1, sigma=0.8, size=self.config.n_features)
        protein_vars = np.random.gamma(shape=2, scale=0.3, size=self.config.n_features)

        X = np.empty((self.config.n_samples, self.config.n_features))

        for j in range(self.config.n_features):
            X[:, j] = np.random.lognormal(
                mean=np.log(protein_means[j]),
                sigma=np.sqrt(protein_vars[j]),
                size=self.config.n_samples,
            )

        return X

    def _add_effects(self, X: np.ndarray, groups: list[str], batches: list[str]) -> np.ndarray:
        """Add biological group and batch effects to expression data.

        Parameters
        ----------
        X : np.ndarray
            Base expression matrix.
        groups : list[str]
            Group assignments.
        batches : list[str]
            Batch assignments.

        Returns
        -------
        np.ndarray
            Expression matrix with added effects.
        """
        X_effect = X.copy()

        # Group effects (for first 20% of proteins)
        n_group_proteins = int(self.config.n_features * _GROUP_PROTEIN_RATIO)
        group_indices = np.random.choice(
            self.config.n_features, size=n_group_proteins, replace=False
        )

        group_labels = [int(g.replace("Group", "")) - 1 for g in groups]
        for i, group_idx in enumerate(group_labels):
            effect = (
                1.0
                + (group_idx - self.config.n_groups / 2) * self.config.group_effect_strength * 0.1
            )
            X_effect[i, group_indices] *= effect

        # Batch effects (for all proteins)
        batch_effects = {}
        for i, batch_name in enumerate(batches):
            if batch_name not in batch_effects:
                batch_effects[batch_name] = np.random.lognormal(
                    mean=0, sigma=self.config.batch_effect_strength * 0.1
                )
            X_effect[i, :] *= batch_effects[batch_name]

        return X_effect

    def _add_noise(self, X: np.ndarray) -> np.ndarray:
        """Add measurement noise to expression data.

        Parameters
        ----------
        X : np.ndarray
            Input expression matrix.

        Returns
        -------
        np.ndarray
            Noisy expression matrix.
        """
        signal_var = np.var(X)
        noise_std = np.sqrt(signal_var / self.config.signal_to_noise_ratio)

        return X + np.random.normal(0, noise_std, size=X.shape)

    def _introduce_missing_values(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Introduce realistic missing value patterns.

        Parameters
        ----------
        X : np.ndarray
            Input expression matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (X_with_missing, mask_matrix) where mask uses MaskCode values.
        """
        X_observed = X.copy()
        M = np.zeros(X.shape, dtype=np.int8)

        n_total = self.config.n_samples * self.config.n_features
        n_lod = int(n_total * self.config.missing_rate * _LOD_MISSING_RATIO)

        # Target lowest abundance proteins and samples
        protein_abundances = np.mean(X, axis=0)
        sample_totals = np.sum(X, axis=1)

        protein_probs = np.maximum(1.0 / (protein_abundances + _EPSILON), 0)
        protein_probs /= protein_probs.sum()

        sample_probs = np.maximum(1.0 / (sample_totals + _EPSILON), 0)
        sample_probs /= sample_probs.sum()

        combined_probs = np.outer(sample_probs, protein_probs).flatten()
        combined_probs = np.maximum(combined_probs, 0)
        combined_probs /= combined_probs.sum()

        lod_indices = np.random.choice(n_total, size=n_lod, replace=False, p=combined_probs)
        lod_rows = lod_indices // self.config.n_features
        lod_cols = lod_indices % self.config.n_features

        X_observed[lod_rows, lod_cols] = 0
        M[lod_rows, lod_cols] = MaskCode.LOD.value

        # Random missing values (MBR) - 40% of missing
        n_random = int(n_total * self.config.missing_rate * (1 - _LOD_MISSING_RATIO))
        valid_indices = np.argwhere(M == 0)

        if len(valid_indices) > n_random:
            random_idx = np.random.choice(len(valid_indices), size=n_random, replace=False)
            random_indices = valid_indices[random_idx]

            X_observed[random_indices[:, 0], random_indices[:, 1]] = 0
            M[random_indices[:, 0], random_indices[:, 1]] = MaskCode.MBR.value

        return X_observed, M

    def _generate_feature_metadata(self) -> pl.DataFrame:
        """Generate feature (protein) metadata.

        Returns
        -------
        pl.DataFrame
            Feature metadata with protein_id and protein_class.
        """
        protein_ids = [f"P{i + 1:04d}" for i in range(self.config.n_features)]
        protein_classes = np.random.choice(
            _PROTEIN_CLASSES,
            size=self.config.n_features,
            p=_CLASS_PROBABILITIES,
        )

        return pl.DataFrame(
            {
                "protein_id": protein_ids,
                "protein_class": protein_classes,
                "_index": protein_ids,
            }
        )

    def get_ground_truth(self) -> dict[str, np.ndarray]:
        """Get ground truth information for validation.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with ground truth labels and effects.
        """
        np.random.seed(self.config.random_seed)

        groups = self._generate_groups()
        batches = self._generate_batches()

        return {
            "groups": np.array(groups),
            "batches": np.array(batches),
            "group_labels": np.array([int(g.replace("Group", "")) - 1 for g in groups]),
            "batch_labels": np.array([int(b.replace("Batch", "")) - 1 for b in batches]),
        }


def create_benchmark_datasets() -> list[ScpContainer]:
    """Create a collection of benchmark datasets for testing.

    Returns
    -------
    list[ScpContainer]
        List of synthetic datasets with varying characteristics.
    """
    configs = [
        (50, 200, 2, 2, 0.2, 0.2, 0.5, 3.0, 42),  # Small, high signal
        (100, 500, 3, 3, 0.3, 0.2, 0.5, 2.0, 123),  # Medium, moderate
        (200, 1000, 4, 4, 0.4, 0.2, 0.5, 1.5, 456),  # Large, challenging
        (150, 800, 2, 3, 0.35, 0.4, 0.5, 2.5, 789),  # High batch effect
    ]

    datasets = []
    for cfg in configs:
        generator = SyntheticDataset(*cfg)
        datasets.append(generator.generate())

    return datasets
