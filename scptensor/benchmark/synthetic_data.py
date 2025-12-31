"""
High-quality synthetic dataset generation for benchmarking.
"""

import numpy as np
import polars as pl
from typing import Optional, Dict, List, Tuple
from scptensor.core.structures import ScpContainer, Assay, ScpMatrix, MaskCode


class SyntheticDataset:
    """
    Generate high-quality synthetic single-cell proteomics datasets for benchmarking.

    Features:
    - Realistic protein abundance distributions
    - Controlled biological groups and batch effects
    - Configurable missing value patterns
    - Ground truth for validation
    """

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
        random_seed: int = 42
    ):
        """
        Initialize synthetic dataset parameters.

        Args:
            n_samples: Number of samples/cells
            n_features: Number of proteins/features
            n_groups: Number of biological groups
            n_batches: Number of experimental batches
            missing_rate: Overall missing value rate
            batch_effect_strength: Strength of batch effect (0-1)
            group_effect_strength: Strength of biological group effects (0-1)
            signal_to_noise_ratio: Ratio of biological signal to noise
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_groups = n_groups
        self.n_batches = n_batches
        self.missing_rate = missing_rate
        self.batch_effect_strength = batch_effect_strength
        self.group_effect_strength = group_effect_strength
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.random_seed = random_seed

    def generate(self) -> ScpContainer:
        """
        Generate a synthetic dataset.

        Returns:
            ScpContainer with synthetic data and metadata
        """
        np.random.seed(self.random_seed)

        # 1. Generate sample metadata
        sample_ids = [f"S{i:03d}" for i in range(self.n_samples)]
        groups = self._generate_groups()
        batches = self._generate_batches()

        obs = pl.DataFrame({
            'sample_id': sample_ids,
            'group': groups,
            'batch': batches,
            '_index': sample_ids
        })

        # 2. Generate true protein expression values
        X_true = self._generate_true_expression()

        # 3. Add batch and group effects
        X_effect = self._add_effects(X_true, groups, batches)

        # 4. Add measurement noise
        X_noisy = self._add_noise(X_effect)

        # 5. Introduce missing values
        X_observed, M = self._introduce_missing_values(X_noisy)

        # 6. Create feature metadata
        var = self._generate_feature_metadata()

        # 7. Create ScpContainer
        matrix = ScpMatrix(X=X_observed, M=M)
        assay = Assay(var=var, layers={'raw': matrix}, feature_id_col='protein_id')

        container = ScpContainer(
            assays={'protein': assay},
            obs=obs,
            sample_id_col='sample_id'
        )

        return container

    def _generate_groups(self) -> List[str]:
        """Generate biological group assignments."""
        samples_per_group = self.n_samples // self.n_groups
        groups = []

        for i in range(self.n_groups):
            group_name = f"Group{i+1}"
            groups.extend([group_name] * samples_per_group)

        # Add remaining samples to last group
        remaining = self.n_samples - len(groups)
        if remaining > 0:
            groups.extend([f"Group{self.n_groups}"] * remaining)

        return groups

    def _generate_batches(self) -> List[str]:
        """Generate batch assignments."""
        samples_per_batch = self.n_samples // self.n_batches
        batches = []

        for i in range(self.n_batches):
            batch_name = f"Batch{i+1}"
            batches.extend([batch_name] * samples_per_batch)

        # Add remaining samples to last batch
        remaining = self.n_samples - len(batches)
        if remaining > 0:
            batches.extend([f"Batch{self.n_batches}"] * remaining)

        # Random shuffle to mix groups within batches
        np.random.shuffle(batches)

        return batches

    def _generate_true_expression(self) -> np.ndarray:
        """Generate true protein expression values."""
        # Generate protein-specific expression levels (log-normal distribution)
        protein_means = np.random.lognormal(mean=1, sigma=0.8, size=self.n_features)

        # Add protein-specific variance
        protein_vars = np.random.gamma(shape=2, scale=0.3, size=self.n_features)

        # Generate expression matrix
        X = np.zeros((self.n_samples, self.n_features))

        for j in range(self.n_features):
            # Sample-specific expression around protein mean
            sample_expression = np.random.lognormal(
                mean=np.log(protein_means[j]),
                sigma=np.sqrt(protein_vars[j]),
                size=self.n_samples
            )
            X[:, j] = sample_expression

        return X

    def _add_effects(
        self,
        X: np.ndarray,
        groups: List[str],
        batches: List[str]
    ) -> np.ndarray:
        """Add biological group and batch effects to expression data."""

        X_effect = X.copy()

        # Add group effects (for first 20% of proteins)
        n_group_proteins = int(self.n_features * 0.2)
        group_protein_indices = np.random.choice(
            self.n_features, size=n_group_proteins, replace=False
        )

        for i, group_name in enumerate(groups):
            for j in group_protein_indices:
                # Different groups have different expression levels for group proteins
                group_effect = 1.0 + (i - self.n_groups/2) * self.group_effect_strength * 0.1
                X_effect[i, j] *= group_effect

        # Add batch effects (for all proteins)
        batch_effects = {}
        for i, batch_name in enumerate(batches):
            if batch_name not in batch_effects:
                # Generate batch-specific multiplicative effect
                batch_effect = np.random.lognormal(
                    mean=0,
                    sigma=self.batch_effect_strength * 0.1
                )
                batch_effects[batch_name] = batch_effect

            X_effect[i, :] *= batch_effects[batch_name]

        return X_effect

    def _add_noise(self, X: np.ndarray) -> np.ndarray:
        """Add measurement noise to expression data."""
        # Calculate signal strength
        signal_var = np.var(X)

        # Add Gaussian noise based on signal-to-noise ratio
        noise_std = np.sqrt(signal_var / self.signal_to_noise_ratio)
        noise = np.random.normal(0, noise_std, size=X.shape)

        return X + noise

    def _introduce_missing_values(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Introduce realistic missing value patterns."""
        X_observed = X.copy()
        M = np.zeros(X.shape, dtype=np.int8)

        # 1. LOD (Limit of Detection) missing values - 60% of missing
        n_lod = int(self.n_samples * self.n_features * self.missing_rate * 0.6)

        # Target lowest abundance proteins and samples
        protein_abundances = np.mean(X, axis=0)
        sample_totals = np.sum(X, axis=1)

        # Probability weighting based on abundance (lower = higher chance)
        protein_probs = 1.0 / (protein_abundances + 1e-8)
        protein_probs = np.maximum(protein_probs, 0)  # Ensure non-negative
        protein_probs = protein_probs / protein_probs.sum()

        sample_probs = 1.0 / (sample_totals + 1e-8)
        sample_probs = np.maximum(sample_probs, 0)  # Ensure non-negative
        sample_probs = sample_probs / sample_probs.sum()

        # Combined probability matrix
        combined_probs = np.outer(sample_probs, protein_probs).flatten()
        combined_probs = np.maximum(combined_probs, 0)  # Ensure non-negative
        combined_probs = combined_probs / combined_probs.sum()

        lod_indices = np.random.choice(
            self.n_samples * self.n_features,
            size=n_lod,
            replace=False,
            p=combined_probs
        )

        lod_row_indices = lod_indices // self.n_features
        lod_col_indices = lod_indices % self.n_features

        X_observed[lod_row_indices, lod_col_indices] = 0
        M[lod_row_indices, lod_col_indices] = MaskCode.LOD.value

        # 2. Random missing values (MBR) - 40% of missing
        n_random = int(self.n_samples * self.n_features * self.missing_rate * 0.4)

        # Get remaining valid indices
        valid_mask = (M == 0)
        valid_indices = np.argwhere(valid_mask)

        if len(valid_indices) > n_random:
            random_choice_idx = np.random.choice(len(valid_indices), size=n_random, replace=False)
            random_indices = valid_indices[random_choice_idx]

            X_observed[random_indices[:, 0], random_indices[:, 1]] = 0
            M[random_indices[:, 0], random_indices[:, 1]] = MaskCode.MBR.value

        return X_observed, M

    def _generate_feature_metadata(self) -> pl.DataFrame:
        """Generate feature (protein) metadata."""
        protein_ids = [f"P{i+1:04d}" for i in range(self.n_features)]

        # Add some protein classifications for testing
        protein_classes = np.random.choice(
            ['Highly_Variable', 'Medium_Variable', 'Low_Variable'],
            size=self.n_features,
            p=[0.2, 0.6, 0.2]
        )

        return pl.DataFrame({
            'protein_id': protein_ids,
            'protein_class': protein_classes,
            '_index': protein_ids
        })

    def get_ground_truth(self) -> Dict[str, np.ndarray]:
        """
        Get ground truth information for validation.

        Returns:
            Dictionary with ground truth labels and effects
        """
        np.random.seed(self.random_seed)

        groups = self._generate_groups()
        batches = self._generate_batches()

        return {
            'groups': np.array(groups),
            'batches': np.array(batches),
            'group_labels': np.array([int(g.replace('Group', '')) - 1 for g in groups]),
            'batch_labels': np.array([int(b.replace('Batch', '')) - 1 for b in batches])
        }


def create_benchmark_datasets() -> List[ScpContainer]:
    """
    Create a collection of benchmark datasets for testing.

    Returns:
        List of synthetic datasets with varying characteristics
    """
    datasets = []

    # Dataset 1: Small, high signal
    dataset1 = SyntheticDataset(
        n_samples=50,
        n_features=200,
        n_groups=2,
        n_batches=2,
        missing_rate=0.2,
        signal_to_noise_ratio=3.0,
        random_seed=42
    )
    datasets.append(dataset1.generate())

    # Dataset 2: Medium size, moderate complexity
    dataset2 = SyntheticDataset(
        n_samples=100,
        n_features=500,
        n_groups=3,
        n_batches=3,
        missing_rate=0.3,
        signal_to_noise_ratio=2.0,
        random_seed=123
    )
    datasets.append(dataset2.generate())

    # Dataset 3: Large, challenging
    dataset3 = SyntheticDataset(
        n_samples=200,
        n_features=1000,
        n_groups=4,
        n_batches=4,
        missing_rate=0.4,
        signal_to_noise_ratio=1.5,
        random_seed=456
    )
    datasets.append(dataset3.generate())

    # Dataset 4: High batch effect
    dataset4 = SyntheticDataset(
        n_samples=150,
        n_features=800,
        n_groups=2,
        n_batches=3,
        missing_rate=0.35,
        batch_effect_strength=0.4,
        signal_to_noise_ratio=2.5,
        random_seed=789
    )
    datasets.append(dataset4.generate())

    return datasets