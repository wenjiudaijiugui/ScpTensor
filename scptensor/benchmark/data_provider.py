"""Data provider for ScpTensor vs Scanpy comparison benchmarks.

This module provides unified access to synthetic and real datasets
for comparative benchmarking.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scptensor.benchmark.synthetic_data import SyntheticDataset

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclasses.dataclass(frozen=True)
class ComparisonDataset:
    name: str
    n_samples: int
    n_features: int
    missing_rate: float
    n_batches: int = 1
    n_groups: int = 3
    noise_level: float = 0.1
    is_synthetic: bool = True


COMPARISON_DATASETS: tuple[ComparisonDataset, ...] = (
    ComparisonDataset("synthetic_small", 500, 100, 0.10, 1, 3, 0.05),
    ComparisonDataset("synthetic_medium", 2000, 500, 0.15, 2, 4, 0.10),
    ComparisonDataset("synthetic_large", 10000, 1000, 0.20, 3, 5, 0.15),
    ComparisonDataset("synthetic_batch", 3000, 500, 0.20, 5, 3, 0.10),
)

PARAMETER_SWEEPS = {
    "missing_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "n_samples": [500, 1000, 2000, 5000, 10000],
    "k_knn": [5, 10, 15, 20, 30, 50],
    "n_components": [10, 20, 50, 100],
    "n_clusters": [3, 5, 8, 10, 15],
}


class DataProvider:
    def __init__(self, cache_dir: str | Path = ".benchmark_cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def get_dataset(self, config: ComparisonDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = f"{config.name}_mr{config.missing_rate}_b{config.n_batches}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        synthetic = SyntheticDataset(
            n_samples=config.n_samples,
            n_features=config.n_features,
            n_groups=config.n_groups,
            n_batches=config.n_batches,
            missing_rate=config.missing_rate,
            batch_effect_strength=0.3 if config.n_batches > 1 else 0.0,
            group_effect_strength=0.5,
            signal_to_noise_ratio=2.0,
            random_seed=42,
        )

        container = synthetic.generate()
        assay = container.assays["protein"]
        X = assay.layers["raw"].X.copy()
        M = assay.layers["raw"].M.copy() if assay.layers["raw"].M is not None else np.zeros_like(X, dtype=np.uint8)

        groups = container.obs["group"].to_numpy()
        batches = container.obs["batch"].to_numpy()

        self._cache[cache_key] = (X, M, batches, groups)
        return (X, M, batches, groups)

    def iter_datasets(self, names: tuple[str, ...] | None = None) -> Iterator[ComparisonDataset]:
        for dataset in COMPARISON_DATASETS:
            if names is None or dataset.name in names:
                yield dataset

    def get_parameter_sweep_dataset(
        self,
        param_name: str,
        param_value: float,
        base_name: str = "synthetic_medium",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base = next((d for d in COMPARISON_DATASETS if d.name == base_name), None)
        if base is None:
            raise ValueError(f"Base dataset '{base_name}' not found")

        if param_name in ("k_knn", "n_components"):
            return self.get_dataset(base)

        fields = {
            "name": f"{base_name}_sweep_{param_name}_{param_value}",
            "n_samples": base.n_samples,
            "n_features": base.n_features,
            "missing_rate": base.missing_rate,
            "n_batches": base.n_batches,
            "n_groups": base.n_groups,
            "noise_level": base.noise_level,
            "is_synthetic": True,
            param_name: param_value,
        }

        config = ComparisonDataset(**fields)
        return self.get_dataset(config)


_default_provider: DataProvider | None = None


def get_provider() -> DataProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = DataProvider()
    return _default_provider
