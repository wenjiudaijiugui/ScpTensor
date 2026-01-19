"""Cache management for benchmark results."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from scptensor.benchmark.modules.base import ModuleResult

__all__ = ["CacheManager"]


class CacheManager:
    """Manager for caching benchmark results."""

    def __init__(self, cache_dir: Path | str = Path("benchmark_results/cache")) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def get_cache_key(self, module: str, dataset: str, params: dict) -> str:
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        hash_input = f"{module}:{dataset}:{sorted_params}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def get(self, module: str, dataset: str, params: dict) -> list[ModuleResult] | None:
        cache_file = self._cache_dir / f"{self.get_cache_key(module, dataset, params)}.pkl"
        if not cache_file.exists():
            return None
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            return None

    def set(self, module: str, dataset: str, params: dict, results: list[ModuleResult]) -> None:
        cache_file = self._cache_dir / f"{self.get_cache_key(module, dataset, params)}.pkl"
        cache_file.write_bytes(pickle.dumps(results))

    def clear(self, module: str | None = None) -> None:
        for f in self._cache_dir.glob("*.pkl"):
            if module is None:
                f.unlink(missing_ok=True)
            else:
                try:
                    results = pickle.loads(f.read_bytes())
                    if results and results[0].module_name == module:
                        f.unlink(missing_ok=True)
                except Exception:
                    continue

    def is_valid(self, module: str, dataset: str, params: dict) -> bool:
        return self.get(module, dataset, params) is not None

    def get_cache_info(self) -> dict:
        cache_files = list(self._cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        module_counts = {}
        for f in cache_files:
            try:
                results = pickle.loads(f.read_bytes())
                if results:
                    name = results[0].module_name
                    module_counts[name] = module_counts.get(name, 0) + 1
            except Exception:
                continue
        return {
            "cache_dir": str(self._cache_dir),
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "module_counts": module_counts,
        }

    def prune_by_size(self, max_size_mb: float) -> int:
        cache_files = sorted(self._cache_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime)
        total_size = sum(f.stat().st_size for f in cache_files)
        max_bytes = max_size_mb * 1024 * 1024
        removed = 0
        for f in cache_files:
            if total_size <= max_bytes:
                break
            size = f.stat().st_size
            f.unlink(missing_ok=True)
            total_size -= size
            removed += 1
        return removed
