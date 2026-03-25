"""Helpers for loading benchmark-local sidecar modules without name collisions."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _module_cache_key(module_path: Path) -> str:
    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    parent = module_path.parent.name.replace("-", "_")
    stem = module_path.stem.replace("-", "_")
    return f"_scptensor_benchmark_{parent}_{stem}_{digest}"


def load_sidecar_module(
    anchor_file: str | Path,
    module_basename: str,
    *,
    relative_dir: str | Path | None = None,
) -> ModuleType:
    """Load a benchmark-local Python sidecar by file path.

    Benchmarks under different directories often use the same helper names
    (for example ``metrics.py`` and ``plots.py``). Importing them by bare
    module name makes the active ``sys.modules`` entry depend on import order.
    This loader gives each sidecar a path-derived cache key so benchmark tests
    and CLI entrypoints stay isolated.
    """

    anchor_path = Path(anchor_file).resolve()
    directory = anchor_path.parent if relative_dir is None else anchor_path.parent / relative_dir
    module_path = directory / f"{module_basename}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Benchmark sidecar module does not exist: {module_path}")

    cache_key = _module_cache_key(module_path)
    cached = sys.modules.get(cache_key)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load benchmark sidecar module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_sidecar_module"]
