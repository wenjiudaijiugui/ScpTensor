"""ScpTensor: DIA-based single-cell proteomics preprocessing framework.

ScpTensor provides an end-to-end analysis stack built around a hierarchical
container model (ScpContainer -> Assay -> ScpMatrix), with strong support for
DIA quant-table workflows.

Current I/O scope focuses on DIA-NN and Spectronaut outputs at protein and
peptide/precursor levels (long and matrix table shapes).

Quick Start:
    >>> from scptensor import aggregate_to_protein, load_diann
    >>> from scptensor.normalization import norm_median
    >>> from scptensor.transformation import log_transform
    >>> container = load_diann("report.tsv", level="peptide", table_format="long")
    >>> container = aggregate_to_protein(
    ...     container, source_assay="peptides", target_assay="proteins"
    ... )
    >>> container = log_transform(
    ...     container,
    ...     assay_name="proteins",
    ...     source_layer="raw",
    ...     new_layer_name="log",
    ... )
    >>> container = norm_median(
    ...     container,
    ...     assay_name="proteins",
    ...     source_layer="log",
    ...     new_layer_name="norm",
    ... )
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__author__ = "ScpTensor Team"

_CORE_EXPORTS = [
    "ScpContainer",
    "Assay",
    "ScpMatrix",
]

_AGGREGATION_EXPORTS = ["aggregate_to_protein"]

_IO_EXPORTS = [
    "load_diann",
    "load_peptide_pivot",
    "load_spectronaut",
]

_EXPORT_GROUPS: dict[str, list[str]] = {
    "scptensor.core": _CORE_EXPORTS,
    "scptensor.aggregation": _AGGREGATION_EXPORTS,
    "scptensor.io": _IO_EXPORTS,
}

_EXPORT_MAP = {
    symbol: (module_name, symbol)
    for module_name, symbols in _EXPORT_GROUPS.items()
    for symbol in symbols
}

__all__ = [
    "__version__",
    *_CORE_EXPORTS,
    *_IO_EXPORTS,
    *_AGGREGATION_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily resolve top-level reexports from their owning subpackages."""
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:  # pragma: no cover - stdlib-facing fallback
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy reexports through interactive discovery tools."""
    return sorted(set(globals()) | set(__all__))
