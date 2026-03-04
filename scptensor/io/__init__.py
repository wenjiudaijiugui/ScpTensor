"""ScpTensor I/O API (mass-spec focused).

Public API only keeps DIA-NN / Spectronaut quant-table importers.
"""

from __future__ import annotations

from scptensor.io.exceptions import IOFormatError, IOPasswordError, IOWriteError
from scptensor.io.mass_spec import (
    aggregate_to_protein,
    load_diann,
    load_peptide_pivot,
    load_quant_table,
    load_spectronaut,
)

__all__ = [
    # Exceptions
    "IOFormatError",
    "IOPasswordError",
    "IOWriteError",
    # Unified mass-spec import
    "load_quant_table",
    "load_diann",
    "load_spectronaut",
    "load_peptide_pivot",
    "aggregate_to_protein",
]
