"""Helpers for resolving stable assay naming aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer

_ASSAY_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("protein", "proteins"),
    ("peptide", "peptides"),
)


def resolve_assay_name(container: ScpContainer, assay_name: str) -> str:
    """Resolve common singular/plural assay aliases against a container."""
    if assay_name in container.assays:
        return assay_name

    for aliases in _ASSAY_ALIAS_GROUPS:
        if assay_name not in aliases:
            continue
        for candidate in aliases:
            if candidate in container.assays:
                return candidate

    return assay_name


__all__ = ["resolve_assay_name"]
