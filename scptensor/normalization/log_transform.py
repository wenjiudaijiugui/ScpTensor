"""Deprecated compatibility wrapper for log transformation.

Use ``scptensor.transformation.log_transform`` instead.
"""

from __future__ import annotations

import warnings

from scptensor.core.structures import ScpContainer
from scptensor.transformation import log_transform as _log_transform


def log_transform(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0,
    use_jit: bool = True,
    detect_logged: bool = True,
    skip_if_logged: bool = True,
) -> ScpContainer:
    """Deprecated wrapper for backward compatibility.

    Notes
    -----
    This entry point is deprecated and will be removed in a future version.
    Import from ``scptensor.transformation`` instead.
    """
    warnings.warn(
        "scptensor.normalization.log_transform is deprecated. "
        "Use scptensor.transformation.log_transform instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _log_transform(
        container=container,
        assay_name=assay_name,
        source_layer=source_layer,
        new_layer_name=new_layer_name,
        base=base,
        offset=offset,
        use_jit=use_jit,
        detect_logged=detect_logged,
        skip_if_logged=skip_if_logged,
    )


__all__ = ["log_transform"]
