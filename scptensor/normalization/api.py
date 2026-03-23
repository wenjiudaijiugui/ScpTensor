"""High-level normalization API for DIA single-cell workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scptensor.core._layer_processing import clone_matrix_data
from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer

from .base import (
    add_result_layer,
    get_layer_name,
    log_operation,
    validate_layer_context,
)
from .mean_normalization import norm_mean
from .median_normalization import norm_median
from .quantile_normalization import norm_quantile
from .trqn_normalization import norm_trqn


def norm_none(
    container: ScpContainer,
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str | None = "no_norm",
) -> ScpContainer:
    """Create a passthrough layer without applying normalization."""
    ctx = validate_layer_context(container, assay_name, source_layer)
    assay = ctx.assay
    input_layer = ctx.layer
    layer_name = get_layer_name(new_layer_name, "no_norm")

    if layer_name != source_layer:
        passthrough = clone_matrix_data(input_layer.X)
        add_result_layer(assay, layer_name, passthrough, input_layer)

    log_operation(
        container,
        action="normalization_none",
        params={
            "assay": ctx.resolved_assay_name,
            "source_layer": source_layer,
            "new_layer_name": layer_name,
        },
        description=f"Normalization skipped on layer '{source_layer}' -> '{layer_name}'.",
    )
    return container


_METHODS: dict[str, Callable[..., ScpContainer]] = {
    "none": norm_none,
    "norm_none": norm_none,
    "mean": norm_mean,
    "norm_mean": norm_mean,
    "median": norm_median,
    "norm_median": norm_median,
    "quantile": norm_quantile,
    "norm_quantile": norm_quantile,
    "trqn": norm_trqn,
    "norm_trqn": norm_trqn,
}

_DEFAULT_LAYER_NAME: dict[str, str] = {
    "none": "no_norm",
    "mean": "sample_mean_norm",
    "median": "median_centered",
    "quantile": "quantile_norm",
    "trqn": "trqn_norm",
}


def normalize(
    container: ScpContainer,
    method: str = "median",
    assay_name: str = "protein",
    source_layer: str = "raw",
    new_layer_name: str | None = None,
    **kwargs: Any,
) -> ScpContainer:
    """Dispatch to a normalization method with a unified entrypoint.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    method : str, default="median"
        Normalization method name.
        Supported values: ``none``, ``mean``, ``median``, ``quantile``, ``trqn``.
        ``norm_*`` aliases are also accepted.
    assay_name : str, default="protein"
        Name of target assay.
    source_layer : str, default="raw"
        Name of source layer.
    new_layer_name : str | None, default=None
        Destination layer. If None, method-specific default is used.
    **kwargs : Any
        Additional method-specific parameters.

    Returns
    -------
    ScpContainer
        Container with normalized layer added.
    """
    method_key = method.strip().lower()
    if method_key not in _METHODS:
        available = sorted(_DEFAULT_LAYER_NAME.keys())
        raise ScpValueError(
            f"Unsupported normalization method '{method}'. "
            f"Supported methods: {', '.join(available)}.",
            parameter="method",
            value=method,
        )

    canonical = method_key.replace("norm_", "")
    layer_name = new_layer_name or _DEFAULT_LAYER_NAME[canonical]
    method_fn = _METHODS[method_key]

    return method_fn(
        container=container,
        assay_name=assay_name,
        source_layer=source_layer,
        new_layer_name=layer_name,
        **kwargs,
    )


__all__ = [
    "norm_none",
    "normalize",
]
