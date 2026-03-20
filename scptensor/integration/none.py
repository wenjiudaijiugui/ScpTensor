"""No-op baseline integration for DIA-based single-cell proteomics.

This method copies the source layer to a new layer without applying batch
correction. It is useful as an explicit baseline in method comparison.
"""

from __future__ import annotations

from scptensor.core.exceptions import ScpValueError
from scptensor.core.structures import ScpContainer
from scptensor.integration.base import (
    clone_layer_matrix,
    log_integration_operation,
    register_integrate_method,
    validate_layer_params,
)


@register_integrate_method("none", integration_level="matrix", recommended_for_de=True)
def integrate_none(
    container: ScpContainer,
    batch_key: str = "batch",
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str | None = "none",
) -> ScpContainer:
    """Create an explicit no-correction baseline layer.

    Parameters
    ----------
    container : ScpContainer
        Input container.
    batch_key : str, default="batch"
        Batch label column in ``obs``. Required for consistency with other
        integration methods and downstream evaluators.
    assay_name : str, default="protein"
        Assay name to operate on (supports ``protein``/``proteins`` aliases).
    base_layer : str, default="raw"
        Source layer to copy.
    new_layer_name : str | None, default="none"
        Target layer name for copied values.
    """
    assay, layer = validate_layer_params(container, assay_name, base_layer)

    if batch_key not in container.obs.columns:
        raise ScpValueError(
            f"Batch key '{batch_key}' not found in obs. Available columns: {container.obs.columns}",
            parameter="batch_key",
            value=batch_key,
        )

    assay.add_layer(new_layer_name or "none", clone_layer_matrix(layer))

    return log_integration_operation(
        container,
        action="integration_none",
        method_name="none",
        params={
            "batch_key": batch_key,
            "assay": assay_name,
            "base_layer": base_layer,
        },
        description="No batch correction baseline (layer copy).",
    )


__all__ = ["integrate_none"]
