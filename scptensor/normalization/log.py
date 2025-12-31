import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix

def log_normalize(
    container: ScpContainer,
    assay_name: str = "protein",
    base_layer: str = "raw",
    new_layer_name: str = "log",
    base: float = 2.0,
    offset: float = 1.0
) -> ScpContainer:
    """
    Apply Log Normalization to a specific layer in an assay.

    Args:
        container: The ScpContainer object.
        assay_name: Name of the assay to transform.
        base_layer: Name of the layer to use as input.
        new_layer_name: Name of the new layer to create.
        base: Log base (default: 2.0).
        offset: Offset to add before logging to handle zeros (default: 1.0).

    Returns:
        The modified ScpContainer (in-place modification of the assay, returning self for chaining).
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
        raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")

    input_matrix = assay.layers[base_layer]
    X = input_matrix.X
    M = input_matrix.M

    # Perform log transformation: log_base(X + offset)
    # Using change of base formula: log_b(x) = ln(x) / ln(b)
    X_log = np.log(X + offset) / np.log(base)

    # Create new matrix
    new_matrix = ScpMatrix(X=X_log, M=M.copy())

    # Add new layer to assay
    assay.add_layer(new_layer_name, new_matrix)

    # Log operation
    container.log_operation(
        action="log_normalize",
        params={
            "assay": assay_name,
            "base_layer": base_layer,
            "new_layer": new_layer_name,
            "base": base,
            "offset": offset
        },
        description=f"Log{base} normalization applied to {assay_name}/{base_layer}."
    )

    return container
