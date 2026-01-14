"""Input validation for visualization functions.

This module provides validation utilities for ScpTensor visualization functions.
All validation functions raise VisualizationError with descriptive messages.
"""

import numpy as np
from scptensor import ScpContainer
from scptensor.core.exceptions import VisualizationError, LayerNotFoundError


__all__ = [
    "validate_container",
    "validate_layer",
    "validate_features",
    "validate_groupby",
    "validate_plot_data",
]


def validate_container(container: ScpContainer) -> None:
    """Validate container is a valid ScpContainer instance.

    Parameters
    ----------
    container : ScpContainer
        Container to validate.

    Raises
    ------
    VisualizationError
        If container is None or not a ScpContainer instance.
    """
    if container is None:
        raise VisualizationError("Container cannot be None")
    if not isinstance(container, ScpContainer):
        raise VisualizationError(
            f"Expected ScpContainer, got {type(container).__name__}"
        )


def validate_layer(container: ScpContainer, assay_name: str, layer: str) -> None:
    """Validate that a layer exists in the specified assay.

    Parameters
    ----------
    container : ScpContainer
        Container containing the assay.
    assay_name : str
        Name of the assay.
    layer : str
        Name of the layer.

    Raises
    ------
    VisualizationError
        If assay does not exist.
    LayerNotFoundError
        If layer does not exist in the assay.
    """
    if assay_name not in container.assays:
        raise VisualizationError(f"Assay '{assay_name}' not found in container")

    assay = container.assays[assay_name]
    if layer not in assay.layers:
        raise LayerNotFoundError(layer_name=layer, assay_name=assay_name)


def validate_features(
    container: ScpContainer,
    assay_name: str,
    var_names: list[str],
) -> None:
    """Validate that feature names exist in the assay's var DataFrame.

    Parameters
    ----------
    container : ScpContainer
        Container containing the assay.
    assay_name : str
        Name of the assay.
    var_names : list[str]
        List of feature names to validate.

    Raises
    ------
    VisualizationError
        If assay does not exist or features are not found.
    """
    if assay_name not in container.assays:
        raise VisualizationError(f"Assay '{assay_name}' not found in container")

    assay = container.assays[assay_name]

    if assay.var.height == 0:
        raise VisualizationError(f"Assay '{assay_name}' has no features in var")

    # Try to find the feature identifier column
    # Prefer 'protein' or 'gene' columns, fall back to first column
    var_col = None
    for preferred in ["protein", "gene", "feature", "name"]:
        if preferred in assay.var.columns:
            var_col = preferred
            break

    if var_col is None:
        var_col = assay.var.columns[0]

    available = set(assay.var[var_col].to_list())
    missing = set(var_names) - available

    if missing:
        raise VisualizationError(
            f"Features not found in assay '{assay_name}': {sorted(missing)}"
        )


def validate_groupby(container: ScpContainer, groupby: str) -> None:
    """Validate that a column exists in obs for grouping operations.

    Parameters
    ----------
    container : ScpContainer
        Container containing obs.
    groupby : str
        Column name to validate.

    Raises
    ------
    VisualizationError
        If column does not exist in obs.
    """
    if groupby not in container.obs.columns:
        available = container.obs.columns
        raise VisualizationError(
            f"Column '{groupby}' not found in obs. "
            f"Available columns: {list(available)}"
        )


def validate_plot_data(X: np.ndarray, n_min: int = 1) -> None:
    """Validate sufficient data for plotting.

    Parameters
    ----------
    X : np.ndarray
        Data array to validate.
    n_min : int, default=1
        Minimum number of elements required.

    Raises
    ------
    VisualizationError
        If data size is less than n_min.
    """
    if X.size < n_min:
        raise VisualizationError(
            f"Insufficient data for plotting: {X.size} elements < {n_min} required"
        )


if __name__ == "__main__":
    print("Testing validation module...")

    import polars as pl

    # Test: validate_container
    print("\n1. Testing validate_container...")
    from scptensor import Assay, ScpContainer, ScpMatrix

    obs = pl.DataFrame({"_index": [f"S{i}" for i in range(10)], "condition": ["A"] * 5 + ["B"] * 5})
    var = pl.DataFrame({"_index": [f"P{i}" for i in range(5)]})
    good_container = ScpContainer(obs=obs)
    validate_container(good_container)
    print("   Valid container: OK")

    try:
        validate_container(None)
        print("   None container: FAILED")
    except VisualizationError as e:
        print(f"   None container: OK (raised: {e})")

    try:
        validate_container("not_a_container")
        print("   Wrong type: FAILED")
    except VisualizationError as e:
        print(f"   Wrong type: OK (raised: {e})")

    # Test: validate_layer
    print("\n2. Testing validate_layer...")

    X = np.random.rand(10, 5)
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
    container = ScpContainer(obs=obs, assays={"proteins": assay})

    validate_layer(container, "proteins", "raw")
    print("   Valid layer: OK")

    try:
        validate_layer(container, "missing", "raw")
        print("   Missing assay: FAILED")
    except VisualizationError as e:
        print(f"   Missing assay: OK (raised: {e})")

    try:
        validate_layer(container, "proteins", "nonexistent")
        print("   Missing layer: FAILED")
    except LayerNotFoundError as e:
        print(f"   Missing layer: OK (raised: {e})")

    # Test: validate_groupby
    print("\n3. Testing validate_groupby...")
    validate_groupby(container, "condition")
    print("   Valid column: OK")

    try:
        validate_groupby(container, "missing")
        print("   Missing column: FAILED")
    except VisualizationError as e:
        print(f"   Missing column: OK (raised: {e})")

    # Test: validate_plot_data
    print("\n4. Testing validate_plot_data...")
    validate_plot_data(np.array([1, 2, 3]), n_min=1)
    print("   Valid data: OK")

    try:
        validate_plot_data(np.array([]), n_min=1)
        print("   Insufficient data: FAILED")
    except VisualizationError as e:
        print(f"   Insufficient data: OK (raised: {e})")

    # Test: validate_features
    print("\n5. Testing validate_features...")
    validate_features(container, "proteins", ["P0", "P2"])
    print("   Valid features: OK")

    try:
        validate_features(container, "proteins", ["P0", "MISSING"])
        print("   Missing features: FAILED")
    except VisualizationError as e:
        print(f"   Missing features: OK (raised: {e})")

    print("\nAll validation tests passed!")
