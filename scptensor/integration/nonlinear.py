from typing import Optional, List
from scptensor.core.structures import ScpContainer, ScpMatrix
from scptensor.core.utils import requires_dependency

@requires_dependency('harmonypy', 'pip install harmonypy')
def harmony(
    container: ScpContainer,
    batch_key: str,
    assay_name: str = 'protein',
    base_layer: str = 'pca', # Harmony typically runs on PCA
    new_layer_name: Optional[str] = 'harmony',
    **kwargs
) -> ScpContainer:
    """
    Harmony integration.
    """
    import harmonypy as hm
    
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")
    
    assay = container.assays[assay_name]
    if base_layer not in assay.layers:
         raise ValueError(f"Layer '{base_layer}' not found in assay '{assay_name}'.")

    data_mat = assay.layers[base_layer].X # PCA embedding: (n_samples, n_pcs)
    meta_data = container.obs.to_pandas()
    
    ho = hm.run_harmony(data_mat, meta_data, batch_key, **kwargs)
    
    res = ho.Z_corr.T # Harmony returns (n_pcs, n_samples), we need (n_samples, n_pcs)
    
    new_matrix = ScpMatrix(X=res, M=assay.layers[base_layer].M.copy())
    container.assays[assay_name].add_layer(new_layer_name, new_matrix)
    
    container.log_operation(
        action="integration_harmony",
        params={"batch_key": batch_key, **kwargs},
        description=f"Harmony integration on layer '{base_layer}'."
    )
    
    return container
