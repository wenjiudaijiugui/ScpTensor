"""
Integration module for batch effect correction in single-cell proteomics data.

This module provides methods for removing batch effects and integrating data
from multiple experiments, runs, or platforms.

Available Methods
-----------------
- combat: ComBat batch correction (empirical Bayes) - built-in
- harmony: Harmony integration (iterative clustering) - requires harmonypy
- mnn_correct: Mutual Nearest Neighbors correction - built-in
- scanorama_integrate: Scanorama integration - requires scanorama

Optional Dependencies
---------------------
Some methods require external packages:

- harmonypy: Install with ``pip install harmonypy``
- scanorama: Install with ``pip install scanorama``

Examples
--------
>>> from scptensor.integration import combat, harmony, mnn_correct, scanorama_integrate
>>>
>>> # ComBat batch correction (built-in, no extra dependencies)
>>> container = combat(container, batch_key='batch')
>>>
>>> # Harmony integration (requires harmonypy)
>>> container = harmony(container, batch_key='batch', base_layer='pca')
>>>
>>> # MNN correction (built-in)
>>> container = mnn_correct(container, batch_key='batch', k=20)
>>>
>>> # Scanorama integration (requires scanorama)
>>> container = scanorama_integrate(container, batch_key='batch', sigma=15.0)

References
----------
- ComBat: Johnson et al. Biostatistics (2007)
- Harmony: Korsunsky et al. Nature Methods (2019)
- MNN: Haghverdi et al. Nature Biotechnology (2018)
- Scanorama: Hie et al. Nature Biotechnology (2019)
"""

from scptensor.integration.combat import combat
from scptensor.integration.harmony import harmony
from scptensor.integration.mnn import integrate_mnn, mnn_correct
from scptensor.integration.scanorama import integrate_scanorama as scanorama_integrate

__all__ = [
    "combat",
    "harmony",
    "integrate_mnn",
    "mnn_correct",  # Deprecated: use integrate_mnn
    "scanorama_integrate",
]
