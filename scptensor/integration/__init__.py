"""Integration module for batch effect correction in single-cell proteomics data.

This module provides methods for removing batch effects and integrating data
from multiple experiments, runs, or platforms.

Available Methods
-----------------
- integrate_combat: ComBat batch correction (empirical Bayes) - built-in
- integrate_harmony: Harmony integration (iterative clustering) - requires harmonypy
- integrate_mnn: Mutual Nearest Neighbors correction - built-in
- integrate_scanorama: Scanorama integration - requires scanorama

Optional Dependencies
---------------------
Some methods require external packages:

- harmonypy: Install with ``pip install harmonypy``
- scanorama: Install with ``pip install scanorama``

Examples
--------
>>> from scptensor.integration import integrate_combat, integrate_harmony, integrate_mnn, integrate_scanorama
>>>
>>> # ComBat batch correction (built-in, no extra dependencies)
>>> container = integrate_combat(container, batch_key='batch')
>>>
>>> # Harmony integration (requires harmonypy)
>>> container = integrate_harmony(container, batch_key='batch', base_layer='pca')
>>>
>>> # MNN correction (built-in)
>>> container = integrate_mnn(container, batch_key='batch', k=20)
>>>
>>> # Scanorama integration (requires scanorama)
>>> container = integrate_scanorama(container, batch_key='batch', sigma=15.0)

References
----------
- ComBat: Johnson et al. Biostatistics (2007)
- Harmony: Korsunsky et al. Nature Methods (2019)
- MNN: Haghverdi et al. Nature Biotechnology (2018)
- Scanorama: Hie et al. Nature Biotechnology (2019)
"""

from scptensor.integration.combat import integrate_combat
from scptensor.integration.harmony import integrate_harmony
from scptensor.integration.mnn import integrate_mnn
from scptensor.integration.scanorama import integrate_scanorama

__all__ = [
    "integrate_combat",
    "integrate_harmony",
    "integrate_mnn",
    "integrate_scanorama",
]
