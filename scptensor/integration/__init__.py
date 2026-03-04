"""Integration module for batch effect correction in DIA-based single-cell proteomics data.

This module provides methods for removing batch effects and integrating data
from multiple experiments, runs, or platforms.

Available Methods
-----------------
- integrate: Unified interface for all integration methods
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
>>> from scptensor.integration import integrate, integrate_combat, integrate_harmony
>>>
>>> # Unified interface - easiest way to use any method
>>> container = integrate(container, method='combat', batch_key='batch')
>>> container = integrate(container, method='harmony', batch_key='batch', theta=2.0)
>>>
>>> # Direct function calls
>>> container = integrate_combat(container, batch_key='batch')
>>> container = integrate_harmony(container, batch_key='batch', base_layer='pca')
>>> container = integrate_mnn(container, batch_key='batch', k=20)
>>> container = integrate_scanorama(container, batch_key='batch', sigma=15.0)

References
----------
- ComBat: Johnson et al. Biostatistics (2007)
- Harmony: Korsunsky et al. Nature Methods (2019)
- MNN: Haghverdi et al. Nature Biotechnology (2018)
- Scanorama: Hie et al. Nature Biotechnology (2019)
"""

from scptensor.integration.base import (
    IntegrateMethod,
    get_integrate_method,
    integrate,
    list_integrate_methods,
    register_integrate_method,
)
from scptensor.integration.combat import integrate_combat
from scptensor.integration.diagnostics import (
    compute_batch_asw,
    compute_batch_mixing_metric,
    compute_lisi_approx,
    integration_quality_report,
)
from scptensor.integration.harmony import integrate_harmony
from scptensor.integration.mnn import integrate_mnn
from scptensor.integration.nonlinear import integrate_harmony as integrate_nonlinear
from scptensor.integration.scanorama import integrate_scanorama

__all__ = [
    # Unified interface
    "integrate",
    "list_integrate_methods",
    "get_integrate_method",
    "IntegrateMethod",
    "register_integrate_method",
    # Individual methods
    "integrate_combat",
    "integrate_harmony",
    "integrate_mnn",
    "integrate_scanorama",
    "integrate_nonlinear",
    # Diagnostics
    "compute_batch_mixing_metric",
    "compute_batch_asw",
    "compute_lisi_approx",
    "integration_quality_report",
]
