"""
Integration module for batch effect correction in single-cell proteomics data.

This module provides methods for removing batch effects and integrating data
from multiple experiments, runs, or platforms.

Available Methods:
- combat: ComBat batch correction (empirical Bayes) - BUILT-IN
- harmony: Harmony integration (iterative clustering) - REQUIRES harmonypy
- mnn_correct: Mutual Nearest Neighbors correction - BUILT-IN
- scanorama_integrate: Scanorama integration - REQUIRES scanorama

Note: Some methods require external packages (harmonypy, scanorama).
"""

from scptensor.integration.combat import combat
from scptensor.integration.nonlinear import harmony as harmony_nonlinear
from scptensor.integration.harmony import harmony
from scptensor.integration.mnn import mnn_correct
from scptensor.integration.scanorama import scanorama_integrate

__all__ = [
    "combat",
    "harmony",
    "mnn_correct",
    "scanorama_integrate",
]
