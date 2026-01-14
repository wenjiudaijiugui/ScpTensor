"""Standardization module for single-cell proteomics data.

.. deprecated:: 0.1.0
    The standardization module has been consolidated into the normalization module.
    Please use ``scptensor.normalization.zscore`` instead.

    This module is kept for backward compatibility and will be removed in v0.2.0.
"""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer

# Re-export from normalization for backward compatibility
from scptensor.normalization.zscore import zscore

__all__ = ["zscore"]


# Emit deprecation warning on import
warnings.warn(
    "The 'scptensor.standardization' module is deprecated. "
    "Use 'scptensor.normalization.zscore' instead. "
    "This module will be removed in v0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)
