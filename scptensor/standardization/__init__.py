"""Standardization module for single-cell proteomics data.

This module provides feature scaling methods that transform data to have
specific statistical properties (e.g., mean=0, std=1 for z-score).

**Standardization vs Normalization:**

- **Standardization** (this module): Feature scaling methods that transform
  features to have zero mean and unit variance. Makes different features
  comparable by removing scale differences. Example: z-score

- **Normalization**: Sample-wise scaling methods that adjust samples to a
  common scale or reference point. Example: median centering, log transform

**When to use:**

- Use z-score standardization when you need to make features comparable
  for machine learning algorithms that assume standardized features
  (e.g., PCA, clustering, SVM).

- Use normalization when you need to remove sample-specific technical
  bias or align samples to a common scale.

**Key Methods:**

- ``zscore``: Z-score standardization (mean=0, std=1) for features

**References:**

    Vanderaa, C., & Gatto, L. (2023). Revisiting the analysis of single-cell
    proteomics data. Expert Review of Proteomics.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scptensor.core.structures import ScpContainer

from .zscore import zscore

__all__ = ["zscore"]
