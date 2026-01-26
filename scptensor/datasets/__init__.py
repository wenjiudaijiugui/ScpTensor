"""Example datasets for ScpTensor tutorials and testing.

This module provides ready-to-use example datasets that users can load
for learning and testing the ScpTensor framework.

Available Datasets:
-------------------

1. **load_toy_example()** - Small synthetic dataset (~100 samples, ~50 features)
   - Multiple batches (3)
   - 20% missing values (mix of MNAR and MCAR)
   - Perfect for quick testing and documentation examples

2. **load_simulated_scrnaseq_like()** - Larger simulated dataset (~500 samples, ~200 features)
   - Multiple cell types (4 groups)
   - Batch effects
   - Realistic missing patterns
   - For comprehensive pipeline testing

3. **load_example_with_clusters()** - Data with known cluster labels
   - Pre-computed cluster labels
   - Distinct cell types for clustering tutorials
   - Includes quality control metrics

Example Usage:
--------------
>>> from scptensor.datasets import load_toy_example
>>> container = load_toy_example()
>>> print(container)
"""

from scptensor.datasets._example import (
    REPRODUCIBILITY_NOTE,
    DatasetSize,
    DatasetType,
    load_example_with_clusters,
    load_simulated_scrnaseq_like,
    load_toy_example,
)

__all__ = [
    "load_toy_example",
    "load_simulated_scrnaseq_like",
    "load_example_with_clusters",
    "DatasetType",
    "DatasetSize",
    "REPRODUCIBILITY_NOTE",
]

__version__ = "0.1.0"
