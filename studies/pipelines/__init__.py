"""Pipeline implementations for single-cell proteomics analysis comparison.

This package provides 5 different analysis pipelines for comparing technical
performance in single-cell proteomics data analysis.

Available Pipelines:
    - PipelineA: Classic Pipeline (most common in literature)
    - PipelineB: Batch Correction Pipeline (for multi-batch data)
    - PipelineC: Advanced Pipeline (with latest methods)
    - PipelineD: Performance-Optimized Pipeline (for large-scale data)
    - PipelineE: Conservative Pipeline (minimal assumptions)

Examples
--------
>>> from scptensor import create_test_container
>>> from studies.pipelines import PipelineA
>>> container = create_test_container()
>>> pipeline = PipelineA()
>>> result = pipeline.run(container)
>>> print(pipeline.get_execution_log())
"""

from .base import BasePipeline, load_pipeline_config
from .pipeline_a import PipelineA
from .pipeline_b import PipelineB
from .pipeline_c import PipelineC
from .pipeline_d import PipelineD
from .pipeline_e import PipelineE

__all__ = [
    "BasePipeline",
    "load_pipeline_config",
    "PipelineA",
    "PipelineB",
    "PipelineC",
    "PipelineD",
    "PipelineE",
]
