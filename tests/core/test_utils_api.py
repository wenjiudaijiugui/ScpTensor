"""Tests for stable utils namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.utils as stable_utils
from scptensor.utils import (
    BatchProcessor,
    ScpDataGenerator,
    apply_by_batch,
    batch_apply_along_axis,
    batch_iterator,
    correlation_matrix,
    cosine_similarity,
    partial_correlation,
    quantile_normalize,
    robust_scale,
    spearman_correlation,
)
from scptensor.utils.batch import (
    BatchProcessor as BatchProcessorCore,
)
from scptensor.utils.batch import (
    apply_by_batch as apply_by_batch_core,
)
from scptensor.utils.batch import (
    batch_apply_along_axis as batch_apply_along_axis_core,
)
from scptensor.utils.batch import batch_iterator as batch_iterator_core
from scptensor.utils.data_generator import ScpDataGenerator as ScpDataGeneratorCore
from scptensor.utils.stats import correlation_matrix as correlation_matrix_core
from scptensor.utils.stats import cosine_similarity as cosine_similarity_core
from scptensor.utils.stats import partial_correlation as partial_correlation_core
from scptensor.utils.stats import spearman_correlation as spearman_correlation_core
from scptensor.utils.transform import quantile_normalize as quantile_normalize_core
from scptensor.utils.transform import robust_scale as robust_scale_core


def test_stable_utils_namespace_all_freezes_package_surface() -> None:
    assert stable_utils.__all__ == [
        "ScpDataGenerator",
        "correlation_matrix",
        "partial_correlation",
        "spearman_correlation",
        "cosine_similarity",
        "quantile_normalize",
        "robust_scale",
        "batch_iterator",
        "apply_by_batch",
        "batch_apply_along_axis",
        "BatchProcessor",
    ]


def test_stable_utils_namespace_reexports_stable_implementations() -> None:
    assert ScpDataGenerator is ScpDataGeneratorCore
    assert correlation_matrix is correlation_matrix_core
    assert partial_correlation is partial_correlation_core
    assert spearman_correlation is spearman_correlation_core
    assert cosine_similarity is cosine_similarity_core
    assert quantile_normalize is quantile_normalize_core
    assert robust_scale is robust_scale_core
    assert batch_iterator is batch_iterator_core
    assert apply_by_batch is apply_by_batch_core
    assert batch_apply_along_axis is batch_apply_along_axis_core
    assert BatchProcessor is BatchProcessorCore


def test_only_scp_data_generator_is_reexported_from_top_level_package() -> None:
    assert scp.ScpDataGenerator is ScpDataGeneratorCore

    assert "ScpDataGenerator" in scp.__all__
    for name in (
        "correlation_matrix",
        "partial_correlation",
        "spearman_correlation",
        "cosine_similarity",
        "quantile_normalize",
        "robust_scale",
        "batch_iterator",
        "apply_by_batch",
        "batch_apply_along_axis",
        "BatchProcessor",
    ):
        assert name not in scp.__all__
