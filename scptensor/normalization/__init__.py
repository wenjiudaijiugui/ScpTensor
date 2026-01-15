# New norm_* prefixed functions (recommended)
from .global_median import norm_global_median
from .log import norm_log
from .median_centering import norm_median_center
from .median_scaling import norm_median_scale
from .sample_mean import norm_sample_mean
from .sample_median import norm_sample_median
from .tmm import norm_tmm
from .upper_quartile import norm_quartile
from .zscore import norm_zscore

# Backward compatibility aliases (deprecated, will be removed in v1.0.0)
from .global_median import global_median_normalization
from .log import log_normalize
from .median_centering import median_centering
from .median_scaling import median_scaling
from .sample_mean import sample_mean_normalization
from .sample_median import sample_median_normalization
from .tmm import tmm_normalization
from .upper_quartile import upper_quartile_normalization
from .zscore import zscore

__all__ = [
    # New norm_* prefixed functions (recommended)
    "norm_log",
    "norm_zscore",
    "norm_median_center",
    "norm_median_scale",
    "norm_sample_median",
    "norm_sample_mean",
    "norm_global_median",
    "norm_tmm",
    "norm_quartile",
    # Backward compatibility aliases (deprecated)
    "log_normalize",
    "zscore",
    "median_centering",
    "median_scaling",
    "sample_median_normalization",
    "sample_mean_normalization",
    "global_median_normalization",
    "tmm_normalization",
    "upper_quartile_normalization",
]
