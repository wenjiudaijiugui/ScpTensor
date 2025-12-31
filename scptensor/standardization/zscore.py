# Specht, H., et al. (2021).
# Single-cell proteomic and transcriptomic analysis of macrophage heterogeneity using SCoPE2.
# Genome Biology.
# 在 SCoPE2 的计算管道(DART-ID / scp)中,处理顺序明确为：
# Filtering (过滤)
# Cell-specific Normalization (细胞间归一化,消除上样差异)
# Imputation (KNN 填补)
# Batch correction (批次校正)
# Scaling / PCA (包含 Z-score 操作)

# Lazar, C., et al. (2016).
# Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data: Comparative Assessment of Imputation Methods.
# Journal of Proteome Research.
# 这篇文章详细讨论了MNAR(非随机缺失)的处理。
# 文中强调填补必须基于原始强度的分布特性。
# 如果在填补前改变数据的分布形态(如 Z-score 强制正态化),会违背像 MinProb 或 Downshifted Normal 这类针对质谱数据设计的填补算法的假设。

# Vanderaa, C., & Gatto, L. (2023).
# Revisiting the analysis of single-cell proteomics data.
# Expert Review of Proteomics.
# Laurent Gatto 组(开发 scp R包的团队)明确指出,
# 标准化(Standardization / Z-score)旨在消除特征间的量级差异,以便进行比较。
# 这必须在完整的数据矩阵(Complete Matrix)上进行。
# 如果在稀疏矩阵上做 Z-score,不仅统计量不稳定,还会引入由缺失模式(Missingness Pattern)驱动的假相关性。

from typing import Optional
import numpy as np
from scptensor.core.structures import ScpContainer, ScpMatrix

def zscore_standardization(
    container: ScpContainer,
    assay_name: str = 'protein',
    base_layer_name: str = 'imputed',
    new_layer_name: Optional[str] = 'zscore',
    axis: int = 0,
    ddof: int = 1  # 增加自由度参数, 默认为1以符合样本标准差
) -> ScpContainer:
    """
    Z-score standardization.

    Mathematical Formulation:
        z = (x - mean) / std

    Args:
        axis: 0 to standardize features (columns), 1 to standardize samples (rows).
        ddof: Delta Degrees of Freedom. Set to 1 for unbiased estimator (sample std),
              0 for population std. R's scale() uses ddof=1.

    Important:
        Must be applied to complete matrix (no missing values) after imputation.
        Reference: Vanderaa, C. & Gatto, L. (2023). Expert Review of Proteomics.
    """
    if assay_name not in container.assays:
        raise ValueError(f"Assay '{assay_name}' not found.")

    assay = container.assays[assay_name]
    if base_layer_name not in assay.layers:
         raise ValueError(f"Layer '{base_layer_name}' not found in assay '{assay_name}'.")

    X = assay.layers[base_layer_name].X.copy()

    # 必须基于完整矩阵，符合 Lazar et al. (2016) 及 Vanderaa & Gatto (2023) 的理论
    if np.isnan(X).any():
        raise ValueError(
            f"Z-score standardization requires a complete matrix (no missing values). "
            f"Layer '{base_layer_name}' contains NaNs. "
            f"Reference: Vanderaa, C. & Gatto, L. (2023). Expert Review of Proteomics."
        )

    # 计算均值和标准差
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True, ddof=ddof) # [修正应用]

    # 处理标准差为0的情况（通常是该维度数值完全相同）
    # 避免 RuntimeWarning: invalid value encountered in true_divide
    std[std == 0] = 1.0

    X_z = (X - mean) / std

    new_matrix = ScpMatrix(X=X_z, M=assay.layers[base_layer_name].M.copy())

    target_layer_name = new_layer_name or 'zscore'

    container.assays[assay_name].add_layer(target_layer_name, new_matrix)

    container.log_operation(
        action="standardization_zscore",
        params={"assay": assay_name, "axis": axis, "ddof": ddof},
        description=f"Z-score standardization on '{base_layer_name}' -> '{target_layer_name}'."
    )

    return container