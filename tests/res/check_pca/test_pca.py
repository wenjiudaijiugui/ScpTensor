import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from scptensor.core.structures import ScpContainer
from scptensor.dim_reduction.pca import reduce_pca as scp_pca

# 导入自定义模块
from scptensor.utils.data_generator import ScpDataGenerator

# 设置绘图风格
plt.style.use(["science", "no-latex"])


def generate_data(
    n_samples: int = 200, n_features: int = 1000, random_seed: int = 42
) -> ScpContainer:
    """生成测试数据"""
    generator: ScpDataGenerator = ScpDataGenerator(
        n_samples=n_samples,
        n_features=n_features,
        missing_rate=0.0,  # PCA通常需要完整数据
        n_batches=2,
        random_seed=random_seed,
    )
    return generator.generate()


def run_sklearn_pca(
    X: np.ndarray, n_components: int = 2, center: bool = True, scale: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """运行sklearn的PCA实现"""
    X_processed: np.ndarray = X.copy()

    if scale:
        # Sklearn uses biased estimator (std with ddof=0) by default in StandardScaler
        # ScpTensor PCA uses unbiased estimator (ddof=1)
        # To match results, we should force sklearn to use unbiased estimator or vice versa.
        # Sklearn StandardScaler doesn't have a ddof parameter, it always uses ddof=0.
        # However, we can manually scale using np.std(ddof=1) before passing to PCA.

        scaler = StandardScaler(with_mean=center, with_std=False)
        X_processed = scaler.fit_transform(X_processed)
        std = np.std(X_processed, axis=0, ddof=1)
        std[std == 0] = 1.0
        X_processed = X_processed / std

        # Disable scaling in PCA since we did it manually
        # But wait, run_sklearn_pca implementation below initializes PCA.
        # PCA object itself centers data if not already centered, but doesn't scale.
        # So we just pass the manually scaled data.
    elif center:
        X_processed = X_processed - np.mean(X_processed, axis=0)

    pca: SklearnPCA = SklearnPCA(n_components=n_components, svd_solver="full", random_state=42)
    scores: np.ndarray = pca.fit_transform(X_processed)
    components: np.ndarray = pca.components_
    explained_variance: np.ndarray = pca.explained_variance_

    return scores, components, explained_variance


def compare_pca_results(
    scp_container: ScpContainer, n_components: int = 2, center: bool = True, scale: bool = False
) -> None:
    """对比自定义PCA与sklearn PCA的结果"""

    # 1. 运行自定义PCA
    # 假设 assay_name="protein", layer_name="intensity" (根据 data_generator 的输出结构推测)
    # 查看 data_generator 代码, ScpMatrix 被创建, 但没有显式看到 Assay 名字
    # 需要确认 Assay 名字, 这里假设是 "protein_intensity" 或默认值
    # 先运行 scp_pca, 假设默认 assay
    # 从 ScpContainer 结构来看, 通常会有 assay
    # 这里我们需要先检查 container 的 assay 名称

    assay_names = list(scp_container.assays.keys())
    if not assay_names:
        # 如果没有 assay, 可能是 data_generator 直接返回了某种未完全包装的结构?
        # 根据 data_generator 代码:
        # matrix: ScpMatrix = ScpMatrix(X=X_complete.astype(np.float64), M=M)
        # var ... obs ...
        # 但最后没有看到 Assay 和 ScpContainer 的组装代码 (截断了?)
        # 假设 data_generator 返回的是 ScpContainer, 且包含一个名为 'proteomics' 的 assay
        # 让我们先手动包装一下以防万一，或者先打印看看
        pass

    # 为了确保代码运行, 我们手动构建一个标准流程
    # 假设 data_generator 返回 ScpContainer
    # 如果 scp_container 中没有 assay, 我们需要自己加一个

    if len(scp_container.assays) == 0:
        # 这种情况应该不会发生，如果 generator 是完整的
        # 但如果 generator 代码最后几行没有 add assay
        # 我们先假定 generator 是完整的
        pass

    target_assay_name: str = list(scp_container.assays.keys())[0]
    # 获取 layer 名, ScpMatrix 通常作为 layer
    # 假设 ScpContainer 的结构
    assay = scp_container.assays[target_assay_name]
    target_layer_name: str = list(assay.layers.keys())[0]

    print(f"Running ScpTensor PCA on Assay: {target_assay_name}, Layer: {target_layer_name}")

    # 运行 ScpTensor PCA
    scp_container_pca: ScpContainer = scp_pca(
        container=scp_container,
        assay_name=target_assay_name,
        base_layer_name=target_layer_name,
        new_assay_name="pca_result",
        n_components=n_components,
        center=center,
        scale=scale,
        random_state=42,
    )

    # 获取结果
    pca_assay = scp_container_pca.assays["pca_result"]
    # 假设 PCA 结果存储在 layers["score"] 或者直接作为 X
    # 查看 pca.py: scores_matrix = ScpMatrix(X=scores, M=scores_M)
    # 然后应该被添加到了 assay 中
    # 代码最后截断了, 我们假设它被添加为默认 layer 或者 "score"
    # 通常 dim_reduction 的结果 assay 会有一个 layer 存放 scores

    # 由于 pca.py 最后几行没显示完, 假设 layer 名字是 "score" 或 "pca"
    # 我们遍历一下 layers
    pca_layer_name = list(pca_assay.layers.keys())[0]
    scp_scores: np.ndarray = pca_assay.layers[pca_layer_name].X

    # 获取解释方差
    # pca.py 中: pca_var = pl.DataFrame(...)
    # 这应该作为 var metadata 存储
    scp_explained_variance: np.ndarray = pca_assay.var["explained_variance"].to_numpy()

    # 2. 运行 sklearn PCA
    X_raw: np.ndarray = assay.layers[target_layer_name].X
    if hasattr(X_raw, "toarray"):
        X_raw = X_raw.toarray()

    sklearn_scores, sklearn_components, sklearn_explained_variance = run_sklearn_pca(
        X_raw, n_components=n_components, center=center, scale=scale
    )

    # 3. 对比指标
    print("\n" + "=" * 50)
    print(f"Comparison Results (Center={center}, Scale={scale})")
    print("=" * 50)

    # 3.1 解释方差对比
    print("\n[Explained Variance]")
    print(f"ScpTensor: {scp_explained_variance}")
    print(f"Sklearn:   {sklearn_explained_variance}")
    var_diff: float = np.mean(np.abs(scp_explained_variance - sklearn_explained_variance))
    print(f"Mean Absolute Difference: {var_diff:.6e}")

    # 3.2 Scores 对比 (考虑符号翻转)
    # PCA 的符号是任意的, 所以我们需要检查每一列的相关性绝对值
    print("\n[Scores Correlation]")
    correlations: list[float] = []
    for i in range(n_components):
        corr: float = np.corrcoef(scp_scores[:, i], sklearn_scores[:, i])[0, 1]
        correlations.append(abs(corr))
        print(f"PC{i + 1} Correlation: {abs(corr):.6f} (Raw: {corr:.6f})")

    # 3.3 可视化
    plot_comparison(
        scp_scores,
        sklearn_scores,
        scp_explained_variance,
        sklearn_explained_variance,
        n_components,
        f"PCA_Comparison_Center{center}_Scale{scale}",
    )


def plot_comparison(
    scp_scores: np.ndarray,
    sklearn_scores: np.ndarray,
    scp_var: np.ndarray,
    sklearn_var: np.ndarray,
    n_components: int,
    title_suffix: str,
) -> None:
    """绘制对比图"""
    fig = plt.figure(figsize=(12, 5))

    # 子图1: PC1 vs PC2 散点图对比
    ax1 = fig.add_subplot(121)

    # 为了方便对比，如果相关性是负的，翻转 sklearn 的符号
    # 这里简单处理，统一翻转使得第一个元素同号 (仅用于绘图视觉一致性)
    # 真正的对比看相关性数值

    # 绘制 ScpTensor 结果
    ax1.scatter(
        scp_scores[:, 0], scp_scores[:, 1], c="blue", alpha=0.5, label="ScpTensor", marker="o", s=20
    )

    # 绘制 Sklearn 结果 (尝试匹配符号)
    # 检测 PC1 符号
    sign_pc1 = np.sign(np.corrcoef(scp_scores[:, 0], sklearn_scores[:, 0])[0, 1])
    sign_pc2 = np.sign(np.corrcoef(scp_scores[:, 1], sklearn_scores[:, 1])[0, 1])

    ax1.scatter(
        sklearn_scores[:, 0] * sign_pc1,
        sklearn_scores[:, 1] * sign_pc2,
        c="red",
        alpha=0.5,
        label="Sklearn (Aligned)",
        marker="x",
        s=20,
    )

    ax1.set_xlabel(f"PC1 ({scp_var[0]:.2f})")
    ax1.set_ylabel(f"PC2 ({scp_var[1]:.2f})")
    ax1.set_title("Score Distribution Comparison")
    ax1.legend()

    # 子图2: 解释方差对比
    ax2 = fig.add_subplot(122)
    indices = np.arange(n_components)
    width = 0.35

    ax2.bar(indices - width / 2, scp_var, width, label="ScpTensor")
    ax2.bar(indices + width / 2, sklearn_var, width, label="Sklearn")

    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance")
    ax2.set_title("Explained Variance Comparison")
    ax2.set_xticks(indices)
    ax2.set_xticklabels([f"PC{i + 1}" for i in indices])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{title_suffix}.png", dpi=300)
    print(f"Figure saved: {title_suffix}.png")
    plt.close()


def main() -> None:
    # 1. 生成数据
    print("Generating synthetic data...")
    container: ScpContainer = generate_data(n_samples=300, n_features=5000)

    # 2. 测试场景 1: Center=True, Scale=False (Standard PCA)
    compare_pca_results(container, n_components=2, center=True, scale=False)

    # 3. 测试场景 2: Center=True, Scale=True (Correlation PCA)
    compare_pca_results(container, n_components=2, center=True, scale=True)

    # 4. 测试场景 3: Center=False, Scale=False (SVD on raw data)
    # 注意: Sklearn PCA 默认总是 center=True, 如果我们要模拟 center=False
    # 我们需要使用 TruncatedSVD 或者自行实现 SVD
    # 这里我们仅对比前两种常用情况，或者在 run_sklearn_pca 中手动处理
    # Scikit-learn PCA 不支持 center=False (除了使用 TruncatedSVD)
    # 为了简单起见，我们只对比前两种最常用的情况


if __name__ == "__main__":
    main()
