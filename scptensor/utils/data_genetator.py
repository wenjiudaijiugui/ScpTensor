# 1. 基于质谱DIA模式的单细胞蛋白组数据的缺失值机制是随机缺失(MNAR)+非随机缺失(MAR)
# Lazar et al. (2016) 指出，质谱的缺失是一个概率过程，低丰度离子的离子流强度越低，被检测到的概率越低，呈现 Sigmoid 曲线特征，而非垂直切断。
# Vanderaa & Gatto (2021) 在审查单细胞数据时指出，SCP数据不仅包含丰度导致的缺失（MNAR），也包含随机的离子采样失败（MCAR）。
# 代码使用了sigmoid函数模拟概率性丢失, 并混合了MNAR+MAR
# _____________________________________________________________________________
# 2. 蛋白质表达并不独立, 而是受到转录因子或复合物化学计量比的约束. Zappia et al., 2017 提出Splatter, 对sc-RNA进行模拟
# 代码引入了潜在因子模型构建协方差矩阵, 有效模拟了生物学变异与协方差
# _____________________________________________________________________________
# 3. 质谱数据普遍存在均值-方差依赖性, 即, 低丰度的信号信噪比低, 变异系数高
# 在物理上，质谱噪声通常被建模为“加性噪声+乘性噪声”的混合（Rocke & Lorenzato, 1995）
# 对数转化后, 这种关系体现为非线性的方差稳定趋势
# 代码采用了较为简单的噪声方差随信号强度递减模型(0.8-0.5*norm_intensity)
# _____________________________________________________________________________
# 4. 在本项目的第一版, 仅处理蛋白定量层级的矩阵, 暂时不考虑肽段层级的处理, 故直接生成了N_samples x M_features的蛋白质矩阵
# _____________________________________________________________________________
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any, Tuple
from scptensor.core.structures import ScpContainer, Assay, ScpMatrix
from scipy.special import expit
import scipy.sparse as sp

class ScpDataGenerator:
    """
    A generator for synthetic single-cell proteomics data in ScpContainer format.
    
    Attributes:
        n_samples (int): Number of samples (cells), range 10-1000.
        n_features (int): Number of features (proteins), range 500-10000.
        missing_rate (float): Total missing rate, range 0.0-0.7.
        lod_ratio (float): Proportion of missing values caused by Probabilistic Dropout (MNAR).
                           The rest will be Random missing (MCAR). 0.0-1.0.
                           Note: Replaces hard LOD cutoff with probabilistic dropout.
        n_batches (int): Number of batches to simulate batch effects.
        random_seed (int): Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 1000,
        missing_rate: float = 0.3,
        lod_ratio: float = 0.6,
        n_batches: int = 3,
        n_groups: int = 4,
        random_seed: int = 42,
        assay_name: str = "proteins",
        layer_name: str = "raw",
        feature_id_col: str = "protein_id",
        sample_id_col: str = "sample_id",
        mask_kind: str = "int8",
        use_sparse_X: bool = False,
        use_sparse_M: bool = False
    ) -> None:
        self.n_samples: int = n_samples
        self.n_features: int = n_features
        self.missing_rate: float = missing_rate
        self.lod_ratio: float = lod_ratio
        self.n_batches: int = n_batches
        self.n_groups: int = n_groups
        self.random_seed: int = random_seed
        self.assay_name: str = assay_name
        self.layer_name: str = layer_name
        self.feature_id_col: str = feature_id_col
        self.sample_id_col: str = sample_id_col
        self.mask_kind: str = mask_kind
        self.use_sparse_X: bool = use_sparse_X
        self.use_sparse_M: bool = use_sparse_M

        self._validate_params()
        self.rng: np.random.Generator = np.random.default_rng(seed=self.random_seed)

    def _validate_params(self) -> None:
        if not (10 <= self.n_samples <= 10000): 
            pass 
        if not (0.0 <= self.missing_rate <= 1.0):
             raise ValueError("missing_rate must be between 0.0 and 1.0")

    def generate(self) -> ScpContainer:
        """
        Generate the ScpContainer with synthetic data.
        
        Returns:
            ScpContainer: The generated data container.
        """
        # 1. Generate complete data matrix (Log-Normal distribution simulation)
        
        # 1.1 Biological Variation (Correlated Protein Expression)
        # Instead of independent proteins, we use a Latent Factor Model (Pathway-like)
        # to introduce biological correlations (Co-expression).
        n_pathways = max(10, self.n_features // 50)  # Heuristic number of latent factors
        
        # Pathway Activity: Each cell has different activation of pathways
        # Shape: (n_samples, n_pathways)
        # Introduce discrete biological groups (Cell Types)
        group_indices = np.array([i % self.n_groups for i in range(self.n_samples)])
        self.rng.shuffle(group_indices) # Randomly assign groups to avoid correlation with batch
        
        # Base pathway activity
        pathway_activity = self.rng.normal(loc=0.0, scale=1.0, size=(self.n_samples, n_pathways))
        
        # Add group-specific shifts to pathway activity
        # Each group activates different pathways
        group_pathway_shifts = self.rng.normal(loc=0.0, scale=2.0, size=(self.n_groups, n_pathways))
        # Make group shifts sparse to simulate distinct cell types
        mask_group_sparsity = self.rng.random(size=(self.n_groups, n_pathways)) < 0.5
        group_pathway_shifts[mask_group_sparsity] = 0.0
        
        # Apply shifts
        pathway_activity += group_pathway_shifts[group_indices]
        
        # Protein Loadings: Each protein belongs to/is affected by pathways
        # Shape: (n_pathways, n_features)
        # We make it somewhat sparse to represent specific pathways
        protein_loadings = self.rng.normal(loc=0.0, scale=1.0, size=(n_pathways, self.n_features))
        mask_sparsity = self.rng.random(size=(n_pathways, self.n_features)) < 0.7 # 70% sparse
        protein_loadings[mask_sparsity] = 0.0
        
        # Correlated Signal
        # Shape: (n_samples, n_features)
        biological_variation = pathway_activity @ protein_loadings
        
        # Normalize biological variation to have a reasonable scale (e.g., std dev approx 2.0)
        bio_std = np.std(biological_variation)
        if bio_std > 0:
            biological_variation = biological_variation / bio_std * 2.0
            
        # Baseline Protein Abundance (Mean)
        # Shape: (1, n_features)
        protein_means: np.ndarray = self.rng.normal(loc=15.0, scale=2.0, size=(1, self.n_features))
        
        # 1.2 Technical Variation - Sample Efficiency (Library Size)
        # Shape: (n_samples, 1)
        sample_efficiencies: np.ndarray = self.rng.normal(loc=0.0, scale=0.5, size=(self.n_samples, 1))
        
        # 1.3 Technical Variation - Batch Effects
        # Assign samples to batches
        batch_indices = np.array([i % self.n_batches for i in range(self.n_samples)])
        
        # Generate random batch shifts (Mean Shift)
        # Adjusted scale from 1.0 to 0.4 to be more realistic (Log2 scale)
        batch_effects_matrix = self.rng.normal(loc=0.0, scale=0.4, size=(self.n_batches, self.n_features))
        
        # Expand batch effects to sample level
        sample_batch_effects = batch_effects_matrix[batch_indices]
        
        # Calculate "Clean" Signal (Expected Value without random noise)
        X_clean = protein_means + biological_variation + sample_efficiencies + sample_batch_effects
        
        # 1.4 Heteroscedastic Noise (Intensity-dependent)
        # Noise is higher for lower intensity signals.
        # Model: sigma = base_sigma + decay * (max_int - current_int)
        # Or simply scaling sigma inversely with intensity.
        
        # We calculate a noise scale for each data point based on its clean intensity
        # Normalize intensity to 0-1 range for scaling calculation
        min_val = np.min(X_clean)
        max_val = np.max(X_clean)
        if max_val > min_val:
            norm_intensity = (X_clean - min_val) / (max_val - min_val)
        else:
            norm_intensity = np.zeros_like(X_clean)
            
        # High intensity (1.0) -> Low Noise (0.3)
        # Low intensity (0.0) -> High Noise (0.8)
        noise_scale_matrix = 0.8 - 0.5 * norm_intensity
        
        # Generate noise
        noise = self.rng.normal(loc=0.0, scale=1.0, size=(self.n_samples, self.n_features)) * noise_scale_matrix
        
        # 1.5 Combine All Components
        X_complete: np.ndarray = X_clean + noise
        
        # 2. Generate Missing Mask
        # Initialize Mask: 0 = Valid
        M: np.ndarray = np.zeros((self.n_samples, self.n_features), dtype=np.int8)
        
        if self.missing_rate > 0:
            # Target counts
            total_elements = self.n_samples * self.n_features
            target_missing = int(total_elements * self.missing_rate)
            target_mnar = int(target_missing * self.lod_ratio) # MNAR (Probabilistic Dropout)
            target_mcar = target_missing - target_mnar        # MCAR (Random)
            
            # 2.1 Probabilistic Dropout (MNAR)
            if target_mnar > 0:
                dropout_slope = 1.0 
                
                target_mnar_rate = target_mnar / total_elements
                # Heuristic calibration for bias (cutoff)
                bias_guess = np.percentile(X_complete, target_mnar_rate * 100)
                
                # P_missing = sigmoid( -slope * (X - bias) )
                # Lower X -> Higher P_missing
                p_missing_mnar = expit(-dropout_slope * (X_complete - bias_guess))
                
                random_matrix = self.rng.random(size=X_complete.shape)
                mnar_mask = random_matrix < p_missing_mnar
                
                M[mnar_mask] = 2
            
            # 2.2 Random Missing (MCAR)
            if target_mcar > 0:
                valid_mask = (M == 0)
                valid_indices = np.where(valid_mask.ravel())[0]
                n_valid = len(valid_indices)
                if n_valid > 0:
                    n_to_mask = min(target_mcar, n_valid)
                    random_choice = self.rng.choice(valid_indices, size=n_to_mask, replace=False)
                    row_indices, col_indices = np.unravel_index(random_choice, (self.n_samples, self.n_features))
                    M[row_indices, col_indices] = 1

        X_out = X_complete.astype(np.float64)
        if self.use_sparse_X:
            X_out = sp.csr_matrix(X_out)

        if self.mask_kind == "none":
            M_out: Optional[Any] = None
        elif self.mask_kind == "bool":
            M_out = (M != 0)
            if self.use_sparse_M:
                M_out = sp.csr_matrix(M_out.astype(np.int8))
        elif self.mask_kind == "int8":
            M_out = M.astype(np.int8)
            if self.use_sparse_M:
                M_out = sp.csr_matrix(M_out)
        else:
            M_out = M.astype(np.int8)

        matrix: ScpMatrix = ScpMatrix(X=X_out, M=M_out)
        
        # 4. Create Metadata
        # 4.1 Feature Metadata (var)
        base_feature_ids: List[str] = [f"Prot_{i:05d}" for i in range(self.n_features)]
        var_cols: Dict[str, List[Any]] = {
            "protein_id": base_feature_ids,
            "gene_name": [f"Gene_{i:05d}" for i in range(self.n_features)],
            "mean_abundance": protein_means.flatten().tolist()
        }
        if self.feature_id_col not in var_cols:
            var_cols[self.feature_id_col] = base_feature_ids
        var: pl.DataFrame = pl.DataFrame(var_cols)
        
        # 4.2 Sample Metadata (obs)
        batch_names = [f"Batch_{i}" for i in batch_indices]
        group_names = [f"Group_{i}" for i in group_indices]
        base_sample_ids: List[str] = [f"Cell_{i:05d}" for i in range(self.n_samples)]
        obs_cols: Dict[str, List[Any]] = {
            "sample_id": base_sample_ids,
            "batch": batch_names,
            "group": group_names,
            "efficiency": sample_efficiencies.flatten().tolist()
        }
        if self.sample_id_col not in obs_cols:
            obs_cols[self.sample_id_col] = base_sample_ids
        obs: pl.DataFrame = pl.DataFrame(obs_cols)
        
        # 5. Create Assay
        assay: Assay = Assay(var=var, feature_id_col=self.feature_id_col)
        assay.add_layer(self.layer_name, matrix)
        
        # 6. Create Container
        container: ScpContainer = ScpContainer(obs=obs, sample_id_col=self.sample_id_col)
        container.add_assay(self.assay_name, assay)
        
        container.log_operation(
            action="generate_synthetic_data",
            params={
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "missing_rate": self.missing_rate,
                "lod_ratio": self.lod_ratio,
                "n_batches": self.n_batches,
                "seed": self.random_seed,
                "missing_mechanism": "Probabilistic Dropout (Sigmoid) + MCAR",
                "noise_model": "Heteroscedastic (Intensity-dependent)",
                "biological_model": "Latent Factor (Pathway-based Co-expression)",
                "n_groups": self.n_groups,
                "assay_name": self.assay_name,
                "layer_name": self.layer_name,
                "feature_id_col": self.feature_id_col,
                "sample_id_col": self.sample_id_col,
                "mask_kind": self.mask_kind,
                "use_sparse_X": self.use_sparse_X,
                "use_sparse_M": self.use_sparse_M
            },
            description="Generated synthetic single-cell proteomics data with biological correlations, heteroscedastic noise, groups, and batch effects."
        )
        
        return container

if __name__ == "__main__":
    # Quick test
    import time
    
    print("Initializing Generator...")
    gen = ScpDataGenerator(
        n_samples=500, 
        n_features=2000, 
        missing_rate=0.4, 
        lod_ratio=0.6,
        n_batches=3
    )
    
    print("Generating Data...")
    start_time = time.time()
    container = gen.generate()
    end_time = time.time()
    
    print(f"Generation took {end_time - start_time:.4f} seconds.")
    print(container)
    
    assay = container.assays["proteins"]
    matrix = assay.layers["raw"]
    
    n_total = matrix.X.size
    n_valid = np.sum(matrix.M == 0)
    n_random = np.sum(matrix.M == 1)
    n_lod = np.sum(matrix.M == 2) # Now MNAR
    
    print(f"Total Elements: {n_total}")
    print(f"Valid: {n_valid} ({n_valid/n_total:.2%})")
    print(f"MCAR (M=1): {n_random} ({n_random/n_total:.2%})")
    print(f"MNAR (Probabilistic) (M=2): {n_lod} ({n_lod/n_total:.2%})")
    print(f"Total Missing: {n_random + n_lod} ({(n_random + n_lod)/n_total:.2%})")
