from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import IntEnum
import copy
import numpy as np
import polars as pl
import scipy.sparse as sp
from datetime import datetime

class MaskCode(IntEnum):
    """
    Enhanced enumeration for data status codes in ScpMatrix.M.
    """
    VALID = 0         # Valid, detected values
    MBR = 1          # Match Between Runs missing
    LOD = 2          # Below Limit of Detection
    FILTERED = 3     # Filtered out (quality control)
    OUTLIER = 4      # Statistical outlier
    IMPUTED = 5      # Imputed/filled value
    UNCERTAIN = 6    # Uncertain data quality

@dataclass
class ProvenanceLog:
    """
    记录对容器执行的操作历史。
    Record of operations performed on the container.
    """
    timestamp: str
    action: str
    params: Dict[str, Any]
    software_version: Optional[str] = None
    description: Optional[str] = None

@dataclass
class MatrixMetadata:
    """
    Enhanced metadata for ScpMatrix containing additional quality information.
    """
    confidence_scores: Optional[Union[np.ndarray, sp.spmatrix]] = None     # [0,1] Data confidence scores
    detection_limits: Optional[Union[np.ndarray, sp.spmatrix]] = None      # Detection limit values
    imputation_quality: Optional[Union[np.ndarray, sp.spmatrix]] = None   # [0,1] Imputation quality scores
    outlier_scores: Optional[Union[np.ndarray, sp.spmatrix]] = None       # Outlier detection scores
    creation_info: Optional[Dict[str, Any]] = None                        # Creation tracking info

@dataclass
class ScpMatrix:
    """
    最小数据单元: 物理存储层，负责数值和状态。
    Minimal data unit: Physical storage layer responsible for values and status.
    
    Attributes:
        X (Union[np.ndarray, sp.spmatrix]): 定量数值矩阵 (f64/f32)。支持稀疏矩阵 (CSR/CSC)。
                                            Quantitative value matrix. Supports sparse matrix.
                                            Shape: (N_samples, M_features_local)
        M (Union[np.ndarray, sp.spmatrix, None]): 状态掩码矩阵 (int8 / bool / Sparse)。
                                            Status mask matrix.
                                            0=Valid, 1=MBR, 2=LOD, 3=Filtered
                                            Shape: (N_samples, M_features_local)
                                            If None, assumes all data is Valid (0) or handled implicitly (Lazy Mask).
    """
    X: Union[np.ndarray, sp.spmatrix]
    M: Union[np.ndarray, sp.spmatrix, None] = None
    metadata: Optional[MatrixMetadata] = None

    def __post_init__(self):
        # Ensure X is float
        if not np.issubdtype(self.X.dtype, np.floating):
             # You might want to cast here or raise a warning. 
             # For strictness, let's assume it should be provided correctly or we cast it.
             self.X = self.X.astype(np.float64)

        if self.M is not None:
            if self.X.shape != self.M.shape:
                raise ValueError(f"Shape mismatch: X {self.X.shape} != M {self.M.shape}")
            
            # Validate M values if it's dense
            if isinstance(self.M, np.ndarray):
                valid_codes = [code.value for code in MaskCode]
                if not np.all(np.isin(self.M, valid_codes)):
                    invalid_values = np.setdiff1d(np.unique(self.M), valid_codes)
                    raise ValueError(f"Invalid mask codes found: {invalid_values}. "
                                   f"Valid codes are: {valid_codes}")
            
            # Ensure M is integer type
            if not np.issubdtype(self.M.dtype, np.integer):
                 self.M = self.M.astype(np.int8)
            elif self.M.dtype != np.int8:
                 self.M = self.M.astype(np.int8)

    def get_m(self) -> Union[np.ndarray, sp.spmatrix]:
        """
        Return the mask matrix M. If M is None, returns a zero matrix of the same shape as X.
        """
        if self.M is not None:
            return self.M
        
        if sp.issparse(self.X):
            return sp.csr_matrix(self.X.shape, dtype=np.int8)
        else:
            return np.zeros(self.X.shape, dtype=np.int8)

    def copy(self) -> ScpMatrix:
        """
        Deep copy of the matrix.
        """
        new_X = self.X.copy()
        new_M = self.M.copy() if self.M is not None else None
        return ScpMatrix(X=new_X, M=new_M)


@dataclass
class AggregationLink:
    """
    描述从源 Assay 到目标 Assay 的特征聚合关系 (例如 Peptide -> Protein)。
    Describes the feature aggregation relationship from source Assay to target Assay.
    """
    source_assay: str
    target_assay: str
    # Linkage table: must contain 'source_id' and 'target_id' columns mapping feature IDs.
    linkage: pl.DataFrame
    
    def __post_init__(self):
        required_cols = {"source_id", "target_id"}
        if not required_cols.issubset(set(self.linkage.columns)):
             raise ValueError(f"Linkage DataFrame must contain columns: {required_cols}")

class Assay:
    """
    特征子对象: 负责管理特定特征空间下的数据。
    Feature SubObject: Manages data under a specific feature space.
    """
    def __init__(
        self, 
        var: pl.DataFrame, 
        layers: Optional[Dict[str, ScpMatrix]] = None,
        feature_id_col: str = "_index"
    ):
        """
        Args:
            var (pl.DataFrame): 局部特征元数据 (Local Feature Meta)。
                                MUST contain a unique ID column specified by feature_id_col.
            layers (Dict[str, ScpMatrix], optional): 数据层字典。 Defaults to None.
            feature_id_col (str): Column name in 'var' that serves as the unique feature identifier.
                                  Defaults to "_index".
        """
        self.feature_id_col = feature_id_col
        
        # Validate ID column existence
        if feature_id_col not in var.columns:
             # If default _index is missing, try to create it or raise error? 
             # Strict mode: raise error.
             raise ValueError(f"Feature ID column '{feature_id_col}' not found in var.")
        
        # Validate ID uniqueness
        if var[feature_id_col].n_unique() != var.height:
            raise ValueError(f"Feature ID column '{feature_id_col}' is not unique.")

        self.var: pl.DataFrame = var
        self.layers: Dict[str, ScpMatrix] = layers if layers is not None else {}
        
        self._validate()

    def _validate(self):
        """
        验证所有 Layer 的特征维度是否与 var 对齐。
        Validate that feature dimensions of all Layers align with var.
        """
        for name, matrix in self.layers.items():
            if matrix.X.shape[1] != self.n_features:
                raise ValueError(
                    f"Feature dimension mismatch in Layer '{name}': "
                    f"Matrix has {matrix.X.shape[1]}, Assay var has {self.n_features}"
                )
        
    @property
    def n_features(self) -> int:
        return self.var.height

    @property
    def feature_ids(self) -> pl.Series:
        return self.var[self.feature_id_col]

    @property
    def X(self) -> Optional[Union[np.ndarray, sp.spmatrix]]:
        """
        Shortcut to access the 'X' layer matrix if it exists.
        """
        if "X" in self.layers:
            return self.layers["X"].X
        return None

    def add_layer(self, name: str, matrix: ScpMatrix) -> None:
        """
        添加新的数据层。
        
        Args:
            name (str): 层名称 (e.g., 'raw', 'log', 'imputed').
            matrix (ScpMatrix): 矩阵对象.
        """
        # Validate feature dimension
        if matrix.X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: Layer has {matrix.X.shape[1]}, "
                f"Assay var has {self.n_features}"
            )
        if name in self.layers:
            # Warn about overwriting?
            pass
        self.layers[name] = matrix

    def __repr__(self) -> str:
        return f"<Assay n_features={self.n_features}, layers={list(self.layers.keys())}>"

    def subset(self, feature_indices: Union[List[int], np.ndarray], copy_data: bool = True) -> Assay:
        """
        Return a new Assay with a subset of features.
        
        Args:
            feature_indices: Indices of features to keep.
            copy_data: Whether to copy the underlying data. Defaults to True.
        """
        # Fix Polars row indexing: df[indices, :]
        new_var = self.var[feature_indices, :]
        new_layers = {}
        for name, matrix in self.layers.items():
            # Handle Matrix slicing
            new_X = matrix.X[:, feature_indices]
            
            # Handle optional/sparse M
            new_M = None
            if matrix.M is not None:
                new_M = matrix.M[:, feature_indices]

            if copy_data:
                # For numpy, slicing creates a view, so we need explicit copy
                if isinstance(new_X, np.ndarray):
                    new_X = new_X.copy()
                # For sparse matrices, slicing often returns a copy, but let's be safe if implementation changes
                # sp.csr_matrix slicing usually returns a copy, but verifying documentation is good.
                # Actually sparse matrix slicing returns a copy.
                elif sp.issparse(new_X):
                    new_X = new_X.copy()

                if new_M is not None:
                    if isinstance(new_M, np.ndarray):
                        new_M = new_M.copy()
                    elif sp.issparse(new_M):
                        new_M = new_M.copy()
                
            new_layers[name] = ScpMatrix(X=new_X, M=new_M)
            
        return Assay(var=new_var, layers=new_layers, feature_id_col=self.feature_id_col)

class ScpContainer:
    """
    顶层容器: 负责全局样本索引的管理和不同模态 (Assay) 的调度。
    Top-level container: Manages global sample index and dispatches different Assays.
    """
    def __init__(
        self, 
        obs: pl.DataFrame, 
        assays: Optional[Dict[str, Assay]] = None,
        links: Optional[List[AggregationLink]] = None,
        history: Optional[List[ProvenanceLog]] = None,
        sample_id_col: str = "_index"
    ):
        """
        Args:
            obs (pl.DataFrame): 全局样本元数据 (Global Meta).
                                MUST contain a unique ID column specified by sample_id_col.
            assays (Dict[str, Assay], optional): 模态仓库 (Assays Registry). Defaults to None.
            links (List[AggregationLink], optional): 特征聚合关系 (e.g. Peptide -> Protein). Defaults to None.
            history (List[ProvenanceLog], optional): 溯源日志. Defaults to None.
            sample_id_col (str): Column name in 'obs' that serves as the unique sample identifier.
                                 Defaults to "_index".
        """
        self.sample_id_col = sample_id_col
        
        # Validate ID column existence
        if sample_id_col not in obs.columns:
             raise ValueError(f"Sample ID column '{sample_id_col}' not found in obs.")
        
        # Validate ID uniqueness
        if obs[sample_id_col].n_unique() != obs.height:
            raise ValueError(f"Sample ID column '{sample_id_col}' is not unique.")

        self.obs: pl.DataFrame = obs
        self.assays: Dict[str, Assay] = assays if assays is not None else {}
        self.links: List[AggregationLink] = links if links is not None else []
        self.history: List[ProvenanceLog] = history if history is not None else []
        
        # Validate consistency upon initialization
        self._validate()
        # Optional: validate links if they exist
        if self.links:
            self.validate_links()

    @property
    def n_samples(self) -> int:
        return self.obs.height

    @property
    def sample_ids(self) -> pl.Series:
        return self.obs[self.sample_id_col]

    def _validate(self):
        """
        验证所有 Assay 的样本维度是否与全局 obs 对齐。
        Validate that sample dimensions of all Assays align with global obs.
        """
        for assay_name, assay in self.assays.items():
            for layer_name, matrix in assay.layers.items():
                if matrix.X.shape[0] != self.n_samples:
                    raise ValueError(
                        f"Sample dimension mismatch in Assay '{assay_name}', Layer '{layer_name}': "
                        f"Matrix has {matrix.X.shape[0]}, Container obs has {self.n_samples}"
                    )

    def validate_links(self):
        """
        Validate that all links connect to existing assays and features.
        """
        for link in self.links:
            if link.source_assay not in self.assays:
                raise ValueError(f"Link source assay '{link.source_assay}' not found.")
            if link.target_assay not in self.assays:
                raise ValueError(f"Link target assay '{link.target_assay}' not found.")
            
            source_ids = set(self.assays[link.source_assay].feature_ids)
            target_ids = set(self.assays[link.target_assay].feature_ids)
            
            # Check if link IDs exist in respective assays
            # This might be expensive for large datasets, so maybe make it optional or optimize
            link_source_ids = set(link.linkage["source_id"])
            link_target_ids = set(link.linkage["target_id"])
            
            if not link_source_ids.issubset(source_ids):
                 # Warn or error? strict mode -> error
                 pass # For now, let's assume it's ok to have links for missing features (maybe filtered out?)
                      # But generally they should match. 
                      # Let's raise warning if not strict.
            
            if not link_target_ids.issubset(target_ids):
                 pass

    def add_assay(self, name: str, assay: Assay) -> None:
        """
        注册新的模态 (Assay)。
        
        Args:
            name (str): 模态名称 (e.g., 'proteins', 'peptides', 'pca').
            assay (Assay): Assay 对象.
        """
        if name in self.assays:
             raise ValueError(f"Assay '{name}' already exists.")

        # Validate sample dimension for all layers in the new assay
        for layer_name, matrix in assay.layers.items():
             if matrix.X.shape[0] != self.n_samples:
                raise ValueError(
                    f"Sample dimension mismatch in new Assay '{name}', Layer '{layer_name}': "
                    f"Matrix has {matrix.X.shape[0]}, Container obs has {self.n_samples}"
                )
        self.assays[name] = assay

    def log_operation(self, action: str, params: Dict[str, Any], description: Optional[str] = None, software_version: Optional[str] = None):
        """
        记录操作日志。
        Log an operation to the history.
        """
        log = ProvenanceLog(
            timestamp=datetime.now().isoformat(),
            action=action,
            params=params,
            software_version=software_version,
            description=description
        )
        self.history.append(log)

    def __repr__(self) -> str:
        assays_desc = ", ".join([f"{k}({v.n_features})" for k, v in self.assays.items()])
        return f"<ScpContainer n_samples={self.n_samples}, assays=[{assays_desc}]>"

    def copy(self, deep: bool = True) -> ScpContainer:
        """
        Copy the container.
        
        Args:
            deep (bool): If True, create a deep copy. If False, create a shallow copy.
                         Defaults to True.
        """
        if deep:
            return self.deepcopy()
        else:
            return self.shallow_copy()

    def shallow_copy(self) -> ScpContainer:
        """
        Create a shallow copy of the container.
        """
        return ScpContainer(
            obs=self.obs, # Share obs
            assays=self.assays.copy(), # Shallow copy of assays dict
            links=list(self.links), # Shallow copy of links list
            history=list(self.history), # Shallow copy of history list
            sample_id_col=self.sample_id_col
        )

    def deepcopy(self) -> ScpContainer:
        """
        Create a deep copy of the container.
        """
        # Deep copy obs (Polars DataFrame)
        new_obs = self.obs.clone()
        
        # Deep copy assays
        new_assays = {}
        for name, assay in self.assays.items():
            # Assay.subset(all) with copy=True is effectively a deep copy
            # Or we can implement a dedicated clone method in Assay
            # For now, let's use subset with all indices
            new_assays[name] = assay.subset(np.arange(assay.n_features), copy_data=True)
            
        # Deep copy links (DataFrame clone)
        new_links = []
        for link in self.links:
            new_links.append(AggregationLink(
                source_assay=link.source_assay,
                target_assay=link.target_assay,
                linkage=link.linkage.clone()
            ))
            
        # Deep copy history
        new_history = [copy.deepcopy(log) for log in self.history]
        
        return ScpContainer(
            obs=new_obs,
            assays=new_assays,
            links=new_links,
            history=new_history,
            sample_id_col=self.sample_id_col
        )

