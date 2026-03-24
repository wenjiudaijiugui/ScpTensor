"""Data extraction utilities for visualization."""

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from scipy import sparse

from scptensor import ScpContainer

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class DataExtractor:
    """Extract plotting data from ScpContainer with unified handling."""

    @staticmethod
    def _resolve_indices_or_raise(
        available_ids: list[str],
        requested_ids: list[str],
        id_type: str,
    ) -> list[int]:
        """Resolve requested IDs to indices with explicit missing-ID errors."""
        index_map = {item_id: idx for idx, item_id in enumerate(available_ids)}
        missing = [item_id for item_id in requested_ids if item_id not in index_map]
        if missing:
            raise ValueError(
                f"{id_type} not found: {missing}. Available {id_type.lower()} count: "
                f"{len(available_ids)}",
            )
        return [index_map[item_id] for item_id in requested_ids]

    @staticmethod
    def get_expression_matrix(
        container: ScpContainer,
        assay_name: str,
        layer: str,
        var_names: list[str] | None = None,
        samples: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get expression matrix with metadata.

        Parameters
        ----------
        container : ScpContainer
            Input container
        assay_name : str
            Assay name
        layer : str
            Layer name
        var_names : list or None
            Feature names to extract (None = all)
        samples : list or None
            Sample names to extract (None = all)

        Returns
        -------
        X : ndarray
            Expression matrix (n_samples x n_features)
        obs : ndarray
            Sample metadata
        var : ndarray
            Feature metadata

        """
        if assay_name not in container.assays:
            raise ValueError(f"Assay '{assay_name}' not found in container")

        assay = container.assays[assay_name]
        if layer not in assay.layers:
            raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'")
        scpmatrix = assay.layers[layer]

        X = scpmatrix.X
        if sparse.issparse(X):
            X = cast("spmatrix", X).toarray()
        X = np.asarray(X)

        M = scpmatrix.M if scpmatrix.M is not None else np.zeros_like(X, dtype=int)
        if sparse.issparse(M):
            M = cast("spmatrix", M).toarray()
        M = np.asarray(M)

        # Filter samples
        if samples is not None:
            sample_ids = container.obs[container.sample_id_col].to_list()
            sample_idx = DataExtractor._resolve_indices_or_raise(
                sample_ids,
                samples,
                "Samples",
            )
            X = X[sample_idx]
            M = M[sample_idx]
            obs = container.obs[sample_idx]
        else:
            obs = container.obs

        # Filter features
        if var_names is not None:
            feature_id_col = assay.feature_id_col
            feature_ids = assay.var[feature_id_col].to_list()
            feature_idx = DataExtractor._resolve_indices_or_raise(
                feature_ids,
                var_names,
                "Features",
            )
            X = X[:, feature_idx]
            M = M[:, feature_idx]
            var = assay.var[feature_idx]
        else:
            var = assay.var

        return X, obs.to_numpy(), var.to_numpy()

    @staticmethod
    def get_group_data(
        container: ScpContainer,
        groupby: str,
    ) -> np.ndarray:
        """Get grouping information.

        Parameters
        ----------
        container : ScpContainer
            Input container
        groupby : str
            Column name in obs

        Returns
        -------
        ndarray
            Group labels for each sample

        """
        if groupby not in container.obs.columns:
            raise ValueError(f"Column '{groupby}' not found in obs")
        return container.obs[groupby].to_numpy()

    @staticmethod
    def handle_missing_values(
        X: np.ndarray,
        M: np.ndarray | None,
        method: Literal["separate", "transparent", "imputed"] = "separate",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle missing values for visualization.

        Parameters
        ----------
        X : ndarray
            Expression matrix
        M : ndarray
            Mask matrix (0=valid, >0=non-valid/masked status)
        method : str
            - 'separate': Return separate arrays for valid/missing
            - 'transparent': Return X with nan for missing
            - 'imputed': Return X as-is (assume already imputed)

        Returns
        -------
        X_valid : ndarray
            Valid values
        X_missing : ndarray
            Missing values (if method='separate')
        M_types : ndarray
            Missing value types

        """
        allowed_methods = {"separate", "transparent", "imputed"}
        if method not in allowed_methods:
            raise ValueError(
                f"Unsupported method '{method}'. Expected one of: {sorted(allowed_methods)}",
            )

        x_array = np.asarray(X)
        m_array = np.zeros_like(x_array, dtype=np.int8) if M is None else np.asarray(M)
        if x_array.shape != m_array.shape:
            raise ValueError(
                f"X and M must have the same shape, got X={x_array.shape}, M={m_array.shape}",
            )

        if method == "imputed":
            return x_array, np.array([]), np.array([])

        valid_mask = m_array == 0
        missing_mask = m_array > 0

        if method == "transparent":
            X_result = x_array.copy().astype(float)
            X_result[missing_mask] = np.nan
            return X_result, np.array([]), m_array[missing_mask]

        # method == 'separate'
        X_valid = x_array[valid_mask]
        X_missing = x_array[missing_mask]
        M_types = m_array[missing_mask]

        return X_valid, X_missing, M_types
