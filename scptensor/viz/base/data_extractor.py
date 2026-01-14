"""Data extraction utilities for visualization."""

from typing import Literal

import numpy as np
from scipy import sparse

from scptensor import ScpContainer


class DataExtractor:
    """Extract plotting data from ScpContainer with unified handling."""

    @staticmethod
    def get_expression_matrix(
        container: ScpContainer,
        assay_name: str,
        layer: str,
        var_names: list[str] | None = None,
        samples: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get expression matrix with metadata.

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
        assay = container.assays[assay_name]
        scpmatrix = assay.layers[layer]

        X = scpmatrix.X  # noqa: N806
        if sparse.issparse(X):
            X = X.toarray()  # noqa: N806
        M = scpmatrix.M if scpmatrix.M is not None else np.zeros_like(X, dtype=int)  # noqa: N806

        # Filter samples
        if samples is not None:
            sample_idx = [
                i for i, s in enumerate(container.obs[container.sample_id_col]) if s in samples
            ]
            X = X[sample_idx]  # noqa: N806
            M = M[sample_idx]  # noqa: N806
            obs = container.obs[sample_idx]
        else:
            obs = container.obs

        # Filter features
        if var_names is not None:
            feature_idx = [i for i, v in enumerate(assay.var["protein"]) if v in var_names]
            X = X[:, feature_idx]  # noqa: N806
            M = M[:, feature_idx]  # noqa: N806
            var = assay.var[feature_idx]
        else:
            var = assay.var

        return X, obs.to_numpy(), var.to_numpy()  # noqa: N806

    @staticmethod
    def get_group_data(
        container: ScpContainer,
        groupby: str,
    ) -> np.ndarray:
        """
        Get grouping information.

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
        X: np.ndarray,  # noqa: N803
        M: np.ndarray,  # noqa: N803
        method: Literal["separate", "transparent", "imputed"] = "separate",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle missing values for visualization.

        Parameters
        ----------
        X : ndarray
            Expression matrix
        M : ndarray
            Mask matrix (0=valid, 1=MBR, 2=LOD, 3=filtered)
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
        if method == "imputed":
            return X, np.array([]), np.array([])

        valid_mask = M == 0
        missing_mask = M > 0

        if method == "transparent":
            X_result = X.copy().astype(float)  # noqa: N806
            X_result[missing_mask] = np.nan
            return X_result, np.array([]), M[missing_mask]

        # method == 'separate'
        X_valid = X[valid_mask]  # noqa: N806
        X_missing = X[missing_mask]  # noqa: N806
        M_types = M[missing_mask]  # noqa: N806

        return X_valid, X_missing, M_types
