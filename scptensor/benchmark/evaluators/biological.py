"""Biological evaluator for single-cell data integration assessment.

This module provides metrics for evaluating biological signal preservation
and batch correction quality in single-cell proteomics data.

The evaluator supports two modes:
1. With scib-metrics: Full kBET, iLISI, cLISI metrics
2. Without scib-metrics: Fallback to sklearn-based metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    pass

# Try to import scib-metrics for advanced biological metrics
try:
    import scib_metrics

    _SCIB_AVAILABLE = True
except ImportError:
    _SCIB_AVAILABLE = False

# =============================================================================
# Type Aliases
# =============================================================================

ArrayFloat = NDArray[np.float64]
ArrayInt = NDArray[np.intp]


# =============================================================================
# Base Evaluator
# =============================================================================


class BaseEvaluator:
    """Base class for all benchmark evaluators.

    Evaluators compute metrics on processed data to assess the quality
    of analysis methods such as normalization, imputation, or integration.

    Subclasses should implement the evaluate() method with their specific
    metric computation logic.

    Examples
    --------
    >>> from scptensor.benchmark.evaluators.biological import BaseEvaluator
    >>>
    >>> class MyEvaluator(BaseEvaluator):
    ...     def evaluate(self, X, labels=None, batches=None):
    ...         return {"custom_metric": 0.5}
    """

    def evaluate(
        self,
        X: ArrayFloat,
        labels: ArrayInt | None = None,
        batches: ArrayInt | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate metrics on the given data.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix of shape (n_samples, n_features).
        labels : ArrayInt | None, default=None
            Cell type or cluster labels.
        batches : ArrayInt | None, default=None
            Batch labels for each sample.
        **kwargs
            Additional evaluator-specific parameters.

        Returns
        -------
        dict[str, float]
            Dictionary of metric names to values.
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


# =============================================================================
# Biological Evaluator
# =============================================================================


class BiologicalEvaluator(BaseEvaluator):
    """Evaluator for biological signal preservation and batch correction.

    This evaluator computes metrics that assess how well biological signals
    are preserved after processing and how effectively batch effects are
    removed.

    Metrics
    -------
    - kBET: k-nearest neighbour batch effect test. Higher is better.
    - iLISI: Inverse label Simpson's diversity (batch mixing). Higher is better.
    - cLISI: Cell label Simpson's diversity (cluster preservation). Higher is better.
    - ARI: Adjusted Rand Index (clustering agreement with truth). Higher is better.
    - NMI: Normalized Mutual Information (clustering agreement). Higher is better.

    Parameters
    ----------
    use_scib : bool, default=True
        Whether to use scib-metrics for kBET, iLISI, cLISI computation.
        Falls back to simplified metrics if scib-metrics is unavailable.
    k_bet : int, default=25
        Number of nearest neighbors for kBET computation.
    k_lisi : int, default=None
        Number of nearest neighbors for LISI computation. Defaults to k_bet.

    Attributes
    ----------
    use_scib : bool
        Whether scib-metrics is being used (may be False even if requested).
    scib_available : bool
        Whether scib-metrics package is installed.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators.biological import BiologicalEvaluator
    >>>
    >>> X = np.random.randn(100, 20)
    >>> labels = np.repeat([0, 1, 2, 3], 25)
    >>> batches = np.repeat([0, 0, 1, 1], 25)
    >>>
    >>> evaluator = BiologicalEvaluator()
    >>> metrics = evaluator.evaluate(X, labels=labels, batches=batches)
    >>> print(f"kBET: {metrics['kbet']:.3f}")
    """

    __slots__ = ("use_scib", "scib_available", "k_bet", "k_lisi")

    def __init__(
        self,
        use_scib: bool = True,
        k_bet: int = 25,
        k_lisi: int | None = None,
    ) -> None:
        """Initialize the biological evaluator.

        Parameters
        ----------
        use_scib : bool, default=True
            Whether to use scib-metrics when available.
        k_bet : int, default=25
            Number of neighbors for kBET computation.
        k_lisi : int | None, default=None
            Number of neighbors for LISI computation.
        """
        self.use_scib = use_scib and _SCIB_AVAILABLE
        self.scib_available = _SCIB_AVAILABLE
        self.k_bet = max(1, k_bet)
        self.k_lisi = k_lisi if k_lisi is not None else self.k_bet

    def evaluate(
        self,
        X: ArrayFloat,
        labels: ArrayInt | None = None,
        batches: ArrayInt | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate biological preservation and batch correction metrics.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix of shape (n_samples, n_features).
        labels : ArrayInt | None, default=None
            Cell type or cluster labels (required for cLISI, ARI, NMI).
        batches : ArrayInt | None, default=None
            Batch labels (required for kBET, iLISI).
        **kwargs
            Additional parameters including:
            - labels_true: True labels for ARI/NMI computation
            - labels_pred: Predicted cluster labels for ARI/NMI

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - "kbet": kBET score (0-1, higher is better)
            - "ilisi": iLISI score (0-1+, higher is better)
            - "clisi": cLISI score (0-1+, higher is better)
            - "ari": ARI score (0-1, higher is better, NaN if no labels)
            - "nmi": NMI score (0-1, higher is better, NaN if no labels)
        """
        X = np.asarray(X, dtype=np.float64)

        result: dict[str, float] = {}

        # kBET: batch mixing quality
        result["kbet"] = self._compute_kbet(X, batches) if batches is not None else np.nan

        # iLISI: batch mixing (inverse perspective)
        result["ilisi"] = self._compute_ilisi(X, batches) if batches is not None else np.nan

        # cLISI: cluster preservation
        result["clisi"] = self._compute_clisi(X, labels) if labels is not None else np.nan

        # ARI and NMI: clustering consistency with ground truth
        labels_true = kwargs.get("labels_true")
        labels_pred = kwargs.get("labels_pred")

        if labels_true is not None and labels_pred is not None:
            result["ari"] = self._compute_ari(
                np.asarray(labels_true), np.asarray(labels_pred)
            )
            result["nmi"] = self._compute_nmi(
                np.asarray(labels_true), np.asarray(labels_pred)
            )
        elif labels is not None and labels_true is not None:
            # Use provided labels as predictions
            result["ari"] = self._compute_ari(np.asarray(labels_true), labels)
            result["nmi"] = self._compute_nmi(np.asarray(labels_true), labels)
        else:
            result["ari"] = np.nan
            result["nmi"] = np.nan

        return result

    def evaluate_batch_correction(
        self,
        X_orig: ArrayFloat,
        X_corr: ArrayFloat,
        batches: ArrayInt,
        labels: ArrayInt | None = None,
    ) -> dict[str, float]:
        """Evaluate batch correction quality comparing original vs corrected.

        Parameters
        ----------
        X_orig : ArrayFloat
            Original data matrix before batch correction.
        X_corr : ArrayFloat
            Corrected data matrix after batch correction.
        batches : ArrayInt
            Batch labels for each sample.
        labels : ArrayInt | None, default=None
            Cell type labels for biological preservation assessment.

        Returns
        -------
        dict[str, float]
            Dictionary with metrics for both original and corrected data,
            plus improvement scores:
            - "kbet_orig": kBET on original data
            - "kbet_corr": kBET on corrected data
            - "kbet_delta": Improvement in kBET
            - "ilisi_orig": iLISI on original data
            - "ilisi_corr": iLISI on corrected data
            - "ilisi_delta": Improvement in iLISI
            - "clisi_corr": cLISI on corrected data (if labels provided)
        """
        orig_metrics = self.evaluate(X_orig, labels=None, batches=batches)
        corr_metrics = self.evaluate(X_corr, labels=labels, batches=batches)

        result: dict[str, float] = {
            "kbet_orig": orig_metrics["kbet"],
            "kbet_corr": corr_metrics["kbet"],
            "kbet_delta": corr_metrics["kbet"] - orig_metrics["kbet"],
            "ilisi_orig": orig_metrics["ilisi"],
            "ilisi_corr": corr_metrics["ilisi"],
            "ilisi_delta": corr_metrics["ilisi"] - orig_metrics["ilisi"],
        }

        if labels is not None:
            result["clisi_corr"] = corr_metrics["clisi"]

        return result

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _compute_kbet(self, X: ArrayFloat, batches: ArrayInt) -> float:
        """Compute kBET score using scib-metrics or fallback implementation.

        kBET tests whether the batch distribution in a cell's neighborhood
        matches the global batch distribution. Higher scores indicate better
        batch mixing.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        batches : ArrayInt
            Batch labels.

        Returns
        -------
        float
            kBET score in [0, 1], higher is better.
        """
        if self.use_scib and self.scib_available:
            return self._kbet_scib(X, batches)
        return self._kbet_fallback(X, batches)

    def _kbet_scib(self, X: ArrayFloat, batches: ArrayInt) -> float:
        """Compute kBET using scib-metrics.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        batches : ArrayInt
            Batch labels.

        Returns
        -------
        float
            kBET score (1 - rejection rate).
        """
        try:
            import anndata

            ad = anndata.AnnData(X)
            ad.obs["batch"] = np.asarray(batches).astype(str)

            score = scib_metrics.metrics.kBET(
                ad, batch_key="batch", embed="X", k=self.k_bet
            )
            return float(score)
        except Exception:
            return self._kbet_fallback(X, batches)

    def _kbet_fallback(self, X: ArrayFloat, batches: ArrayInt) -> float:
        """Fallback kBET implementation using nearest neighbors.

        Simplified version that computes the fraction of cells whose
        neighborhood batch distribution matches the global distribution.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        batches : ArrayInt
            Batch labels.

        Returns
        -------
        float
            kBET approximation in [0, 1].
        """
        n_samples = X.shape[0]
        if n_samples < self.k_bet + 1:
            return 0.0

        batches_arr = np.asarray(batches)
        unique_batches = np.unique(batches_arr)
        n_batches = len(unique_batches)

        if n_batches < 2:
            return 0.0

        # Global batch distribution
        global_batch_freq = np.array([np.mean(batches_arr == b) for b in unique_batches])

        try:
            nn = NearestNeighbors(n_neighbors=min(self.k_bet + 1, n_samples))
            nn.fit(X)
            distances, indices = nn.kneighbors(X)

            # Count acceptances (neighborhood matches global distribution)
            accept_count = 0
            for i in range(n_samples):
                neighbor_batches = batches_arr[indices[i][1:]]  # Exclude self
                local_batch_freq = np.array(
                    [np.mean(neighbor_batches == b) for b in unique_batches]
                )
                # Chi-squared-like test: sum of squared differences
                chi2 = np.sum((local_batch_freq - global_batch_freq) ** 2)
                if chi2 < 0.1:  # Tolerance threshold
                    accept_count += 1

            return accept_count / n_samples
        except Exception:
            return 0.0

    def _compute_ilisi(self, X: ArrayFloat, batches: ArrayInt) -> float:
        """Compute iLISI (inverse label Simpson's diversity) for batch mixing.

        iLISI measures the diversity of batches in each cell's neighborhood.
        Higher values indicate better batch mixing.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        batches : ArrayInt
            Batch labels.

        Returns
        -------
        float
            iLISI score (0 to n_batches, typically 1-2 is good).
        """
        if self.use_scib and self.scib_available:
            return self._ilisi_scib(X, batches)
        return self._ilisi_fallback(X, batches)

    def _ilisi_scib(self, X: ArrayFloat, batches: ArrayInt) -> float:
        """Compute iLISI using scib-metrics.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        batches : ArrayInt
            Batch labels.

        Returns
        -------
        float
            Mean iLISI score.
        """
        try:
            import anndata

            ad = anndata.AnnData(X)
            ad.obs["batch"] = np.asarray(batches).astype(str)

            scores = scib_metrics.metrics.lisi_graph(
                ad,
                batch_key="batch",
                label_key="batch",  # Use batch as both for iLISI
                k0=self.k_lisi,
                type="embed",
                use_raw=False,
            )
            return float(np.mean(scores))
        except Exception:
            return self._ilisi_fallback(X, batches)

    def _ilisi_fallback(self, X: ArrayFloat, batches: ArrayInt) -> float:
        """Fallback iLISI implementation.

        Simplified version computing Simpson's diversity index for
        batch labels in each neighborhood.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        batches : ArrayInt
            Batch labels.

        Returns
        -------
        float
            Mean iLISI score across all cells.
        """
        n_samples = X.shape[0]
        if n_samples < self.k_lisi + 1:
            return 0.0

        batches_arr = np.asarray(batches)

        try:
            nn = NearestNeighbors(n_neighbors=min(self.k_lisi + 1, n_samples))
            nn.fit(X)
            _, indices = nn.kneighbors(X)

            lisi_scores = []
            for i in range(n_samples):
                neighbor_batches = batches_arr[indices[i][1:]]  # Exclude self
                unique, counts = np.unique(neighbor_batches, return_counts=True)
                proportions = counts / np.sum(counts)
                simpson = 1.0 / np.sum(proportions**2)  # Inverse Simpson index
                lisi_scores.append(simpson)

            return float(np.mean(lisi_scores))
        except Exception:
            return 0.0

    def _compute_clisi(self, X: ArrayFloat, labels: ArrayInt) -> float:
        """Compute cLISI for cluster preservation.

        cLISI measures how well cell type clusters are preserved.
        Unlike iLISI, we want low diversity here (cells of same type
        should be near each other).

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        labels : ArrayInt
            Cell type labels.

        Returns
        -------
        float
            cLISI score (lower values indicate better separated clusters).
        """
        if self.use_scib and self.scib_available:
            return self._clisi_scib(X, labels)
        return self._clisi_fallback(X, labels)

    def _clisi_scib(self, X: ArrayFloat, labels: ArrayInt) -> float:
        """Compute cLISI using scib-metrics.

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        labels : ArrayInt
            Cell type labels.

        Returns
        -------
        float
            Mean cLISI score.
        """
        try:
            import anndata

            ad = anndata.AnnData(X)
            ad.obs["label"] = np.asarray(labels).astype(str)
            ad.obs["batch"] = "batch"  # Dummy batch for scib-metrics

            scores = scib_metrics.metrics.lisi_graph(
                ad,
                batch_key="batch",
                label_key="label",
                k0=self.k_lisi,
                type="embed",
                use_raw=False,
            )
            return float(np.mean(scores))
        except Exception:
            return self._clisi_fallback(X, labels)

    def _clisi_fallback(self, X: ArrayFloat, labels: ArrayInt) -> float:
        """Fallback cLISI implementation.

        For cLISI, we want cells of the same type to be close together,
        so we compute the inverse of label diversity (lower = better clusters).

        Parameters
        ----------
        X : ArrayFloat
            Data matrix.
        labels : ArrayInt
            Cell type labels.

        Returns
        -------
        float
            Mean cLISI score (lower is better for cluster separation).
        """
        n_samples = X.shape[0]
        if n_samples < self.k_lisi + 1:
            return 0.0

        labels_arr = np.asarray(labels)

        try:
            nn = NearestNeighbors(n_neighbors=min(self.k_lisi + 1, n_samples))
            nn.fit(X)
            _, indices = nn.kneighbors(X)

            # For cLISI, we want same-label neighbors (low diversity)
            purity_scores = []
            for i in range(n_samples):
                neighbor_labels = labels_arr[indices[i][1:]]  # Exclude self
                most_common = np.bincount(neighbor_labels).max()
                purity = most_common / len(neighbor_labels)
                purity_scores.append(purity)

            # Return purity (higher is better for cluster preservation)
            return float(np.mean(purity_scores))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_ari(labels_true: ArrayInt, labels_pred: ArrayInt) -> float:
        """Compute Adjusted Rand Index.

        ARI measures the similarity between two clusterings, adjusted
        for chance. Range is [-1, 1], with 1 indicating perfect agreement.

        Parameters
        ----------
        labels_true : ArrayInt
            Ground truth labels.
        labels_pred : ArrayInt
            Predicted cluster labels.

        Returns
        -------
        float
            ARI score in [-1, 1], higher is better.
        """
        try:
            unique_true = np.unique(labels_true)
            unique_pred = np.unique(labels_pred)

            if len(unique_true) < 2 or len(unique_pred) < 2:
                return 0.0

            return float(adjusted_rand_score(labels_true, labels_pred))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_nmi(labels_true: ArrayInt, labels_pred: ArrayInt) -> float:
        """Compute Normalized Mutual Information.

        NMI measures the mutual information between two clusterings,
        normalized by the entropy of each. Range is [0, 1].

        Parameters
        ----------
        labels_true : ArrayInt
            Ground truth labels.
        labels_pred : ArrayInt
            Predicted cluster labels.

        Returns
        -------
        float
            NMI score in [0, 1], higher is better.
        """
        try:
            unique_true = np.unique(labels_true)
            unique_pred = np.unique(labels_pred)

            if len(unique_true) < 2 or len(unique_pred) < 2:
                return 0.0

            return float(adjusted_mutual_info_score(labels_true, labels_pred))
        except Exception:
            return 0.0


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_biological(
    X: ArrayFloat,
    labels: ArrayInt | None = None,
    batches: ArrayInt | None = None,
    use_scib: bool = True,
    **kwargs,
) -> dict[str, float]:
    """Convenience function to evaluate biological metrics.

    Parameters
    ----------
    X : ArrayFloat
        Data matrix of shape (n_samples, n_features).
    labels : ArrayInt | None, default=None
        Cell type or cluster labels.
    batches : ArrayInt | None, default=None
        Batch labels.
    use_scib : bool, default=True
        Whether to use scib-metrics when available.
    **kwargs
        Additional parameters including labels_true, labels_pred for ARI/NMI.

    Returns
    -------
    dict[str, float]
        Dictionary of biological metric scores.

    Examples
    --------
    >>> import numpy as np
    >>> from scptensor.benchmark.evaluators.biological import evaluate_biological
    >>>
    >>> X = np.random.randn(100, 20)
    >>> labels = np.repeat([0, 1], 50)
    >>> batches = np.repeat([0, 1], 50)
    >>> metrics = evaluate_biological(X, labels=labels, batches=batches)
    """
    evaluator = BiologicalEvaluator(use_scib=use_scib)
    return evaluator.evaluate(X, labels=labels, batches=batches, **kwargs)


__all__ = ["BaseEvaluator", "BiologicalEvaluator", "evaluate_biological"]
