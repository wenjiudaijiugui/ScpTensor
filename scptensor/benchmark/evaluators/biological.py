"""Biological evaluator for single-cell data integration assessment."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


class BaseEvaluator:
    """Base class for all benchmark evaluators."""

    def evaluate(self, X: np.ndarray, labels: np.ndarray | None = None,
                 batches: np.ndarray | None = None, **kwargs) -> dict[str, float]:
        raise NotImplementedError("Subclasses must implement evaluate()")

try:
    import scib_metrics
    _SCIB_AVAILABLE = True
except ImportError:
    _SCIB_AVAILABLE = False


class BiologicalEvaluator:
    """Evaluator for biological signal preservation and batch correction."""

    __slots__ = ("use_scib", "k_bet", "k_lisi")

    def __init__(self, use_scib: bool = True, k_bet: int = 25, k_lisi: int | None = None):
        self.use_scib = use_scib and _SCIB_AVAILABLE
        self.k_bet = max(1, k_bet)
        self.k_lisi = k_lisi if k_lisi is not None else self.k_bet

    def evaluate(self, X: np.ndarray, labels: np.ndarray | None = None,
                 batches: np.ndarray | None = None, **kwargs) -> dict[str, float]:
        """Evaluate biological preservation and batch correction metrics."""
        X = np.asarray(X, dtype=np.float64)
        result = {}

        result["kbet"] = self._kbet(X, batches) if batches is not None else np.nan
        result["ilisi"] = self._lisi(X, batches) if batches is not None else np.nan
        result["clisi"] = self._lisi(X, labels) if labels is not None else np.nan

        labels_true = kwargs.get("labels_true")
        labels_pred = kwargs.get("labels_pred")

        if labels_true is not None and labels_pred is not None:
            result["ari"] = float(adjusted_rand_score(labels_true, labels_pred))
            result["nmi"] = float(adjusted_mutual_info_score(labels_true, labels_pred))
        else:
            result["ari"] = np.nan
            result["nmi"] = np.nan

        return result

    def evaluate_batch_correction(self, X_orig: np.ndarray, X_corr: np.ndarray,
                                  batches: np.ndarray, labels: np.ndarray | None = None) -> dict[str, float]:
        """Evaluate batch correction quality comparing original vs corrected."""
        orig = self.evaluate(X_orig, labels=None, batches=batches)
        corr = self.evaluate(X_corr, labels=labels, batches=batches)

        result = {
            "kbet_orig": orig["kbet"], "kbet_corr": corr["kbet"],
            "kbet_delta": corr["kbet"] - orig["kbet"],
            "ilisi_orig": orig["ilisi"], "ilisi_corr": corr["ilisi"],
            "ilisi_delta": corr["ilisi"] - orig["ilisi"],
        }
        if labels is not None:
            result["clisi_corr"] = corr["clisi"]
        return result

    def _kbet(self, X: np.ndarray, batches: np.ndarray) -> float:
        """Compute kBET score."""
        if self.use_scib:
            try:
                import anndata
                ad = anndata.AnnData(X)
                ad.obs["batch"] = np.asarray(batches).astype(str)
                return float(scib_metrics.metrics.kBET(ad, batch_key="batch", embed="X", k=self.k_bet))
            except Exception:
                pass

        n, k = X.shape[0], min(self.k_bet + 1, X.shape[0])
        if n < k + 1:
            return 0.0

        batches_arr = np.asarray(batches)
        unique_batches = np.unique(batches_arr)
        if len(unique_batches) < 2:
            return 0.0

        global_freq = np.array([np.mean(batches_arr == b) for b in unique_batches])

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)

        neighbor_batches = batches_arr[indices[:, 1:]]
        local_freq = np.stack([np.mean(neighbor_batches == b, axis=1) for b in unique_batches], axis=1)
        chi2 = np.sum((local_freq - global_freq) ** 2, axis=1)
        return float(np.mean(chi2 < 0.1))

    def _lisi(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute LISI score (Simpson's diversity)."""
        if self.use_scib:
            try:
                import anndata
                ad = anndata.AnnData(X)
                ad.obs["label"] = np.asarray(labels).astype(str)
                ad.obs["batch"] = "batch"
                scores = scib_metrics.metrics.lisi_graph(ad, batch_key="batch", label_key="label",
                                                         k0=self.k_lisi, type="embed", use_raw=False)
                return float(np.mean(scores))
            except Exception:
                pass

        n, k = X.shape[0], min(self.k_lisi + 1, X.shape[0])
        if n < k + 1:
            return 0.0

        labels_arr = np.asarray(labels)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)

        neighbor_labels = labels_arr[indices[:, 1:]]
        simpson_vals = np.array([
            1.0 / np.sum((np.bincount(neighbor_labels[i], minlength=labels_arr.max() + 1) /
                          len(neighbor_labels[i])) ** 2)
            for i in range(len(neighbor_labels))
        ])
        return float(np.mean(simpson_vals))


def evaluate_biological(X: np.ndarray, labels: np.ndarray | None = None,
                       batches: np.ndarray | None = None, use_scib: bool = True,
                       **kwargs) -> dict[str, float]:
    """Convenience function to evaluate biological metrics."""
    return BiologicalEvaluator(use_scib=use_scib).evaluate(X, labels, batches, **kwargs)


__all__ = ["BaseEvaluator", "BiologicalEvaluator", "evaluate_biological"]
