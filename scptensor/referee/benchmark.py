"""Referee module for evaluating integration methods."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

from scptensor.core.structures import ScpContainer

from .metrics import compute_batch_mixing, compute_cluster_separation

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


# Constants
DEFAULT_LAYER: str = "X"
METHOD_COLUMN: str = "method"
BATCH_MIXING_COLUMN: str = "batch_mixing_score"
BIO_CONSERVATION_COLUMN: str = "bio_conservation_score"


class IntegrationReferee:
    """
    Referee for evaluating integration methods.

    Evaluates batch correction quality using two metrics:
    - Batch mixing score: Higher is better (batches should be mixed)
    - Bio conservation score: Higher is better (biological clusters should separate)

    Parameters
    ----------
    container : ScpContainer
        Input container with integration results.
    batch_key : str
        Column name in obs for batch information.
    label_key : str, optional
        Column name in obs for biological labels. If provided, computes
        bio conservation score.

    Examples
    --------
    >>> referee = IntegrationReferee(container, batch_key="batch", label_key="cell_type")
    >>> scores = referee.score("proteins", "corrected")
    >>> comparison = referee.compare([("proteins", "corrected"), ("proteins", "harmony")])
    """

    def __init__(
        self,
        container: ScpContainer,
        batch_key: str,
        label_key: str | None = None,
    ) -> None:
        self._container = container
        self._batch_key = batch_key
        self._label_key = label_key

    @property
    def container(self) -> ScpContainer:
        """Get the container."""
        return self._container

    @property
    def batch_key(self) -> str:
        """Get the batch key."""
        return self._batch_key

    @property
    def label_key(self) -> str | None:
        """Get the label key."""
        return self._label_key

    def score(self, assay_name: str, layer: str = DEFAULT_LAYER) -> dict[str, float]:
        """
        Calculate integration scores for a specific assay layer.

        Parameters
        ----------
        assay_name : str
            Name of the assay to evaluate.
        layer : str, optional
            Layer name within the assay. Default is "X".

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - ``batch_mixing_score``: float in [0, 1], higher is better
            - ``bio_conservation_score``: float in [-1, 1], only if label_key is set

        Raises
        ------
        ValueError
            If assay_name or layer is not found.
        KeyError
            If batch_key is not in obs.

        Examples
        --------
        >>> referee = IntegrationReferee(container, batch_key="batch")
        >>> scores = referee.score("proteins", "corrected")
        >>> print(f"Batch mixing: {scores['batch_mixing_score']:.3f}")
        """
        # Validate assay and layer exist
        assays = self._container.assays
        if assay_name not in assays:
            available = ", ".join(assays.keys())
            raise ValueError(f"Assay '{assay_name}' not found. Available: {available}")

        assay = assays[assay_name]
        if layer not in assay.layers:
            available = ", ".join(assay.layers.keys())
            raise ValueError(
                f"Layer '{layer}' not found in assay '{assay_name}'. Available: {available}"
            )

        # Extract data
        X = assay.layers[layer].X
        batches = self._get_obs_column(self._batch_key).to_numpy()

        # Compute batch mixing (always required)
        scores = {BATCH_MIXING_COLUMN: compute_batch_mixing(X, batches)}

        # Compute bio conservation (optional)
        if self._label_key:
            try:
                labels = self._get_obs_column(self._label_key).to_numpy()
                scores[BIO_CONSERVATION_COLUMN] = compute_cluster_separation(X, labels)
            except KeyError:
                logger.warning(
                    f"Label key '{self._label_key}' not found in obs. "
                    "Skipping bio conservation score."
                )

        return scores

    def compare(
        self,
        candidates: Sequence[tuple[str, str]],
    ) -> "pd.DataFrame":
        """
        Compare multiple integration results.

        Parameters
        ----------
        candidates : Sequence[tuple[str, str]]
            Sequence of (assay_name, layer_name) tuples to evaluate.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: method, batch_mixing_score, bio_conservation_score.
            Returns empty DataFrame if all candidates fail.

        Examples
        --------
        >>> candidates = [("proteins", "combat"), ("proteins", "harmony")]
        >>> df = referee.compare(candidates)
        >>> print(df)
        """
        import pandas as pd

        # Compute scores with list comprehension (faster than explicit loop)
        results = [
            self._create_score_entry(assay, layer)
            for assay, layer in candidates
            if self._is_valid_candidate(assay, layer)
        ]

        if not results:
            return pd.DataFrame()

        # Build DataFrame with consistent column order
        df = pd.DataFrame(results)
        columns = [METHOD_COLUMN, BATCH_MIXING_COLUMN]
        if BIO_CONSERVATION_COLUMN in df.columns:
            columns.append(BIO_CONSERVATION_COLUMN)

        return df.reindex(columns=columns)

    # Internal helpers

    def _get_obs_column(self, key: str) -> pl.Series:
        """Get a column from obs, raising KeyError with helpful message."""
        try:
            return self._container.obs[key]
        except KeyError:
            available = ", ".join(self._container.obs.columns)
            raise KeyError(f"Column '{key}' not found in obs. Available: {available}") from None

    def _is_valid_candidate(self, assay: str, layer: str) -> bool:
        """Check if candidate exists, logging failures."""
        if assay not in self._container.assays:
            logger.warning(f"Skipping {assay}_{layer}: assay '{assay}' not found")
            return False

        if layer not in self._container.assays[assay].layers:
            logger.warning(
                f"Skipping {assay}_{layer}: layer '{layer}' not found in assay '{assay}'"
            )
            return False

        return True

    def _create_score_entry(self, assay: str, layer: str) -> dict[str, float | str]:
        """Create score dictionary entry for a candidate."""
        scores: dict[str, float | str] = self.score(assay, layer)
        scores[METHOD_COLUMN] = f"{assay}_{layer}"
        return scores
