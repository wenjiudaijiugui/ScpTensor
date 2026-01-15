"""Shared pytest fixtures for ScpTensor tests.

This module provides reusable fixtures for testing ScpTensor core structures.
Fixtures are organized by data structure: obs, var, matrices, assays, and containers.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import (
    AggregationLink,
    Assay,
    MaskCode,
    ProvenanceLog,
    ScpContainer,
    ScpMatrix,
)


@pytest.fixture
def sample_obs() -> pl.DataFrame:
    """Create sample obs DataFrame with 5 samples.

    Returns
    -------
    pl.DataFrame
        DataFrame with sample IDs, batch assignments, and group labels.
    """
    return pl.DataFrame({
        "_index": ["S1", "S2", "S3", "S4", "S5"],
        "batch": ["batch1", "batch1", "batch2", "batch2", "batch1"],
        "group": ["A", "A", "B", "B", "A"]
    })


@pytest.fixture
def sample_var() -> pl.DataFrame:
    """Create sample var DataFrame with 5 features.

    Returns
    -------
    pl.DataFrame
        DataFrame with feature IDs, protein names, and chromosome locations.
    """
    return pl.DataFrame({
        "_index": ["P1", "P2", "P3", "P4", "P5"],
        "protein_name": ["Protein1", "Protein2", "Protein3", "Protein4", "Protein5"],
        "chromosome": ["chr1", "chr2", "chr3", "chr1", "chr2"]
    })


@pytest.fixture
def sample_dense_X() -> np.ndarray:
    """Create sample dense data matrix (5x5).

    Returns
    -------
    np.ndarray
        Random matrix with shape (5, 5), seeded for reproducibility.
    """
    rng = np.random.default_rng(42)
    return rng.random((5, 5))


@pytest.fixture
def sample_sparse_X() -> sparse.csr_matrix:
    """Create sample sparse data matrix (5x5).

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix with approximately 60% zeros, seeded for reproducibility.
    """
    rng = np.random.default_rng(42)
    X_dense = rng.random((5, 5))
    X_dense[X_dense < 0.6] = 0
    return sparse.csr_matrix(X_dense)


@pytest.fixture
def sample_mask_M() -> np.ndarray:
    """Create sample mask matrix with various mask codes.

    Returns
    -------
    np.ndarray
        int8 array with mask codes: 0=VALID, 1=MBR, 2=LOD, 3=FILTERED,
        4=OUTLIER, 5=IMPUTED.
    """
    M = np.zeros((5, 5), dtype=np.int8)
    M[0, 0] = MaskCode.MBR
    M[1, 1] = MaskCode.LOD
    M[2, 2] = MaskCode.FILTERED
    M[3, 3] = MaskCode.IMPUTED
    M[4, 4] = MaskCode.OUTLIER
    return M


@pytest.fixture
def sample_matrix(sample_dense_X: np.ndarray) -> ScpMatrix:
    """Create sample ScpMatrix without mask.

    Parameters
    ----------
    sample_dense_X : np.ndarray
        Dense data matrix.

    Returns
    -------
    ScpMatrix
        Matrix without mask.
    """
    return ScpMatrix(X=sample_dense_X)


@pytest.fixture
def sample_matrix_with_mask(
    sample_dense_X: np.ndarray,
    sample_mask_M: np.ndarray,
) -> ScpMatrix:
    """Create sample ScpMatrix with mask.

    Parameters
    ----------
    sample_dense_X : np.ndarray
        Dense data matrix.
    sample_mask_M : np.ndarray
        Mask matrix.

    Returns
    -------
    ScpMatrix
        Matrix with mask codes.
    """
    return ScpMatrix(X=sample_dense_X, M=sample_mask_M)


@pytest.fixture
def sample_sparse_matrix(sample_sparse_X: sparse.csr_matrix) -> ScpMatrix:
    """Create sample ScpMatrix with sparse data.

    Parameters
    ----------
    sample_sparse_X : sparse.csr_matrix
        Sparse data matrix.

    Returns
    -------
    ScpMatrix
        Matrix with sparse storage.
    """
    return ScpMatrix(X=sample_sparse_X)


@pytest.fixture
def sample_assay(sample_var: pl.DataFrame, sample_dense_X: np.ndarray) -> Assay:
    """Create sample Assay with single layer.

    Parameters
    ----------
    sample_var : pl.DataFrame
        Feature metadata.
    sample_dense_X : np.ndarray
        Data matrix.

    Returns
    -------
    Assay
        Assay with 'raw' layer.
    """
    return Assay(var=sample_var, layers={"raw": ScpMatrix(X=sample_dense_X)})


@pytest.fixture
def sample_assay_with_mask(
    sample_var: pl.DataFrame,
    sample_dense_X: np.ndarray,
    sample_mask_M: np.ndarray,
) -> Assay:
    """Create sample Assay with masked matrix.

    Parameters
    ----------
    sample_var : pl.DataFrame
        Feature metadata.
    sample_dense_X : np.ndarray
        Data matrix.
    sample_mask_M : np.ndarray
        Mask matrix.

    Returns
    -------
    Assay
        Assay with masked 'raw' layer.
    """
    return Assay(
        var=sample_var,
        layers={"raw": ScpMatrix(X=sample_dense_X, M=sample_mask_M)}
    )


@pytest.fixture
def sample_assay_multi_layer(
    sample_var: pl.DataFrame,
    sample_dense_X: np.ndarray,
) -> Assay:
    """Create sample Assay with multiple transformation layers.

    Parameters
    ----------
    sample_var : pl.DataFrame
        Feature metadata.
    sample_dense_X : np.ndarray
        Raw data matrix.

    Returns
    -------
    Assay
        Assay with 'raw', 'log', and 'normalized' layers.
    """
    X_log = np.log1p(sample_dense_X)
    X_norm = (X_log - X_log.mean()) / (X_log.std() + 1e-8)
    return Assay(
        var=sample_var,
        layers={
            "raw": ScpMatrix(X=sample_dense_X),
            "log": ScpMatrix(X=X_log),
            "normalized": ScpMatrix(X=X_norm)
        }
    )


@pytest.fixture
def sample_container(sample_obs: pl.DataFrame, sample_assay: Assay) -> ScpContainer:
    """Create sample ScpContainer with single assay.

    Parameters
    ----------
    sample_obs : pl.DataFrame
        Sample metadata.
    sample_assay : Assay
        Protein assay.

    Returns
    -------
    ScpContainer
        Container with proteins assay.
    """
    return ScpContainer(obs=sample_obs, assays={"proteins": sample_assay})


@pytest.fixture
def sample_container_multi_assay(
    sample_obs: pl.DataFrame,
    sample_var: pl.DataFrame,
    sample_dense_X: np.ndarray,
) -> ScpContainer:
    """Create sample ScpContainer with multiple assays.

    Parameters
    ----------
    sample_obs : pl.DataFrame
        Sample metadata.
    sample_var : pl.DataFrame
        Feature metadata for protein assay.
    sample_dense_X : np.ndarray
        Data matrix.

    Returns
    -------
    ScpContainer
        Container with proteins and peptides assays.
    """
    assay1 = Assay(var=sample_var, layers={"X": ScpMatrix(X=sample_dense_X)})
    var_peptide = pl.DataFrame({"_index": [f"PEP{i}" for i in range(10)]})
    X_peptide = np.random.default_rng(42).random((5, 10))
    assay2 = Assay(var=var_peptide, layers={"X": ScpMatrix(X=X_peptide)})
    return ScpContainer(
        obs=sample_obs,
        assays={"proteins": assay1, "peptides": assay2}
    )


@pytest.fixture
def sample_aggregation_link() -> AggregationLink:
    """Create sample AggregationLink for peptide-to-protein mapping.

    Returns
    -------
    AggregationLink
        Link mapping peptides to their parent proteins.
    """
    linkage = pl.DataFrame({
        "source_id": ["PEP1", "PEP2", "PEP3", "PEP4", "PEP5"],
        "target_id": ["PROT1", "PROT1", "PROT2", "PROT2", "PROT3"]
    })
    return AggregationLink(
        source_assay="peptides",
        target_assay="proteins",
        linkage=linkage
    )


@pytest.fixture
def minimal_obs() -> pl.DataFrame:
    """Create minimal obs DataFrame with single sample.

    Returns
    -------
    pl.DataFrame
        DataFrame with one row.
    """
    return pl.DataFrame({"_index": ["S1"]})


@pytest.fixture
def minimal_var() -> pl.DataFrame:
    """Create minimal var DataFrame with single feature.

    Returns
    -------
    pl.DataFrame
        DataFrame with one row.
    """
    return pl.DataFrame({"_index": ["P1"]})


@pytest.fixture
def minimal_X() -> np.ndarray:
    """Create minimal 1x1 data matrix.

    Returns
    -------
    np.ndarray
        Array with shape (1, 1).
    """
    return np.array([[1.0]])


@pytest.fixture
def empty_container_data() -> pl.DataFrame:
    """Create obs DataFrame for empty container.

    Returns
    -------
    pl.DataFrame
        DataFrame with samples but no assays.
    """
    return pl.DataFrame({"_index": ["S1", "S2"]})


@pytest.fixture
def valid_mask_codes() -> list[int]:
    """Return all valid mask code values.

    Returns
    -------
    list[int]
        List of valid mask code integers.
    """
    return [0, 1, 2, 3, 4, 5, 6]


@pytest.fixture
def mask_code_names() -> dict[int, str]:
    """Return mapping of mask codes to names.

    Returns
    -------
    dict[int, str]
        Dictionary mapping code values to descriptive names.
    """
    return {
        0: "VALID",
        1: "MBR",
        2: "LOD",
        3: "FILTERED",
        4: "OUTLIER",
        5: "IMPUTED",
        6: "UNCERTAIN"
    }
