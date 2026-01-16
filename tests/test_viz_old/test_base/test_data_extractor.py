# tests/test_viz/test_base/test_data_extractor.py
import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor import Assay, ScpContainer, ScpMatrix
from scptensor.viz.base.data_extractor import DataExtractor


@pytest.fixture
def test_container():
    """Create test container with layers."""
    X_raw = np.random.rand(100, 50) * 10
    M_raw = np.zeros_like(X_raw, dtype=int)
    M_raw[X_raw < 2] = 1  # Some missing values

    obs = pl.DataFrame(
        {
            "_index": [f"S{i}" for i in range(100)],
            "sample_id": [f"sample_{i}" for i in range(100)],
            "cluster": np.random.choice(["A", "B", "C"], 100),
            "batch": np.random.choice(["batch1", "batch2"], 100),
        }
    )

    var = pl.DataFrame(
        {
            "_index": [f"P{i}" for i in range(50)],
            "protein": [f"P{i}" for i in range(50)],
        }
    )

    assay = Assay(var=var)
    assay.layers["raw"] = ScpMatrix(X=X_raw, M=M_raw)
    assay.layers["normalized"] = ScpMatrix(X=np.log1p(X_raw), M=M_raw)

    container = ScpContainer(obs=obs, assays={"proteins": assay}, sample_id_col="sample_id")
    return container


def test_get_expression_matrix_dense(test_container):
    """Test extracting dense expression matrix."""
    X, obs, var = DataExtractor.get_expression_matrix(test_container, "proteins", "normalized")
    assert X.shape == (100, 50)
    assert isinstance(X, np.ndarray)


def test_get_expression_matrix_sparse(test_container):
    """Test extracting sparse expression matrix."""
    X_sparse = sparse.csr_matrix(np.random.rand(100, 50) * 10)
    M_sparse = np.zeros((100, 50), dtype=int)
    test_container.assays["proteins"].layers["sparse"] = ScpMatrix(X=X_sparse, M=M_sparse)

    X, _, _ = DataExtractor.get_expression_matrix(test_container, "proteins", "sparse")
    assert isinstance(X, np.ndarray)


def test_get_expression_matrix_with_features(test_container):
    """Test extracting subset of features."""
    features = ["P0", "P1", "P2"]
    X, obs, var = DataExtractor.get_expression_matrix(
        test_container, "proteins", "normalized", var_names=features
    )
    assert X.shape == (100, 3)


def test_get_expression_matrix_with_samples(test_container):
    """Test extracting subset of samples using sample_id_col."""
    # Select first 10 samples by their sample_id
    selected_samples = [f"sample_{i}" for i in range(10)]
    X, obs, var = DataExtractor.get_expression_matrix(
        test_container, "proteins", "normalized", samples=selected_samples
    )
    assert X.shape == (10, 50)
    assert obs.shape[0] == 10
    # Verify correct samples were selected
    assert set(obs[:, 1]) == set(selected_samples)  # sample_id is column 1


def test_get_group_data(test_container):
    """Test extracting group information."""
    groups = DataExtractor.get_group_data(test_container, "cluster")
    assert groups is not None
    assert len(groups) == 100


def test_handle_missing_values_separate():
    """Test separating missing values."""
    X = np.array([[1, 2, 0], [4, 0, 6]])
    M = np.array([[0, 0, 1], [0, 2, 0]])  # 1=MBR, 2=LOD

    X_valid, X_missing, M_types = DataExtractor.handle_missing_values(X, M, method="separate")
    # M == 0 at positions [0,0], [0,1], [1,0], [1,2] = 4 valid values
    assert X_valid.shape[0] == 4  # 4 valid values
    assert X_missing.shape[0] == 2  # 2 missing values
