#!/usr/bin/env python3
"""
Quick test to verify the refactoring is working correctly.
"""

import numpy as np
import polars as pl

# Test new imports
try:
    from scptensor.core import ScpContainer, ScpMatrix, MaskCode, MatrixOps
    from scptensor.normalization import (
        sample_median_normalization,
        sample_mean_normalization,
        global_median_normalization,
        tmm_normalization,
        upper_quartile_normalization
    )
    from scptensor.standardization import zscore_standardization
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def create_test_container():
    """Create a simple test container."""
    n_samples, n_features = 10, 20
    X = np.random.lognormal(mean=1, sigma=0.5, size=(n_samples, n_features))

    # Add some missing values
    M = np.zeros((n_samples, n_features), dtype=np.int8)
    missing_indices = np.random.choice(n_samples * n_features, size=int(n_samples * n_features * 0.1), replace=False)
    row_indices = missing_indices // n_features
    col_indices = missing_indices % n_features
    M[row_indices, col_indices] = MaskCode.MBR.value
    X[row_indices, col_indices] = 0

    obs = pl.DataFrame({
        'sample_id': [f'S{i+1:03d}' for i in range(n_samples)],
        'group': ['A'] * 5 + ['B'] * 5,
        '_index': [f'S{i+1:03d}' for i in range(n_samples)]
    })

    var = pl.DataFrame({
        'protein_id': [f'P{i+1:04d}' for i in range(n_features)],
        '_index': [f'P{i+1:04d}' for i in range(n_features)]
    })

    matrix = ScpMatrix(X=X, M=M)
    from scptensor.core import Assay
    assay = Assay(var=var, layers={'raw': matrix}, feature_id_col='protein_id')

    container = ScpContainer(
        assays={'protein': assay},
        obs=obs,
        sample_id_col='sample_id'
    )

    return container

def test_normalization_methods():
    """Test all normalization methods."""
    container = create_test_container()

    print("Testing normalization methods...")

    # Test sample median normalization
    container = sample_median_normalization(container)
    assert 'sample_median_norm' in container.assays['protein'].layers
    print("✅ Sample median normalization")

    # Test sample mean normalization
    container = sample_mean_normalization(container)
    assert 'sample_mean_norm' in container.assays['protein'].layers
    print("✅ Sample mean normalization")

    # Test global median normalization
    container = global_median_normalization(container)
    assert 'global_median_norm' in container.assays['protein'].layers
    print("✅ Global median normalization")

    # Test TMM normalization
    container = tmm_normalization(container)
    assert 'tmm_norm' in container.assays['protein'].layers
    print("✅ TMM normalization")

    # Test upper quartile normalization
    container = upper_quartile_normalization(container)
    assert 'upper_quartile_norm' in container.assays['protein'].layers
    print("✅ Upper quartile normalization")

def test_matrix_ops():
    """Test matrix operations."""
    container = create_test_container()
    matrix = container.assays['protein'].layers['raw']

    print("Testing matrix operations...")

    # Test mask statistics
    stats = MatrixOps.get_mask_statistics(matrix)
    assert isinstance(stats, dict)
    print("✅ Mask statistics")

    # Test valid mask
    valid_mask = MatrixOps.get_valid_mask(matrix)
    assert valid_mask.shape == matrix.X.shape
    print("✅ Valid mask")

    # Test mark imputed
    new_matrix = MatrixOps.mark_imputed(matrix, (np.array([0, 1]), np.array([0, 1])))
    assert new_matrix.get_m()[0, 0] == MaskCode.IMPUTED.value
    print("✅ Mark imputed")

def test_standardization():
    """Test standardization method."""
    container = create_test_container()

    # First impute to create complete matrix
    from scptensor.impute import knn
    container = knn(container, assay_name='protein', base_layer='raw', new_layer_name='imputed')

    print("Testing standardization...")
    container = zscore_standardization(container, base_layer_name='imputed')
    assert 'zscore' in container.assays['protein'].layers
    print("✅ Z-score standardization")

if __name__ == "__main__":
    print("🧪 Testing ScpTensor Refactoring\n")

    test_normalization_methods()
    print()
    test_matrix_ops()
    print()
    test_standardization()

    print("\n🎉 All tests passed! Refactoring successful!")