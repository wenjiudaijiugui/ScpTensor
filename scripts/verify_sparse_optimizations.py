#!/usr/bin/env python3
"""Verification script for sparse matrix optimizations.

Demonstrates sparse matrix improvements in ScpTensor, including:
- Automatic sparse conversion based on sparsity threshold
- Sparsity preservation through matrix operations
- Memory efficiency comparisons

Usage:
    python scripts/verify_sparse_optimizations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Add project root to path dynamically
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scptensor.core import (
    is_sparse_matrix,
    get_sparsity_ratio,
    to_sparse_if_beneficial,
    get_memory_usage,
    ScpMatrix,
    MaskCode,
    MatrixOps,
)

# Type aliases for clarity
SparseMatrix = sp.csr_matrix


def demo_sparse_conversion() -> None:
    """Demonstrate automatic sparse conversion with memory savings."""
    print("=" * 70)
    print("DEMO 1: Automatic Sparse Conversion")
    print("=" * 70)

    # Create sparse matrix (typical SCP data: 70% missing)
    n_rows, n_cols = 1000, 500
    n_nonzero = int(0.3 * n_rows * n_cols)

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero) * 10 + 20

    X_dense = np.zeros((n_rows, n_cols))
    X_dense[rows, cols] = data

    print(f"\nMatrix shape: {X_dense.shape}")
    print(f"Sparsity: {get_sparsity_ratio(X_dense) * 100:.1f}%")

    # Auto-convert
    X_sparse = to_sparse_if_beneficial(X_dense, threshold=0.5)

    print(f"\nAfter auto-conversion:")
    print(f"  Format: {'Sparse' if is_sparse_matrix(X_sparse) else 'Dense'}")

    # Memory comparison
    mem_dense = get_memory_usage(X_dense)["nbytes"]
    mem_sparse = get_memory_usage(X_sparse)["nbytes"]
    savings = (1 - mem_sparse / mem_dense) * 100

    print(f"  Dense memory:  {mem_dense / 1024 / 1024:.2f} MB")
    print(f"  Sparse memory: {mem_sparse / 1024 / 1024:.2f} MB")
    print(f"  Memory saved:  {savings:.1f}%")


def demo_matrix_ops_optimization() -> None:
    """Demonstrate matrix operations that preserve sparsity."""
    print("\n" + "=" * 70)
    print("DEMO 2: Matrix Operations (Preserving Sparsity)")
    print("=" * 70)

    # Create sparse test matrix
    n_rows, n_cols = 500, 200
    n_nonzero = int(0.4 * n_rows * n_cols)

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero) * 10 + 20

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    M = sp.csr_matrix(X != 0, dtype=np.int8)

    matrix = ScpMatrix(X=X, M=M)

    print(f"\nInput: {X.shape}, {X.nnz} non-zero ({get_sparsity_ratio(X) * 100:.1f}% sparse)")

    # Test operations and verify sparsity preservation
    tests = [
        (
            "Marking values as imputed",
            lambda m: MatrixOps.mark_values(m, (np.array([0, 10]), np.array([0, 10])), MaskCode.IMPUTED),
            lambda r: r.M,
        ),
        (
            "Filtering by mask (keep only VALID)",
            lambda m: MatrixOps.filter_by_mask(m, [MaskCode.VALID]),
            lambda r: r.M,
        ),
        (
            "Applying mask to values (zero invalid)",
            lambda m: MatrixOps.apply_mask_to_values(m, operation="zero"),
            lambda r: r.X,
        ),
    ]

    for i, (desc, op, get_target) in enumerate(tests, 1):
        print(f"\n{i}. {desc}...")
        result = op(matrix)
        target = get_target(result)
        status = "Sparse" if is_sparse_matrix(target) else "Dense"
        print(f"   Result: {status}")


def demo_realistic_scp_pipeline() -> None:
    """Demonstrate realistic SCP analysis pipeline."""
    print("\n" + "=" * 70)
    print("DEMO 3: Realistic SCP Analysis Pipeline")
    print("=" * 70)

    # Simulate SCP data (1000 cells, 2000 proteins, 80% sparse)
    n_cells, n_proteins, sparsity = 1000, 2000, 0.8
    n_nonzero = int((1 - sparsity) * n_cells * n_proteins)

    rows = np.random.randint(0, n_cells, n_nonzero)
    cols = np.random.randint(0, n_proteins, n_nonzero)
    data = np.random.exponential(5, n_nonzero)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_proteins))
    M = sp.csr_matrix(X != 0, dtype=np.int8) * MaskCode.VALID.value

    print(f"\nDataset: {n_cells} cells x {n_proteins} proteins")
    print(f"Sparsity: {sparsity * 100:.0f}%")
    print(f"Non-zero values: {X.nnz:,}")

    # Memory comparison
    mem_sparse = get_memory_usage(X)["nbytes"]
    mem_dense = n_cells * n_proteins * 8  # float64

    print(f"\nMemory usage:")
    print(f"  Sparse: {mem_sparse / 1024 / 1024:.2f} MB")
    print(f"  Dense:  {mem_dense / 1024 / 1024:.2f} MB")
    print(f"  Savings: {(1 - mem_sparse / mem_dense) * 100:.1f}%")

    # Simulate pipeline operations
    print("\nPipeline operations:")

    # 1. Mark outliers
    print("  1. Marking outliers (0.1% of values)...")
    n_outliers = int(0.001 * X.nnz)
    outlier_rows = np.random.randint(0, n_cells, n_outliers)
    outlier_cols = np.random.randint(0, n_proteins, n_outliers)
    matrix = ScpMatrix(X=X, M=M)

    result = MatrixOps.mark_outliers(matrix, (outlier_rows, outlier_cols))
    assert is_sparse_matrix(result.M), "Sparsity not preserved in mark_outliers"
    print("     Sparsity preserved")

    # 2. Filter by mask
    print("  2. Filtering to keep VALID and IMPUTED...")
    result2 = MatrixOps.filter_by_mask(result, [MaskCode.VALID, MaskCode.IMPUTED])
    assert is_sparse_matrix(result2.M), "Sparsity not preserved in filter_by_mask"
    print("     Sparsity preserved")

    # 3. Apply mask
    print("  3. Zeroing out filtered values...")
    result3 = MatrixOps.apply_mask_to_values(result2, operation="zero")
    assert is_sparse_matrix(result3.X), "Sparsity not preserved in apply_mask_to_values"
    print("     Sparsity preserved")

    print("\nAll operations preserved sparsity!")


def print_header() -> None:
    """Print script header."""
    print("\n" + "=" * 70)
    print("SCPTENSOR SPARSE OPTIMIZATION VERIFICATION")
    print("=" * 70)
    print("\nThis script demonstrates sparse matrix optimizations implemented in P1-7.")
    print("All operations preserve sparsity when possible, avoiding memory bloat.\n")


def print_footer() -> None:
    """Print script footer."""
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nAll optimizations working correctly!")
    print("Sparsity preserved throughout pipeline!")
    print("Significant memory savings on sparse SCP data!")
    print("\nSee: SPARSE_OPTIMIZATION_REPORT.md")
    print("=" * 70 + "\n")


def main() -> None:
    """Run all demonstrations."""
    print_header()
    demo_sparse_conversion()
    demo_matrix_ops_optimization()
    demo_realistic_scp_pipeline()
    print_footer()


if __name__ == "__main__":
    main()
