#!/usr/bin/env python3
"""
Comprehensive benchmarks for sparse matrix operations.

This module benchmarks various sparse operations to verify that:
1. Sparse operations preserve sparsity (no unnecessary densification)
2. Sparse operations are faster than dense equivalents for sparse data
3. Memory usage is significantly reduced for sparse data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import scipy.sparse as sp
from scptensor.core.sparse_utils import (
    is_sparse_matrix,
    get_sparsity_ratio,
    get_memory_usage,
    sparse_multiply_rowwise,
    sparse_multiply_colwise,
    sparse_safe_log1p,
    optimal_format_for_operation,
    auto_convert_for_operation,
    sparse_row_operation,
    sparse_col_operation,
)
from scptensor.core.structures import ScpMatrix, MaskCode


def benchmark_sparse_vs_dense_memory():
    """Benchmark memory usage of sparse vs dense matrices."""
    print("\n" + "="*60)
    print("BENCHMARK: Sparse vs Dense Memory Usage")
    print("="*60)

    results = []

    for sparsity in [0.5, 0.7, 0.9, 0.95, 0.99]:
        n_rows, n_cols = 1000, 500
        n_elements = n_rows * n_cols
        n_nonzero = int((1 - sparsity) * n_elements)

        # Create sparse matrix
        rows = np.random.randint(0, n_rows, n_nonzero)
        cols = np.random.randint(0, n_cols, n_nonzero)
        data = np.random.randn(n_nonzero).astype(np.float64)

        X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        X_dense = X_sparse.toarray()

        mem_sparse = get_memory_usage(X_sparse)
        mem_dense = get_memory_usage(X_dense)

        compression_ratio = mem_dense['nbytes'] / mem_sparse['nbytes']

        print(f"\nSparsity: {sparsity*100:.0f}%")
        print(f"  Sparse: {mem_sparse['nbytes'] / 1024 / 1024:.2f} MB")
        print(f"  Dense:  {mem_dense['nbytes'] / 1024 / 1024:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.1f}x")

        results.append({
            'sparsity': sparsity,
            'compression_ratio': compression_ratio,
        })

    return results


def benchmark_sparse_multiply_rowwise():
    """Benchmark row-wise multiplication for sparse vs dense."""
    print("\n" + "="*60)
    print("BENCHMARK: Row-wise Multiplication (Sparse vs Dense)")
    print("="*60)

    results = []

    for sparsity in [0.7, 0.9, 0.95]:
        n_rows, n_cols = 1000, 500
        n_nonzero = int((1 - sparsity) * n_rows * n_cols)

        rows = np.random.randint(0, n_rows, n_nonzero)
        cols = np.random.randint(0, n_cols, n_nonzero)
        data = np.random.randn(n_nonzero).astype(np.float64)

        X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        X_dense = X_sparse.toarray()
        factors = np.random.randn(n_rows).astype(np.float64)

        # Benchmark sparse
        start = time.time()
        for _ in range(100):
            _ = sparse_multiply_rowwise(X_sparse, factors)
        time_sparse = time.time() - start

        # Benchmark dense
        start = time.time()
        for _ in range(100):
            _ = X_dense * factors[:, np.newaxis]
        time_dense = time.time() - start

        speedup = time_dense / time_sparse

        print(f"\nSparsity: {sparsity*100:.0f}%")
        print(f"  Sparse: {time_sparse*1000:.2f} ms")
        print(f"  Dense:  {time_dense*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        results.append({
            'sparsity': sparsity,
            'sparse_time': time_sparse,
            'dense_time': time_dense,
            'speedup': speedup,
        })

    return results


def benchmark_sparse_log_transform():
    """Benchmark log transformation for sparse vs dense."""
    print("\n" + "="*60)
    print("BENCHMARK: Log Transformation (Sparse vs Dense)")
    print("="*60)

    results = []

    for sparsity in [0.7, 0.9, 0.95]:
        n_rows, n_cols = 1000, 500
        n_nonzero = int((1 - sparsity) * n_rows * n_cols)

        rows = np.random.randint(0, n_rows, n_nonzero)
        cols = np.random.randint(0, n_cols, n_nonzero)
        data = np.abs(np.random.randn(n_nonzero)).astype(np.float64) + 0.1

        X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        X_dense = X_sparse.toarray()

        # Benchmark sparse (only transforms non-zero elements)
        start = time.time()
        for _ in range(100):
            _ = sparse_safe_log1p(X_sparse)
        time_sparse = time.time() - start

        # Benchmark dense (transforms all elements)
        start = time.time()
        for _ in range(100):
            _ = np.log1p(X_dense)
        time_dense = time.time() - start

        speedup = time_dense / time_sparse

        print(f"\nSparsity: {sparsity*100:.0f}%")
        print(f"  Sparse: {time_sparse*1000:.2f} ms (only {n_nonzero} elements)")
        print(f"  Dense:  {time_dense*1000:.2f} ms (all {n_rows*n_cols} elements)")
        print(f"  Speedup: {speedup:.2f}x")

        results.append({
            'sparsity': sparsity,
            'sparse_time': time_sparse,
            'dense_time': time_dense,
            'speedup': speedup,
            'elements_processed': f"{n_nonzero} vs {n_rows*n_cols}",
        })

    return results


def benchmark_format_conversion():
    """Benchmark sparse format conversion overhead."""
    print("\n" + "="*60)
    print("BENCHMARK: Format Conversion Overhead")
    print("="*60)

    n_rows, n_cols = 1000, 500
    n_nonzero = int(0.1 * n_rows * n_cols)

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero).astype(np.float64)

    # Create in different formats
    X_csr = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    X_csc = sp.csc_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    X_coo = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Benchmark conversions
    conversions = [
        ('CSR to CSC', X_csr, 'csc'),
        ('CSC to CSR', X_csc, 'csr'),
        ('COO to CSR', X_coo, 'csr'),
        ('COO to CSC', X_coo, 'csc'),
    ]

    for name, X, target_fmt in conversions:
        start = time.time()
        for _ in range(1000):
            if target_fmt == 'csr':
                _ = X.tocsr()
            else:
                _ = X.tocsc()
        elapsed = time.time() - start

        print(f"  {name}: {elapsed*1000:.2f} ms (1000 conversions)")

    return {}


def benchmark_row_col_operations():
    """Benchmark sparse row and column operations."""
    print("\n" + "="*60)
    print("BENCHMARK: Row/Column Operations")
    print("="*60)

    n_rows, n_cols = 1000, 500
    n_nonzero = int(0.1 * n_rows * n_cols)

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero).astype(np.float64)

    X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    X_dense = X_sparse.toarray()

    # Benchmark row sum (sparse)
    start = time.time()
    for _ in range(100):
        _ = sparse_row_operation(X_sparse, np.sum)
    time_sparse_row = time.time() - start

    # Benchmark row sum (dense)
    start = time.time()
    for _ in range(100):
        _ = np.sum(X_dense, axis=1)
    time_dense_row = time.time() - start

    print(f"  Row sum (sparse): {time_sparse_row*1000:.2f} ms")
    print(f"  Row sum (dense):  {time_dense_row*1000:.2f} ms")
    print(f"  Speedup: {time_dense_row/time_sparse_row:.2f}x")

    # Benchmark col sum (sparse)
    start = time.time()
    for _ in range(100):
        _ = sparse_col_operation(X_sparse, np.sum)
    time_sparse_col = time.time() - start

    # Benchmark col sum (dense)
    start = time.time()
    for _ in range(100):
        _ = np.sum(X_dense, axis=0)
    time_dense_col = time.time() - start

    print(f"  Col sum (sparse): {time_sparse_col*1000:.2f} ms")
    print(f"  Col sum (dense):  {time_dense_col*1000:.2f} ms")
    print(f"  Speedup: {time_dense_col/time_sparse_col:.2f}x")

    return {}


def benchmark_auto_convert():
    """Benchmark automatic format conversion for operations."""
    print("\n" + "="*60)
    print("BENCHMARK: Auto Convert for Operation")
    print("="*60)

    n_rows, n_cols = 1000, 500
    n_nonzero = int(0.1 * n_rows * n_cols)

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero).astype(np.float64)

    X_csc = sp.csc_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Benchmark auto convert for row operation
    start = time.time()
    for _ in range(1000):
        _ = auto_convert_for_operation(X_csc, 'row_wise')
    time_auto = time.time() - start

    # Benchmark manual convert
    start = time.time()
    for _ in range(1000):
        _ = X_csc.tocsr()
    time_manual = time.time() - start

    print(f"  Auto convert (1000x): {time_auto*1000:.2f} ms")
    print(f"  Manual tocsr (1000x): {time_manual*1000:.2f} ms")
    print(f"  Overhead: {(time_auto/time_manual - 1)*100:.1f}%")

    return {}


def verify_no_densification():
    """Verify that operations don't densify sparse matrices."""
    print("\n" + "="*60)
    print("VERIFICATION: No Densification Test")
    print("="*60)

    n_rows, n_cols = 1000, 500
    n_nonzero = int(0.1 * n_rows * n_cols)

    rows = np.random.randint(0, n_rows, n_nonzero)
    cols = np.random.randint(0, n_cols, n_nonzero)
    data = np.random.randn(n_nonzero).astype(np.float64)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Track memory before
    mem_before = get_memory_usage(X)

    # Apply operations
    X1 = sparse_multiply_rowwise(X, np.ones(n_rows))
    X2 = sparse_safe_log1p(X.copy())
    X3 = auto_convert_for_operation(X, 'row_wise')

    # Check memory after each operation
    mem1 = get_memory_usage(X1)
    mem2 = get_memory_usage(X2)
    mem3 = get_memory_usage(X3)

    print(f"  Original memory: {mem_before['nbytes'] / 1024:.2f} KB")
    print(f"  After multiply:  {mem1['nbytes'] / 1024:.2f} KB ({mem1['nbytes']/mem_before['nbytes']:.2f}x)")
    print(f"  After log1p:     {mem2['nbytes'] / 1024:.2f} KB ({mem2['nbytes']/mem_before['nbytes']:.2f}x)")
    print(f"  After convert:   {mem3['nbytes'] / 1024:.2f} KB ({mem3['nbytes']/mem_before['nbytes']:.2f}x)")

    # Verify all are still sparse
    assert is_sparse_matrix(X1), "multiply_rowwise should preserve sparsity"
    assert is_sparse_matrix(X2), "sparse_safe_log1p should preserve sparsity"
    assert is_sparse_matrix(X3), "auto_convert should preserve sparsity"

    # Check memory ratio (should be similar, within 2x)
    assert mem1['nbytes'] / mem_before['nbytes'] < 2.0, "Memory increased too much"
    assert mem2['nbytes'] / mem_before['nbytes'] < 2.0, "Memory increased too much"
    assert mem3['nbytes'] / mem_before['nbytes'] < 2.0, "Memory increased too much"

    print("\n  All operations preserve sparsity!")

    return {}


def run_all_benchmarks():
    """Run all benchmarks and return summary."""
    print("\n" + "="*60)
    print("SCP TENSOR: SPARSE OPERATIONS BENCHMARK SUITE")
    print("="*60)

    results = {}

    results['memory'] = benchmark_sparse_vs_dense_memory()
    results['multiply'] = benchmark_sparse_multiply_rowwise()
    results['log'] = benchmark_sparse_log_transform()
    results['conversion'] = benchmark_format_conversion()
    results['rowcol'] = benchmark_row_col_operations()
    results['auto_convert'] = benchmark_auto_convert()
    results['no_densification'] = verify_no_densification()

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    avg_compression = np.mean([r['compression_ratio'] for r in results['memory']])
    avg_multiply_speedup = np.mean([r['speedup'] for r in results['multiply']])
    avg_log_speedup = np.mean([r['speedup'] for r in results['log']])

    print(f"\nAverage memory compression: {avg_compression:.1f}x")
    print(f"Average multiply speedup: {avg_multiply_speedup:.2f}x")
    print(f"Average log transform speedup: {avg_log_speedup:.2f}x")

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    run_all_benchmarks()
