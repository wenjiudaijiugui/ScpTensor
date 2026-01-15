ScpTensor vs Scanpy - Phase 1 Comparison Report

## Test Results

| Module | Status | Pearson Correlation | Speedup |
|--------|--------|---------------------|---------|
| Normalization (log1p) | ✅ PASS | 1.0000 | 0.10x (Scanpy faster) |
| PCA (50 components) | ✅ PASS | 1.0000 | **1.70x (ScpTensor faster)** |
| UMAP | ⚠️ Known Difference | 0.24 | 0.60x (Scanpy faster) |

## Analysis

### Normalization
- **Perfect numerical agreement** (r = 1.0)
- ScpTensor correctly implements log1p transformation
- Scanpy is ~10x faster due to C-optimized internals

### PCA
- **Perfect numerical agreement** (r = 1.0)
- **ScpTensor is 1.7x faster** than Scanpy!
- Variance explained: r = 1.0

### UMAP
- Low correlation is **expected behavior**:
  - Scanpy uses pre-computed neighbor graph (sc.pp.neighbors)
  - Direct UMAP recomputes neighbors internally
  - Different computation paths → different results even with same seed
- This is a known algorithmic difference, not a bug

## Conclusion

ScpTensor achieves **numerical parity with Scanpy** for deterministic operations (normalization, PCA).
The PCA implementation is **70% faster** while maintaining identical results.

Generated: Tue Jan 13 22:26:30 CST 2026

