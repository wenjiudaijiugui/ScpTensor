# ScpTensor Migration Guide
## v0.1.0-alpha → v0.1.0-beta

**For:** All users currently using v0.1.0-alpha
**Migration Difficulty:** Medium (1-2 hours)
**Data Compatibility:** ⚠️ Some breaking changes
**Last Updated:** 2025-01-05

---

## Quick Start

**Estimated Time:** 1-2 hours
**Prerequisites:** Backup current environment and code

**3-Step Migration:**
1. Update environment: `uv pip install --upgrade scptensor`
2. Update import statements in your code
3. Validate with provided test script

---

## Checklist

### Pre-Migration
- [ ] Backup current virtual environment
- [ ] Backup custom code using ScpTensor
- [ ] Note down all ScpTensor APIs used
- [ ] Read breaking changes section

### Post-Migration
- [ ] All import statements updated
- [ ] Code runs without errors
- [ ] Saved data objects load correctly
- [ ] Analysis results match (allow numerical tolerance)

---

## Breaking Changes

### 1. Module Import Paths

#### Change
Previously required direct file imports due to missing `__init__.py`:
```python
# v0.1.0-alpha (workaround)
from scptensor.integration.combat import combat
from scptensor.qc.basic import basic_qc
from scptensor.impute.knn import knn
```

Now use standard imports:
```python
# v0.1.0-beta (correct way)
from scptensor.integration import combat
from scptensor.qc import basic_qc
from scptensor.impute import knn
```

#### Migration Steps

**Step 1:** Find all direct imports
```bash
grep -r "from scptensor\." --include="*.py" . > imports_audit.txt
```

**Step 2:** Replace imports
```python
# Replace patterns:
from scptensor.integration.combat import combat
↓
from scptensor.integration import combat

from scptensor.qc.basic import basic_qc
↓
from scptensor.qc import basic_qc

from scptensor.impute.knn import knn
↓
from scptensor.impute import knn
```

**Step 3:** Verify imports
```bash
python -c "from scptensor.integration import combat; print('✅')"
python -c "from scptensor.qc import basic_qc; print('✅')"
python -c "from scptensor.impute import knn; print('✅')"
```

---

### 2. Function Signatures (Type Enforcement)

#### Change
Functions now have strict type annotations and validation:

```python
# v0.1.0-alpha (no types, silent failures)
def combat(container, batch_key, assay_name, base_layer, new_layer_name):
    pass

# v0.1.0-beta (type-safe, validates input)
def combat(
    container: ScpContainer,
    batch_key: str,
    assay_name: str,
    base_layer: str,
    new_layer_name: str
) -> ScpContainer:
    # Validates: container is ScpContainer
    # Validates: batch_key exists in obs
    # Validates: assay exists
    # Validates: layers exist
    ...
```

#### Impact
- Type mismatches now raise clear errors
- Invalid parameters detected early
- Better IDE support (autocomplete, type checking)

#### Migration Steps

**Step 1:** Run your code with v0.1.0-beta
```bash
python your_analysis.py
```

**Step 2:** Fix type errors as they appear
```python
# Example error:
# TypeError: batch_key must be str, got int

# Fix:
container = combat(container, batch_key="batch", ...)  # Correct
```

---

### 3. Error Handling

#### Change
Silent failures replaced with explicit exceptions:

```python
# v0.1.0-alpha (ignores invalid method)
container.normalize(method='invalid')
# Does nothing, hard to debug

# v0.1.0-beta (raises clear error)
container.normalize(method='invalid')
# ValueError: Invalid normalization method 'invalid'.
# Valid options: ['log', 'median', 'tmm', 'zscore', 'mean', 'upper_quartile']
```

#### Migration Steps

**Step 1:** Check for invalid parameters in your code
```python
# Review all function calls
normalize(container, method=...)
combat(container, batch_key=...)
knn(container, k=...)
```

**Step 2:** Update to valid parameters
```python
# Old (deprecated)
container.normalize(method='median_centering')

# New (correct)
container.normalize(method='median')
# OR use sample_median_normalization() directly
```

---

### 4. Dependency Versions

#### Minimum Version Requirements

```toml
# v0.1.0-beta minimum versions
polars >= 1.35.2      # (up from 1.0.0)
numpy >= 2.3.5        # (up from 1.24.0)
scipy >= 1.16.3       # (up from 1.10.0)
```

#### Migration Steps

**Step 1:** Update dependencies
```bash
uv pip install --upgrade scptensor
```

**Step 2:** Verify versions
```bash
python -c "import polars; print(polars.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import scipy; print(scipy.__version__)"
```

---

## Data Compatibility

### ScpContainer Serialization

#### Compatibility: Partially Compatible

```python
# Saving in v0.1.0-alpha
import pickle
with open('container_alpha.pkl', 'wb') as f:
    pickle.dump(container, f)

# Loading in v0.1.0-beta
with open('container_alpha.pkl', 'rb') as f:
    container = pickle.load(f)  # ✅ Works for base structures
```

#### New Fields (Backward Compatible)

```python
# New in v0.1.0-beta
container.history        # Operation log (was empty/None before)
container.assays['X'].metadata  # Quality metadata (new)

# Handle missing fields gracefully
if not hasattr(container, 'history'):
    container.history = []
```

#### Layer Naming

Layer naming convention now enforced:
```python
# Recommended names
'raw'       # Original data
'log'       # Log-transformed
'imputed'   # After imputation
'corrected' # After batch correction
'scaled'    # After scaling

# Your custom names still work
'my_custom_layer'  # ✅ Allowed
```

---

## Migration Strategies

### Strategy 1: Incremental Migration (Recommended)

**Best for:** Large projects, teams, production code

**Steps:**

1. **Create migration branch**
```bash
git checkout -b feature/migrate-to-beta
```

2. **Install v0.1.0-beta in new environment**
```bash
uv venv .venv-beta
source .venv-beta/bin/activate
uv pip install scptensor==0.1.0b1
```

3. **Migrate one module at a time**
```bash
# Start with non-critical scripts
# Test each module before proceeding
```

4. **Run test suite after each change**
```bash
pytest tests/unit/ -v
```

5. **Validate results match**
```python
# Compare old vs new results
compare_results(old_container, new_container)
```

6. **Merge when all tests pass**
```bash
git checkout main
git merge feature/migrate-to-beta
```

---

### Strategy 2: Parallel Validation

**Best for:** Risk-averse projects, scientific reproducibility

**Steps:**

1. **Keep v0.1.0-alpha environment**
```bash
# Old environment stays intact
source .venv-alpha/bin/activate
```

2. **Run analysis, save results**
```python
# v0.1.0-alpha
container_alpha = run_analysis(...)
save_results(container_alpha, 'results_alpha.npz')
```

3. **Repeat in v0.1.0-beta**
```bash
source .venv-beta/bin/activate
```

```python
# v0.1.0-beta
container_beta = run_analysis(...)
save_results(container_beta, 'results_beta.npz')
```

4. **Compare results**
```python
# Validation script
import numpy as np

X_alpha = load_results('results_alpha.npz')
X_beta = load_results('results_beta.npz')

# Allow numerical tolerance
if np.allclose(X_alpha, X_beta, rtol=1e-5, atol=1e-8):
    print("✅ Results match within tolerance")
else:
    max_diff = np.abs(X_alpha - X_beta).max()
    print(f"⚠️  Max difference: {max_diff}")
    print("Review differences before deploying")
```

---

### Strategy 3: Complete Rewrite

**Best for:** Small projects, new users, code needing refactoring

**Advantages:**
- Clean code using latest APIs
- Type safety from start
- Best practices built-in

**Disadvantages:**
- Higher upfront effort
- Risk of introducing bugs

**When to Use:**
- Codebase < 1000 lines
- Project not in production
- Learning opportunity for team

---

## Step-by-Step Migration

### Step 1: Environment Setup (15 min)

```bash
# 1. Backup current environment
uv pip freeze > requirements_alpha.txt
cp -r .venv .venv_backup

# 2. Uninstall old version
uv pip uninstall scptensor

# 3. Install v0.1.0-beta
uv pip install scptensor==0.1.0b1

# 4. Verify installation
python -c "import scptensor; print(scptensor.__version__)"
# Expected: 0.1.0b1
```

---

### Step 2: Code Audit (30 min)

```bash
# 1. Find all ScpTensor imports
grep -r "from scptensor" --include="*.py" . > imports.txt

# 2. Find all direct submodule imports
grep -r "from scptensor\." --include="*.py" . > direct_imports.txt

# 3. Audit the files
cat imports.txt
cat direct_imports.txt
```

**Checklist:**
- [ ] List all modules used (normalization, impute, integration, etc.)
- [ ] Identify direct submodule imports (need fixing)
- [ ] Note any internal API usage (may need updates)
- [ ] Check for deprecated function names

---

### Step 3: Update Imports (15 min)

**Automated replacement script:**

```python
import re
from pathlib import Path

def fix_imports(file_path):
    """Update import statements to v0.1.0-beta style"""
    content = file_path.read_text()

    replacements = [
        # Integration module
        (r'from scptensor\.integration\.combat import',
         'from scptensor.integration import'),
        (r'from scptensor\.integration\.harmony import',
         'from scptensor.integration import'),

        # QC module
        (r'from scptensor\.qc\.basic import',
         'from scptensor.qc import'),
        (r'from scptensor\.qc\.outlier import',
         'from scptensor.qc import'),

        # Impute module
        (r'from scptensor\.impute\.knn import',
         'from scptensor.impute import'),
        (r'from scptensor\.impute\.ppca import',
         'from scptensor.impute import'),
        (r'from scptensor\.impute\.svd import',
         'from scptensor.impute import'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    file_path.write_text(content)
    print(f"✅ Fixed: {file_path}")

# Apply to all Python files
for py_file in Path('.').rglob('*.py'):
    if 'venv' not in str(py_file):
        try:
            fix_imports(py_file)
        except Exception as e:
            print(f"❌ Error in {py_file}: {e}")
```

---

### Step 4: Run Tests (30 min)

```bash
# 1. Run unit tests
pytest tests/unit/ -v

# 2. Run integration tests
pytest tests/integration/ -v

# 3. Run your analysis scripts
python scripts/my_analysis.py
```

**Common Errors and Fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'scptensor.integration.combat'` | Old import path | Change to `from scptensor.integration import combat` |
| `TypeError: 'str' object cannot be interpreted as an integer` | Type checking now strict | Ensure parameter types match signature |
| `ValueError: Invalid normalization method` | Deprecated method name | Use correct method name |
| `AttributeError: 'ScpContainer' object has no attribute 'history'` | Old data format | Re-create container from raw data |

---

### Step 5: Data Validation (15 min)

```python
# Test saved data compatibility
import pickle
import numpy as np

# Load old container
with open('my_container.pkl', 'rb') as f:
    container = pickle.load(f)

# Validate structure
try:
    assert container.n_samples > 0, "No samples"
    assert len(container.assays) > 0, "No assays"
    print("✅ Container structure valid")
except AssertionError as e:
    print(f"❌ Validation error: {e}")
    print("Re-create container from raw data")

# Check data integrity
for assay_name, assay in container.assays.items():
    for layer_name, layer in assay.layers.items():
        X = layer.X
        if np.any(np.isnan(X)):
            print(f"⚠️  NaN found in {assay_name}.{layer_name}")
        if np.any(np.isinf(X)):
            print(f"⚠️  Inf found in {assay_name}.{layer_name}")

print("✅ Data validation complete")
```

---

### Step 6: Performance Check (15 min)

```python
import time

# Benchmark key operations
start = time.time()
container = log_normalize(container, 'raw', 'log')
normalize_time = time.time() - start
print(f"Normalization: {normalize_time:.2f}s")

start = time.time()
container = knn(container, 'protein', 'log', 'imputed', k=5)
knn_time = time.time() - start
print(f"KNN imputation: {knn_time:.2f}s")

# Expected improvements:
# Normalization: ~10% faster
# KNN: ~20% faster (with Numba JIT)
```

---

## Rollback Procedure

If migration fails:

```bash
# 1. Uninstall v0.1.0-beta
uv pip uninstall scptensor

# 2. Restore v0.1.0-alpha environment
rm -rf .venv
cp -r .venv_backup .venv
source .venv/bin/activate

# 3. Reinstall v0.1.0-alpha
uv pip install -r requirements_alpha.txt

# 4. Restore code from git
git checkout main
git branch -D feature/migrate-to-beta

# 5. Verify rollback
pytest tests/  # Should pass
python scripts/my_analysis.py  # Should work
```

---

## FAQ

### Q1: My saved ScpContainer won't load in v0.1.0-beta

**A:** Check for new optional fields:

```python
# Handle missing fields
if not hasattr(container, 'history'):
    container.history = []

# Re-attach metadata if missing
for assay in container.assays.values():
    for layer in assay.layers.values():
        if layer.metadata is None:
            layer.metadata = MatrixMetadata()
```

---

### Q2: Migration is slower than expected

**A:** v0.1.0-beta should be faster, not slower. Check:
1. Numba is installed and working: `python -c "import numba; print(numba.__version__)"`
2. Using sparse matrices for sparse data
3. Run profiler: `python -m cProfile -o profile.stats your_script.py`

---

### Q3: Function names have changed

**A:** Check the [API_REFERENCE.md](./API_REFERENCE.md) for current function names. Common renames:
- `median_centering()` → `sample_median_normalization()`
- `global_scaling()` → `global_median_normalization()`

Old names may still work but are deprecated.

---

### Q4: How long until v0.1.0-alpha is deprecated?

**A:** v0.1.0-alpha will be supported for 6 months after v0.1.0-beta release. After that, security updates only. Plan to migrate within 3 months.

---

## Getting Help

### Resources

- **Documentation:** [API_REFERENCE.md](./API_REFERENCE.md)
- **Architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Roadmap:** [ROADMAP.md](./ROADMAP.md)
- **Issues:** [GitHub Issues](https://github.com/yourorg/scptensor/issues)

### Reporting Migration Problems

When reporting issues, include:

1. ScpTensor version:
```bash
python -c "import scptensor; print(scptensor.__version__)"
```

2. Error message and stack trace:
```bash
python your_script.py 2>&1 | tee error.log
```

3. Minimal reproducible code:
```python
# Shortest code that reproduces the issue
from scptensor.core import ScpContainer
# ... your code ...
```

4. System info:
```bash
python -m sysconfig
```

---

## Success Criteria

Migration is successful when:

- [ ] All import statements updated
- [ ] Code runs without errors
- [ ] Tests pass (pytest)
- [ ] Saved data loads correctly
- [ ] Results match v0.1.0-alpha (within numerical tolerance)
- [ ] Performance is equal or better

---

## Appendix

### A. Complete Import Mapping

| Old Import (v0.1.0-alpha) | New Import (v0.1.0-beta) |
|---------------------------|--------------------------|
| `from scptensor.integration.combat import combat` | `from scptensor.integration import combat` |
| `from scptensor.qc.basic import basic_qc` | `from scptensor.qc import basic_qc` |
| `from scptensor.impute.knn import knn` | `from scptensor.impute import knn` |
| `from scptensor.impute.p pca import ppca` | `from scptensor.impute import ppca` |
| `from scptensor.normalization.log import log_normalize` | `from scptensor.normalization import log_normalize` |

---

### B. Type Annotation Example

```python
# v0.1.0-beta enforces types
from typing import Optional
import numpy as np
from scptensor.core import ScpContainer

def my_function(
    container: ScpContainer,
    assay_name: str,
    layer: str,
    param: Optional[float] = None
) -> ScpContainer:
    """
    Your custom function with type hints.

    Parameters
    ----------
    container : ScpContainer
        Input container
    assay_name : str
        Assay to process
    layer : str
        Layer to use
    param : float, optional
        Your parameter

    Returns
    -------
    ScpContainer
        Processed container
    """
    # Implementation here
    return container
```

---

### C. Validation Script

```python
#!/usr/bin/env python
"""
validate_migration.py - Validate v0.1.0-beta migration
"""
import sys
import numpy as np
from scptensor.core import ScpContainer

def test_imports():
    """Test all imports work"""
    try:
        from scptensor.integration import combat
        from scptensor.qc import basic_qc
        from scptensor.impute import knn
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_container_creation():
    """Test container creation"""
    try:
        import polars as pl
        from scptensor.core import ScpContainer, Assay, ScpMatrix

        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["P1", "P2"]})
        X = np.random.rand(3, 2)
        matrix = ScpMatrix(X=X)
        assay = Assay(var=var, layers={"X": matrix})
        container = ScpContainer(obs=obs, assays={"test": assay})

        print("✅ Container creation successful")
        return True
    except Exception as e:
        print(f"❌ Container error: {e}")
        return False

def main():
    """Run all validation tests"""
    tests = [
        test_imports,
        test_container_creation,
    ]

    results = [test() for test in tests]

    if all(results):
        print("\n✅ All validation tests passed!")
        print("Migration successful.")
        return 0
    else:
        print("\n❌ Some validation tests failed")
        print("Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run validation:
```bash
python validate_migration.py
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-05
**Next Review:** v0.1.0-beta release

**End of MIGRATION.md**
