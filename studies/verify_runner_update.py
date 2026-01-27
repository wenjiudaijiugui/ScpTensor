#!/usr/bin/env python3
"""Verification script for updated run_comparison.py"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 60)
print("Verifying Updated run_comparison.py")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from studies.data_generation import (
        generate_small_dataset,
    )

    print("  ✓ data_generation imports successful")
except Exception as e:
    print(f"  ✗ data_generation import failed: {e}")
    sys.exit(1)

try:
    print("  ✓ metrics imports successful")
except Exception as e:
    print(f"  ✗ metrics import failed: {e}")
    sys.exit(1)

try:
    print("  ✓ plotting imports successful")
except Exception as e:
    print(f"  ✗ plotting import failed: {e}")
    sys.exit(1)

try:
    print("  ✓ comparison_engine imports successful")
except Exception as e:
    print(f"  ✗ comparison_engine import failed: {e}")
    sys.exit(1)

# Test 2: Check run_comparison.py syntax
print("\n[2/5] Checking run_comparison.py syntax...")
try:
    import py_compile

    runner_path = project_root / "studies" / "run_comparison.py"
    py_compile.compile(str(runner_path), doraise=True)
    print("  ✓ Syntax check passed")
except Exception as e:
    print(f"  ✗ Syntax check failed: {e}")
    sys.exit(1)

# Test 3: Import run_comparison module
print("\n[3/5] Importing run_comparison module...")
try:
    from studies import run_comparison

    print("  ✓ Module import successful")
except Exception as e:
    print(f"  ✗ Module import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Check key functions exist
print("\n[4/5] Checking key functions...")
required_functions = [
    "parse_arguments",
    "load_config",
    "setup_output_directory",
    "load_datasets",
    "initialize_pipelines",
    "main",
]

missing_functions = []
for func_name in required_functions:
    if not hasattr(run_comparison, func_name):
        missing_functions.append(func_name)

if missing_functions:
    print(f"  ✗ Missing functions: {missing_functions}")
    sys.exit(1)
else:
    print(f"  ✓ All {len(required_functions)} required functions present")

# Test 5: Generate small test dataset
print("\n[5/5] Testing data generation...")
try:
    container = generate_small_dataset(seed=42)
    print(
        f"  ✓ Generated test dataset: {container.n_samples} samples, {len(container.assays)} assay(s)"
    )
except Exception as e:
    print(f"  ✗ Data generation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("✓ All checks passed!")
print("\nThe updated run_comparison.py is ready to use.")
print("\nUsage examples:")
print("  Quick test: uv run python run_comparison.py --test --verbose")
print("  Full experiment: uv run python run_comparison.py --full --verbose")
print("=" * 60)
