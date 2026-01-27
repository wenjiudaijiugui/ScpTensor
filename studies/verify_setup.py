#!/usr/bin/env python3
"""
Verification script to check if all required modules can be imported.

Run this script to verify the pipeline comparison study setup before
running the main experiment.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def verify_imports():
    """Verify all required modules can be imported."""
    print("\n" + "=" * 60)
    print("Pipeline Comparison Study - Setup Verification")
    print("=" * 60)

    all_success = True

    # Test 1: Import data module
    print("\n[1/5] Testing data module...")
    try:
        from studies import data  # noqa: F401

        print("✓ Data module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data module: {e}")
        all_success = False

    # Test 2: Import pipelines module
    print("\n[2/5] Testing pipelines module...")
    try:
        from studies.pipelines import (
            PipelineA,
            PipelineB,
            PipelineC,
            PipelineD,
            PipelineE,
        )

        print("✓ Pipelines module imported successfully")
        print(f"  - PipelineA: {PipelineA}")
        print(f"  - PipelineB: {PipelineB}")
        print(f"  - PipelineC: {PipelineC}")
        print(f"  - PipelineD: {PipelineD}")
        print(f"  - PipelineE: {PipelineE}")
    except ImportError as e:
        print(f"✗ Failed to import pipelines module: {e}")
        all_success = False

    # Test 3: Import evaluation module
    print("\n[3/5] Testing evaluation module...")
    try:
        from studies import evaluation  # noqa: F401

        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation module: {e}")
        all_success = False

    # Test 4: Import visualization module
    print("\n[4/5] Testing visualization module...")
    try:
        from studies import visualization  # noqa: F401

        print("✓ Visualization module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visualization module: {e}")
        all_success = False

    # Test 5: Import runner functions
    print("\n[5/5] Testing runner script...")
    try:
        from studies import run_comparison  # noqa: F401

        print("✓ Runner functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import runner functions: {e}")
        all_success = False

    # Test 6: Check configuration files
    print("\n[6/6] Checking configuration files...")
    base_dir = Path(__file__).parent
    configs_ok = True

    pipeline_config = base_dir / "configs" / "pipeline_configs.yaml"
    eval_config = base_dir / "configs" / "evaluation_config.yaml"

    if not pipeline_config.exists():
        print(f"✗ Missing: {pipeline_config}")
        configs_ok = False
    else:
        print(f"✓ Found: {pipeline_config}")

    if not eval_config.exists():
        print(f"✗ Missing: {eval_config}")
        configs_ok = False
    else:
        print(f"✓ Found: {eval_config}")

    if not configs_ok:
        all_success = False

    # Summary
    print("\n" + "=" * 60)
    if all_success:
        print("✓ ALL CHECKS PASSED - Setup is ready!")
        print("\nNext steps:")
        print(
            "  1. Run quick test: python studies/run_comparison.py --test --verbose"
        )
        print(
            "  2. Run full experiment: python studies/run_comparison.py --full --verbose"
        )
    else:
        print("✗ SOME CHECKS FAILED - Please fix errors above")
        print("\nCommon solutions:")
        print("  1. Ensure all dependencies are installed:")
        print("     uv pip install -e '.[dev]'")
        print("  2. Verify all modules are implemented")
        print("  3. Check configuration files exist")
    print("=" * 60 + "\n")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(verify_imports())
