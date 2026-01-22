#!/bin/bash
# Test script for PSM QC module

echo "Running PSM QC tests..."
cd /home/shenshang/projects/ScpTensor

# Run the quick test
uv run python run_psm_tests.py

echo ""
echo "Running pytest suite..."
uv run pytest tests/test_psm_qc.py -v
