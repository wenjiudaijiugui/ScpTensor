"""Smoke tests for runtime baseline tooling."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_runtime_baseline_quick_smoke(tmp_path: Path) -> None:
    script = Path("scripts/perf/run_runtime_baseline.py")
    output_dir = tmp_path / "runtime_baseline"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "quick",
            "--scenario",
            "import_diann_protein_long",
            "--scenario",
            "stable_chain_dense",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    stage_runs = output_dir / "stage_runs.csv"
    scenario_summary = output_dir / "scenario_summary.json"
    environment = output_dir / "environment.json"
    errors = output_dir / "errors.json"

    assert stage_runs.exists()
    assert scenario_summary.exists()
    assert environment.exists()
    assert errors.exists()

    with stage_runs.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert any(row["scenario"] == "import_diann_protein_long" for row in rows)
    assert any(row["scenario"] == "stable_chain_dense" for row in rows)
    assert any(row["stage"] == "log_transform" for row in rows)

    with scenario_summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert any(item["scenario"] == "stable_chain_dense" for item in summary)

    with errors.open("r", encoding="utf-8") as handle:
        error_rows = json.load(handle)
    assert error_rows == []
