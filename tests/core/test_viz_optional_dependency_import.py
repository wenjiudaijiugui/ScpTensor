"""Regression tests for optional visualization dependencies."""

from __future__ import annotations

import json
import subprocess
import sys


def test_stable_viz_import_succeeds_without_seaborn_or_scienceplots() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import json

real_import = builtins.__import__

def guarded(name, *args, **kwargs):
    if name in {"seaborn", "scienceplots"}:
        raise ImportError(f"blocked {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded
try:
    import scptensor.viz as viz
    payload = {
        "has_plot_data_overview": hasattr(viz, "plot_data_overview"),
        "has_plot_imputation_comparison": hasattr(viz, "plot_imputation_comparison"),
        "has_plot_sensitivity_summary": hasattr(viz, "plot_sensitivity_summary"),
    }
finally:
    builtins.__import__ = real_import

print(json.dumps(payload))
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "has_plot_data_overview": True,
        "has_plot_imputation_comparison": True,
        "has_plot_sensitivity_summary": False,
    }
