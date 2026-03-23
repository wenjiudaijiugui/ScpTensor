"""Regression tests for the lazy top-level package facade."""

from __future__ import annotations

import json
import subprocess
import sys


def _run_python(code: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_root_import_does_not_eager_load_heavy_optional_surfaces() -> None:
    payload = _run_python(
        """
import json
import sys

import scptensor  # noqa: F401

print(json.dumps({
    "scptensor_viz": "scptensor.viz" in sys.modules,
    "scptensor_integration": "scptensor.integration" in sys.modules,
    "matplotlib": any(name.startswith("matplotlib") for name in sys.modules),
    "seaborn": any(name.startswith("seaborn") for name in sys.modules),
    "sklearn": any(name.startswith("sklearn") for name in sys.modules),
}))
"""
    )

    assert payload == {
        "scptensor_viz": False,
        "scptensor_integration": False,
        "matplotlib": False,
        "seaborn": False,
        "sklearn": False,
    }


def test_root_attribute_access_lazily_loads_viz_and_integration_exports() -> None:
    payload = _run_python(
        """
import json
import sys

import scptensor as scp

_ = scp.plot_data_overview
after_viz = {
    "scptensor_viz": "scptensor.viz" in sys.modules,
    "matplotlib": any(name.startswith("matplotlib") for name in sys.modules),
}

_ = scp.integrate_limma
after_integration = {
    "scptensor_integration": "scptensor.integration" in sys.modules,
}

print(json.dumps({
    "after_viz": after_viz,
    "after_integration": after_integration,
}))
"""
    )

    assert payload == {
        "after_viz": {
            "scptensor_viz": True,
            "matplotlib": True,
        },
        "after_integration": {
            "scptensor_integration": True,
        },
    }
