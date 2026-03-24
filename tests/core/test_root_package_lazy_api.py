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
""",
    )

    assert payload == {
        "scptensor_viz": False,
        "scptensor_integration": False,
        "matplotlib": False,
        "seaborn": False,
        "sklearn": False,
    }


def test_root_package_surface_is_small_and_explicit() -> None:
    payload = _run_python(
        """
import json

import scptensor

print(json.dumps({"all": scptensor.__all__}))
""",
    )

    assert payload == {
        "all": [
            "__version__",
            "ScpContainer",
            "Assay",
            "ScpMatrix",
            "load_diann",
            "load_peptide_pivot",
            "load_spectronaut",
            "aggregate_to_protein",
        ],
    }


def test_root_attribute_access_only_lazily_loads_kept_exports() -> None:
    payload = _run_python(
        """
import json
import sys

import scptensor as scp

_ = scp.load_diann
after_io = {
    "scptensor_io": "scptensor.io" in sys.modules,
}

_ = scp.aggregate_to_protein
after_aggregation = {
    "scptensor_aggregation": "scptensor.aggregation" in sys.modules,
}

print(json.dumps({
    "after_io": after_io,
    "after_aggregation": after_aggregation,
}))
""",
    )

    assert payload == {
        "after_io": {
            "scptensor_io": True,
        },
        "after_aggregation": {
            "scptensor_aggregation": True,
        },
    }


def test_removed_root_exports_do_not_trigger_heavy_module_imports() -> None:
    payload = _run_python(
        """
import json
import sys

import scptensor as scp

try:
    _ = scp.plot_data_overview
except AttributeError:
    pass

try:
    _ = scp.integrate_limma
except AttributeError:
    pass

print(json.dumps({
    "has_plot_data_overview": hasattr(scp, "plot_data_overview"),
    "has_integrate_limma": hasattr(scp, "integrate_limma"),
    "scptensor_viz": "scptensor.viz" in sys.modules,
    "scptensor_integration": "scptensor.integration" in sys.modules,
    "matplotlib": any(name.startswith("matplotlib") for name in sys.modules),
}))
""",
    )

    assert payload == {
        "has_plot_data_overview": False,
        "has_integrate_limma": False,
        "scptensor_viz": False,
        "scptensor_integration": False,
        "matplotlib": False,
    }
