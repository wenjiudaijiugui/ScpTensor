"""Regression tests for dependency boundary layout in pyproject.toml."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _dep_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~\\[]", requirement, maxsplit=1)[0].strip() for requirement in requirements
    }


def _load_pyproject() -> dict[str, object]:
    with Path("pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_default_runtime_dependencies_stay_aligned_to_stable_core() -> None:
    pyproject = _load_pyproject()
    project = pyproject["project"]
    dependencies = _dep_names(project["dependencies"])

    assert {
        "matplotlib",
        "numpy",
        "polars",
        "pyarrow",
        "scikit-learn",
        "scipy",
    }.issubset(dependencies)

    assert dependencies.isdisjoint(
        {
            "numba",
            "pandas",
            "plotly",
            "psutil",
            "requests",
            "igraph",
            "leidenalg",
            "scienceplots",
            "seaborn",
            "umap-learn",
            "h5py",
        }
    )


def test_optional_dependency_groups_capture_noncore_surfaces() -> None:
    pyproject = _load_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    assert {"anndata"} <= _dep_names(optional["io"])
    assert {"numba"} <= _dep_names(optional["accel"])
    assert {"harmonypy", "scanorama"} <= _dep_names(optional["integration"])
    assert {"seaborn", "scienceplots"} <= _dep_names(optional["viz"])
    assert {"umap-learn"} <= _dep_names(optional["experimental"])
    assert {"igraph", "leidenalg"} <= _dep_names(optional["graph"])
    assert {"psutil"} <= _dep_names(optional["perf"])
    assert {"pandas", "requests", "scanpy", "seaborn"} <= _dep_names(optional["benchmark"])
