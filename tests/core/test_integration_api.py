"""Tests for stable integration namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.integration as stable_integration
from scptensor.integration import (
    integrate,
    integrate_combat,
    integrate_harmony,
    integrate_limma,
    integrate_mnn,
    integrate_none,
    integrate_scanorama,
)
from scptensor.integration.base import IntegrateMethod as IntegrateMethodCore
from scptensor.integration.base import IntegrationMethodInfo as IntegrationMethodInfoCore
from scptensor.integration.base import get_integrate_method as get_integrate_method_core
from scptensor.integration.base import (
    get_integrate_method_info as get_integrate_method_info_core,
)
from scptensor.integration.base import integrate as integrate_core
from scptensor.integration.base import (
    list_integrate_method_info as list_integrate_method_info_core,
)
from scptensor.integration.base import list_integrate_methods as list_integrate_methods_core
from scptensor.integration.base import register_integrate_method as register_integrate_method_core
from scptensor.integration.combat import integrate_combat as integrate_combat_core
from scptensor.integration.diagnostics import compute_batch_asw as compute_batch_asw_core
from scptensor.integration.diagnostics import (
    compute_batch_mixing_metric as compute_batch_mixing_metric_core,
)
from scptensor.integration.diagnostics import compute_ilisi as compute_ilisi_core
from scptensor.integration.diagnostics import compute_kbet as compute_kbet_core
from scptensor.integration.diagnostics import compute_lisi_approx as compute_lisi_approx_core
from scptensor.integration.diagnostics import (
    integration_quality_report as integration_quality_report_core,
)
from scptensor.integration.harmony import integrate_harmony as integrate_harmony_core
from scptensor.integration.limma import integrate_limma as integrate_limma_core
from scptensor.integration.mnn import integrate_mnn as integrate_mnn_core
from scptensor.integration.none import integrate_none as integrate_none_core
from scptensor.integration.scanorama import integrate_scanorama as integrate_scanorama_core


def test_stable_integration_namespace_all_freezes_package_surface() -> None:
    assert stable_integration.__all__ == [
        "integrate",
        "integrate_none",
        "integrate_combat",
        "integrate_limma",
        "integrate_harmony",
        "integrate_mnn",
        "integrate_scanorama",
    ]


def test_stable_integration_namespace_reexports_stable_implementations() -> None:
    assert integrate is integrate_core
    assert integrate_none is integrate_none_core
    assert integrate_combat is integrate_combat_core
    assert integrate_limma is integrate_limma_core
    assert integrate_harmony is integrate_harmony_core
    assert integrate_mnn is integrate_mnn_core
    assert integrate_scanorama is integrate_scanorama_core


def test_integration_base_and_diagnostics_modules_expose_explicit_helpers() -> None:
    assert IntegrateMethodCore is not None
    assert IntegrationMethodInfoCore is not None
    assert list_integrate_methods_core is not None
    assert get_integrate_method_core is not None
    assert register_integrate_method_core is not None
    assert get_integrate_method_info_core is not None
    assert list_integrate_method_info_core is not None
    assert compute_batch_mixing_metric_core is not None
    assert compute_batch_asw_core is not None
    assert compute_lisi_approx_core is not None
    assert compute_kbet_core is not None
    assert compute_ilisi_core is not None
    assert integration_quality_report_core is not None


def test_stable_integration_namespace_does_not_reexport_registry_or_diagnostics() -> None:
    for name in (
        "list_integrate_methods",
        "get_integrate_method",
        "IntegrateMethod",
        "IntegrationMethodInfo",
        "register_integrate_method",
        "get_integrate_method_info",
        "list_integrate_method_info",
        "compute_batch_mixing_metric",
        "compute_batch_asw",
        "compute_lisi_approx",
        "compute_kbet",
        "compute_ilisi",
        "integration_quality_report",
    ):
        assert name not in stable_integration.__all__
        assert not hasattr(stable_integration, name)


def test_integration_api_is_not_reexported_from_top_level_package() -> None:
    for name in (
        "integrate_none",
        "integrate_combat",
        "integrate_limma",
        "integrate_harmony",
        "integrate_mnn",
        "integrate_scanorama",
    ):
        assert name not in scp.__all__
        assert not hasattr(scp, name)

    for name in (
        "integrate",
        "list_integrate_methods",
        "get_integrate_method",
        "register_integrate_method",
        "get_integrate_method_info",
        "list_integrate_method_info",
        "compute_batch_mixing_metric",
        "compute_batch_asw",
        "compute_lisi_approx",
        "compute_kbet",
        "compute_ilisi",
        "integration_quality_report",
    ):
        assert name not in scp.__all__
        assert not hasattr(scp, name)
