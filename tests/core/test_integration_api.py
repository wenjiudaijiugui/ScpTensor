"""Tests for stable integration namespace exports."""

from __future__ import annotations

import scptensor as scp
import scptensor.integration as stable_integration
from scptensor.integration import (
    IntegrateMethod,
    IntegrationMethodInfo,
    compute_batch_asw,
    compute_batch_mixing_metric,
    compute_lisi_approx,
    get_integrate_method,
    get_integrate_method_info,
    integrate,
    integrate_combat,
    integrate_harmony,
    integrate_limma,
    integrate_mnn,
    integrate_none,
    integrate_scanorama,
    integration_quality_report,
    list_integrate_method_info,
    list_integrate_methods,
    register_integrate_method,
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
        "list_integrate_methods",
        "get_integrate_method",
        "IntegrateMethod",
        "IntegrationMethodInfo",
        "register_integrate_method",
        "get_integrate_method_info",
        "list_integrate_method_info",
        "integrate_none",
        "integrate_combat",
        "integrate_limma",
        "integrate_harmony",
        "integrate_mnn",
        "integrate_scanorama",
        "compute_batch_mixing_metric",
        "compute_batch_asw",
        "compute_lisi_approx",
        "integration_quality_report",
    ]


def test_stable_integration_namespace_reexports_stable_implementations() -> None:
    assert IntegrateMethod is IntegrateMethodCore
    assert IntegrationMethodInfo is IntegrationMethodInfoCore
    assert integrate is integrate_core
    assert list_integrate_methods is list_integrate_methods_core
    assert get_integrate_method is get_integrate_method_core
    assert register_integrate_method is register_integrate_method_core
    assert get_integrate_method_info is get_integrate_method_info_core
    assert list_integrate_method_info is list_integrate_method_info_core
    assert integrate_none is integrate_none_core
    assert integrate_combat is integrate_combat_core
    assert integrate_limma is integrate_limma_core
    assert integrate_harmony is integrate_harmony_core
    assert integrate_mnn is integrate_mnn_core
    assert integrate_scanorama is integrate_scanorama_core
    assert compute_batch_mixing_metric is compute_batch_mixing_metric_core
    assert compute_batch_asw is compute_batch_asw_core
    assert compute_lisi_approx is compute_lisi_approx_core
    assert integration_quality_report is integration_quality_report_core


def test_only_direct_integration_methods_are_reexported_from_top_level_package() -> None:
    assert scp.integrate_none is integrate_none_core
    assert scp.integrate_combat is integrate_combat_core
    assert scp.integrate_limma is integrate_limma_core
    assert scp.integrate_harmony is integrate_harmony_core
    assert scp.integrate_mnn is integrate_mnn_core
    assert scp.integrate_scanorama is integrate_scanorama_core

    for name in (
        "integrate_none",
        "integrate_combat",
        "integrate_limma",
        "integrate_harmony",
        "integrate_mnn",
        "integrate_scanorama",
    ):
        assert name in scp.__all__

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
        "integration_quality_report",
    ):
        assert name not in scp.__all__
