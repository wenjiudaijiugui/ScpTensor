"""Public API guardrails for scptensor.io."""

import scptensor.io as io


def test_public_mass_spec_api_is_exposed() -> None:
    assert callable(io.load_quant_table)
    assert callable(io.load_diann)
    assert callable(io.load_spectronaut)
    assert callable(io.load_peptide_pivot)
    assert callable(io.aggregate_to_protein)


def test_non_mass_spec_io_api_is_not_exposed() -> None:
    assert not hasattr(io, "save_hdf5")
    assert not hasattr(io, "load_hdf5")
    assert not hasattr(io, "save_npz")
    assert not hasattr(io, "load_npz")
    assert not hasattr(io, "save_csv")
    assert not hasattr(io, "load_csv")


def test_legacy_private_compat_api_is_not_exposed() -> None:
    assert not hasattr(io, "_detect_format")
    assert not hasattr(io, "_extract_sample_columns")
    assert not hasattr(io, "_load_diann_bgs")
    assert not hasattr(io, "_load_spectronaut_pivot")
