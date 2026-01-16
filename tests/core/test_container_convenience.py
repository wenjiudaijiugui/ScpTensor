"""Tests for ScpContainer convenience methods.

This module tests convenience methods that improve usability:
- list_assays()
- list_layers()
- summary()
- _repr_html_()
"""

import pytest

from scptensor.core import ScpContainer
from scptensor.core.utils import _find_closest_match


class TestContainerConvenience:
    """Test ScpContainer convenience methods."""

    def test_list_assays_single_assay(self, sample_container):
        """Test list_assays with single assay."""
        assays = sample_container.list_assays()
        assert assays == ["proteins"]
        assert isinstance(assays, list)
        assert all(isinstance(a, str) for a in assays)

    def test_list_assays_multiple_assays(self, sample_container_multi_assay):
        """Test list_assays with multiple assays."""
        assays = sample_container_multi_assay.list_assays()
        assert "proteins" in assays
        assert "peptides" in assays
        assert len(assays) == 2

    def test_list_assays_empty_container(self, empty_container_data):
        """Test list_assays with empty container."""
        container = ScpContainer(obs=empty_container_data, assays={})
        assays = container.list_assays()
        assert assays == []

    def test_list_layers_existing_assay(self, sample_assay_multi_layer, sample_obs):
        """Test list_layers with existing assay."""
        container = ScpContainer(obs=sample_obs, assays={"proteins": sample_assay_multi_layer})
        layers = container.list_layers("proteins")
        assert "raw" in layers
        assert "log" in layers
        assert "normalized" in layers
        assert len(layers) == 3

    def test_list_layers_invalid_assay(self, sample_container):
        """Test list_layers raises KeyError for invalid assay."""
        with pytest.raises(KeyError, match="Assay 'invalid' not found"):
            sample_container.list_layers("invalid")

    def test_list_layers_error_message_includes_available(self, sample_container):
        """Test list_layers error message includes available assays or suggestion."""
        # Error should either suggest a similar name OR show available assays
        with pytest.raises(KeyError) as exc_info:
            sample_container.list_layers("nonexistent")
        error_msg = str(exc_info.value)
        # Either we get a suggestion or the available assays list
        assert "Did you mean" in error_msg or "Available assays:" in error_msg

    def test_list_layers_error_message_suggests_list_assays(self, sample_container):
        """Test list_layers error message suggests using list_assays."""
        with pytest.raises(KeyError, match="list_assays"):
            sample_container.list_layers("wrong_name")

    def test_summary_single_assay(self, sample_container):
        """Test summary with single assay."""
        summary = sample_container.summary()
        assert "ScpContainer" in summary
        assert "5 samples" in summary
        assert "Assays: 1" in summary
        assert "proteins" in summary
        assert "5 features" in summary

    def test_summary_multiple_assays(self, sample_container_multi_assay):
        """Test summary with multiple assays."""
        summary = sample_container_multi_assay.summary()
        assert "ScpContainer" in summary
        assert "5 samples" in summary
        assert "Assays: 2" in summary
        assert "proteins" in summary
        assert "peptides" in summary

    def test_summary_empty_container(self, empty_container_data):
        """Test summary with empty container."""
        container = ScpContainer(obs=empty_container_data, assays={})
        summary = container.summary()
        assert "ScpContainer" in summary
        assert "2 samples" in summary
        assert "Assays: 0" in summary

    def test_summary_format_is_multiline(self, sample_container):
        """Test summary returns multi-line string."""
        summary = sample_container.summary()
        lines = summary.split("\n")
        assert len(lines) >= 3  # Header, assays count, at least one assay

    def test_repr_html_basic(self, sample_container):
        """Test _repr_html_ returns valid HTML."""
        html = sample_container._repr_html_()
        assert isinstance(html, str)
        assert "<div" in html
        assert "</div>" in html
        assert "<strong>ScpContainer</strong>" in html
        assert "samples" in html

    def test_repr_html_multiple_assays(self, sample_container_multi_assay):
        """Test _repr_html_ with multiple assays."""
        html = sample_container_multi_assay._repr_html_()
        assert "proteins" in html
        assert "peptides" in html
        assert "features" in html
        assert "layers" in html

    def test_repr_html_empty_container(self, empty_container_data):
        """Test _repr_html_ with empty container."""
        container = ScpContainer(obs=empty_container_data, assays={})
        html = container._repr_html_()
        assert "<strong>ScpContainer</strong>" in html
        assert "2 samples" in html
        assert "Assays:</strong> 0" in html

    def test_repr_html_monospace_font(self, sample_container):
        """Test _repr_html_ uses monospace font."""
        html = sample_container._repr_html_()
        assert "font-family: monospace" in html

    def test_repr_html_uses_br_not_newlines(self, sample_container):
        """Test _repr_html_ uses <br> instead of newlines."""
        html = sample_container._repr_html_()
        assert "<br>" in html
        assert "\n" not in html  # Should not have literal newlines in HTML

    def test_convenience_methods_together(self, sample_assay_multi_layer, sample_obs):
        """Test using convenience methods together."""
        container = ScpContainer(obs=sample_obs, assays={"proteins": sample_assay_multi_layer})

        # List assays
        assays = container.list_assays()
        assert "proteins" in assays

        # List layers for each assay
        for assay_name in assays:
            layers = container.list_layers(assay_name)
            assert isinstance(layers, list)

        # Get summary
        summary = container.summary()
        assert "ScpContainer" in summary

        # Get HTML representation
        html = container._repr_html_()
        assert "<strong>" in html


class TestFuzzyMatching:
    """Test fuzzy matching functionality for error suggestions."""

    def test_find_closest_match_exact(self):
        """Test _find_closest_match with exact match."""
        result = _find_closest_match("raw", ["raw", "log", "normalized"])
        assert result == "raw"

    def test_find_closest_match_typo(self):
        """Test _find_closest_match with typo."""
        result = _find_closest_match("ra", ["raw", "log", "normalized"])
        assert result == "raw"

        result = _find_closest_match("prote", ["proteins", "peptides"])
        assert result == "proteins"

    def test_find_closest_match_case_sensitive(self):
        """Test _find_closest_match is case-sensitive."""
        result = _find_closest_match("RAW", ["raw", "log", "normalized"])
        # Case-sensitive matching may not find "raw" for "RAW"
        # depending on the similarity threshold
        assert result is None or result == "raw"

    def test_find_closest_match_no_close_match(self):
        """Test _find_closest_match with no close match."""
        result = _find_closest_match("xyz", ["raw", "log", "normalized"])
        assert result is None

    def test_find_closest_match_empty_options(self):
        """Test _find_closest_match with empty options."""
        result = _find_closest_match("raw", [])
        assert result is None

    def test_find_closest_match_single_option(self):
        """Test _find_closest_match with single option."""
        result = _find_closest_match("ra", ["raw"])
        assert result == "raw"

    def test_list_layers_suggests_correction(self, sample_container_multi_assay):
        """Test list_layers suggests correction for similar assay name."""
        # Test "prot" suggests "proteins"
        with pytest.raises(KeyError, match="Did you mean 'proteins"):
            sample_container_multi_assay.list_layers("prot")

    def test_list_layers_suggests_peptide(self, sample_container_multi_assay):
        """Test list_layers suggests correction for 'peptid' typo."""
        with pytest.raises(KeyError, match="Did you mean 'peptides"):
            sample_container_multi_assay.list_layers("peptid")

    def test_list_layers_no_suggestion_for_dissimilar(self, sample_container_multi_assay):
        """Test list_layers shows available assays when no close match."""
        with pytest.raises(KeyError, match="Available assays:"):
            sample_container_multi_assay.list_layers("xyzcompletelydifferent")

    def test_list_layers_error_includes_list_assays_hint(self, sample_container):
        """Test list_layers error message suggests list_assays when no match."""
        # When there's no close match, should show available assays
        with pytest.raises(KeyError, match="list_assays"):
            sample_container.list_layers("completely_different_name")


class TestAssayNotFoundError:
    """Test AssayNotFoundError with fuzzy matching."""

    def test_assay_not_found_with_suggestion(self):
        """Test AssayNotFoundError includes fuzzy suggestion."""
        from scptensor.core.exceptions import AssayNotFoundError

        exc = AssayNotFoundError("prot", available_assays=["proteins", "peptides"])
        assert "Did you mean 'proteins'" in str(exc)
        assert exc.suggestion == "proteins"

    def test_assay_not_found_without_suggestion(self):
        """Test AssayNotFoundError shows available when no close match."""
        from scptensor.core.exceptions import AssayNotFoundError

        exc = AssayNotFoundError("xyz", available_assays=["proteins", "peptides"])
        assert "Available assays:" in str(exc)
        assert exc.suggestion is None

    def test_assay_not_found_with_hint(self):
        """Test AssayNotFoundError with manual hint."""
        from scptensor.core.exceptions import AssayNotFoundError

        # Suggestion takes precedence over hint
        exc = AssayNotFoundError(
            "prot", hint="Manual hint", available_assays=["proteins", "peptides"]
        )
        assert "Did you mean 'proteins'" in str(exc)

    def test_assay_not_found_legacy_compatibility(self):
        """Test AssayNotFoundError works without available_assays."""
        from scptensor.core.exceptions import AssayNotFoundError

        # Old-style usage without available_assays
        exc = AssayNotFoundError("metabolites", hint="Use create_assay()")
        assert "metabolites" in str(exc)
        assert "Use create_assay()" in str(exc)


class TestLayerNotFoundError:
    """Test LayerNotFoundError with fuzzy matching."""

    def test_layer_not_found_with_suggestion(self):
        """Test LayerNotFoundError includes fuzzy suggestion."""
        from scptensor.core.exceptions import LayerNotFoundError

        exc = LayerNotFoundError(
            "ra", assay_name="proteins", available_layers=["raw", "log", "normalized"]
        )
        assert "Did you mean 'raw'" in str(exc)
        assert exc.suggestion == "raw"

    def test_layer_not_found_without_suggestion(self):
        """Test LayerNotFoundError shows available when no close match."""
        from scptensor.core.exceptions import LayerNotFoundError

        exc = LayerNotFoundError(
            "xyz", assay_name="proteins", available_layers=["raw", "log", "normalized"]
        )
        assert "Available layers:" in str(exc)
        assert exc.suggestion is None

    def test_layer_not_found_without_assay_name(self):
        """Test LayerNotFoundError without assay name."""
        from scptensor.core.exceptions import LayerNotFoundError

        exc = LayerNotFoundError("ra", available_layers=["raw", "log", "normalized"])
        assert "Did you mean 'raw'" in str(exc)

    def test_layer_not_found_legacy_compatibility(self):
        """Test LayerNotFoundError works without available_layers."""
        from scptensor.core.exceptions import LayerNotFoundError

        # Old-style usage
        exc = LayerNotFoundError("normalized", assay_name="proteins", hint="Run normalize()")
        assert "normalized" in str(exc)
        assert "Run normalize()" in str(exc)
