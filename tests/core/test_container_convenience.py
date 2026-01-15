"""Tests for ScpContainer convenience methods.

This module tests convenience methods that improve usability:
- list_assays()
- list_layers()
- summary()
- _repr_html_()
"""

import pytest

from scptensor.core import ScpContainer


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
        """Test list_layers error message includes available assays."""
        with pytest.raises(KeyError, match="Available assays:"):
            sample_container.list_layers("nonexistent")

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
