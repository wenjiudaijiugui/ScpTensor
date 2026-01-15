"""
Tests for ProvenanceLog and AggregationLink helper structures.

This module contains tests for ProvenanceLog and AggregationLink functionality.
"""

import pytest
from datetime import datetime
import numpy as np
import polars as pl


class TestProvenanceLog:
    """Test ProvenanceLog functionality."""

    def test_provenance_log_creation(self):
        """Test creating a ProvenanceLog entry."""
        from scptensor.core import ProvenanceLog

        log = ProvenanceLog(
            timestamp="2025-01-06T10:00:00",
            action="normalize",
            params={"method": "log", "base": 2},
            description="Log normalization"
        )

        assert log.timestamp == "2025-01-06T10:00:00"
        assert log.action == "normalize"
        assert log.params == {"method": "log", "base": 2}
        assert log.description == "Log normalization"

    def test_provenance_log_with_optional_fields(self):
        """Test ProvenanceLog with optional software version."""
        from scptensor.core import ProvenanceLog

        log = ProvenanceLog(
            timestamp="2025-01-06T10:00:00",
            action="impute",
            params={"method": "knn", "k": 5},
            software_version="0.1.0",
            description="KNN imputation"
        )

        assert log.software_version == "0.1.0"
        assert log.action == "impute"
        assert log.params["k"] == 5

    def test_provenance_log_without_description(self):
        """Test ProvenanceLog without description (optional)."""
        from scptensor.core import ProvenanceLog

        log = ProvenanceLog(
            timestamp="2025-01-06T10:00:00",
            action="filter",
            params={"threshold": 0.5}
        )

        assert log.action == "filter"
        assert log.description is None

    def test_provenance_log_with_complex_params(self):
        """Test ProvenanceLog with complex parameter dict."""
        from scptensor.core import ProvenanceLog

        log = ProvenanceLog(
            timestamp="2025-01-06T10:00:00",
            action="batch_correction",
            params={
                "method": "combat",
                "batch_key": "batch",
                "covariates": ["age", "gender"]
            }
        )

        assert len(log.params["covariates"]) == 2
        assert "age" in log.params["covariates"]


class TestAggregationLink:
    """Test AggregationLink functionality."""

    def test_aggregation_link_creation(self):
        """Test creating an AggregationLink between assays."""
        from scptensor.core import AggregationLink

        linkage = pl.DataFrame({
            "source_id": ["PEP1", "PEP2", "PEP3"],
            "target_id": ["PROT1", "PROT1", "PROT2"]
        })

        link = AggregationLink(
            source_assay="peptide",
            target_assay="protein",
            linkage=linkage
        )

        assert link.source_assay == "peptide"
        assert link.target_assay == "protein"
        assert link.linkage.shape == (3, 2)

    def test_aggregation_link_validation(self):
        """Test AggregationLink validates required columns."""
        from scptensor.core import AggregationLink

        # Missing required column
        invalid_linkage = pl.DataFrame({
            "source_id": ["PEP1", "PEP2"]
            # Missing "target_id"
        })

        with pytest.raises(ValueError, match="must contain columns"):
            AggregationLink(
                source_assay="peptide",
                target_assay="protein",
                linkage=invalid_linkage
            )

    def test_aggregation_link_with_many_to_one(self):
        """Test AggregationLink with many-to-one mapping."""
        from scptensor.core import AggregationLink

        linkage = pl.DataFrame({
            "source_id": ["PEP1", "PEP2", "PEP3", "PEP4", "PEP5"],
            "target_id": ["PROT1", "PROT1", "PROT2", "PROT2", "PROT3"]
        })

        link = AggregationLink(
            source_assay="peptide",
            target_assay="protein",
            linkage=linkage
        )

        assert link.source_assay == "peptide"
        assert len(link.linkage) == 5

        # Verify many-to-one mapping
        prot1_peps = link.linkage.filter(pl.col("target_id") == "PROT1")["source_id"].to_list()
        assert len(prot1_peps) == 2

    def test_aggregation_link_with_one_to_one(self):
        """Test AggregationLink with one-to-one mapping."""
        from scptensor.core import AggregationLink

        linkage = pl.DataFrame({
            "source_id": ["TX1", "TX2", "TX3"],
            "target_id": ["GENE1", "GENE2", "GENE3"]
        })

        link = AggregationLink(
            source_assay="transcript",
            target_assay="gene",
            linkage=linkage
        )

        assert link.source_assay == "transcript"
        assert link.target_assay == "gene"
        assert len(link.linkage) == 3

    def test_aggregation_link_query_mapping(self):
        """Test querying feature mapping through linkage."""
        from scptensor.core import AggregationLink

        linkage = pl.DataFrame({
            "source_id": ["PEP1", "PEP2", "PEP3"],
            "target_id": ["PROT1", "PROT1", "PROT2"]
        })

        link = AggregationLink(
            source_assay="peptide",
            target_assay="protein",
            linkage=linkage
        )

        # Query mapping for PEP1
        pep1_target = link.linkage.filter(pl.col("source_id") == "PEP1")["target_id"][0]
        assert pep1_target == "PROT1"

        # Query all peptides mapping to PROT1
        prot1_sources = link.linkage.filter(pl.col("target_id") == "PROT1")["source_id"].to_list()
        assert set(prot1_sources) == {"PEP1", "PEP2"}

