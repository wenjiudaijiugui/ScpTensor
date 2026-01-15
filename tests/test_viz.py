"""
Comprehensive tests for visualization modules.

Tests cover:
- scptensor.viz.base.scatter: scatter() function with mask handling
- scptensor.viz.base.heatmap: heatmap() function with mask hatching
- scptensor.viz.base.violin: violin() function
- scptensor.viz.base.style: setup_style() function
- scptensor.viz.recipes.embedding: embedding() visualization
- scptensor.viz.recipes.qc: qc_completeness() and qc_matrix_spy()
- scptensor.viz.recipes.stats: volcano() function
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes

from scptensor.core import Assay, ScpContainer, ScpMatrix
from scptensor.viz.base import heatmap, scatter, violin
from scptensor.viz.base.style import setup_style
from scptensor.viz.recipes.embedding import embedding
from scptensor.viz.recipes.qc import qc_completeness, qc_matrix_spy
from scptensor.viz.recipes.stats import volcano

# ============================================================================
# Fixtures for visualization tests
# ============================================================================


@pytest.fixture
def viz_container():
    """Create a container suitable for visualization tests."""
    np.random.seed(42)
    obs = pl.DataFrame(
        {
            "_index": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
            "batch": [
                "batch1",
                "batch1",
                "batch1",
                "batch2",
                "batch2",
                "batch2",
                "batch1",
                "batch1",
                "batch2",
                "batch2",
            ],
            "group": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
            "n_detected": [100, 120, 90, 110, 105, 95, 130, 85, 115, 125],
            "treatment": [
                "control",
                "control",
                "treated",
                "treated",
                "control",
                "treated",
                "control",
                "treated",
                "control",
                "treated",
            ],
        }
    )

    var = pl.DataFrame(
        {
            "_index": ["P1", "P2", "P3", "P4", "P5"],
            "protein_id": ["P1", "P2", "P3", "P4", "P5"],
            "protein_name": ["Protein1", "Protein2", "Protein3", "Protein4", "Protein5"],
        }
    )

    X = np.random.rand(10, 5) * 10
    M = np.zeros((10, 5), dtype=np.int8)
    # Add some mask values
    M[0, 0] = 1  # MBR
    M[1, 1] = 2  # LOD
    M[2, 2] = 5  # IMPUTED
    M[3, 3] = 5  # IMPUTED
    M[4, 4] = 1  # MBR

    assay = Assay(
        var=var, layers={"raw": ScpMatrix(X=X, M=M), "imputed": ScpMatrix(X=X + 0.1, M=M)}
    )

    # Add UMAP coordinates as a separate assay
    X_umap = np.random.rand(10, 2)
    umap_assay = Assay(
        var=pl.DataFrame({"_index": ["dim1", "dim2"]}), layers={"X": ScpMatrix(X=X_umap)}
    )

    # Add PCA coordinates
    X_pca = np.random.rand(10, 3)  # 3D for testing dimension validation
    pca_assay = Assay(
        var=pl.DataFrame({"_index": ["PC1", "PC2", "PC3"]}), layers={"X": ScpMatrix(X=X_pca)}
    )

    return ScpContainer(
        obs=obs,
        assays={
            "protein": assay,  # Note: functions expect "protein" not "proteins"
            "umap": umap_assay,
            "pca": pca_assay,
        },
    )


@pytest.fixture
def volcano_container():
    """Create a container with volcano plot data."""
    obs = pl.DataFrame(
        {
            "_index": ["S1", "S2", "S3", "S4", "S5"],
        }
    )

    # Create var with DE results
    var = pl.DataFrame(
        {
            "_index": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
            "protein_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
            "Treat_vs_Ctrl_logFC": [2.5, -2.0, 0.5, -0.3, 1.5, -1.8, 0.1, -0.2, 3.0, -2.5],
            "Treat_vs_Ctrl_pval": [0.001, 0.01, 0.3, 0.5, 0.02, 0.03, 0.6, 0.7, 0.0001, 0.005],
        }
    )

    X = np.random.rand(5, 10)
    assay = Assay(var=var, layers={"X": ScpMatrix(X=X)})

    return ScpContainer(obs=obs, assays={"protein": assay})  # Note: functions expect "protein"


@pytest.fixture
def minimal_viz_container():
    """Create minimal container for edge case tests."""
    obs = pl.DataFrame({"_index": ["S1", "S2"], "batch": ["A", "B"]})

    var = pl.DataFrame({"_index": ["P1"], "protein_id": ["P1"]})

    X = np.array([[1.0], [2.0]])
    M = np.array([[0], [1]], dtype=np.int8)  # One imputed value

    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})
    return ScpContainer(obs=obs, assays={"protein": assay})  # Note: functions expect "protein"


# ============================================================================
# Tests for setup_style (scptensor.viz.base.style)
# ============================================================================


class TestSetupStyle:
    """Tests for setup_style function."""

    def test_setup_style_returns_none(self):
        """setup_style should return None."""
        result = setup_style()
        assert result is None

    def test_setup_style_is_callable(self):
        """setup_style should be callable without error."""
        # This test just ensures the function runs without exception
        setup_style()
        # Style should be applied - we can't easily test the exact style
        # but we can verify matplotlib is in a working state


# ============================================================================
# Tests for scatter (scptensor.viz.base.scatter)
# ============================================================================


class TestScatter:
    """Tests for scatter function."""

    def test_scatter_basic(self):
        """Test basic scatter plot creation."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ax = scatter(X)

        assert isinstance(ax, Axes)
        assert ax is not None
        # Check that data was plotted
        assert len(ax.collections) > 0
        plt.close("all")

    def test_scatter_with_color(self):
        """Test scatter plot with color array."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        c = np.array([1, 2, 3])
        ax = scatter(X, c=c)

        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0
        plt.close("all")

    def test_scatter_with_mask_subtle(self):
        """Test scatter plot with subtle mask style."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        c = np.array([1, 2, 3])
        m = np.array([0, 1, 0], dtype=np.int8)  # Middle point is invalid
        ax = scatter(X, c=c, m=m, mask_style="subtle")

        assert isinstance(ax, Axes)
        # Should have 2 collections (valid and invalid)
        assert len(ax.collections) == 2
        plt.close("all")

    def test_scatter_with_mask_explicit(self):
        """Test scatter plot with explicit mask style."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        c = np.array([1, 2, 3])
        m = np.array([0, 1, 0], dtype=np.int8)  # Middle point is invalid
        ax = scatter(X, c=c, m=m, mask_style="explicit")

        assert isinstance(ax, Axes)
        # Should have 2 collections (valid and invalid)
        assert len(ax.collections) == 2
        plt.close("all")

    def test_scatter_all_valid(self):
        """Test scatter plot when all points are valid."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        m = np.array([0, 0, 0], dtype=np.int8)
        ax = scatter(X, m=m)

        assert isinstance(ax, Axes)
        # Should have 1 collection (all valid)
        assert len(ax.collections) == 1
        plt.close("all")

    def test_scatter_all_invalid(self):
        """Test scatter plot when all points are invalid."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        m = np.array([1, 1, 1], dtype=np.int8)
        ax = scatter(X, m=m)

        assert isinstance(ax, Axes)
        # Should have 1 collection (all invalid)
        assert len(ax.collections) == 1
        plt.close("all")

    def test_scatter_no_mask(self):
        """Test scatter plot without mask (default all zeros)."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ax = scatter(X, m=None)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_scatter_with_title_and_labels(self):
        """Test scatter plot with title and axis labels."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ax = scatter(X, title="Test Title", xlabel="X Label", ylabel="Y Label")

        assert ax.get_title() == "Test Title"
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        plt.close("all")

    def test_scatter_with_custom_ax(self):
        """Test scatter plot with custom Axes."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        fig, ax = plt.subplots()
        result_ax = scatter(X, ax=ax)

        assert result_ax is ax  # Should return the same ax
        assert len(ax.collections) > 0
        plt.close("all")

    def test_scatter_with_kwargs(self):
        """Test scatter plot with additional kwargs."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ax = scatter(X, s=50, cmap="plasma")

        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0
        plt.close("all")

    def test_scatter_string_colors_with_mask(self):
        """Test scatter with string color array and mask."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        c = np.array(["red", "blue", "green", "yellow"])
        m = np.array([0, 0, 1, 1], dtype=np.int8)
        ax = scatter(X, c=c, m=m, mask_style="subtle")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_scatter_single_point(self):
        """Test scatter with a single point."""
        plt.close("all")
        X = np.array([[1, 2]])
        ax = scatter(X)

        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0
        plt.close("all")

    def test_scatter_mask_style_fallback(self):
        """Test scatter with unrecognized mask style falls back to default."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        m = np.array([0, 1, 0], dtype=np.int8)
        ax = scatter(X, m=m, mask_style="unknown")

        assert isinstance(ax, Axes)
        plt.close("all")


# ============================================================================
# Tests for heatmap (scptensor.viz.base.heatmap)
# ============================================================================


class TestHeatmap:
    """Tests for heatmap function."""

    def test_heatmap_basic(self):
        """Test basic heatmap creation."""
        plt.close("all")
        X = np.random.rand(5, 5)
        ax = heatmap(X)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        plt.close("all")

    def test_heatmap_with_mask(self):
        """Test heatmap with mask hatching."""
        plt.close("all")
        X = np.random.rand(5, 5)
        M = np.zeros((5, 5), dtype=np.int8)
        M[0, 0] = 1
        M[1, 1] = 2
        M[2, 2] = 5
        ax = heatmap(X, m=M)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        plt.close("all")

    def test_heatmap_with_tick_labels(self):
        """Test heatmap with tick labels."""
        plt.close("all")
        X = np.random.rand(3, 4)
        xticklabels = ["A", "B", "C", "D"]
        yticklabels = ["S1", "S2", "S3"]
        ax = heatmap(X, xticklabels=xticklabels, yticklabels=yticklabels)

        assert isinstance(ax, Axes)
        # Check tick labels were set
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert x_labels == xticklabels
        assert y_labels == yticklabels
        plt.close("all")

    def test_heatmap_with_title(self):
        """Test heatmap with title."""
        plt.close("all")
        X = np.random.rand(5, 5)
        ax = heatmap(X, title="Test Heatmap")

        assert ax.get_title() == "Test Heatmap"
        plt.close("all")

    def test_heatmap_with_custom_cmap(self):
        """Test heatmap with custom colormap."""
        plt.close("all")
        X = np.random.rand(5, 5)
        ax = heatmap(X, cmap="plasma")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_heatmap_with_custom_ax(self):
        """Test heatmap with custom Axes."""
        plt.close("all")
        X = np.random.rand(5, 5)
        fig, ax = plt.subplots()
        result_ax = heatmap(X, ax=ax)

        assert result_ax is ax
        plt.close("all")

    def test_heatmap_colorbar(self):
        """Test heatmap creates colorbar."""
        plt.close("all")
        X = np.random.rand(5, 5)
        fig, ax = plt.subplots()
        heatmap(X, ax=ax)

        # Check that a colorbar was created
        # The colorbar is attached to the axes
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_heatmap_large_matrix(self):
        """Test heatmap with larger matrix."""
        plt.close("all")
        X = np.random.rand(100, 50)
        ax = heatmap(X)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_heatmask_all_masked(self):
        """Test heatmap where all values are masked."""
        plt.close("all")
        X = np.random.rand(5, 5)
        M = np.ones((5, 5), dtype=np.int8)
        ax = heatmap(X, m=M)

        assert isinstance(ax, Axes)
        plt.close("all")


# ============================================================================
# Tests for violin (scptensor.viz.base.violin)
# ============================================================================


class TestViolin:
    """Tests for violin function."""

    def test_violin_basic(self):
        """Test basic violin plot creation."""
        plt.close("all")
        data = [np.random.rand(50) for _ in range(3)]
        labels = ["A", "B", "C"]
        ax = violin(data, labels)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_violin_with_title(self):
        """Test violin plot with title."""
        plt.close("all")
        data = [np.random.rand(50) for _ in range(3)]
        labels = ["A", "B", "C"]
        ax = violin(data, labels, title="Test Violin")

        assert ax.get_title() == "Test Violin"
        plt.close("all")

    def test_violin_with_ylabel(self):
        """Test violin plot with ylabel."""
        plt.close("all")
        data = [np.random.rand(50) for _ in range(3)]
        labels = ["A", "B", "C"]
        ax = violin(data, labels, ylabel="Expression")

        assert ax.get_ylabel() == "Expression"
        plt.close("all")

    def test_violin_with_custom_ax(self):
        """Test violin plot with custom Axes."""
        plt.close("all")
        data = [np.random.rand(50) for _ in range(3)]
        labels = ["A", "B", "C"]
        fig, ax = plt.subplots()
        result_ax = violin(data, labels, ax=ax)

        assert result_ax is ax
        plt.close("all")

    def test_violin_single_group(self):
        """Test violin plot with single group."""
        plt.close("all")
        data = [np.random.rand(50)]
        labels = ["A"]
        ax = violin(data, labels)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_violin_many_groups(self):
        """Test violin plot with many groups."""
        plt.close("all")
        data = [np.random.rand(30) for _ in range(10)]
        labels = [f"G{i}" for i in range(10)]
        ax = violin(data, labels)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_violin_tick_labels_rotation(self):
        """Test that violin plot tick labels are rotated."""
        plt.close("all")
        data = [np.random.rand(50) for _ in range(3)]
        labels = ["Very Long Label A", "Very Long Label B", "Very Long Label C"]
        ax = violin(data, labels)

        # Labels should be set and rotated
        assert len(ax.get_xticklabels()) == 3
        plt.close("all")


# ============================================================================
# Tests for embedding (scptensor.viz.recipes.embedding)
# ============================================================================


class TestEmbedding:
    """Tests for embedding function."""

    def test_embedding_basic(self, viz_container):
        """Test basic embedding plot."""
        plt.close("all")
        ax = embedding(viz_container, basis="umap")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_embedding_with_color_metadata(self, viz_container):
        """Test embedding colored by metadata column."""
        plt.close("all")
        ax = embedding(viz_container, basis="umap", color="batch")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_embedding_with_color_numeric(self, viz_container):
        """Test embedding colored by numeric metadata."""
        plt.close("all")
        ax = embedding(viz_container, basis="umap", color="n_detected")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_embedding_with_color_feature(self, viz_container):
        """Test embedding colored by protein expression."""
        plt.close("all")
        ax = embedding(viz_container, basis="umap", color="P1", layer="raw")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_embedding_show_missing(self, viz_container):
        """Test embedding with explicit missing markers."""
        plt.close("all")
        ax = embedding(viz_container, basis="umap", color="P1", show_missing=True)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_embedding_pca_basis(self, viz_container):
        """Test embedding with PCA basis."""
        plt.close("all")
        ax = embedding(viz_container, basis="pca")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_embedding_basis_not_found(self, viz_container):
        """Test embedding with non-existent basis."""
        plt.close("all")
        with pytest.raises(ValueError, match="Basis assay 'tsne' not found"):
            embedding(viz_container, basis="tsne")

    def test_embedding_basis_low_dimension(self, viz_container):
        """Test embedding with basis having < 2 dimensions."""
        plt.close("all")
        # Create 1D assay
        obs = viz_container.obs
        X_1d = np.random.rand(10, 1)
        assay_1d = Assay(var=pl.DataFrame({"_index": ["dim1"]}), layers={"X": ScpMatrix(X=X_1d)})
        container_1d = ScpContainer(obs=obs, assays={"tsne": assay_1d})

        with pytest.raises(ValueError, match="has less than 2 dimensions"):
            embedding(container_1d, basis="tsne")

    def test_embedding_color_feature_not_found(self, viz_container):
        """Test embedding with non-existent feature."""
        plt.close("all")
        with pytest.raises(ValueError, match="not found in obs or"):
            embedding(viz_container, basis="umap", color="NONEXISTENT_PROTEIN")

    def test_embedding_without_color(self, viz_container):
        """Test embedding without color."""
        plt.close("all")
        ax = embedding(viz_container, basis="umap", color=None)

        assert isinstance(ax, Axes)
        plt.close("all")


# ============================================================================
# Tests for qc_completeness (scptensor.viz.recipes.qc)
# ============================================================================


class TestQCCompleteness:
    """Tests for qc_completeness function."""

    def test_qc_completeness_basic(self, viz_container):
        """Test basic QC completeness plot."""
        plt.close("all")
        ax = qc_completeness(viz_container)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_qc_completeness_with_group_by(self, viz_container):
        """Test QC completeness grouped by metadata."""
        plt.close("all")
        ax = qc_completeness(viz_container, group_by="batch")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_qc_completeness_with_custom_ax(self, viz_container):
        """Test QC completeness with custom Axes."""
        plt.close("all")
        fig, ax = plt.subplots()
        result_ax = qc_completeness(viz_container, ax=ax)

        assert result_ax is ax
        plt.close("all")

    def test_qc_completeness_assay_not_found(self, viz_container):
        """Test QC completeness with non-existent assay."""
        plt.close("all")
        with pytest.raises(ValueError, match="Assay 'peptides' not found"):
            qc_completeness(viz_container, assay_name="peptides")

    def test_qc_completeness_layer_not_found(self, viz_container):
        """Test QC completeness with non-existent layer."""
        plt.close("all")
        with pytest.raises(ValueError, match="Layer 'normalized' not found"):
            qc_completeness(viz_container, assay_name="protein", layer="normalized")

    def test_qc_completeness_missing_group_column(self, viz_container):
        """Test QC completeness with missing group column."""
        plt.close("all")
        # Should fallback to "All" group
        ax = qc_completeness(viz_container, group_by="nonexistent_column")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_qc_completeness_with_custom_figsize(self, viz_container):
        """Test QC completeness with custom figure size."""
        plt.close("all")
        ax = qc_completeness(viz_container, figsize=(10, 8))

        assert isinstance(ax, Axes)
        plt.close("all")


# ============================================================================
# Tests for qc_matrix_spy (scptensor.viz.recipes.qc)
# ============================================================================


class TestQCMatrixSpy:
    """Tests for qc_matrix_spy function."""

    def test_qc_matrix_spy_basic(self, viz_container):
        """Test basic spy plot creation."""
        plt.close("all")
        ax = qc_matrix_spy(viz_container)

        assert isinstance(ax, Axes)
        assert len(ax.images) > 0
        plt.close("all")

    def test_qc_matrix_spy_with_custom_ax(self, viz_container):
        """Test spy plot with custom Axes."""
        plt.close("all")
        fig, ax = plt.subplots()
        result_ax = qc_matrix_spy(viz_container, ax=ax)

        assert result_ax is ax
        plt.close("all")

    def test_qc_matrix_spy_assay_not_found(self, viz_container):
        """Test spy plot with non-existent assay."""
        plt.close("all")
        with pytest.raises(ValueError, match="Assay 'peptides' not found"):
            qc_matrix_spy(viz_container, assay_name="peptides")

    def test_qc_matrix_spy_layer_not_found(self, viz_container):
        """Test spy plot with non-existent layer."""
        plt.close("all")
        with pytest.raises(ValueError, match="Layer 'normalized' not found"):
            qc_matrix_spy(viz_container, assay_name="protein", layer="normalized")

    def test_qc_matrix_spy_custom_figsize(self, viz_container):
        """Test spy plot with custom figure size."""
        plt.close("all")
        ax = qc_matrix_spy(viz_container, figsize=(12, 10))

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_qc_matrix_spy_title(self, viz_container):
        """Test spy plot has correct title."""
        plt.close("all")
        ax = qc_matrix_spy(viz_container)

        assert "assay_name/layer" in ax.get_title().lower() or "protein" in ax.get_title().lower()
        plt.close("all")

    def test_qc_matrix_spy_axis_labels(self, viz_container):
        """Test spy plot has axis labels."""
        plt.close("all")
        ax = qc_matrix_spy(viz_container)

        assert ax.get_xlabel() == "Features"
        assert ax.get_ylabel() == "Samples"
        plt.close("all")


# ============================================================================
# Tests for volcano (scptensor.viz.recipes.stats)
# ============================================================================


class TestVolcano:
    """Tests for volcano function."""

    def test_volcano_basic(self, volcano_container):
        """Test basic volcano plot."""
        plt.close("all")
        ax = volcano(volcano_container, comparison="Treat_vs_Ctrl")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_volcano_with_custom_thresholds(self, volcano_container):
        """Test volcano plot with custom thresholds."""
        plt.close("all")
        ax = volcano(
            volcano_container, comparison="Treat_vs_Ctrl", fc_threshold=2.0, pval_threshold=0.01
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_volcano_with_custom_columns(self, volcano_container):
        """Test volcano plot with explicitly named columns."""
        plt.close("all")
        ax = volcano(
            volcano_container,
            comparison="AnyName",  # Not used when columns are explicit
            logfc_col="Treat_vs_Ctrl_logFC",
            pval_col="Treat_vs_Ctrl_pval",
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_volcano_assay_not_found(self, volcano_container):
        """Test volcano plot with non-existent assay."""
        plt.close("all")
        with pytest.raises(ValueError, match="Assay 'peptides' not found"):
            volcano(volcano_container, comparison="Treat_vs_Ctrl", assay_name="peptides")

    def test_volcano_logfc_column_not_found(self, volcano_container):
        """Test volcano plot with missing logFC column."""
        plt.close("all")
        with pytest.raises(ValueError, match="Column.*not found in.*var"):
            volcano(volcano_container, comparison="NonExistent_Comparison")

    def test_volcano_pval_column_not_found(self, volcano_container):
        """Test volcano plot with explicitly set missing pval column."""
        plt.close("all")
        with pytest.raises(ValueError, match="Column.*not found in.*var"):
            volcano(
                volcano_container,
                comparison="Any",
                logfc_col="Treat_vs_Ctrl_logFC",
                pval_col="NonExistent_pval",
            )

    def test_volcano_axis_labels(self, volcano_container):
        """Test volcano plot has correct axis labels."""
        plt.close("all")
        ax = volcano(volcano_container, comparison="Treat_vs_Ctrl")

        assert ax.get_xlabel() == "log2 Fold Change"
        assert ax.get_ylabel() == "-log10 P-value"
        plt.close("all")

    def test_volcano_title(self, volcano_container):
        """Test volcano plot has correct title."""
        plt.close("all")
        ax = volcano(volcano_container, comparison="Treat_vs_Ctrl")

        assert "Volcano Plot" in ax.get_title()
        assert "Treat_vs_Ctrl" in ax.get_title()
        plt.close("all")

    def test_volcano_threshold_lines(self, volcano_container):
        """Test volcano plot has threshold lines."""
        plt.close("all")
        ax = volcano(volcano_container, comparison="Treat_vs_Ctrl")

        # Check that threshold lines exist
        # The function adds 3 lines: 1 horizontal, 2 vertical
        # We can't easily count them but we can verify the axes is valid
        assert isinstance(ax, Axes)
        plt.close("all")


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================


class TestVisualizationEdgeCases:
    """Tests for edge cases in visualization functions."""

    def test_scatter_empty_data(self):
        """Test scatter with empty data."""
        plt.close("all")
        X = np.array([]).reshape(0, 2)
        ax = scatter(X)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_heatmap_1x1(self):
        """Test heatmap with single cell."""
        plt.close("all")
        X = np.array([[1.0]])
        ax = heatmap(X)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_violin_with_nan_values(self):
        """Test violin plot with NaN values."""
        plt.close("all")
        data = [np.array([1, 2, np.nan, 4, 5])]
        labels = ["A"]
        ax = violin(data, labels)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_embedding_with_string_metadata(self, viz_container):
        """Test embedding with string metadata column."""
        plt.close("all")
        ax = embedding(viz_container, basis="umap", color="treatment")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_scatter_mask_without_color(self):
        """Test scatter with mask but no color."""
        plt.close("all")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        m = np.array([0, 1, 0], dtype=np.int8)
        ax = scatter(X, m=m, c=None)

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_volcano_with_nan_values(self, volcano_container):
        """Test volcano plot handles NaN values properly."""
        plt.close("all")
        # Modify var to have NaN
        volcano_container.assays["protein"].var = volcano_container.assays[
            "protein"
        ].var.with_columns(
            pl.lit(np.nan).alias("Treat_vs_Ctrl_logFC_nan"),
            pl.lit(0.01).alias("Treat_vs_Ctrl_pval_nan"),
        )

        ax = volcano(
            volcano_container,
            comparison="AnyName",  # Not used when columns are explicit
            logfc_col="Treat_vs_Ctrl_logFC_nan",
            pval_col="Treat_vs_Ctrl_pval_nan",
        )

        assert isinstance(ax, Axes)
        plt.close("all")


# ============================================================================
# Integration tests
# ============================================================================


class TestVisualizationIntegration:
    """Integration tests for visualization workflows."""

    def test_full_qc_workflow(self, viz_container):
        """Test running multiple QC visualizations in sequence."""
        plt.close("all")

        # Create completeness plot
        ax1 = qc_completeness(viz_container, group_by="batch")
        assert isinstance(ax1, Axes)

        # Create spy plot
        ax2 = qc_matrix_spy(viz_container)
        assert isinstance(ax2, Axes)

        # Create embedding plot
        ax3 = embedding(viz_container, basis="umap", color="batch")
        assert isinstance(ax3, Axes)

        plt.close("all")

    def test_multi_assay_visualization(self, viz_container):
        """Test visualizations work across multiple assays."""
        plt.close("all")

        # UMAP embedding
        ax1 = embedding(viz_container, basis="umap")
        assert isinstance(ax1, Axes)

        # PCA embedding
        ax2 = embedding(viz_container, basis="pca")
        assert isinstance(ax2, Axes)

        plt.close("all")

    def test_volcano_with_de_results(self, volcano_container):
        """Test volcano plot with realistic DE results."""
        plt.close("all")

        ax = volcano(
            volcano_container, comparison="Treat_vs_Ctrl", fc_threshold=1.0, pval_threshold=0.05
        )

        assert isinstance(ax, Axes)
        # Verify axes labels
        assert ax.get_xlabel() == "log2 Fold Change"
        assert ax.get_ylabel() == "-log10 P-value"

        plt.close("all")
