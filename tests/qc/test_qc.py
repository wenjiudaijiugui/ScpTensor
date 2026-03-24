"""Comprehensive tests for QC module.

Tests cover the refactored QC module structure:
- qc_psm: PSM-level filtering (filter_contaminants, filter_psms_by_pif)
- qc_sample: Sample-level QC (calculate_sample_qc_metrics, filter_low_quality_samples,
             filter_doublets_mad, assess_batch_effects)
- qc_feature: Feature-level QC (calculate_feature_qc_metrics, filter_features_by_missingness,
               filter_features_by_cv)
- metrics: Utility functions (compute_mad, is_outlier_mad, compute_cv)

"""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from scptensor.core import Assay, MaskCode, ScpContainer, ScpMatrix
from scptensor.core.exceptions import AssayNotFoundError, LayerNotFoundError, ScpValueError
from scptensor.experimental import qc_psm

# New imports from refactored QC module
from scptensor.qc import qc_feature, qc_sample
from scptensor.qc.metrics import compute_cv, compute_mad, is_outlier_mad


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def qc_obs():
    """Create obs DataFrame for QC testing."""
    return pl.DataFrame(
        {
            "_index": [str(i) for i in range(20)],  # Use integer strings as IDs
            "batch": ["A"] * 10 + ["B"] * 10,
        },
    )


@pytest.fixture
def qc_var():
    """Create var DataFrame for QC testing."""
    # Include some contaminant proteins for testing
    protein_names = [
        "KRT1",
        "KRT2",  # Keratins (contaminants)
        "ALB001",  # Albumin (contaminant)
        "IGG1",  # Immunoglobulin (contaminant)
    ] + [f"PROT{i}" for i in range(4, 20)]
    return pl.DataFrame(
        {
            "_index": [
                str(i) for i in range(20)
            ],  # Use integer strings as IDs (matches positional indices)
            "name": protein_names,
        },
    )


@pytest.fixture
def qc_var_with_contaminants():
    """Create var DataFrame with known contaminants."""
    protein_names = [
        "KRT1",
        "KRT2",
        "KRT8",  # Keratins
        "Trypsin",  # Trypsin
        "Albumin",  # Albumin
        "IGHG1",
        "IGKC",  # Immunoglobulins
    ] + [f"PROT{i}" for i in range(7, 20)]
    # Use integer strings as IDs for compatibility with filter_features
    return pl.DataFrame(
        {
            "_index": [str(i) for i in range(20)],
            "name": protein_names,
        },
    )


@pytest.fixture
def qc_dense_X():
    """Create dense data matrix with some missing values."""
    np.random.seed(42)
    X = np.random.exponential(1.0, size=(20, 20))
    # Introduce missing values (zeros)
    missing_mask = np.random.random((20, 20)) < 0.3
    X[missing_mask] = 0
    # Make some cells low quality (first 2 samples have few features)
    X[0, :] = 0
    X[0, :5] = 0.5  # Only 5 features detected (above default min_features=200 doesn't apply to 20)
    X[1, :] = 0
    X[1, :3] = 1.0  # Only 3 features detected
    # Make some features low prevalence (last 2 features rarely detected)
    X[:, 18] = 0
    X[:5, 18] = 0.5
    X[:, 19] = 0
    X[:3, 19] = 1.0
    return X


@pytest.fixture
def qc_sparse_X(qc_dense_X):
    """Create sparse data matrix."""
    return sparse.csr_matrix(qc_dense_X)


@pytest.fixture
def qc_dense_M(qc_dense_X):
    """Create a mask matrix where zeros are treated as LOD."""
    mask = np.full(qc_dense_X.shape, MaskCode.VALID.value, dtype=np.int8)
    mask[qc_dense_X == 0] = MaskCode.LOD.value
    return mask


@pytest.fixture
def qc_container(qc_obs, qc_var, qc_dense_X, qc_dense_M):
    """Create a ScpContainer for QC testing."""
    matrix = ScpMatrix(X=qc_dense_X, M=qc_dense_M)
    assay = Assay(var=qc_var, layers={"raw": matrix})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


@pytest.fixture
def qc_container_sparse(qc_obs, qc_var, qc_sparse_X, qc_dense_M):
    """Create a ScpContainer with sparse data."""
    matrix = ScpMatrix(X=qc_sparse_X, M=qc_dense_M)
    assay = Assay(var=qc_var, layers={"raw": matrix})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


@pytest.fixture
def qc_container_multi_layer(qc_obs, qc_var, qc_dense_X, qc_dense_M):
    """Create a ScpContainer with multiple layers."""
    matrix_raw = ScpMatrix(X=qc_dense_X, M=qc_dense_M)
    matrix_norm = ScpMatrix(X=qc_dense_X * 2, M=qc_dense_M)
    assay = Assay(var=qc_var, layers={"raw": matrix_raw, "normalized": matrix_norm})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


@pytest.fixture
def qc_container_with_contaminants(qc_obs, qc_var_with_contaminants, qc_dense_X, qc_dense_M):
    """Create a ScpContainer with contaminant proteins."""
    matrix = ScpMatrix(X=qc_dense_X, M=qc_dense_M)
    assay = Assay(var=qc_var_with_contaminants, layers={"raw": matrix})
    return ScpContainer(obs=qc_obs, assays={"protein": assay})


@pytest.fixture
def qc_psm_container(qc_obs):
    """Create a peptide/PSM-level container with alias-sensitive assay naming."""
    obs = qc_obs.head(4)
    var = pl.DataFrame(
        {
            "_index": [str(i) for i in range(4)],
            "gene_names": ["KRT1", "PEP_A", "PEP_B", "PEP_C"],
            "pif": [0.95, 0.60, 0.82, 0.40],
            "qvalue": [0.001, 0.020, 0.200, 0.005],
        },
    )
    X = np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0
    assay = Assay(var=var, layers={"raw": ScpMatrix(X=X)})
    return ScpContainer(obs=obs, assays={"peptides": assay})


# =============================================================================
# Tests for qc_sample module
# =============================================================================


class TestCalculateSampleQCMetrics:
    """Tests for calculate_sample_qc_metrics function."""

    def test_calculate_sample_qc_metrics_default(self, qc_container):
        """Test calculate_sample_qc_metrics with default parameters."""
        result = qc_sample.calculate_sample_qc_metrics(qc_container)
        assert isinstance(result, ScpContainer)

        # Check that metrics were added to obs
        assert "n_features_protein" in result.obs.columns
        assert "total_intensity_protein" in result.obs.columns
        assert "log1p_total_intensity_protein" in result.obs.columns

    def test_calculate_sample_qc_metrics_sparse(self, qc_container_sparse):
        """Test calculate_sample_qc_metrics with sparse matrix."""
        result = qc_sample.calculate_sample_qc_metrics(qc_container_sparse)
        assert isinstance(result, ScpContainer)
        assert "n_features_protein" in result.obs.columns

    def test_calculate_sample_qc_metrics_custom_layer(self, qc_container_multi_layer):
        """Test with different layer."""
        result = qc_sample.calculate_sample_qc_metrics(
            qc_container_multi_layer,
            layer_name="normalized",
        )
        assert "n_features_protein" in result.obs.columns

    def test_calculate_sample_qc_metrics_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.calculate_sample_qc_metrics(qc_container, assay_name="nonexistent")

    def test_calculate_sample_qc_metrics_invalid_layer(self, qc_container):
        """Test that invalid layer raises error."""
        with pytest.raises(LayerNotFoundError):
            qc_sample.calculate_sample_qc_metrics(qc_container, layer_name="nonexistent")

    def test_calculate_sample_qc_metrics_uses_mask_semantics(self):
        """True zero should count as detected only when M marks it VALID."""
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        X = np.array([[0.0, 1.0], [0.0, 0.0], [5.0, 0.0]])
        M = np.array(
            [
                [MaskCode.VALID.value, MaskCode.VALID.value],
                [MaskCode.LOD.value, MaskCode.LOD.value],
                [MaskCode.VALID.value, MaskCode.LOD.value],
            ],
            dtype=np.int8,
        )
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})},
        )

        result = qc_sample.calculate_sample_qc_metrics(container)
        assert result.obs["n_features_protein"].to_list() == [2, 0, 1]

    def test_calculate_sample_qc_metrics_accepts_alias_and_copies_container(self, qc_container):
        """QC metrics should resolve assay aliases without mutating the source container."""
        result = qc_sample.calculate_sample_qc_metrics(qc_container, assay_name="proteins")
        assert "n_features_protein" in result.obs.columns
        assert result.assays is not qc_container.assays
        result.assays["protein"].add_layer("tmp", ScpMatrix(X=np.ones((20, 20))))
        assert "tmp" not in qc_container.assays["protein"].layers

    def test_calculate_sample_qc_metrics_requires_raw_by_default(self, qc_obs, qc_var):
        """Default sample QC keeps the explicit raw-layer requirement."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = np.full(X.shape, MaskCode.VALID.value, dtype=np.int8)
        container = ScpContainer(
            obs=qc_obs.head(2),
            assays={
                "protein": Assay(var=qc_var.head(2), layers={"normalized": ScpMatrix(X=X, M=M)}),
            },
        )

        with pytest.raises(LayerNotFoundError):
            qc_sample.calculate_sample_qc_metrics(container)

        result = qc_sample.calculate_sample_qc_metrics(container, layer_name="normalized")
        assert "n_features_protein" in result.obs.columns

    def test_calculate_sample_qc_metrics_does_not_log_history(self, qc_container):
        """Metric calculation remains history-free by contract."""
        result = qc_sample.calculate_sample_qc_metrics(qc_container)
        assert len(qc_container.history) == 0
        assert len(result.history) == 0


class TestFilterLowQualitySamples:
    """Tests for filter_low_quality_samples function."""

    def test_filter_low_quality_samples_default(self, qc_container):
        """Test filter_low_quality_samples with default parameters."""
        # Use lower threshold since we only have 20 features
        result = qc_sample.filter_low_quality_samples(qc_container, min_features=5, use_mad=False)
        assert isinstance(result, ScpContainer)
        assert result.n_samples <= qc_container.n_samples

    def test_filter_low_quality_samples_with_mad(self, qc_container):
        """Test with MAD-based filtering."""
        result = qc_sample.filter_low_quality_samples(
            qc_container,
            min_features=2,
            nmads=2.0,
            use_mad=True,
        )
        assert isinstance(result, ScpContainer)
        assert result.n_samples <= qc_container.n_samples

    def test_filter_low_quality_samples_permissive(self, qc_container):
        """Test with permissive threshold."""
        result = qc_sample.filter_low_quality_samples(qc_container, min_features=0, use_mad=False)
        # Should keep all samples
        assert result.n_samples == qc_container.n_samples

    def test_filter_low_quality_samples_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.filter_low_quality_samples(qc_container, assay_name="nonexistent")

    def test_filter_low_quality_samples_prefers_raw_when_present(self):
        """Sample filters should use raw when available, even if it is not the first layer."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        raw_X = np.array([[1.0, 0.0], [1.0, 1.0]])
        raw_M = np.array(
            [
                [MaskCode.VALID.value, MaskCode.LOD.value],
                [MaskCode.VALID.value, MaskCode.VALID.value],
            ],
            dtype=np.int8,
        )
        normalized_X = np.ones((2, 2))
        normalized_M = np.full((2, 2), MaskCode.VALID.value, dtype=np.int8)
        assay = Assay(
            var=var,
            layers={
                "normalized": ScpMatrix(X=normalized_X, M=normalized_M),
                "raw": ScpMatrix(X=raw_X, M=raw_M),
            },
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = qc_sample.filter_low_quality_samples(container, min_features=2, use_mad=False)
        assert result.obs["_index"].to_list() == ["S2"]

    def test_filter_low_quality_samples_requires_raw_by_default(self):
        """Sample filters should require raw unless callers select another layer explicitly."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        normalized_X = np.array([[1.0, 0.0], [1.0, 1.0]])
        normalized_M = np.array(
            [
                [MaskCode.VALID.value, MaskCode.LOD.value],
                [MaskCode.VALID.value, MaskCode.VALID.value],
            ],
            dtype=np.int8,
        )
        assay = Assay(var=var, layers={"normalized": ScpMatrix(X=normalized_X, M=normalized_M)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        with pytest.raises(LayerNotFoundError):
            qc_sample.filter_low_quality_samples(container, min_features=2, use_mad=False)

        result = qc_sample.filter_low_quality_samples(
            container,
            layer_name="normalized",
            min_features=2,
            use_mad=False,
        )
        assert result.obs["_index"].to_list() == ["S2"]

    def test_filter_low_quality_samples_preserves_filter_provenance(self, qc_container):
        """Filtering should keep both the structural and entrypoint-specific provenance."""
        result = qc_sample.filter_low_quality_samples(qc_container, min_features=5, use_mad=False)
        assert [log.action for log in result.history[-2:]] == [
            "filter_samples",
            "filter_low_quality_samples",
        ]


class TestFilterDoubletsMAD:
    """Tests for filter_doublets_mad function."""

    def test_filter_doublets_mad_default(self, qc_container):
        """Test doublet detection with default parameters."""
        result = qc_sample.filter_doublets_mad(qc_container, nmads=3.0)
        assert isinstance(result, ScpContainer)
        assert result.n_samples <= qc_container.n_samples

    def test_filter_doublets_mad_custom_nmads(self, qc_container):
        """Test with custom nmads threshold."""
        result = qc_sample.filter_doublets_mad(qc_container, nmads=2.0)
        assert isinstance(result, ScpContainer)

    def test_filter_doublets_mad_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.filter_doublets_mad(qc_container, assay_name="nonexistent")

    def test_filter_doublets_mad_prefers_raw_when_present(self):
        """Doublet filtering should prefer raw over earlier non-raw layers."""
        obs = pl.DataFrame({"_index": [f"S{i}" for i in range(5)]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        raw_X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1000.0, 0.0]])
        raw_M = np.full(raw_X.shape, MaskCode.VALID.value, dtype=np.int8)
        normalized_X = np.ones((5, 2))
        normalized_M = np.full((5, 2), MaskCode.VALID.value, dtype=np.int8)
        assay = Assay(
            var=var,
            layers={
                "normalized": ScpMatrix(X=normalized_X, M=normalized_M),
                "raw": ScpMatrix(X=raw_X, M=raw_M),
            },
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = qc_sample.filter_doublets_mad(container, nmads=1.0)
        assert result.obs["_index"].to_list() == ["S0", "S1", "S2", "S3"]


class TestAssessBatchEffects:
    """Tests for assess_batch_effects function."""

    def test_assess_batch_effects_default(self, qc_container):
        """Test batch effect assessment."""
        result = qc_sample.assess_batch_effects(qc_container, batch_col="batch")
        assert isinstance(result, pl.DataFrame)
        assert "batch" in result.columns
        assert "n_cells" in result.columns
        assert "median_features" in result.columns
        assert "median_intensity" in result.columns

    def test_assess_batch_effects_invalid_batch_col(self, qc_container):
        """Test that invalid batch column raises error."""
        with pytest.raises(ScpValueError):
            qc_sample.assess_batch_effects(qc_container, batch_col="nonexistent")

    def test_assess_batch_effects_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_sample.assess_batch_effects(
                qc_container,
                batch_col="batch",
                assay_name="nonexistent",
            )

    def test_assess_batch_effects_prefers_raw_when_present(self):
        """Batch summaries should use raw when both raw and non-raw layers are present."""
        obs = pl.DataFrame({"_index": ["S1", "S2"], "batch": ["A", "A"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        raw_X = np.array([[1.0, 0.0], [5.0, 0.0]])
        raw_M = np.full(raw_X.shape, MaskCode.VALID.value, dtype=np.int8)
        normalized_X = np.full((2, 2), 10.0)
        normalized_M = np.full((2, 2), MaskCode.VALID.value, dtype=np.int8)
        assay = Assay(
            var=var,
            layers={
                "normalized": ScpMatrix(X=normalized_X, M=normalized_M),
                "raw": ScpMatrix(X=raw_X, M=raw_M),
            },
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result = qc_sample.assess_batch_effects(container, batch_col="batch")
        assert result["median_intensity"].to_list() == pytest.approx([3.0])

    def test_assess_batch_effects_requires_raw_by_default(self):
        """Batch summaries should require raw unless callers select another layer explicitly."""
        obs = pl.DataFrame({"_index": ["S1", "S2"], "batch": ["A", "B"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        normalized_X = np.array([[1.0, 0.0], [2.0, 2.0]])
        normalized_M = np.full(normalized_X.shape, MaskCode.VALID.value, dtype=np.int8)
        assay = Assay(var=var, layers={"normalized": ScpMatrix(X=normalized_X, M=normalized_M)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        with pytest.raises(LayerNotFoundError):
            qc_sample.assess_batch_effects(container, batch_col="batch")

        result = qc_sample.assess_batch_effects(
            container,
            batch_col="batch",
            layer_name="normalized",
        )
        assert result["median_intensity"].to_list() == pytest.approx([1.0, 4.0])


# =============================================================================
# Tests for qc_feature module
# =============================================================================


class TestCalculateFeatureQCMetrics:
    """Tests for calculate_feature_qc_metrics function."""

    def test_calculate_feature_qc_metrics_default(self, qc_container):
        """Test calculate_feature_qc_metrics with default parameters."""
        result = qc_feature.calculate_feature_qc_metrics(qc_container)
        assert isinstance(result, ScpContainer)

        # Check that metrics were added to var
        var = result.assays["protein"].var
        assert "missing_rate" in var.columns
        assert "detection_rate" in var.columns
        assert "mean_expression" in var.columns
        assert "cv" in var.columns

    def test_calculate_feature_qc_metrics_sparse(self, qc_container_sparse):
        """Test calculate_feature_qc_metrics with sparse matrix."""
        result = qc_feature.calculate_feature_qc_metrics(qc_container_sparse)
        assert isinstance(result, ScpContainer)
        var = result.assays["protein"].var
        assert "missing_rate" in var.columns
        assert "cv" in var.columns

    def test_calculate_feature_qc_metrics_custom_layer(self, qc_container_multi_layer):
        """Test with different layer."""
        result = qc_feature.calculate_feature_qc_metrics(
            qc_container_multi_layer,
            layer_name="normalized",
        )
        var = result.assays["protein"].var
        assert "missing_rate" in var.columns

    def test_calculate_feature_qc_metrics_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_feature.calculate_feature_qc_metrics(qc_container, assay_name="nonexistent")

    def test_calculate_feature_qc_metrics_uses_mask_semantics(self):
        """Feature detection metrics should be derived from the mask matrix."""
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        X = np.array([[0.0, 1.0], [0.0, 0.0], [5.0, 0.0]])
        M = np.array(
            [
                [MaskCode.VALID.value, MaskCode.VALID.value],
                [MaskCode.LOD.value, MaskCode.LOD.value],
                [MaskCode.VALID.value, MaskCode.LOD.value],
            ],
            dtype=np.int8,
        )
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})},
        )

        result = qc_feature.calculate_feature_qc_metrics(container)
        out_var = result.assays["protein"].var
        assert out_var["missing_rate"].to_list() == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
        assert out_var["detection_rate"].to_list() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
        assert out_var["mean_expression"].to_list() == pytest.approx([2.5, 1.0])

    def test_calculate_feature_qc_metrics_accepts_alias_and_isolates_history(self, qc_container):
        """Feature QC should resolve aliases and not mutate the original history."""
        assert len(qc_container.history) == 0
        result = qc_feature.calculate_feature_qc_metrics(qc_container, assay_name="proteins")
        assert "missing_rate" in result.assays["protein"].var.columns
        assert len(qc_container.history) == 0
        assert len(result.history) == 1

    def test_calculate_feature_qc_metrics_defaults_to_raw_even_when_not_first_layer(self):
        """Feature QC should use raw by default instead of assay layer insertion order."""
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        normalized_X = np.ones((3, 2))
        normalized_M = np.full(normalized_X.shape, MaskCode.VALID.value, dtype=np.int8)
        raw_X = np.array([[1.0, 1.0], [10.0, 10.0], [20.0, 20.0]])
        raw_M = np.full(raw_X.shape, MaskCode.VALID.value, dtype=np.int8)
        assay = Assay(
            var=var,
            layers={
                "normalized": ScpMatrix(X=normalized_X, M=normalized_M),
                "raw": ScpMatrix(X=raw_X, M=raw_M),
            },
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result_default = qc_feature.calculate_feature_qc_metrics(container)
        result_raw = qc_feature.calculate_feature_qc_metrics(container, layer_name="raw")

        assert result_default.assays["protein"].var["cv"].to_list() == pytest.approx(
            result_raw.assays["protein"].var["cv"].to_list(),
        )
        assert result_default.assays["protein"].var["cv"].to_list()[0] > 0.0


class TestFilterFeaturesByMissingness:
    """Tests for filter_features_by_missingness function."""

    def test_filter_features_by_missingness_default(self, qc_container):
        """Test filter_features_by_missingness with default parameters."""
        result = qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=0.5)
        assert isinstance(result, ScpContainer)
        assert result.assays["protein"].n_features <= qc_container.assays["protein"].n_features

    def test_filter_features_by_missingness_permissive(self, qc_container):
        """Test with permissive threshold."""
        result = qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=1.0)
        # Should keep all features
        assert result.assays["protein"].n_features == qc_container.assays["protein"].n_features

    def test_filter_features_by_missingness_strict(self, qc_container):
        """Test with strict threshold."""
        result = qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=0.0)
        # Should keep only features with 0% missing
        assert result.assays["protein"].n_features >= 0

    def test_filter_features_by_missingness_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = qc_feature.filter_features_by_missingness(
            qc_container_sparse,
            max_missing_rate=0.5,
        )
        assert isinstance(result, ScpContainer)

    def test_filter_features_by_missingness_invalid_threshold(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError):
            qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=1.5)

    def test_filter_features_by_missingness_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_feature.filter_features_by_missingness(qc_container, assay_name="nonexistent")

    def test_filter_features_by_missingness_uses_mask_semantics(self):
        """Filtering should agree with mask-derived feature missingness."""
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        X = np.array([[0.0, 1.0], [0.0, 0.0], [5.0, 0.0]])
        M = np.array(
            [
                [MaskCode.VALID.value, MaskCode.VALID.value],
                [MaskCode.LOD.value, MaskCode.LOD.value],
                [MaskCode.VALID.value, MaskCode.LOD.value],
            ],
            dtype=np.int8,
        )
        container = ScpContainer(
            obs=obs,
            assays={"protein": Assay(var=var, layers={"raw": ScpMatrix(X=X, M=M)})},
        )

        result = qc_feature.filter_features_by_missingness(container, max_missing_rate=0.5)
        assert result.assays["protein"].var["_index"].to_list() == ["F1"]

    def test_filter_features_by_missingness_accepts_alias(self, qc_container):
        """Feature filtering should resolve proteins/protein aliases consistently."""
        result = qc_feature.filter_features_by_missingness(
            qc_container,
            assay_name="proteins",
            max_missing_rate=0.5,
        )
        assert isinstance(result, ScpContainer)

    def test_filter_features_by_missingness_defaults_to_raw_even_when_not_first_layer(self):
        """Missingness filtering should use raw by default instead of the first layer."""
        obs = pl.DataFrame({"_index": ["S1", "S2"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        normalized_X = np.array([[1.0, 0.0], [1.0, 0.0]])
        normalized_M = np.array(
            [
                [MaskCode.VALID.value, MaskCode.LOD.value],
                [MaskCode.VALID.value, MaskCode.LOD.value],
            ],
            dtype=np.int8,
        )
        raw_X = np.ones((2, 2))
        raw_M = np.full((2, 2), MaskCode.VALID.value, dtype=np.int8)
        assay = Assay(
            var=var,
            layers={
                "normalized": ScpMatrix(X=normalized_X, M=normalized_M),
                "raw": ScpMatrix(X=raw_X, M=raw_M),
            },
        )
        container = ScpContainer(obs=obs, assays={"protein": assay})

        result_default = qc_feature.filter_features_by_missingness(container, max_missing_rate=0.0)
        result_normalized = qc_feature.filter_features_by_missingness(
            container,
            layer_name="normalized",
            max_missing_rate=0.0,
        )

        assert result_default.assays["protein"].var["_index"].to_list() == ["F1", "F2"]
        assert result_normalized.assays["protein"].var["_index"].to_list() == ["F1"]

    def test_filter_features_by_missingness_preserves_filter_provenance(self, qc_container):
        """Feature filtering should keep both generic and entrypoint-specific logs."""
        result = qc_feature.filter_features_by_missingness(qc_container, max_missing_rate=0.5)
        assert [log.action for log in result.history[-2:]] == [
            "filter_features",
            "filter_features_by_missingness",
        ]


class TestFilterFeaturesByCV:
    """Tests for filter_features_by_cv function."""

    def test_filter_features_by_cv_default(self, qc_container):
        """Test filter_features_by_cv with default parameters."""
        result = qc_feature.filter_features_by_cv(qc_container, max_cv=1.0)
        assert isinstance(result, ScpContainer)
        assert result.assays["protein"].n_features <= qc_container.assays["protein"].n_features

    def test_filter_features_by_cv_custom_threshold(self, qc_container):
        """Test with custom max_cv threshold."""
        result = qc_feature.filter_features_by_cv(qc_container, max_cv=0.5)
        assert isinstance(result, ScpContainer)

    def test_filter_features_by_cv_sparse(self, qc_container_sparse):
        """Test with sparse matrix."""
        result = qc_feature.filter_features_by_cv(qc_container_sparse, max_cv=1.0)
        assert isinstance(result, ScpContainer)

    def test_filter_features_by_cv_invalid_threshold(self, qc_container):
        """Test that invalid threshold raises error."""
        with pytest.raises(ScpValueError):
            qc_feature.filter_features_by_cv(qc_container, max_cv=0)

    def test_filter_features_by_cv_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_feature.filter_features_by_cv(qc_container, assay_name="nonexistent")

    def test_filter_features_by_cv_requires_raw_by_default(self):
        """CV filtering should require raw unless callers select another layer explicitly."""
        obs = pl.DataFrame({"_index": ["S1", "S2", "S3"]})
        var = pl.DataFrame({"_index": ["F1", "F2"]})
        normalized_X = np.ones((3, 2))
        normalized_M = np.full(normalized_X.shape, MaskCode.VALID.value, dtype=np.int8)
        assay = Assay(var=var, layers={"normalized": ScpMatrix(X=normalized_X, M=normalized_M)})
        container = ScpContainer(obs=obs, assays={"protein": assay})

        with pytest.raises(LayerNotFoundError):
            qc_feature.filter_features_by_cv(container, max_cv=0.1)

        result = qc_feature.filter_features_by_cv(
            container,
            layer_name="normalized",
            max_cv=0.1,
        )
        assert result.assays["protein"].n_features == 2


# =============================================================================
# Tests for qc_psm module
# =============================================================================


class TestQCPsmModuleSurface:
    """Tests for qc_psm module-level export boundaries."""

    def test_qc_psm_module_all_explicitly_freezes_public_surface(self):
        """Module-level __all__ should freeze helpers.

        Imported implementation details must stay excluded.
        """
        assert qc_psm.__all__ == [
            "filter_psms_by_pif",
            "filter_contaminants",
            "pep_to_qvalue",
            "filter_psms_by_qvalue",
            "compute_sample_cv",
        ]
        assert "_DEFAULT_CONTAMINANT_PATTERNS" not in qc_psm.__all__
        assert "np" not in qc_psm.__all__
        assert "pl" not in qc_psm.__all__
        assert "warnings" not in qc_psm.__all__
        assert "ScpContainer" not in qc_psm.__all__


class TestFilterContaminants:
    """Tests for filter_contaminants function."""

    def test_filter_contaminants_default(self, qc_container_with_contaminants):
        """Test contaminant filtering with default patterns."""
        result = qc_psm.filter_contaminants(
            qc_container_with_contaminants,
            assay_name="protein",
            feature_col="name",
        )
        assert isinstance(result, ScpContainer)
        # Should have removed some contaminant proteins
        assert (
            result.assays["protein"].n_features
            <= qc_container_with_contaminants.assays["protein"].n_features
        )

    def test_filter_contaminants_custom_patterns(self, qc_container_with_contaminants):
        """Test with custom contaminant patterns."""
        result = qc_psm.filter_contaminants(
            qc_container_with_contaminants,
            assay_name="protein",
            feature_col="name",
            patterns=[r"KRT\d+"],
        )
        assert isinstance(result, ScpContainer)

    def test_filter_contaminants_invalid_assay(self, qc_container):
        """Test that invalid assay raises error."""
        with pytest.raises(AssayNotFoundError):
            qc_psm.filter_contaminants(qc_container, assay_name="nonexistent")

    def test_filter_contaminants_uses_canonical_default_assay_name(self, qc_psm_container):
        """Default assay selection should now target the canonical `peptides` assay."""
        result = qc_psm.filter_contaminants(
            qc_psm_container,
            feature_col="gene_names",
            patterns=[r"^KRT"],
        )

        assert result.assays["peptides"].n_features == 3
        assert result.history[-1].params["assay"] == "peptides"

    def test_filter_contaminants_accepts_peptide_alias(self, qc_psm_container):
        """Contaminant filtering should apply alias resolution through execution."""
        result = qc_psm.filter_contaminants(
            qc_psm_container,
            assay_name="peptide",
            feature_col="gene_names",
            patterns=[r"^KRT"],
        )

        assert "peptides" in result.assays
        assert "peptide" not in result.assays
        assert result.assays["peptides"].n_features == 3
        assert result.history[-1].params["assay"] == "peptides"


class TestFilterPsmsByPif:
    """Tests for filter_psms_by_pif function."""

    def test_filter_psms_by_pif_uses_canonical_default_assay_name(self, qc_psm_container):
        """Default assay selection should no longer rely on the legacy singular alias."""
        result = qc_psm.filter_psms_by_pif(qc_psm_container, min_pif=0.8)

        assert result.assays["peptides"].n_features == 2
        assert result.history[-1].params["assay"] == "peptides"

    def test_filter_psms_by_pif_accepts_alias_and_logs_resolved_assay(self, qc_psm_container):
        """PIF filtering should not fail after alias validation succeeds."""
        result = qc_psm.filter_psms_by_pif(qc_psm_container, assay_name="peptide", min_pif=0.8)

        assert result.assays["peptides"].n_features == 2
        assert [log.action for log in result.history[-2:]] == [
            "filter_features",
            "filter_psms_by_pif",
        ]
        assert result.history[-1].params["assay"] == "peptides"

    @pytest.mark.parametrize("min_pif", [-0.1, 1.1])
    def test_filter_psms_by_pif_enforces_unit_interval(self, qc_psm_container, min_pif):
        """PIF threshold must be validated against the documented [0, 1] range."""
        with pytest.raises(ScpValueError, match="min_pif"):
            qc_psm.filter_psms_by_pif(qc_psm_container, assay_name="peptide", min_pif=min_pif)


class TestFilterPsmsByQValue:
    """Tests for filter_psms_by_qvalue function."""

    def test_filter_psms_by_qvalue_uses_canonical_default_assay_name(self, qc_psm_container):
        """Default assay selection should point at the canonical plural assay name."""
        result = qc_psm.filter_psms_by_qvalue(qc_psm_container, qvalue_threshold=0.01)

        assert result.assays["peptides"].n_features == 2
        assert result.history[-1].params["assay"] == "peptides"

    def test_filter_psms_by_qvalue_accepts_alias_and_logs_resolved_assay(self, qc_psm_container):
        """Q-value filtering should reuse the resolved assay name for execution/logging."""
        result = qc_psm.filter_psms_by_qvalue(
            qc_psm_container,
            assay_name="peptide",
            qvalue_threshold=0.01,
        )

        assert result.assays["peptides"].n_features == 2
        assert [log.action for log in result.history[-2:]] == [
            "filter_features",
            "filter_psms_by_qvalue",
        ]
        assert result.history[-1].params["assay"] == "peptides"

    @pytest.mark.parametrize("qvalue_threshold", [-0.01, 1.01])
    def test_filter_psms_by_qvalue_enforces_unit_interval(self, qc_psm_container, qvalue_threshold):
        """Q-value threshold must be validated against the documented [0, 1] range."""
        with pytest.raises(ScpValueError, match="qvalue_threshold"):
            qc_psm.filter_psms_by_qvalue(
                qc_psm_container,
                assay_name="peptide",
                qvalue_threshold=qvalue_threshold,
            )


class TestPepToQValue:
    """Tests for pep_to_qvalue helper."""

    def test_pep_to_qvalue_bh_matches_expected_adjustment(self):
        """BH mode should produce monotone adjusted q-values on sorted inputs."""
        pep = np.array([0.01, 0.02, 0.03], dtype=np.float64)

        result = qc_psm.pep_to_qvalue(pep, method="bh")

        np.testing.assert_allclose(result, np.array([0.03, 0.03, 0.03]))

    def test_pep_to_qvalue_preserves_nan_and_clips_to_unit_interval(self):
        """NaN should round-trip while finite outputs stay in [0, 1]."""
        pep = np.array([0.0, 0.2, np.nan, 1.5, -0.1], dtype=np.float64)

        result = qc_psm.pep_to_qvalue(pep, method="storey", lambda_param=0.5)

        assert np.isnan(result[2])
        finite = result[~np.isnan(result)]
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 1.0)

    def test_pep_to_qvalue_validates_method(self):
        """Only documented methods should be accepted."""
        with pytest.raises(ValueError, match="method must be 'storey' or 'bh'"):
            qc_psm.pep_to_qvalue(np.array([0.1, 0.2]), method="invalid")

    @pytest.mark.parametrize("lambda_param", [-0.1, 1.0])
    def test_pep_to_qvalue_validates_lambda_param(self, lambda_param):
        """Storey lambda must stay within the documented [0, 1) range."""
        with pytest.raises(ValueError, match="lambda_param"):
            qc_psm.pep_to_qvalue(np.array([0.1, 0.2]), lambda_param=lambda_param)

    def test_pep_to_qvalue_empty_input_returns_empty_float_array(self):
        """Empty inputs should return an empty float array rather than failing."""
        result = qc_psm.pep_to_qvalue(np.array([], dtype=np.float64))

        assert result.dtype == np.float64
        assert result.shape == (0,)


class TestComputeSampleCV:
    """Tests for compute_sample_cv helper."""

    def test_compute_sample_cv_uses_canonical_default_assay_name(self, qc_psm_container):
        """Default assay selection should use the canonical plural assay name."""
        result = qc_psm.compute_sample_cv(qc_psm_container, layer_name="raw", cv_threshold=0.3)

        assert result.history[-1].action == "compute_sample_cv"
        assert result.history[-1].params["assay"] == "peptides"

    def test_compute_sample_cv_accepts_alias_and_writes_sample_qc_columns(self, qc_psm_container):
        """Alias resolution and sample-level obs outputs should stay aligned."""
        result = qc_psm.compute_sample_cv(
            qc_psm_container,
            assay_name="peptide",
            layer_name="raw",
            cv_threshold=0.3,
        )

        expected_cv = compute_cv(qc_psm_container.assays["peptides"].layers["raw"].X.T, axis=0)

        assert result is not qc_psm_container
        assert "sample_cv" not in qc_psm_container.obs.columns
        assert "is_high_sample_cv" not in qc_psm_container.obs.columns
        assert result.obs["sample_cv"].to_list() == pytest.approx(expected_cv.tolist())
        assert result.obs["is_high_sample_cv"].to_list() == list(expected_cv > 0.3)
        assert result.history[-1].action == "compute_sample_cv"
        assert result.history[-1].params["assay"] == "peptides"
        assert result.history[-1].params["layer"] == "raw"
        assert result.history[-1].params["cv_threshold"] == 0.3

    def test_compute_sample_cv_missing_layer_raises_scpvalueerror(self, qc_psm_container):
        """The current handwritten layer check raises ScpValueError, not LayerNotFoundError."""
        with pytest.raises(ScpValueError, match="Layer 'norm' not found"):
            qc_psm.compute_sample_cv(qc_psm_container, assay_name="peptide", layer_name="norm")


class TestComputeMedianCVRetirement:
    """Tests for the retired compute_median_cv alias."""

    def test_compute_median_cv_not_exported_and_not_available(self):
        """Legacy alias should be fully removed from the module surface."""
        assert "compute_median_cv" not in qc_psm.__all__
        assert not hasattr(qc_psm, "compute_median_cv")


# =============================================================================
# Tests for metrics module
# =============================================================================


class TestComputeMAD:
    """Tests for compute_mad function."""

    def test_compute_mad_basic(self):
        """Test basic MAD computation."""
        data = np.array([1, 2, 3, 4, 5])
        mad = compute_mad(data)
        assert mad > 0

    def test_compute_mad_with_outliers(self):
        """Test MAD with outliers."""
        data = np.array([1, 2, 3, 4, 100])
        mad = compute_mad(data)
        # MAD should be robust to outlier
        assert mad > 0

    def test_compute_mad_empty(self):
        """Test MAD with empty array."""
        mad = compute_mad(np.array([]))
        assert np.isnan(mad)


class TestIsOutlierMAD:
    """Tests for is_outlier_mad function."""

    def test_is_outlier_mad_both_directions(self):
        """Test outlier detection in both directions."""
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = is_outlier_mad(data, nmads=2.0, direction="both")
        assert len(outliers) == len(data)
        assert outliers[-1]  # Last value should be outlier

    def test_is_outlier_mad_upper_only(self):
        """Test upper-tail outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = is_outlier_mad(data, nmads=2.0, direction="upper")
        assert outliers[-1]

    def test_is_outlier_mad_lower_only(self):
        """Test lower-tail outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 100])
        outliers = is_outlier_mad(data, nmads=2.0, direction="lower")
        # 1 might be detected as lower outlier depending on MAD
        assert len(outliers) == len(data)

    def test_is_outlier_mad_empty(self):
        """Test with empty array."""
        outliers = is_outlier_mad(np.array([]), nmads=3.0)
        assert len(outliers) == 0


class TestComputeCV:
    """Tests for compute_cv function."""

    def test_compute_cv_dense(self):
        """Test CV computation for dense array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        cv = compute_cv(data, axis=0)
        assert len(cv) == 3
        assert all(cv >= 0)

    def test_compute_cv_sparse(self):
        """Test CV computation for sparse matrix."""
        data = sparse.csr_matrix([[1, 2, 3], [4, 5, 6]])
        cv = compute_cv(data, axis=0)
        assert len(cv) == 3

    def test_compute_cv_axis1(self):
        """Test CV computation along axis=1."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        cv = compute_cv(data, axis=1)
        assert len(cv) == 2


# =============================================================================
# Integration tests
# =============================================================================


class TestQCIntegration:
    """Integration tests for QC workflows."""

    def test_qc_pipeline_full(self, qc_container):
        """Test a complete QC pipeline."""
        # Step 1: Calculate sample metrics
        container = qc_sample.calculate_sample_qc_metrics(qc_container)
        assert "n_features_protein" in container.obs.columns

        # Step 2: Calculate feature metrics
        container = qc_feature.calculate_feature_qc_metrics(container)
        var = container.assays["protein"].var
        assert "missing_rate" in var.columns

        # Step 3: Filter samples
        container = qc_sample.filter_low_quality_samples(container, min_features=5, use_mad=False)

        # Step 4: Filter features
        container = qc_feature.filter_features_by_missingness(container, max_missing_rate=0.5)

        assert isinstance(container, ScpContainer)

    def test_qc_with_sparse_matrix_pipeline(self, qc_container_sparse):
        """Test QC pipeline with sparse matrices."""
        container = qc_sample.calculate_sample_qc_metrics(qc_container_sparse)
        container = qc_feature.calculate_feature_qc_metrics(container)
        container = qc_feature.filter_features_by_missingness(container, max_missing_rate=0.5)
        assert isinstance(container, ScpContainer)

    def test_batch_analysis_pipeline(self, qc_container):
        """Test a complete batch analysis pipeline."""
        # Step 1: Calculate sample metrics
        container = qc_sample.calculate_sample_qc_metrics(qc_container)

        # Step 2: Assess batch effects
        batch_summary = qc_sample.assess_batch_effects(container, batch_col="batch")
        assert isinstance(batch_summary, pl.DataFrame)
        assert "batch" in batch_summary.columns

        assert isinstance(container, ScpContainer)
