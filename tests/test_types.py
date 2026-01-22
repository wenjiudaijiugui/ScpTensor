"""Tests for scptensor.core.types module.

Tests verify that type aliases work correctly with various input formats
and can be used for type checking and runtime validation.
"""

import json

import numpy as np
import polars as pl

from scptensor.core.structures import MaskCode
from scptensor.core.types import (
    BooleanMask,
    DenseMatrix,
    FeatureIDs,
    Indices,
    JsonArray,
    JsonObject,
    JsonValue,
    LayerMetadataDict,
    MaskMatrix,
    Matrix,
    MatrixOperation,
    MetadataDict,
    MetadataValue,
    ProvenanceParams,
    RowFunction,
    SampleIDs,
    SerializableDict,
    SparseMatrix,
)


class TestMatrixTypes:
    """Tests for matrix type aliases."""

    def test_dense_matrix_type(self):
        """Test DenseMatrix type alias."""
        matrix: DenseMatrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
        assert matrix.dtype == np.float64

    def test_sparse_matrix_type(self):
        """Test SparseMatrix type alias."""
        from scipy import sparse

        matrix: SparseMatrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 4.0]])
        assert sparse.issparse(matrix)
        assert matrix.shape == (2, 2)

    def test_matrix_union_dense(self):
        """Test Matrix type alias with dense matrix."""
        matrix: Matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(matrix, np.ndarray)

    def test_matrix_union_sparse(self):
        """Test Matrix type alias with sparse matrix."""
        from scipy import sparse

        matrix: Matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 4.0]])
        from scptensor.core.sparse_utils import is_sparse_matrix

        assert is_sparse_matrix(matrix)

    def test_mask_matrix_dense(self):
        """Test MaskMatrix type alias with dense array."""
        mask: MaskMatrix = np.array(
            [[MaskCode.VALID, MaskCode.MBR], [MaskCode.LOD, MaskCode.VALID]],
            dtype=np.int8,
        )
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.int8

    def test_mask_matrix_sparse(self):
        """Test MaskMatrix type alias with sparse matrix."""
        from scipy import sparse

        mask: MaskMatrix = sparse.csr_matrix(
            [[MaskCode.VALID, 0], [MaskCode.LOD, MaskCode.VALID]], dtype=np.int8
        )
        assert sparse.issparse(mask)


class TestSerializationTypes:
    """Tests for serialization type aliases."""

    def test_json_value_primitives(self):
        """Test JsonValue type alias with primitive types."""
        null_val: JsonValue = None
        bool_val: JsonValue = True
        int_val: JsonValue = 42
        float_val: JsonValue = 3.14
        str_val: JsonValue = "test"

        # All should be JSON serializable
        assert json.dumps(null_val) == "null"
        assert json.dumps(bool_val) == "true"
        assert json.dumps(int_val) == "42"
        assert json.dumps(float_val) == "3.14"
        assert json.dumps(str_val) == '"test"'

    def test_json_array(self):
        """Test JsonArray type alias."""
        arr: JsonArray = [1, 2, 3, "four", True]
        json_str = json.dumps(arr)
        assert json.loads(json_str) == arr

    def test_json_object(self):
        """Test JsonObject type alias."""
        obj: JsonObject = {"key1": "value1", "key2": 42, "key3": True}
        json_str = json.dumps(obj)
        assert json.loads(json_str) == obj

    def test_json_nested(self):
        """Test JsonValue with nested structures."""
        nested: JsonValue = {
            "array": [1, 2, 3],
            "nested_obj": {"a": 1, "b": 2},
            "mixed": [1, "two", True],
        }
        json_str = json.dumps(nested)
        recovered = json.loads(json_str)
        assert recovered == nested

    def test_serializable_dict(self):
        """Test SerializableDict type alias."""
        data: SerializableDict = {
            "version": "1.0",
            "data": [1, 2, 3],
            "nested": {"key": "value"},
            "flag": True,
            "count": 42,
        }
        json_str = json.dumps(data)
        assert json_str is not None
        recovered = json.loads(json_str)
        assert recovered == data

    def test_provenance_params(self):
        """Test ProvenanceParams type alias."""
        params: ProvenanceParams = {
            "action": "normalize",
            "method": "log",
            "base": 2.0,
            "offset": 1.0,
            "metadata": {"key": "value", "nested": [1, 2, 3]},
        }
        assert params["action"] == "normalize"
        assert params["base"] == 2.0
        # Should be JSON serializable
        json.dumps(params)

    def test_layer_metadata_dict(self):
        """Test LayerMetadataDict type alias."""
        metadata: LayerMetadataDict = {
            "creation_time": "2024-01-01",
            "method": "knn_impute",
            "parameters": {"n_neighbors": 5, "weights": "distance"},
            "quality_score": 0.95,
        }
        assert metadata["method"] == "knn_impute"
        # Should be JSON serializable
        json.dumps(metadata)


class TestFunctionTypes:
    """Tests for function type aliases."""

    def test_row_function_type(self):
        """Test RowFunction type alias."""

        def my_sum(arr: np.ndarray) -> float:
            return float(np.sum(arr))

        func: RowFunction = my_sum
        result = func(np.array([1.0, 2.0, 3.0]))
        assert result == 6.0

    def test_row_function_lambda(self):
        """Test RowFunction with lambda."""
        func: RowFunction = lambda arr: float(np.mean(arr))
        result = func(np.array([1.0, 2.0, 3.0, 4.0]))
        assert result == 2.5

    def test_row_function_builtin(self):
        """Test RowFunction with numpy function."""
        func: RowFunction = lambda arr: float(np.max(arr))
        result = func(np.array([1.0, 5.0, 3.0]))
        assert result == 5.0

    def test_matrix_operation_return_matrix(self):
        """Test MatrixOperation returning matrix."""

        def normalize(mat: Matrix) -> Matrix:
            if isinstance(mat, np.ndarray):
                return mat / np.max(mat)
            return mat

        op: MatrixOperation = normalize
        result = op(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert isinstance(result, np.ndarray)
        assert np.max(result) == 1.0

    def test_matrix_operation_return_float(self):
        """Test MatrixOperation returning float."""

        def total_sum(mat: Matrix) -> float:
            if isinstance(mat, np.ndarray):
                return float(np.sum(mat))

            return float(mat.sum())

        op: MatrixOperation = total_sum
        result = op(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert result == 10.0


class TestIDAndIndexTypes:
    """Tests for ID and index type aliases."""

    def test_sample_ids_list(self):
        """Test SampleIDs with list format."""
        ids: SampleIDs = ["sample1", "sample2", "sample3"]
        assert len(ids) == 3
        assert ids[0] == "sample1"

    def test_sample_ids_numpy(self):
        """Test SampleIDs with NumPy array."""
        ids: SampleIDs = np.array(["sample1", "sample2", "sample3"])
        assert isinstance(ids, np.ndarray)
        assert len(ids) == 3

    def test_sample_ids_polars(self):
        """Test SampleIDs with Polars Series."""
        ids: SampleIDs = pl.Series(["sample1", "sample2", "sample3"])
        assert isinstance(ids, pl.Series)
        assert len(ids) == 3

    def test_feature_ids_list(self):
        """Test FeatureIDs with list format."""
        ids: FeatureIDs = ["P123", "P456", "P789"]
        assert len(ids) == 3
        assert ids[0] == "P123"

    def test_feature_ids_numpy(self):
        """Test FeatureIDs with NumPy array."""
        ids: FeatureIDs = np.array(["feat1", "feat2", "feat3"])
        assert isinstance(ids, np.ndarray)
        assert len(ids) == 3

    def test_feature_ids_polars(self):
        """Test FeatureIDs with Polars Series."""
        ids: FeatureIDs = pl.Series(["protein1", "protein2", "protein3"])
        assert isinstance(ids, pl.Series)
        assert len(ids) == 3

    def test_indices_list(self):
        """Test Indices with list format."""
        idx: Indices = [0, 5, 10, 15]
        assert len(idx) == 4
        assert idx[0] == 0

    def test_indices_numpy(self):
        """Test Indices with NumPy array."""
        idx: Indices = np.array([0, 1, 2, 3])
        assert isinstance(idx, np.ndarray)
        assert idx.dtype in [np.int32, np.int64]

    def test_boolean_mask_numpy(self):
        """Test BooleanMask with NumPy array."""
        mask: BooleanMask = np.array([True, False, True, True, False])
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_boolean_mask_polars(self):
        """Test BooleanMask with Polars Series."""
        mask: BooleanMask = pl.Series([True, False, True, True, False], dtype=pl.Boolean)
        assert isinstance(mask, pl.Series)
        assert mask.dtype == pl.Boolean


class TestMetadataTypes:
    """Tests for metadata type aliases."""

    def test_metadata_value_none(self):
        """Test MetadataValue with None."""
        val: MetadataValue = None
        assert val is None

    def test_metadata_value_bool(self):
        """Test MetadataValue with bool."""
        val: MetadataValue = True
        assert val is True

    def test_metadata_value_int(self):
        """Test MetadataValue with int."""
        val: MetadataValue = 42
        assert val == 42

    def test_metadata_value_float(self):
        """Test MetadataValue with float."""
        val: MetadataValue = 3.14
        assert val == 3.14

    def test_metadata_value_str(self):
        """Test MetadataValue with str."""
        val: MetadataValue = "test string"
        assert val == "test string"

    def test_metadata_value_list(self):
        """Test MetadataValue with list."""
        val: MetadataValue = [1, 2, 3, "four"]
        assert len(val) == 4

    def test_metadata_value_dict(self):
        """Test MetadataValue with dict."""
        val: MetadataValue = {"key1": "value1", "key2": 42}
        assert val["key1"] == "value1"

    def test_metadata_dict_basic(self):
        """Test MetadataDict with basic types."""
        metadata: MetadataDict = {
            "name": "test",
            "count": 10,
            "flag": True,
            "value": 3.14,
            "missing": None,
        }
        assert metadata["name"] == "test"
        assert metadata["count"] == 10

    def test_metadata_dict_nested(self):
        """Test MetadataDict with nested structures."""
        metadata: MetadataDict = {
            "simple": "value",
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "mixed": [1, "two", True],
        }
        assert metadata["simple"] == "value"
        assert len(metadata["list"]) == 3
        assert metadata["nested"]["a"] == 1


class TestTypeCompatibility:
    """Tests for type compatibility and conversions."""

    def test_sample_ids_conversion_list_to_numpy(self):
        """Test converting SampleIDs from list to NumPy."""
        ids_list: SampleIDs = ["s1", "s2", "s3"]
        ids_array: SampleIDs = np.array(ids_list)
        assert isinstance(ids_array, np.ndarray)
        assert len(ids_array) == 3

    def test_sample_ids_conversion_numpy_to_list(self):
        """Test converting SampleIDs from NumPy to list."""
        ids_array: SampleIDs = np.array(["s1", "s2", "s3"])
        ids_list: SampleIDs = ids_array.tolist()
        assert isinstance(ids_list, list)
        assert len(ids_list) == 3

    def test_boolean_mask_conversions(self):
        """Test BooleanMask conversions between formats."""
        # NumPy to Polars
        mask_np: BooleanMask = np.array([True, False, True])
        mask_pl: BooleanMask = pl.Series(mask_np)
        assert isinstance(mask_pl, pl.Series)

        # Polars to NumPy
        mask_np2: BooleanMask = mask_pl.to_numpy()
        assert isinstance(mask_np2, np.ndarray)
        assert np.array_equal(mask_np, mask_np2)

    def test_indices_conversions(self):
        """Test Indices conversions between formats."""
        # List to NumPy
        idx_list: Indices = [0, 5, 10]
        idx_array: Indices = np.array(idx_list)
        assert isinstance(idx_array, np.ndarray)

        # NumPy to list
        idx_list2: Indices = idx_array.tolist()
        assert isinstance(idx_list2, list)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_sample_ids(self):
        """Test SampleIDs with empty list."""
        ids: SampleIDs = []
        assert len(ids) == 0

    def test_empty_feature_ids(self):
        """Test FeatureIDs with empty array."""
        ids: FeatureIDs = np.array([])
        assert len(ids) == 0

    def test_empty_indices(self):
        """Test Indices with empty list."""
        idx: Indices = []
        assert len(idx) == 0

    def test_empty_boolean_mask(self):
        """Test BooleanMask with empty array."""
        mask: BooleanMask = np.array([], dtype=bool)
        assert len(mask) == 0

    def test_empty_metadata_dict(self):
        """Test MetadataDict with empty dict."""
        metadata: MetadataDict = {}
        assert len(metadata) == 0

    def test_json_value_empty_structures(self):
        """Test JsonValue with empty structures."""
        empty_array: JsonValue = []
        empty_object: JsonValue = {}
        assert json.dumps(empty_array) == "[]"
        assert json.dumps(empty_object) == "{}"
