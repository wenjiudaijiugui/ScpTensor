.. ScpTensor documentation master file

ScpTensor: Single-Cell Proteomics Analysis Framework
=====================================================

Welcome to ScpTensor's documentation!

ScpTensor is a Python library for single-cell proteomics (SCP) data analysis,
featuring a hierarchical data structure and comprehensive analysis tools.

**Version:** |version|
**Python:** >=3.12

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api/index

Overview
--------

ScpTensor provides a complete toolkit for single-cell proteomics data analysis:

* **Hierarchical Data Structure**: `ScpContainer` → `Assay` → `ScpMatrix`
* **Quality Control**: Outlier detection, filtering, quality metrics
* **Normalization**: 9+ normalization methods (CLR, TSS, VST, etc.)
* **Imputation**: KNN, MissForest, PPCA, SVD
* **Batch Correction**: ComBat, Harmony, MNN, Scanorama
* **Dimensionality Reduction**: PCA, UMAP
* **Clustering**: KMeans, graph-based clustering
* **Visualization**: Publication-quality plots with SciencePlots

Key Features
------------

* **Provenance Tracking**: Every operation is logged with mask codes
* **Type-Safe**: Complete type annotations with mypy support
* **Fast**: Numba JIT compilation, sparse matrix support
* **Modular**: Use individual functions or complete pipelines

Installation
------------

.. code-block:: bash

   pip install scptensor

Quick Start
-----------

.. code-block:: python

   import scptensor as scp

   # Load data
   container = scp.read_csv("data.csv", metadata="meta.csv")

   # Quality control
   container = scp.detect_outliers(container, n_std=3)

   # Normalize
   container = scp.clr_normalize(container, assay_name="proteins")

   # Impute missing values
   container = scp.knn_impute(container, assay_name="proteins", k=10)

   # Dimensionality reduction
   container = scp.pca(container, assay_name="proteins", n_components=50)

   # Cluster
   container = scp.kmeans_cluster(container, n_clusters=8)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
