Quick Start Guide
=================

This guide will help you get started with ScpTensor.

Basic Usage
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install scptensor

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   import scptensor as scp

   # Load from CSV files
   container = scp.read_csv(
       data_path="expression.csv",
       metadata_path="metadata.csv"
   )

   # Or create a container programmatically
   import polars as pl
   import numpy as np

   container = scp.ScpContainer(
       obs=pl.DataFrame({"sample": ["A", "B", "C"]}),
       assays={
           "proteins": scp.Assay(
               var=pl.DataFrame({"feature": ["P1", "P2", "P3"]}),
               layers={
                   "counts": scp.ScpMatrix(
                       X=np.random.rand(3, 3),
                       M=np.zeros((3, 3), dtype=np.uint8)
                   )
               }
           )
       }
   )

Quality Control
~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect outliers
   container = scp.basic_qc(container)
   container = scp.detect_outliers(
       container,
       n_std=3,
       metrics=["n_proteins", "total_intensity"]
   )

   # Filter samples
   container = scp.filter_samples(
       container,
       min_n_proteins=100,
       min_total_intensity=1000
   )

Normalization
~~~~~~~~~~~~~

.. code-block:: python

   # CLR normalization (recommended for proteomics)
   container = scp.clr_normalize(
       container,
       assay_name="proteins",
       layer_name="counts",
       new_layer_name="clr"
   )

   # TSS (Total Sum Scaling)
   container = scp.tss_normalize(
       container,
       assay_name="proteins",
       target_depth=10000
   )

   # VST (Variance Stabilizing Transformation)
   container = scp.vst_normalize(
       container,
       assay_name="proteins"
   )

Imputation
~~~~~~~~~~

.. code-block:: python

   # KNN imputation
   container = scp.knn_impute(
       container,
       assay_name="proteins",
       layer_name="clr",
       k=10
   )

   # MissForest imputation
   container = scp.missforest_impute(
       container,
       assay_name="proteins",
       layer_name="clr",
       n_estimators=100
   )

Batch Correction
~~~~~~~~~~~~~~~

.. code-block:: python

   # ComBat batch correction
   container = scp.combat_correct(
       container,
       assay_name="proteins",
       batch_key="batch"
   )

   # Harmony integration
   container = scp.harmony_integrate(
       container,
       assay_name="proteins",
       batch_key="batch"
   )

Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # PCA
   container = scp.pca(
       container,
       assay_name="proteins",
       n_components=50
   )

   # UMAP
   container = scp.umap(
       container,
       assay_name="proteins",
       n_neighbors=15,
       min_dist=0.1
   )

Clustering
~~~~~~~~~~

.. code-block:: python

   # KMeans clustering
   container = scp.kmeans_cluster(
       container,
       assay_name="proteins",
       layer_name="pca",
       n_clusters=8
   )

   # Graph-based clustering
   container = scp.graph_cluster(
       container,
       assay_name="proteins",
       layer_name="umap",
       resolution=1.0
   )

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   # PCA plot
   scp.pca_plot(container, color_by="cluster")

   # UMAP plot
   scp.umap_plot(container, color_by="batch")

   # Heatmap
   scp.heatmap_plot(
       container,
       features=["P1", "P2", "P3"],
       cluster_samples=True
   )

Next Steps
----------

* Read the :doc:`api/index` for detailed API documentation
* Check out the examples in the repository
* Review the data structures in :mod:`scptensor.core`
