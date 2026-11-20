.. _kmeans_clusters:

########################
 K-means cluster nodes
########################

This method defines nodes at the centroids of a weighted spherical
K-means clustering of an existing set of nodes. It is intended for
domains that are irregular or only cover part of the globe, such as the
ocean, where a uniform global mesh would place most of its nodes outside
the region of interest.

The number of nodes is set indirectly through the ``ratio`` argument:
the clustering produces ``round(num_reference_points / ratio)``
centroids, so a larger ratio gives a coarser mesh.

.. code:: yaml

   nodes:
     data:  # name of the reference nodes
       node_builder:
         _target_: anemoi.graphs.nodes.AnemoiDatasetNodes
         dataset: /path/to/ocean-dataset.zarr
       attributes:
         sea_mask:
           _target_: anemoi.graphs.nodes.attributes.NonmissingAnemoiDatasetVariable
           variable: sst
     hidden:
       node_builder:
         _target_: anemoi.graphs.nodes.KMeansClusteredNodes
         reference_node_name: data
         ratio: 20
         mask_attr_name: sea_mask  # optional
         kmeans_kwargs:            # optional
           random_state: 0

The ``mask_attr_name`` argument restricts the clustering to a subset of
the reference nodes, given by a boolean node attribute. Only the nodes
where the attribute is ``True`` are clustered, so the resulting mesh
follows the shape of the area of interest.

By default every reference node contributes equally to the clustering.
The ``node_weight_attribute_name`` argument points at a node attribute
used to weight them instead, which is useful to compensate for the
varying density of the reference grid:

.. code:: yaml

   nodes:
     hidden:
       node_builder:
         _target_: anemoi.graphs.nodes.KMeansClusteredNodes
         reference_node_name: data
         ratio: 20
         mask_attr_name: sea_mask
         node_weight_attribute_name: area_weight

The ``reference_node_name`` can refer to another set of
``KMeansClusteredNodes``, which gives the next level of a hierarchy of
meshes:

.. code:: yaml

   nodes:
     hidden_1:
       node_builder:
         _target_: anemoi.graphs.nodes.KMeansClusteredNodes
         reference_node_name: data
         ratio: 20
     hidden_2:
       node_builder:
         _target_: anemoi.graphs.nodes.KMeansClusteredNodes
         reference_node_name: hidden_1
         ratio: 4

The clustering is performed with `scikit-learn's MiniBatchKMeans
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html>`_
on the 3D cartesian coordinates of the nodes. Any of its arguments can
be passed through ``kmeans_kwargs``.

.. note::

   K-means is initialised randomly, so the resulting nodes are only
   reproducible if a seed is set with ``kmeans_kwargs.random_state``.

The nodes produced by this builder have no regular structure, so the
usual multi-scale edge builders do not apply to them. Use
:ref:`delaunay` to build the connections within the mesh.
