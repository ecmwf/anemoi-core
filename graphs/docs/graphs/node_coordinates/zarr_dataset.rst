.. _zarr-file:

###################
 From Zarr dataset
###################

This class builds a set of nodes from a Zarr dataset. The nodes are
defined by the coordinates of the dataset. The ZarrDataset class
supports operations compatible with :ref:`anemoi-datasets
<anemoi-datasets:index-page>`.

To define the `node coordinates` based on a Zarr dataset, you can use
the following YAML configuration:

.. code:: yaml

   nodes:
     data: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.ZarrDatasetNodes
         dataset: /path/to/dataset.zarr
       attributes: ...

where `dataset` is the path to the Zarr dataset.

The ``ZarrDatasetNodes`` class supports operations over multiple
datasets. For example, the `cutout` operation supports combining a
regional dataset and a global dataset to enable both limited area and
stretched grids. To define the `node coordinates` that combine multiple
Zarr datasets, you can use the following YAML configuration:

.. code:: yaml

   nodes:
     data:  # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.ZarrDatasetNodes
         cutout:
         - dataset: /path/to/lam_dataset.zarr
           thinning: 25 # sample every n-th point (only for lam_dataset), optional
         - dataset: /path/to/boundary_forcing.zarr
       attributes: ...
