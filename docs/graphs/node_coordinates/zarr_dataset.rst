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
     data:
       node_builder:
         _target_: anemoi.graphs.nodes.ZarrDatasetNodes
         dataset: /path/to/dataset.zarr
       attributes: ...

where `dataset` is the path to the Zarr dataset. The
``ZarrDatasetNodes`` class supports operations compatible with
:ref:`anemoi-datasets <anemoi-datasets:index-page>`, such as "cutout".
Below, an example of how to use the "cutout" operation directly within
:ref:`anemoi-graphs <anemoi-graphs:index-page>`.

.. code:: yaml

   nodes:
     data:
       node_builder:
         _target_: anemoi.graphs.nodes.ZarrDatasetNodes
         dataset:
           cutout:
             dataset: /path/to/lam_dataset.zarr
             dataset: /path/to/boundary_forcing.zarr
           adjust: "all"
       attributes: ...
