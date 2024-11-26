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

The ``CutOutZarrDatasetNodes`` class supports 2 input datasets, one for
the LAM model and one for the boundary forcing. To define the `node
coordinates` combining multiple Zarr datasets, you can use the following
YAML configuration:

.. code:: yaml

   nodes:
     data:  # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.CutOutZarrDatasetNodes
         lam_dataset: /path/to/lam_dataset.zarr
         forcing_dataset: /path/to/boundary_forcing.zarr
         thinning: 25 # sample every n-th point (only for lam_dataset)
       attributes: ...
