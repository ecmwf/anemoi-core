.. _usage-limited_area:

#############################
 Limited Area Modeling (LAM)
#############################

Anemoi Graphs brings another level of flexibility to the user by
allowing the definition of limited area graphs.

*****************************************
 Define hidden nodes in area of interest
*****************************************

The user can use a regional dataset to define the `data` nodes over the
region of interest. Then, it can define the hidden nodes only over the
region of interest using any of the ``LimitedArea_____Nodes`` classes.

.. literalinclude:: yaml/lam_nodes_wo_boundary.yaml
   :language: yaml

************************************************
 Cut out regional dataset into a global dataset
************************************************

In this case, the user may want to include boundary forcings to the
region of interest. AnemoI Graphs allows the user to use two datasets to
build the `data` nodes, combining nodes from the LAM dataset and the
global dataset (as boundary forcings). The class ``AnemoiDatasetNodes``
allows this functionality:

.. literalinclude:: yaml/cutout_dataset.yaml
   :language: yaml

The ``AnemoiDatasetNodes`` supports an optional ``thinning`` argument
which can be used to sampling points from the regional dataset to reduce
computation during development stage.

In addition, this node builder class will create an additional node
attribute with a mask showing which node correspond to each of the two
datasets.

.. code:: console

   >>> graph
   HeteroData(
      data={
         x=[40320, 2],
         node_type='AnemoiDatasetNodes',
         area_weight=[40320, 1],
         cutout_mask=[40320, 1],
      }
   )

*********************************************
 Define hidden nodes over region of interest
*********************************************

Once the `data` nodes are defined, the user can define the hidden nodes
only over the region of interest. In this case, the area of interest is
defined by the `data` nodes masked by the ``cutout`` attribute.

.. literalinclude:: yaml/limited_area_nodes.yaml
   :language: yaml

.. code:: console

   >>> graph
   HeteroData(
      data={
         x=[40320, 2],
         node_type='AnemoiDatasetNodes',
         area_weight=[40320, 1],
         cutout_mask=[40320, 1],
      },
      hidden={
         x=[10242, 2],
         node_type='TriNodes',
      }
   )

**************
 Adding edges
**************

The user may define the edges using the same configuration as for the
global graphs.

.. literalinclude:: yaml/global.yaml
   :language: yaml
