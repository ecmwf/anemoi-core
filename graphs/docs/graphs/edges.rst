.. _graphs-edges:

#####################
 Edges - Connections
#####################

Once the `nodes`, :math:`V`, are defined, you can create the `edges`,
:math:`E`, that will connect them. These connections are listed in the
``edges`` section of the recipe file, and they are created independently
for each (`source name`, `target name`) pair specified.

.. code:: yaml

   edges:
     - source_name: data
       target_name: hidden
       edge_builders:
       - _target_: anemoi.graphs.edges.CutOffEdges
         cutoff_factor: 0.7

Below are the available methods for defining the edges:

.. toctree::
   :maxdepth: 1

   edges/cutoff
   edges/knn
   edges/multi_scale

Additionally, there are 2 extra arguments (``source_mask_attr_name`` and
``target_mask_attr_name``) that can be used in the edge configuration to
mask source and/or target nodes. This can be useful to different use
cases, such as Limited Area Modeling (LAM) where your decoder edges
should only connect to the nodes in the limited area.

.. code:: yaml

   edges:
     - source_name: hidden
       target_name: data
       edge_builders:
       - _target_: anemoi.graphs.edges.KNNEdges
         num_nearest_neighbours: 5
         target_mask_attr_name: cutout
