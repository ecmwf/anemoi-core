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
       edge_builder:
         _target_: anemoi.graphs.edges.CutOff
         cutoff_factor: 0.7

Below are the available methods for defining the edges:

.. toctree::
   :maxdepth: 1

   edges/cutoff
   edges/knn
   edges/multi_scale
