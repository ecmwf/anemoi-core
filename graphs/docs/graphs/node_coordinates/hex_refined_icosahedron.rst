###############################
 Hexagonal refined Icosahedron
###############################

This method allows us to define the nodes based on the Hexagonal
Hierarchical Geospatial Indexing System, which uses hexagons to divide
the sphere. With each refinement, each hexagon into seven smaller
hexagons.

To define the `node coordinates` based on the hexagonal refinements of
an icosahedron, you can use the following YAML configuration:

***************
 Global graphs
***************

The class `HexNodes` allows us to define the nodes over the entire
globe.

.. code:: yaml

   nodes:
     hidden: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.HexNodes
         resolution: 4
       attributes: ...

where `resolution` is the number of refinements to be applied.

*********************
 Limited Area graphs
*********************

The class `LimitedAreaHexNodes` allows us to define the nodes only for a
specific area of interest.

.. code:: yaml

   nodes:
     hidden: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.LimitedAreaHexNodes
         resolution: 4
         reference_node_name: nodes_name
         mask_attr_name: mask_name  # optional
         margin_radius_km: 100  # optional
       attributes: ...

where `reference_node_name` is the name of the nodes to define the area
of interest.

.. csv-table:: Hexagonal Hierarchical refinements specifications
   :file: ./hex_refined.csv
   :header-rows: 1

.. warning::

   This class will require the `h3 <https://h3.org>`_ package to be
   installed. You can install it with `pip install h3`.
