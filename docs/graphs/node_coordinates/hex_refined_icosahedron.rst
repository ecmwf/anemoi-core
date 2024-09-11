###############################
 Hexagonal refined Icosahedron
###############################

This method allows us to define the nodes based on the Hexagonal
Hierarchical Geospatial Indexing System, which uses hexagons to divide
the sphere. Each refinement level divides each hexagon into seven
smaller hexagons.

To define the `node coordinates` based on the hexagonal refinements of
an icosahedron, you can use the following YAML configuration:

.. code:: yaml

   nodes:
     data:
       node_builder:
         _target_: anemoi.graphs.nodes.HexNodes
         resolution: 4
       attributes: ...

where resolution is the number of refinements to be applied.

.. csv-table:: Hexagonal Hierarchical refinements specifications
   :file: ./hex_refined.csv
   :header-rows: 1

Note that the refinement level is the parameter used to control the
resolution of the nodes, but the resolution also depends on the
refinement method. Then, for the same refinement level, ``HexNodes``
will have a higher resolution than ``TriNodes``.

.. warning::

   This class will require the `h3 <https://h3.org>`_ package to be
   installed. You can install it with `pip install h3`.
