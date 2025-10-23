####################################
 Triangular Mesh with ICON Topology
####################################

The classes `ICONMultimeshNodes` and `ICONCellGridNodes` define node
sets based on an ICON icosahedral mesh:

-  class `ICONCellGridNodes`: data grid, representing cell circumcenters
-  class `ICONMultimeshNodes`: hidden mesh, representing the vertices of
   a grid hierarchy

Both classes, together with the corresponding edge builders

-  class `ICONTopologicalProcessorEdges`
-  class `ICONTopologicalEncoderEdges`
-  class `ICONTopologicalDecoderEdges`

are based on the mesh hierarchy that is reconstructed from an ICON mesh
file in NetCDF format, making use of the `refinement_level_v` and
`refinement_level_c` property contained therein.

-  `refinement_level_v[vertex] = 0,1,2, ...`,
      where 0 denotes the vertices of the base grid, ie. the icosahedron
      including the step of root subdivision RXXB00.

-  `refinement_level_c[cell]`: cell refinement level index such that
   value 0 denotes the cells of the base grid, ie. the icosahedron
   including the step of root subdivision RXXB00.

See the following YAML example:

.. code:: yaml

   nodes:
     # Data nodes
     data:
       node_builder:
         _target_: anemoi.graphs.nodes.ICONCellGridNodes
         grid_filename: "icon_grid_0026_R03B07_G.nc"
         max_level: 3
     # Hidden nodes
     hidden:
       node_builder:
         _target_: anemoi.graphs.nodes.ICONMultimeshNodes
         grid_filename: "icon_grid_0026_R03B07_G.nc"
         max_level: 3

   edges:
     # Processor configuration
     - source_name: "hidden"
       target_name: "hidden"
       edge_builders:
       - _target_: anemoi.graphs.edges.ICONTopologicalProcessorEdges
