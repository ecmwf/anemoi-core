###############
 From NPZ file
###############

To define the `node coordinates` based on a NPZ file, you can use the
following YAML configuration:

.. code:: yaml

   nodes:
     data: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.NPZFileNodes
         grid_definition_path: /path/to/folder/with/grids/
         resolution: o48

where `grid_definition_path` is the path to the folder containing the
grid definition files and `resolution` is the resolution of the grid to
be used.

By default, the grid files are supposed to be in the `grids` folder in
the same directory as the recipe file. The grid definition files are
expected to be name `"grid_{resolution}.npz"`.

.. note::

   The NPZ file should contain the following keys:

   -  `longitudes`: The longitudes of the grid.
   -  `latitudes`: The latitudes of the grid.
