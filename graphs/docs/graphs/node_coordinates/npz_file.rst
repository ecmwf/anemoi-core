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
         npz_file: /path/to/folder/with/grids/my_grid.npz
         lat_key: latitudes
         lon_key: longitudes

where `npz_file` is the path to the NPZ file and `lat_key` and `lon_key`
are optional arguments with the key names of the latitude and longitude
arrays.
