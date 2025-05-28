##################
 From XArray file
##################

To define the `node coordinates` based on a XArray file, you can use the
following YAML configuration:

.. code:: yaml

   nodes:
     data: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.XArrayNodes
         dataset: /path/to/folder/with/grids/my_grid.zarr
         lat_key: latitudes
         lon_key: longitudes

where `dataset` is the path to the xarray compatible file and `lat_key`
and `lon_key` are optional arguments with the key names of the latitude
and longitude arrays.
