####################################
 Regular latitude-longitude grid
####################################

A regular latitude-longitude grid has the same spacing in latitude and
longitude everywhere. The nodes sit at the cell centres, so no node lies
on a pole. With a resolution of `r` degrees the grid has `180 / r`
latitudes, `90 - (i + 0.5) * r`, and `360 / r` longitudes, `j * r`.

To define `node coordinates` on a regular latitude-longitude grid, you
can use the following YAML configuration:

.. code:: yaml

   nodes:
     hidden: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.RegularLatLonNodes
         resolution: 1.0

Here, `resolution` is the grid spacing in degrees. `180 / resolution`
must be a whole number. For example, `1.0` gives 180 x 360 nodes and
`0.5` gives 360 x 720 nodes.

.. note::

   The nodes are stored row by row, from north to south, with longitudes
   running east from zero within each row. Processors that treat the
   hidden grid as an image, such as the advection-diffusion-reaction
   processor in anemoi-models, rely on this order and read the grid size
   from the node coordinates.
