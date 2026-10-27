##################
 Rectilinear grids
##################

A rectilinear grid arranges the nodes in rows and columns. Processors
that treat the hidden grid as an image need such a grid. Each node keeps
its row and column, which the
:doc:`../node_attributes/rectilinear_layout` attribute exposes.

********************************************
 Global regular latitude-longitude grid
********************************************

A regular latitude-longitude grid has the same spacing in latitude and
longitude everywhere. With a spacing of `dlat` degrees in latitude and
`dlon` degrees in longitude, the grid has `360 / dlon` longitudes,
`j * dlon`. By default the nodes sit at the cell centres, at the
`180 / dlat` latitudes `90 - (i + 0.5) * dlat`, so no node lies on a
pole.

.. code:: yaml

   nodes:
     hidden: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.RectilinearNodes
         resolution: 1.0

Here, `resolution` is the grid spacing in degrees, or a list
`[dlat, dlon]`. `180 / dlat` and `360 / dlon` must be whole numbers. For
example, `1.0` gives 180 x 360 nodes and `0.5` gives 360 x 720 nodes.

With ``include_poles: True``, the grid has the `180 / dlat + 1`
latitudes `90 - i * dlat`, so the first and last rows sit on the poles,
as in the 721 x 1440 grid of ERA5 at 0.25 degrees:

.. code:: yaml

   nodes:
     hidden:
       node_builder:
         _target_: anemoi.graphs.nodes.RectilinearNodes
         resolution: 0.25
         include_poles: True

.. note::

   The nodes are stored row by row, from north to south, with longitudes
   running east from zero within each row. Processors that treat the
   hidden grid as an image, such as the advection-diffusion-reaction
   processor in anemoi-models, rely on this order and read the grid size
   from the node coordinates. These processors only support the grid
   without poles.

.. warning::

   With ``include_poles``, each pole row has many nodes at the same
   point. ``SphericalAreaWeights`` cannot handle repeated points, so use
   ``IsolatitudeAreaWeights`` or ``CosineLatWeightedAttribute`` instead.

*****************************
 Limited-area rotated grid
*****************************

For a limited area, the grid is regular in a rotated
latitude-longitude frame, which moves the centre of the area to (0, 0).

The grid covers the reference nodes, for example the data nodes of a
limited-area dataset, plus a margin. The nodes are stored row by row,
from rotated north to south, with rotated longitudes running east
within each row.

.. code:: yaml

   nodes:
     hidden:
       node_builder:
         _target_: anemoi.graphs.nodes.LimitedAreaRectilinearNodes
         resolution: 0.25
         reference_node_name: data
         margin_radius_km: 200
         mask_attr_name: null
         crop_to_margin: False
         rotation: 0.0

With ``crop_to_margin: True``, the grid points farther than
``margin_radius_km`` from every reference node are left out. For a round
area, for example, this keeps a disc of nodes inside the rectangle. The
remaining nodes keep their row and column in the full rectangle, so
fields can still be put back on it, with gaps where nodes were left out.

With ``rotation``, the grid axes are turned anticlockwise about the
centre by that many degrees. Defaults to ``rotation: 0``.

.. warning::

  An area that reaches a pole of the rotated frame, or goes all the way around
  it, is too large for a limited-area grid (use ``RectilinearNodes`` instead).
