###########
 Orography
###########

These attributes read a 2-D orography field on a regular latitude and
longitude grid from a NetCDF file and interpolate it to the nodes. The
variable must be named one of ``orog``, ``z``, ``orography``, ``hsurf``,
``topography`` or ``fi``; geopotential is converted to metres. Nodes
outside the orography grid get 0. They require `xarray
<https://xarray.dev>`_.

***********
 Elevation
***********

``OrogElevationAttribute`` gives the elevation of each node. It is
normalised by its maximum by default (``norm: unit-max``).

.. code:: yaml

   nodes:
     data:
       node_builder: ...
       attributes:
         z_orog:
           _target_: anemoi.graphs.nodes.attributes.OrogElevationAttribute
           orography_path: /path/to/orography.nc
           norm: unit-max

The elevation can be copied to the edges with
``FloatAttributeFromSourceNode`` or ``FloatAttributeFromTargetNode``,
see :ref:`edge-attributes`.

*******
 Slope
*******

``OrogGradientAttribute`` gives the slope of the terrain at each node,
scaled to [0, 1]: 0 for flat terrain and 1 for nodes at or above the
``percentile_clip`` percentile of the slopes.

.. code:: yaml

   nodes:
     data:
       node_builder: ...
       attributes:
         orog_slope:
           _target_: anemoi.graphs.nodes.attributes.OrogGradientAttribute
           orography_path: /path/to/orography.nc
           percentile_clip: 95  # optional

``OrogGradientWeights`` turns the same slope ``g`` into weights ``1 +
alpha * g``, for example to give steep terrain up to ``1 + alpha`` times
more weight in the loss.

.. code:: yaml

   nodes:
     data:
       node_builder: ...
       attributes:
         orog_weights:
           _target_: anemoi.graphs.nodes.attributes.OrogGradientWeights
           orography_path: /path/to/orography.nc
           alpha: 2.0
           percentile_clip: 95  # optional
