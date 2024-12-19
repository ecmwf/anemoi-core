.. _graphs-node_coordinates:

#####################
 Nodes - Coordinates
#####################

.. warning::

   This is still a work in progress. More classes will be added in the
   future.

The `nodes` :math:`V` correspond to locations on the earth's surface.

The `nodes` are defined in the ``nodes`` section of the recipe file. The
keys are the names of the sets of `nodes` that will later be used to
build the connections. Each `nodes` configuration must include a
``node_builder`` section describing how to define the `nodes`.

The `nodes` can be defined based on the coordinates already available in
a file:

.. toctree::
   :maxdepth: 1

   node_coordinates/zarr_dataset
   node_coordinates/npz_file
   node_coordinates/icon_mesh
   node_coordinates/text_file
   node_coordinates/latlon_arrays

or based on other algorithms. A commonn approach is to use an
icosahedron to project the earth's surface, and refine it iteratively to
reach the desired resolution.

.. toctree::
   :maxdepth: 1

   node_coordinates/tri_refined_icosahedron
   node_coordinates/hex_refined_icosahedron
   node_coordinates/healpix
