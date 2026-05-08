.. _graphs-node_coordinates:

#####################
 Nodes - Coordinates
#####################

.. warning::

   This is still a work in progress. More classes will be added in the
   future.

The `nodes` are defined in the ``nodes`` section of the recipe file. The
keys are the names of the sets of `nodes` that will later be used to
build the connections. Each `nodes` configuration must include a
``node_builder`` section describing how to define the `nodes`.

The `nodes` can be defined based on the coordinates already available in
a file using the following methods:

.. toctree::
   :maxdepth: 1

   node_coordinates/anemoi_dataset
   node_coordinates/npz_file
   node_coordinates/reduced_gaussian
   node_coordinates/icon_mesh
   node_coordinates/text_file
   node_coordinates/latlon_arrays

Alternatively, nodes can be designed based on other algorithms. A common
approach is to project an icosahedron onto the earth's surface, and
refine it iteratively to reach the desired resolution. Currently
available methods of this kind are:

.. toctree::
   :maxdepth: 1

   node_coordinates/tri_refined_icosahedron
   node_coordinates/hex_refined_icosahedron
   node_coordinates/healpix
