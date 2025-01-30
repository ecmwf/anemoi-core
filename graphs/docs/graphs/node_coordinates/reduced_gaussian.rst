#######################
 Reduced Gaussian grid
#######################

A gaussian grid is a latitude/longitude grid where the spacing of the
latitudes is not regular but is symmetrical about the Equator. The grid
is identified by a grid code, which specifies the type (`n/N` for the
original ECMWF reduced Gaussian grid or `o/O` for the octahedral ECMWF
reduced Gaussian grid) and the resolution. The resolution is defined by
the number of latitude lines (`XXX`) between the pole and the Equator.

To define `node coordinates` based on a reduced gaussian grid, you can
use the following YAML configuration:

.. code:: yaml

   nodes:
     data: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.ReducedGaussianGridNodes
         grid: o48

Here, `grid` specifies the type and resolution of the reduced Gaussian
grid in the format `[o|n]XXX`. For example, `o48` represents an
octahedral Gaussian grid with 48 latitude lines between the pole and the
Equator.

.. note::

   The reduced Gaussian grids are stored in NPZ files with the keys
   latitudes and longitudes. These files are downloaded and cached in a
   local directory. Initially, only a subset of grids is available. If
   you require a new Gaussian grid to be added, please contact the
   administrators.

   Currently available reduced Gaussian grids: - o16 - o32 - o48 - o96 -
   o160 - o256 - o320 - n320 - o1280
