#######################################
 From latitude & longitude coordinates
#######################################

Nodes can also be created directly using latitude and longitude
coordinates. Below is an example demonstrating how to add these nodes to
a graph:

.. code:: python

   from anemoi.graphs.nodes import LatLonNodes

   ...

   lats = np.array([45.0, 45.0, 40.0, 40.0])
   lons = np.array([5.0, 10.0, 10.0, 5.0])

   graph = LatLonNodes(latitudes=lats, longitudes=lons, name="my_nodes").update_graph(graph)
