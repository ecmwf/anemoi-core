.. _edge-attributes:

####################
 Edges - Attributes
####################

There are 2 main edge attributes implemented in the `anemoi-graphs`
package:

*************
 Edge length
*************

The `edge length` is a scalar value representing the distance between
the source and target nodes. This attribute is calculated using the
Haversine formula, which is a method of calculating the distance between
two points on the Earth's surface given their latitude and longitude
coordinates.

.. code:: yaml

   edges:
     - source_name: ...
       target_name: ...
       edge_builders: ...
       attributes:
         edge_length:
            _target_: anemoi.graphs.edges.attributes.EdgeLength

****************
 Edge direction
****************

The `edge direction` is a 2D vector representing the direction of the
edge. This attribute is calculated from the difference between the
latitude and longitude coordinates of the source and target nodes.

.. code:: yaml

   edges:
     - source_name: ...
       target_name: ...
       edge_builders: ...
       attributes:
         edge_length:
            _target_: anemoi.graphs.edges.attributes.EdgeDirection
