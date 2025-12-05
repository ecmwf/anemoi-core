.. _edge-attributes:

####################
 Edges - Attributes
####################

There are few edge attributes implemented in the `anemoi-graphs`
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

***********************
 Directional Harmonics
***********************

The `Directional Harmonics` attribute computes harmonic features from
edge directions, providing a periodic encoding of the angle between
source and target nodes. For each order :math:`m` from 1 to the
specified maximum, it computes :math:`\sin(m\psi)` and
:math:`\cos(m\psi)` where :math:`\psi` is the edge direction angle.

.. code:: yaml

   edges:
     - source_name: ...
       target_name: ...
       edge_builders: ...
       attributes:
         dir_harmonics:
            _target_: anemoi.graphs.edges.attributes.DirectionalHarmonics
            order: 3

***********************
 Radial Basis Features
***********************

The `Radial Basis Features` attribute computes Gaussian radial basis
function (RBF) features from edge distances. It evaluates a set of
Gaussian basis functions centered at different scaled distances. By
default, per-node adaptive scaling is used.

.. code:: yaml

   edges:
     - source_name: ...
       target_name: ...
       edge_builders: ...
       attributes:
         rbf_features:
            _target_: anemoi.graphs.edges.attributes.RadialBasisFeatures
            r_scale: auto
            centers: [0.0, 0.25, 0.5, 0.75, 1.0]
            sigma: 0.2

******************
 Gaussian Weights
******************

The `Gaussian Weights` attribute assigns a weight to each edge based on
the distance between its source and target nodes, using a Gaussian
(normal) function of the edge length. This is useful for encoding
spatial locality or for constructing weighted adjacency matrices.

The Gaussian weight for an edge is computed as:

.. math::

   w_{ij} = \exp\left(-\frac{(\ell_{ij})^2}{2\sigma^2}\right)

where:

-  :math:`w_{ij}` is the weight assigned to the edge from node :math:`i`
   to node :math:`j`
-  :math:`\ell_{ij}` is the edge length (distance between nodes, as
   computed by the ``EdgeLength`` attribute)
-  :math:`\sigma` is a configurable parameter controlling the width of
   the Gaussian (the "spread" of the weights)

This means that edges connecting closer nodes will have higher weights,
while those connecting distant nodes will have lower weights.

Example configuration:

.. code:: yaml

   edges:
     - source_name: ...
       target_name: ...
       edge_builders: ...
       attributes:
         gaus_weight:
            _target_: anemoi.graphs.edges.attributes.GaussianDistanceWeights
            sigma: 1.0

.. note::

   This edge attribute normalisation is applied independently for each
   target node, and the default normalisation is :math:`L_1` for this.

*********************
 Attribute from Node
*********************

Attributes can also be copied from nodes to edges. This is done using
the `AttributeFromNode` base class, with specialized versions for source
and target nodes.

From Source
===========

This attribute copies a specific property of the source node to the
edge. Example usage for copying the cutout mask from nodes to edges in
the encoder:

.. code:: yaml

   edges:
     # Encoder
   - source_name: data
     target_name: hidden
     edge_builders: ...
     attributes:
       comes_from_cutout: # Assigned name to the edge attribute, can be different than node_attr_name
         _target_: anemoi.graphs.edges.attributes.AttributeFromSourceNode
         node_attr_name: cutout

From Target
===========

This attribute copies a specific property of the target node to the
edge. Example usage for copying the coutout mask from nodes to edges in
the decoder:

.. code:: yaml

   edges:
      # Decoder
    - source_name: hidden
      target_name: data
      edge_builders: ...
      attributes:
        comes_from_cutout: # Assigned name to the edge attribute, can be different than node_attr_name
          _target_: anemoi.graphs.edges.attributes.AttributeFromTargetNode
          node_attr_name: cutout
