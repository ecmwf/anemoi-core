#########
 Weights
#########

The `weights` are node attributes useful for defining the importance of
a node in the loss function. You can set the weights to follow an
uniform distribution or to match the area associated with that node.

****************
 Uniform weight
****************

The user can define a node attribute with a constant value of 1 as shown
in the example below:

.. literalinclude:: ../yaml/attributes_uniform_weights.yaml
   :language: yaml

*************
 Area weight
*************

The most common approach to weighting the output field of meteorological
data-driven models is to compute the area weight associated with each
data node. For this purpose, we provide 2 classes to compute this area.
The first one, `SphericalAreaWeights`, is recommended for global models
and it computes the area of regions over the sphere.

.. literalinclude:: ../yaml/attributes_spherical_area_weights.yaml
   :language: yaml

In contrast, for limited area graphs, we recommend that users use
`PlanarAreaWeights` to avoid some errors that may occur depending on the
dataset configuration.

.. literalinclude:: ../yaml/attributes_planar_area_weights.yaml
   :language: yaml
