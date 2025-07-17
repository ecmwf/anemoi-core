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
The first one, ``SphericalAreaWeights``, is recommended for global
models and it computes the area of regions over the sphere.

.. literalinclude:: ../yaml/attributes_spherical_area_weights.yaml
   :language: yaml

.. warning::

   The ``SphericalAreaWeights`` method will fail when used with
   rectilinear grids, as multiple nodes are projected to the poles,
   leading to incorrect area assignments.

   For rectilinear grids, please use ``CosineLatWeightedAttribute`` or
   ``IsolatitudeAreaWeights`` instead.

In contrast, for limited area graphs, we recommend that users use
``PlanarAreaWeights`` to avoid some errors that may occur depending on
the dataset configuration.

.. literalinclude:: ../yaml/attributes_planar_area_weights.yaml
   :language: yaml

This method can be extended to ``MaskedPlanarAreaWeights``, which
computes planar area weights but restricts the computation to a masked
region by setting the weights outside the mask to zero.

This is useful for comparing the performance of a limited area model
(LAM) with that of a stretched model, since the masking occurs before
normalisation. If you use ``norm: unit-sum``, all the weights inside the
area of interest will add up to 1.

.. literalinclude:: ../yaml/attributes_masked_planar_area_weights.yaml
   :language: yaml

Also, ``CosineLatWeightedAttribute`` assigns weights to nodes based only
on their latitude. This method is useful for approximating the area
associated with each node on a sphere when only latitude is available or
relevant.

.. literalinclude:: ../yaml/attributes_cosine_lat_weighted.yaml
   :language: yaml

All nodes at the same latitude get the same weight with
``CosineLatWeightedAttribute``, no matter how many nodes are in that
band. This means it does not account for differences in node density. On
the other hand, ``IsolatitudeAreaWeights`` calculates the area of each
latitude band, based on the curvature of the sphere, and splits that
area equally among all nodes in the band.

.. literalinclude:: ../yaml/attributes_isolatitude_area_weights.yaml
   :language: yaml
