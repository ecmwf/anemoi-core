####################
 Boolean operations
####################

_anemoi-graphs_ package implements a set of boolean opearations to
support these operations when defining node attributes. Below, an
attribute `mask` is computed as the intersection of two other masks,
that are generated as the non-missing values in 2 different variables in
a Zarr dataset.

.. literalinclude:: ../yaml/attributes_boolean_operation.yaml
   :language: yaml
