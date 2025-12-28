###########
 Area mask
###########

The `LimitedAreaMask` node attribute builder creates a mask over the
nodes covering the limited area.

The configuration for these masks, is specified in the YAML file:

.. literalinclude:: ../yaml/attributes_lam_mask.yaml
   :language: yaml

.. note::

   This node attribute builder is only supported for nodes created using
   subclasses of ``StretchedIcosahedronNodes``. Currently, it is
   available exclusively for nodes built with the ``StretchedTriNodes``
   subclass.
