###########
 Area mask
###########

The `LimitedAreaMask` node attribte builder creates a mask over the
nodes covering the limited area.

The configuration for these masks, is specified in the YAML file:

.. literalinclude:: ../yaml/attributes_lam_mask.yaml
   :language: yaml

.. note::

   This node attribute builder is only supported for nodes created using
   subclasses of ``StretchedIcosahedroNodes``. Currently, it is
   available exclusively for nodes built with the ``StretchedTriNodes``
   subclass.
