.. _graphs-node_attributes:

####################
 Nodes - Attributes
####################

.. warning::

   This is still a work in progress. More classes will be added in the
   future.

As well as defining their geographical locations, the `nodes` can
contain additional attributes, which should be defined in the
``attributes`` section of the `nodes` configuration. For example, a
`weights` attribute can be used to define the importance of each node in
the loss function, or a `masks` attribute can be used to build
connections only between subsets of nodes.

.. toctree::
   :maxdepth: 1

   node_attributes/weights
   node_attributes/anemoi_dataset_attribute
   node_attributes/area_masks

Additionally, different boolean operations have been implemented to
support more complex use cases:

.. toctree::
   :maxdepth: 1

   node_attributes/boolean_operations
