.. _graphs-node-attributes-timedeltas:

####################
 Timedelta features
####################

``Timedeltas`` encodes runtime observation offsets for dynamic data
nodes. Input values are signed seconds relative to the sample reference
date. The first feature is the offset divided by ``scale_seconds``; each
configured period then adds a sine/cosine pair.

For example, this configuration produces signed hours plus daily and
weekly Fourier features:

.. code:: yaml

   nodes:
     observations:
       node_builder: ...
       attributes:
         timedeltas:
           _target_: anemoi.graphs.nodes.attributes.Timedeltas
           scale_seconds: 3600
           periods: [24, 168]

This attribute is runtime-only. Configure it only on dynamic observation
nodes consumed by ``AnemoiModelEncProcDec``; it cannot be materialised
in a static graph. An empty ``periods`` list produces only the scaled
scalar.

Changing the configured periods changes the model input width and is
therefore not compatible with checkpoints trained with a different
timedelta encoding.
