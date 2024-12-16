########
 Models
########

*********************************
 Encoder Processor Decoder Model
*********************************

The model defines a graph neural network architecture with configurable
encoder, processor, and decoder.

.. automodule:: anemoi.models.models.encoder_processor_decoder
   :members:
   :no-undoc-members:
   :show-inheritance:

**********************************************
 Encoder Hierarchical processor Decoder Model
**********************************************

This model extends the standard encoder-processor-decoder architecture
by introducing a **hierarchical processor**.

Compared to the AnemoiModelEncProcDec model, this architecture requires
a predefined list of hidden nodes, `[hidden_1, ..., hidden_n]`. These
nodes must be sorted to match the expected flow of information `data ->
hidden_1 -> ... -> hidden_n -> ... -> hidden_1 -> data`.

A new argument is added to the configuration file:
`enable_hierarchical_level_processing`. This argument determines whether
a processor is added at each hierarchy level or only at the final level.

By default, the number of channels for the mappers is defined as `2^n *
config.num_channels`, where `n` represents the hierarchy level. This
scaling ensures that the processing capacity grows proportionally with
the depth of the hierarchy, enabling efficient handling of data.

.. automodule:: anemoi.models.models.hierarchical
   :members:
   :no-undoc-members:
   :show-inheritance:
