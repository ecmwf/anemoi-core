Embedder comparison experiment
==================================

Description
--------------
Compares five input embeddings on the same O96 ERA5 setup: baseline,
Pooling Multihead Attention (PMA) embedder, Deep Sets embedder,
Hierarchical embedder, and Linear grouped embedder. Each is a pointwise
embedding of a node's features, run after the residual skip (`x_skip`) is
built from the raw input and before the result feeds into the encoder.

The broader goal is finding a usable, variable-agnostic embedding strategy
that keeps working if a variable is dropped from the dataset. This
comparison trains next-step forecasting on the full O96 ERA5 variable set:
input is the prognostic and forcing variables, output is the prognostic
variables one step ahead - no variable is dropped from either input or
output here. Loss function is MSE. All five use the same joint variable
normalization, described below.

Each embedding is computed simultaneously, nodewise, across all nodes -
every node's embedding is independent of every other node's.

Setup
---------
- **Model:** GraphTransformer, 512 channels, 8 processor layers.
- **Data:** O96 ERA5.
- **Learning rate:** 1.25e-4.
- **Hardware:** 4 GPUs, DDP, 24h time limit.
- **Batch size:** 4 per GPU.

Joint variable normalization
---------------------------------
We use a new variable normalization that pools mean/stdev per physical
variable across all pressure levels (e.g. one shared mean/stdev for `t`
across `t_50` through `t_1000`), instead of normalizing each level
independently. Independent per-level normalization would rescale every
level to the same unit variance, erasing the real physical difference in
magnitude between levels (e.g. wind at 1000 hPa doesn't behave like wind
at 50 hPa). The same MHA/Linear weights are reused across levels for a
given physical variable, which only makes sense if that variable is
represented consistently across levels - per-level normalization would
break that consistency.
