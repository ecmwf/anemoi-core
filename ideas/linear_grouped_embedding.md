Linear grouped embedder: fixed variable set per level, no attention
======================================================================

Description
--------------
`LinearGroupedEmbedder` relies on a fixed assumption for within-level
pooling: every pressure level always has the same fixed set of variables,
in a known order (in O96 ERA5: u, v, q, t, w, z at every level). Their
values are stacked in that fixed order and projected with a plain
`nn.Linear`. A multihead attention (MHA) call is kept only for the second
stage, combining the much smaller per-level tokens into the final node
embedding.

Algorithm
--------------
```
for level in pressure_levels:
    input = stack(value[param] for param in level.params)   # fixed order, e.g. u, v, q, t, w, z
    level_token[level] = Linear(input) + level_pe[level]

surface_input = stack(value[param] for param in surface.params)   # e.g. lsm, 2t, 10u, 10v, ...
surface_token = Linear(surface_input)

tokens = concat(level_token for each level, surface_token)   # extends the existing sequence axis
node_embedding = multihead_attention(tokens)

output = concat(node_embedding, node_attributes_data)
```

Timing
----------
~1.2x slower than baseline.

Limitations
---------------
Nodewise, this is only agnostic with respect to which pressure levels are
used - the level and surface tokens wouldn't make sense if the physical
variables making them up weren't consistent across levels, since
`level_proj` and `surface_proj` are fixed-width `nn.Linear`s.
