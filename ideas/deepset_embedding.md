Deep Sets embedder: mean-pooling instead of attention
======================================================

Description
--------------
`DeepSetEmbedder` uses Deep Sets (Zaheer et al. 2017) to pool a node's
variables: transform each variable independently, then aggregate with a
permutation-invariant mean - cheap and simple to train.

Algorithm
--------------
```
tokens = []
for param in variables:
    level_encoding = level_pe[param] if param.has_level else no_level_embedding
    token = concat(value[param], value_encoder(value[param]), embedding(param), level_encoding)
    tokens.append(token)

mlp_out = MLP(tokens)                    # per variable
agg = mean(mlp_out)

node_embedding = agg + MLP(LayerNorm(agg))

output = concat(node_embedding, node_attributes_data)
```

Timing
----------
~1.3x slower than baseline.
