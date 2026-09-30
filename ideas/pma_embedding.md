PMA embedder: pool with a learned query
===========================================

Description
--------------
`PMAEmbedder` pools a node's variables with PMA (Pooling by Multihead
Attention, Lee et al. 2019): a learned query vector cross-attends over the
variable tokens (the query is a parameter, not derived from the input),
and the variable tokens do not attend back to it or to each other - only
the query's output is kept as the node embedding.

Algorithm
--------------
```
tokens = []
for param in variables:
    level_encoding = level_pe[param] if param.has_level else no_level_embedding
    token = concat(value[param], value_encoder(value[param]), embedding(param), level_encoding, frame_id)
    tokens.append(token)

tokens_proj = Linear(tokens) 

# learned_query is a learned parameter, not built from the input
node_embedding = MHA(key=tokens_proj, value=tokens_proj, query=learned_query)

output = concat(node_embedding, node_attributes_data)
```

Timing
----------
~1.6x slower than baseline.
