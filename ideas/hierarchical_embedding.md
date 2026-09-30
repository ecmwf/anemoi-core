Hierarchical embedder: pool per level first, then pool across levels
=====================================================================

Description
--------------
`HierarchicalEmbedder` pools a node's variables in two stages, each one a
multihead attention (MHA) call: once within each pressure level, then
again across levels - so the model can treat physical variable which are
level-independent, and levels as separate concerns.

Algorithm
--------------
```
level_tokens = []   # stage 1
for level, level_group in level_groups:
    tokens = []
    for param in level_group:
        token = Linear(value[param]) + Linear(embedding(param))   # two separate Linears, added
        tokens.append(token)
    level_tokens.append(multihead_attention(tokens) + level_pe[level])

level_tokens = stack(level_tokens)
node_embedding = multihead_attention(level_tokens)      # stage 2

output = concat(node_embeddings, node_attributes_data)
```

Timing
----------
~9.7x slower than baseline.
