# Future work — not now

## Avoid resolving graph-score edges twice

`GraphScoreGraph.from_definition()` currently reads the graph edge index and
resolves its weights so that it can filter and validate them. It passes the
resulting `edge_mask` to `ProjectionGraphProvider`, which reads the original
edge index and weights again, applies that exact mask, and constructs the CSR
matrix.

The self-edge check itself is not duplicated: the mask is calculated once and
passed to the provider. The duplicated edge and weight resolution happens only
when the loss is constructed, not in the training hot path.

For now, keep the `edge_mask` argument on `ProjectionGraphProvider`. It allows a
caller to select graph edges without copying or mutating the source graph and
keeps graph-score validation separate from generic CSR construction.

A possible future refactor is to add a provider construction path that accepts
already resolved `edge_index`, `edge_weights`, and matrix shape tensors. The
flow could then be:

1. Resolve edge indices and weights.
2. Filter self-edges when required.
3. Validate the filtered tensors.
4. Build the CSR matrix directly from those same tensors.

The provider's existing graph-based path could resolve its tensors and delegate
to the same lower-level CSR builder. This would remove the small amount of
duplicated initialization work and guarantee that validation and CSR
construction consume exactly the same resolved tensors.
