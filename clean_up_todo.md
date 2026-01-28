# Clean-up TODO

- Tests currently assume `FullGrid` grid indices (fixtures create a minimal `HeteroData` with `num_nodes` only). If configs switch to `MaskedGrid`, these fixtures may break. Verify and adjust later.
