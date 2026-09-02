#!/usr/bin/env python
# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Add a (data, to, data) nearest-neighbour edge set, and optionally static node attributes,
to an existing per-dataset graph file. Writes a NEW file; never rebuilds from the recipe and
never overwrites.

Fine-scale epic, 2026-09-02. The edge set feeds the optional local branch of the diffusion
downscaler (``model.model.hres_branch``); the node attributes feed the static-input arm.

Edges: for every data node, its k nearest neighbours by great-circle distance (k includes
the node itself, which the graph transformer wants as a self-match). Attributes ``edge_length``
and ``edge_dirs`` are computed with the anemoi-graphs builders with ``norm: unit-std``, exactly
as every other edge set in the file. The edge type string is ``KNNEdges``.

Verification: after writing, the new file is reloaded and every pre-existing node and edge
store is compared tensor by tensor with the original; the script exits non-zero on any
difference.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from hydra.utils import instantiate
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0
ATTRS = {
    "edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": "unit-std"},
    "edge_dirs": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection", "norm": "unit-std"},
}


def log(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def knn_edge_index(latlon_rad: np.ndarray, k: int):
    la, lo = latlon_rad[:, 0].astype(np.float64), latlon_rad[:, 1].astype(np.float64)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
    tree = cKDTree(xyz, balanced_tree=False, compact_nodes=False)
    n = xyz.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float32)
    ch = 400_000
    for s in range(0, n, ch):
        e = min(s + ch, n)
        dd, ii = tree.query(xyz[s:e], k=k, workers=-1)
        idx[s:e] = ii
        dist[s:e] = 2.0 * R_EARTH_KM * np.arcsin(np.clip(dd / 2.0, 0, 1))
    src = idx.reshape(-1)
    dst = np.repeat(np.arange(n, dtype=np.int64), k)
    return np.stack([src, dst]), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="existing per-dataset graph .pt")
    ap.add_argument("--out", required=True, help="new graph .pt (must not exist)")
    ap.add_argument("--node", default="data")
    ap.add_argument("--k", type=int, default=7, help="neighbours INCLUDING the self-match")
    ap.add_argument("--static-npz", default=None, help="npz with static fields on the node order")
    ap.add_argument("--static-vars", default="", help="comma-separated variables to attach as node attributes")
    a = ap.parse_args()
    t0 = time.time()

    import os
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    g = torch.load(a.graph, weights_only=False, map_location="cpu")
    key = (a.node, "to", a.node)
    if key in g.edge_types:
        raise SystemExit(f"{key} already present in {a.graph}")
    orig_nodes = {n: {k: v.clone() for k, v in g[n].items() if torch.is_tensor(v)} for n in g.node_types}
    orig_edges = {e: {k: v.clone() for k, v in g[e].items() if torch.is_tensor(v)} for e in g.edge_types}
    orig_edge_types = list(g.edge_types)
    log(f"loaded {a.graph}: nodes {g.node_types} edges {orig_edge_types}", t0)

    x = g[a.node].x.numpy()
    edge_index, dist = knn_edge_index(x, a.k)
    log(f"knn k={a.k}: {edge_index.shape[1]} edges; neighbour distance (excluding self) mean "
        f"{dist[:, 1:].mean():.3f} km, p05 {np.percentile(dist[:, 1:], 5):.3f}, p95 {np.percentile(dist[:, 1:], 95):.3f}; "
        f"self-match distance max {dist[:, 0].max():.4f} km", t0)
    ei64 = torch.from_numpy(edge_index)
    g[key].edge_index = ei64.to(torch.int32)
    g[key].edge_type = "KNNEdges"
    for name, cfg in ATTRS.items():
        builder = instantiate(cfg)
        g[key][name] = builder(x=(g[a.node], g[a.node]), edge_index=ei64)
        v = g[key][name]
        log(f"{name}: shape {tuple(v.shape)} dtype {v.dtype} mean {v.mean(0).tolist()} std {v.std(0).tolist()}", t0)

    scaling = {}
    if a.static_npz and a.static_vars:
        z = np.load(a.static_npz)
        for var in a.static_vars.split(","):
            v = z[var].astype(np.float64)
            if var == "fsr":
                vv = np.log10(np.maximum(v, 1e-6))
                mode = "log10 then z-score"
            else:
                vv = v
                mode = "z-score"
            mu, sd = float(vv.mean()), float(vv.std())
            g[a.node][var] = torch.from_numpy(((vv - mu) / sd).astype(np.float32))[:, None]
            scaling[var] = dict(mode=mode, mean=mu, std=sd)
            log(f"node attribute {var}: {mode}, mean {mu:.4g} std {sd:.4g}", t0)

    torch.save(g, a.out)
    log(f"wrote {a.out}", t0)
    if scaling:
        with open(a.out + ".static_scaling.json", "w") as fh:
            json.dump(scaling, fh, indent=1)

    # ---- verification: existing stores unchanged ----
    g2 = torch.load(a.out, weights_only=False, map_location="cpu")
    bad = []
    for n, store in orig_nodes.items():
        for k, v in store.items():
            if not torch.equal(g2[n][k], v):
                bad.append(f"node {n}.{k}")
    for e, store in orig_edges.items():
        for k, v in store.items():
            if not torch.equal(g2[e][k], v):
                bad.append(f"edge {e}.{k}")
        if g2[e].edge_type != g[e].edge_type:
            bad.append(f"edge {e}.edge_type string")
    new_edges = [e for e in g2.edge_types if e not in orig_edge_types]
    log(f"verification: pre-existing stores identical={not bad}; new edge sets {new_edges}; "
        f"new edges {int(g2[key].edge_index.shape[1])}", t0)
    if bad:
        print("DIFFERENCES:\n  " + "\n  ".join(bad))
        raise SystemExit(1)
    print("GATE3_OK")


if __name__ == "__main__":
    main()
