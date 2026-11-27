# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Copy the lat-blend node weights onto the temporal-downscaler N320 graph.

Keeps the TD graph's own edges and edge features untouched (the checkpoint was trained
on them) and only adds the node attributes `oper_lat_weight` and `imerg_lat_weight`.
Refuses to write if the data-node coordinates of the two graphs differ.
usage: graft_node_attrs.py <td_graph.pt> <latblend_node_attrs.pt> <out_graph.pt>
"""
import sys
import torch

td_path, attrs_path, out_path = sys.argv[1:4]
g = torch.load(td_path, weights_only=False, map_location="cpu")
a = torch.load(attrs_path, weights_only=True, map_location="cpu")

d = g["data"]
print("TD graph data node attrs before:", [k for k in d.keys()])
assert d.x.shape == a["x"].shape, f"data node count differs: {tuple(d.x.shape)} vs {tuple(a['x'].shape)}"
max_dx = (d.x - a["x"]).abs().max().item()
print(f"max |data.x difference| (radians) = {max_dx:.3e}")
assert max_dx < 1e-5, "data node coordinates differ -> refusing to graft"

if "area_weight" in d:
    print("area_weight max |TD - latblend graph| =", (d["area_weight"] - a["area_weight"]).abs().max().item(), "(keeping TD graph's own)")
else:
    d["area_weight"] = a["area_weight"]
for k in ("oper_lat_weight", "imerg_lat_weight"):
    d[k] = a[k].clone()
    print(k, tuple(d[k].shape), "min %.4g max %.4g mean %.4g" % (d[k].min(), d[k].max(), d[k].mean()))
print("sum check imerg+oper:", (d["imerg_lat_weight"] + d["oper_lat_weight"]).min().item(), (d["imerg_lat_weight"] + d["oper_lat_weight"]).max().item())
torch.save(g, out_path)
print("wrote", out_path)
