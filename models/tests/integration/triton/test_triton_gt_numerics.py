# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Numerical quality control for the Triton GraphTransformer attention.

Unlike test_triton_gt.py (fp32, random inputs vs PyG), these tests use low-precision inputs and compare against
an fp64 reference evaluated on the *same rounded inputs*, so the measured error is the kernel's own. Each test
pins a property that a previous or plausible kernel change breaks:

- bf16/fp16 gradients stay at the output-rounding floor, also for common key/value/edge offsets and sharp
  attention. The pre-#1172 kernel (bf16 saved output and D) fails these by 1-4 orders of magnitude.
- exact identities: degree-1 rows have dQ = dK = 0, isolated nodes give zeros, dO = 0 gives zeros.
- no out-of-bounds reads for non-power-of-2 heads/channels (inputs are followed by NaNs).
- determinism and exact homogeneity in dO, compile parity.
- resources: no zero-filled gradients for the op's internal outputs; no out-of-bounds reads for padded heads
  (compute-sanitizer memcheck, skipped if it is not installed).
"""

import math
import os
import resource
import shutil
import subprocess
import sys

import pytest
import torch

from anemoi.models.triton.utils import edge_index_to_csc
from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.gt import graph_transformer_attention_conv

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_triton_available(), reason="Triton/CUDA not available"),
]

DEVICE = "cuda"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
# relative L2 error bounds; low-precision bounds are ~2x the error of rounding exact results to the dtype
TOL = {"fp32": 1e-5, "bf16": 4e-3, "fp16": 1e-3}
GRADS = ("dq", "dk", "dv", "de")


# --------------------------------------------------------------------------------------------- helpers


def reference_fp64(q, k, v, e, edge_index, n_dst):
    """Exact GraphTransformer attention (out = sum_i softmax_i(<q, k_i + e_i>/sqrt(C)) (v_i + e_i))."""
    src, dst = edge_index
    ke, ve = k[src] + e, v[src] + e
    s = (q[dst] * ke).sum(-1) / math.sqrt(q.shape[-1])
    smax = torch.full((n_dst, s.shape[1]), -math.inf, dtype=s.dtype, device=s.device)
    smax = smax.scatter_reduce(0, dst[:, None].expand_as(s), s.detach(), reduce="amax")
    w = torch.exp(s - smax[dst])
    p = w / torch.zeros_like(smax).index_add(0, dst, w)[dst]
    return torch.zeros((n_dst,) + q.shape[1:], dtype=q.dtype, device=q.device).index_add(0, dst, p[..., None] * ve)


def nan_guard(x):
    """Copy of ``x`` whose storage is directly followed by NaNs (makes out-of-bounds reads visible)."""
    buf = torch.full((x.numel() + 4096,), float("nan"), dtype=x.dtype, device=x.device)
    buf[: x.numel()] = x.reshape(-1)
    return buf[: x.numel()].view_as(x)


def run(q, k, v, e, edge_index, d_out, dtype, dout_scale=1.0):
    """Triton GT on inputs rounded to ``dtype`` and the fp64 reference on the same values.

    Returns (res, ref, rounded_inputs); res/ref hold out, dq, dk, dv, de (de in edge_index order).
    """
    n_src, n_dst = k.shape[0], q.shape[0]
    lp = {n: t.to(dtype) for n, t in dict(q=q, k=k, v=v, e=e, d_out=d_out).items()}

    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)
    perm = torch.arange(edge_index.shape[1], device=DEVICE) if perm is None else perm
    xs = [nan_guard(lp[n]).requires_grad_(True) for n in ("q", "k", "v")]
    e_csc = nan_guard(lp["e"][perm]).requires_grad_(True)
    out = graph_transformer_attention_conv(*xs, e_csc, csc, reverse)
    dq, dk, dv, de_csc = torch.autograd.grad(out, (*xs, e_csc), nan_guard(lp["d_out"] * dout_scale))
    de = torch.empty_like(de_csc)
    de[perm] = de_csc
    res = dict(out=out.detach(), dq=dq, dk=dk, dv=dv, de=de)

    ref_in = [lp[n].double().requires_grad_(True) for n in ("q", "k", "v", "e")]
    out_ref = reference_fp64(*ref_in, edge_index, n_dst)
    grads_ref = torch.autograd.grad(out_ref, ref_in, lp["d_out"].double())
    ref = dict(out=out_ref.detach(), **dict(zip(GRADS, grads_ref)))
    return res, ref, lp


def rel_err(a, ref):
    a, ref = a.double(), ref.double()
    if not torch.isfinite(a).all():
        return math.inf
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def randn(*shape, gen, scale=1.0):
    return (torch.randn(*shape, generator=gen, dtype=torch.float64, device="cpu") * scale).to(DEVICE)


def random_graph(n_src, n_dst, max_deg, gen):
    """Random bipartite graph; dst 0 and src n_src - 1 are isolated."""
    edges = []
    for dst in range(1, n_dst):
        deg = int(torch.randint(1, max_deg + 1, (1,), generator=gen, device="cpu"))
        edges += [(int(s), dst) for s in torch.randperm(n_src - 1, generator=gen, device="cpu")[:deg]]
    return torch.tensor(edges, dtype=torch.long, device=DEVICE).t().contiguous()


def neighbourhood_graph(n_dst, deg):
    """n_dst disjoint neighbourhoods of ``deg`` sources each (src i*deg .. (i+1)*deg - 1 -> dst i)."""
    src = torch.arange(n_dst * deg, device=DEVICE)
    return torch.stack((src, src // deg))


def random_inputs(edge_index, n_src, n_dst, H, C, gen):
    M = edge_index.shape[1]
    return (
        randn(n_dst, H, C, gen=gen),
        randn(n_src, H, C, gen=gen),
        randn(n_src, H, C, gen=gen),
        randn(M, H, C, gen=gen),
        randn(n_dst, H, C, gen=gen),
    )


# --------------------------------------------------------------------------------------------- accuracy


@pytest.mark.parametrize("dname", list(DTYPES))
@pytest.mark.parametrize("H,C", [(4, 32), (6, 12), (3, 6), (2, 24)])
def test_accuracy_vs_fp64(dname, H, C):
    """Random graph incl. isolated nodes and non-power-of-2 shapes (with NaN-guarded inputs)."""
    gen = torch.Generator().manual_seed(H * 100 + C)
    n_src, n_dst = 40, 24
    ei = random_graph(n_src, n_dst, 12, gen)
    q, k, v, e, d_out = random_inputs(ei, n_src, n_dst, H, C, gen)
    res, ref, _ = run(q, k, v, e, ei, d_out, DTYPES[dname])
    for name in ("out",) + GRADS:
        assert rel_err(res[name], ref[name]) < TOL[dname], f"{name}: {rel_err(res[name], ref[name]):.2e}"


def _offset_case(kind, offset, H=2, C=64, n_dst=8, deg=16, seed=0):
    """Common offset in coordinate 0 of all keys / edges, or in all value coordinates. Exact gradients do not
    depend on it (softmax translation invariance; q[..., 0] = 0 for key/edge offsets).
    """
    gen = torch.Generator().manual_seed(seed)
    ei = neighbourhood_graph(n_dst, deg)
    q, k, v, e, d_out = random_inputs(ei, n_dst * deg, n_dst, H, C, gen)
    if kind in ("key", "edge"):
        q[..., 0] = 0.0
        (k if kind == "key" else e)[..., 0] = offset
    else:
        v = v + offset
    return q, k, v, e, ei, d_out


@pytest.mark.parametrize("dname", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "kind,offset",
    [
        ("key", 2.0**4),
        ("key", 2.0**8),
        ("key", 2.0**12),
        ("value", 2.0**4),
        ("value", 2.0**8),
        ("edge", 2.0**4),
    ],
)
def test_common_offsets_low_precision(dname, kind, offset):
    """Large common key/value/edge offsets (e.g. from projection biases) must not degrade low-precision
    gradients beyond the output-rounding floor. The saved-output channel (bf16 O / D, pre-#1172) fails here.
    """
    q, k, v, e, ei, d_out = _offset_case(kind, offset)
    res, ref, _ = run(q, k, v, e, ei, d_out, DTYPES[dname])
    for name in GRADS:
        assert rel_err(res[name], ref[name]) < TOL[dname], f"{name}: {rel_err(res[name], ref[name]):.2e}"


@pytest.mark.parametrize("dname", ["bf16", "fp16"])
def test_sharp_attention_low_precision(dname):
    """Near one-hot rows (score gap 8): the true dQ/dK are small (~e^-8), the error must stay relative to them."""
    q, k, v, e, ei, d_out = _offset_case("key", 256.0, H=1, deg=8, seed=1)
    first = torch.arange(q.shape[0], device=DEVICE) * 8
    k[first] = k[first] + q * (8.0 * math.sqrt(q.shape[-1]) / (q**2).sum(-1, keepdim=True))
    res, ref, _ = run(q, k, v, e, ei, d_out, DTYPES[dname])
    for name in ("dq", "dk"):
        assert rel_err(res[name], ref[name]) < TOL[dname], f"{name}: {rel_err(res[name], ref[name]):.2e}"


def test_high_in_degree_bf16():
    """A dst with 1024 neighbours: accumulation must not happen in bf16."""
    gen = torch.Generator().manual_seed(2)
    ei = neighbourhood_graph(1, 1024)
    q, k, v, e, d_out = random_inputs(ei, 1024, 1, 2, 64, gen)
    res, ref, _ = run(q, k, v, e, ei, d_out, DTYPES["bf16"])
    for name in ("out",) + GRADS:
        assert rel_err(res[name], ref[name]) < TOL["bf16"], f"{name}: {rel_err(res[name], ref[name]):.2e}"


# --------------------------------------------------------------------------------------------- identities


@pytest.mark.parametrize("dname", list(DTYPES))
def test_singleton_rows_exact(dname):
    """Degree-1 dst: out = v + e (rounded once), dQ = dK = 0 and dV = dE = dO exactly. Breaks if the backward
    reconstructs D from a low-precision saved output (v + e is formed in fp32 and is not bf16-representable).
    """
    gen = torch.Generator().manual_seed(3)
    n, H, C = 8, 2, 64
    ei = neighbourhood_graph(n, 1)
    q, k, v, e, d_out = random_inputs(ei, n, n, H, C, gen)
    res, _, lp = run(q, k, v, e, ei, d_out, DTYPES[dname])
    assert torch.equal(res["dq"], torch.zeros_like(res["dq"]))
    assert torch.equal(res["dk"], torch.zeros_like(res["dk"]))
    assert torch.equal(res["dv"], lp["d_out"])
    assert torch.equal(res["de"], lp["d_out"])
    assert torch.equal(res["out"], (lp["v"].float() + lp["e"].float()).to(DTYPES[dname]))


@pytest.mark.parametrize("dname", list(DTYPES))
def test_isolated_nodes_and_zero_upstream_exact(dname):
    """Isolated dst -> out = dQ = 0; isolated src -> dK = dV = 0; dO = 0 -> all gradients exactly 0."""
    gen = torch.Generator().manual_seed(4)
    n_src, n_dst, H, C = 20, 10, 2, 32
    ei = random_graph(n_src, n_dst, 6, gen)
    q, k, v, e, d_out = random_inputs(ei, n_src, n_dst, H, C, gen)
    res, _, _ = run(q, k, v, e, ei, d_out, DTYPES[dname])
    for name, t in (
        ("out", res["out"][0]),
        ("dq", res["dq"][0]),
        ("dk", res["dk"][-1]),
        ("dv", res["dv"][-1]),
    ):
        assert torch.equal(t, torch.zeros_like(t)), name
    res, _, _ = run(q, k, v, e, ei, torch.zeros_like(d_out), DTYPES[dname])
    for name in GRADS:
        assert torch.equal(res[name], torch.zeros_like(res[name])), name


@pytest.mark.parametrize("dname", ["fp32", "bf16"])
def test_upstream_scaling_homogeneity(dname):
    """grads(2^-20 dO) == 2^-20 grads(dO) bitwise (no hidden absolute thresholds / flush-to-zero)."""
    gen = torch.Generator().manual_seed(5)
    ei = random_graph(30, 16, 8, gen)
    inputs = random_inputs(ei, 30, 16, 2, 64, gen)
    res, _, _ = run(*inputs[:4], ei, inputs[4], DTYPES[dname])
    res_s, _, _ = run(*inputs[:4], ei, inputs[4], DTYPES[dname], dout_scale=2.0**-20)
    for name in GRADS:
        assert torch.equal(res_s[name].double() * 2.0**20, res[name].double()), name


def test_deterministic():
    """No atomics: repeated fwd + bwd are bitwise identical."""
    gen = torch.Generator().manual_seed(6)
    ei = random_graph(64, 48, 16, gen)
    inputs = random_inputs(ei, 64, 48, 4, 32, gen)
    a, _, _ = run(*inputs[:4], ei, inputs[4], torch.bfloat16)
    b, _, _ = run(*inputs[:4], ei, inputs[4], torch.bfloat16)
    for name in ("out",) + GRADS:
        assert torch.equal(a[name], b[name]), name


@pytest.mark.parametrize("dname", list(DTYPES))
def test_output_and_gradient_dtypes(dname):
    gen = torch.Generator().manual_seed(7)
    ei = random_graph(10, 6, 4, gen)
    q, k, v, e, d_out = random_inputs(ei, 10, 6, 2, 16, gen)
    res, _, _ = run(q, k, v, e, ei, d_out, DTYPES[dname])
    for name in ("out",) + GRADS:
        assert res[name].dtype == DTYPES[dname], name


def test_torch_compile_matches_eager():
    gen = torch.Generator().manual_seed(8)
    n_src, n_dst = 40, 24
    ei = random_graph(n_src, n_dst, 8, gen)
    q, k, v, e, d_out = (t.to(torch.bfloat16) for t in random_inputs(ei, n_src, n_dst, 4, 32, gen))
    csc, perm, reverse = edge_index_to_csc(ei, num_nodes=(n_src, n_dst), reverse=True)
    e = e[perm] if perm is not None else e
    xs = [t.clone().requires_grad_(True) for t in (q, k, v, e)]

    def f(q, k, v, e):
        return graph_transformer_attention_conv(q, k, v, e, csc, reverse)

    out = f(*xs)
    g = torch.autograd.grad(out, xs, d_out)
    out_c = torch.compile(f, fullgraph=True)(*xs)
    g_c = torch.autograd.grad(out_c, xs, d_out)
    assert torch.equal(out, out_c)
    for a, b in zip(g, g_c):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------------------------- resources


def test_backward_does_not_materialise_internal_output_gradients():
    """Only the gradient of the returned output is used. The float32 copy of the output and the softmax statistics
    are internal outputs of the op; autograd must not allocate zero-filled gradients for them (for the float32
    output that is an (N_dst, H, C) float32 tensor, twice the size of a bfloat16 gradient)."""
    n, deg, H, C = 32768, 4, 4, 64
    gen = torch.Generator().manual_seed(9)
    ei = neighbourhood_graph(n, deg)
    ei = torch.stack((torch.randint(0, n, (n * deg,), generator=gen, device="cpu").to(DEVICE), ei[1]))
    csc, perm, reverse = edge_index_to_csc(ei, num_nodes=(n, n), reverse=True)
    q, k, v = (torch.randn(n, H, C, device=DEVICE, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    e = torch.randn(n * deg, H, C, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    out = graph_transformer_attention_conv(q, k, v, e, csc, reverse)
    d_out = torch.randn_like(out)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    grads = torch.autograd.grad(out, (q, k, v, e), d_out)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base

    grad_bytes = sum(g.numel() * g.element_size() for g in grads)
    fp32_output_bytes = n * H * C * 4
    # grads + the (N_dst, 2, H) buffer for D and lam + slack; a materialised float32 output gradient adds 2x dQ
    assert extra < grad_bytes + fp32_output_bytes // 2, f"backward allocated {extra / 2**20:.1f} MiB"


@pytest.mark.skipif(shutil.which("compute-sanitizer") is None, reason="compute-sanitizer not on PATH")
def test_no_out_of_bounds_reads_for_padded_heads(tmp_path):
    """Non-power-of-2 heads and channels are padded; the padded lanes must not read past any buffer, including
    the per-destination statistics. Runs the forward and backward under compute-sanitizer memcheck with the
    caching allocator disabled, so that every tensor has its exact size."""
    script = tmp_path / "padded_heads.py"
    script.write_text(
        "import torch\n"
        "from anemoi.models.triton.gt import graph_transformer_attention_conv\n"
        "from anemoi.models.triton.utils import edge_index_to_csc\n"
        "torch.manual_seed(0)\n"
        "H, C, n = 6, 12, 8\n"
        "dst = torch.arange(n, device='cuda').repeat_interleave(3)\n"
        "src = torch.randint(0, n, (3 * n,), device='cuda')\n"
        "csc, perm, rev = edge_index_to_csc(torch.stack((src, dst)), (n, n), reverse=True)\n"
        "q, k, v = (torch.randn(n, H, C, device='cuda', requires_grad=True) for _ in range(3))\n"
        "e = torch.randn(3 * n, H, C, device='cuda', requires_grad=True)\n"
        "graph_transformer_attention_conv(q, k, v, e, csc, rev).sum().backward()\n"
        "torch.cuda.synchronize()\n"
    )
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")
    result = subprocess.run(
        ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1", sys.executable, str(script)],
        env=env,
        # no core files: the process can abort at teardown under the sanitizer and dump ~GBs
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
