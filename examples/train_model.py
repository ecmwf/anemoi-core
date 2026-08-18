# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build a model via the DI ModelBuilder and train it for a few steps.

Runs on GPU if available (submit through SLURM on this HPC), else CPU. Demonstrates that a
model constructed by object injection is an ordinary ``nn.Module``: forward, backward and
an optimizer step all work with no Hydra / config involved at runtime.

    srun --mem=32G --partition=gpu_debug --gres=gpu:1 \
        /lus/h2resw01/hpcperm/mab/git/anemoi-core/.venv/bin/python examples/train_model.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_model import build_from_config  # noqa: E402

N_STEPS = 30
GRID = 4  # data nodes in the toy graph
N_VARS = 2  # prognostic variables


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(0)

    # Build the model via object injection (ModelBuilder), then move it to the device.
    # Graph providers / node attributes register their tensors as buffers, so .to(device)
    # moves them along with the parameters.
    model = build_from_config().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"built {type(model).__name__} with {n_params} parameters")

    # Toy batch: dict[dataset] -> (batch, time, ensemble, grid, vars)
    x = {"data": torch.rand(1, 1, 1, GRID, N_VARS, device=device)}
    target = torch.rand(1, 1, 1, GRID, N_VARS, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    first_loss = None
    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(x)  # dict[dataset] -> (batch, time, ensemble, grid, vars)
        loss = F.mse_loss(out["data"], target)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"  step {step:2d}  loss {loss.item():.6f}")

    print(
        f"loss: {first_loss:.6f} -> {loss.item():.6f}  ({'decreased' if loss.item() < first_loss else 'NOT decreased'})"
    )
    if device == "cuda":
        print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("OK")


if __name__ == "__main__":
    main()
