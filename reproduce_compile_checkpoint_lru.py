"""Reproduce the activation-checkpointing Dynamo LRU bug from pytorch#166926.

The first call compiles a static graph. A different input shape then creates an
automatic-dynamic graph whose guards also accept the first shape. The diagnostic
backend gives those otherwise equivalent graphs different autograd save
signatures, making cache-entry selection observable and deterministic.
"""

from __future__ import annotations

import argparse
import os

os.environ["TORCH_LOGS"] = "recompiles"

import torch
from torch.utils.checkpoint import CheckpointError
from torch.utils.checkpoint import checkpoint


class SaveIdentity(torch.autograd.Function):
    """Identity whose backward causes one additional tensor to be saved."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad


class Block(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sin() * x


class SaveSignatureBackend:
    """Give the first compiled graph one extra saved tensor without changing its result."""

    def __init__(self) -> None:
        self.compile_count = 0

    def __call__(self, graph: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        del example_inputs
        graph_index = self.compile_count
        self.compile_count += 1

        def compiled(*args):
            outputs = graph(*args)
            if graph_index == 0:
                outputs = (SaveIdentity.apply(outputs[0]), *outputs[1:])
            return outputs

        return compiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_case(*, use_lru: bool, device: str) -> None:
    torch._dynamo.reset()
    torch._C._dynamo.eval_frame._set_lru_cache(use_lru)

    backend = SaveSignatureBackend()
    block = torch.compile(Block().to(device), backend=backend)

    def forward(batch_size: int) -> torch.Tensor:
        inputs = torch.randn(batch_size, 4, device=device, requires_grad=True)
        return checkpoint(block, inputs, use_reentrant=False).sum()

    static_loss = forward(3)
    dynamic_loss = forward(5)

    entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(Block.forward)
    compile_ids = [str(entry.compile_id) for entry in entries]
    print(f"LRU {'enabled ' if use_lru else 'disabled'}: cache order {compile_ids}")
    assert backend.compile_count == 2, "Expected one static and one automatic-dynamic graph"

    try:
        static_loss.backward()
    except CheckpointError as error:
        if not use_lru:
            raise AssertionError("Disabling LRU should preserve the static graph") from error
        print("LRU enabled : reproduced CheckpointError")
        print(error)
    else:
        if use_lru:
            raise AssertionError("LRU should select the incompatible automatic-dynamic graph")
        print("LRU disabled: static backward passed")

    dynamic_loss.backward()
    print(f"LRU {'enabled ' if use_lru else 'disabled'}: dynamic backward passed")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu to run on CPU")

    try:
        run_case(use_lru=True, device=args.device)
        run_case(use_lru=False, device=args.device)
    finally:
        torch._C._dynamo.eval_frame._set_lru_cache(True)


if __name__ == "__main__":
    main()