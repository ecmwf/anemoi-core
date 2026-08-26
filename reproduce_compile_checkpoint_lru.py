"""Reproduce a checkpoint mismatch after a torch.compile global-state recompile."""

from __future__ import annotations

import argparse
import os

os.environ["TORCH_LOGS"] = "recompiles"

import torch
from torch.utils.checkpoint import CheckpointError
from torch.utils.checkpoint import checkpoint

INITIAL_NUM_THREADS = torch.get_num_threads()


class ThreadSpecializedBlock(torch.nn.Module):
    """Make the graph selected by the ATen thread state save one extra tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.sin()
        if torch.get_num_threads() == INITIAL_NUM_THREADS:
            output = output.cos()
        return output * x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-debug", action="store_true", help="print checkpoint's operator-by-operator trace")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu to run on CPU")

    torch._dynamo.reset()
    #torch._C._dynamo.eval_frame._set_lru_cache(False)
    initial_num_threads = INITIAL_NUM_THREADS
    changed_num_threads = 2 if initial_num_threads == 1 else initial_num_threads - 1
    block = torch.compile(ThreadSpecializedBlock().to(args.device))

    def forward(batch_size: int) -> torch.Tensor:
        inputs = torch.randn(batch_size, 16, device=args.device, requires_grad=True)
        return checkpoint(block, inputs, use_reentrant=False, debug=True).sum()

    try:
        losses = [forward(3)]
        torch.set_num_threads(changed_num_threads)
        losses.extend(forward(batch_size) for batch_size in (5, 3, 5))
        print(f"Completed 4 forwards; num_threads changed {initial_num_threads} -> {changed_num_threads}")

        try:
            losses[0].backward()
        except CheckpointError as error:
            print("REPRODUCED: oldest forward selected an incompatible graph during recomputation")
            if args.full_debug:
                raise
            print(error.__cause__ or error)
        else:
            raise RuntimeError("CheckpointError was not reproduced")

        for index in (1, 2, 3):
            losses[index].backward()
            print(f"Backward {index} completed")
    finally:
        torch.set_num_threads(initial_num_threads)


if __name__ == "__main__":
    main()