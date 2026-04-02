# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Process-global acceleration flags.

Set once in train.py before model instantiation via `configure()`.
Deep model layers (block.py, mapper.py, etc.) read these flags at init/forward time.
"""

TRITON_GT_ENABLED: bool = False
TORCH_COMPILE_ENABLED: bool = False


def configure(*, triton_gt: bool = False, torch_compile: bool = False) -> None:
    """Set the global acceleration flags.

    Parameters
    ----------
    triton_gt : bool
        Enable the custom Triton graph-transformer attention kernel.
    torch_compile : bool
        Enable torch.compile wrapping for hot-path methods and the full model.
    """
    global TRITON_GT_ENABLED, TORCH_COMPILE_ENABLED
    TRITON_GT_ENABLED = triton_gt
    TORCH_COMPILE_ENABLED = torch_compile
