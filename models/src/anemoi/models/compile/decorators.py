# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

LOG = logging.getLogger(__name__)


def torch_compile(dynamic=True, **kwargs):
    """Compile a function using torch.compile.

    This requires torch>=2.6, and torch_geometric>=2.6,
    and will silently return the original function if the requirements are not met.

    All arguments are passed to torch.compile.
    """

    def decorator(func):
        import torch
        import torch_geometric

        req = torch.__version__ >= "2.6" and torch_geometric.__version__ >= "2.6"
        if req:
            compile_func = torch.compile(dynamic=dynamic, **kwargs)
            return compile_func(func)

        LOG.debug("Requirements not met for torch.compile, returning original function.")
        return func

    return decorator
