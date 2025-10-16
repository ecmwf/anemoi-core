# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch


def make_truncation_matrix(A, data_type=torch.float32):
    A_ = torch.sparse_coo_tensor(
        torch.tensor(np.vstack(A.nonzero()), dtype=torch.long),
        torch.tensor(A.data, dtype=data_type),
        size=A.shape,
    ).coalesce()
    return A_


def truncate_fields(x, A, batch_size=None, auto_cast=False):
    if not batch_size:
        batch_size = x.shape[0]
    out = []
    with torch.amp.autocast(device_type="cuda", enabled=auto_cast):
        for i in range(batch_size):
            out.append(multiply_sparse(x[i, ...], A))
    return torch.stack(out)


def multiply_sparse(x, A):
    if torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out = torch.sparse.mm(A, x)
    else:
        with torch.amp.autocast(device_type="cpu", enabled=False):
            out = torch.sparse.mm(A, x)
    return out
