# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import healpy as hp
import torch


def get_healpix_edgeindex(resolution: int) -> torch.Tensor:
    """Get the HEALPix edge index for a given resolution."""
    npix = hp.nside2npix(2**resolution)
    edge_index = torch.zeros((2, npix * 8))
    edge_index[0] = torch.repeat_interleave(torch.arange(npix), 8)
    for i in range(npix):
        edge_index[1, i * 8 : (i + 1) * 8] = torch.from_numpy(hp.get_all_neighbours(2**resolution, i, nest=True))
    return edge_index
