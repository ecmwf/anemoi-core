# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from torch import nn


class GLU(nn.Module):
    def __init__(self, dim, variation=nn.Sigmoid()):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.variation = variation

    def forward(self, X):
        return self.variation(self.W(X)) * self.V(X)


class SwiGLU(GLU):
    def __init__(self, dim):
        super().__init__(dim, nn.SiLU())


class ReGLU(GLU):
    def __init__(self, dim):
        super().__init__(dim, nn.ReLU())


class GEGLU(GLU):
    def __init__(self, dim):
        super().__init__(dim, nn.GELU())
