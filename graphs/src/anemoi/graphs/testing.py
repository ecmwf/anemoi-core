# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np


class DatasetMock:
    """This datasets emulates the most primitive unstructured grid with
    refinement.

    Enumeration of cells , edges and vertices in netCDF file is 1 based.
    C: cell
    E: edge
    V: vertex

    Cell C2 with its additional vertex V4 and edges E4 and E4 were added as
    a first refinement.

    [V1: 0, 1]🢀-E3--[V3: 1, 1]
      🢁      ╲             🢁
      |       ╲ [C1: ⅔, ⅔] |
      |        ╲           |
      E5        E1         E2
      |          ╲         |
      |           ╲        |
      | [C2: ⅓, ⅓] ╲       |
      |             🢆     |
    [V4: 0, 1]🢀-E4--[V2: 1, 1]

    Note: Triangular refinement does not actually work like this. This grid
    mock serves testing purposes only.

    """

    def __init__(self, *args, **kwargs):

        class MockVariable:
            def __init__(self, data, units, dimensions):
                self.data = np.ma.asarray(data)
                self.shape = data.shape
                self.units = units
                self.dimensions = dimensions

            def __getitem__(self, key):
                return self.data[key]

        self.variables = {
            "vlon": MockVariable(np.array([0, 1, 1, 0]), "radian", ("vertex",)),
            "vlat": MockVariable(np.array([1, 0, 1, 0]), "radian", ("vertex",)),
            "clon": MockVariable(np.array([0.66, 0.33]), "radian", ("cell",)),
            "clat": MockVariable(np.array([0.66, 0.33]), "radian", ("cell",)),
            "edge_vertices": MockVariable(np.array([[1, 2], [2, 3], [3, 1], [2, 4], [4, 1]]).T, "", ("nc", "edge")),
            "vertex_of_cell": MockVariable(np.array([[1, 2, 3], [1, 2, 4]]).T, "", ("nv", "cell")),
            "refinement_level_v": MockVariable(np.array([0, 0, 0, 1]), "", ("vertex",)),
            "refinement_level_c": MockVariable(np.array([0, 1]), "", ("cell",)),
        }
        """common array dimensions:
            nc: 2, # constant
            nv: 3, # constant
            vertex: 4,
            edge: 5,
            cell: 2,
        """
        self.uuidOfHGrid = "__test_data__"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
