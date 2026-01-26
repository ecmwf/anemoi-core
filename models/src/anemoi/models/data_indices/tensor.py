# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch


class BaseTensorIndex:
    """Indexing for variables in index as Tensor."""

    def __init__(
        self,
        *,
        prognostic: list[str],
        diagnostic: list[str],
        forcing: list[str],
        includes: list[str],
        name_to_index: dict[str, int],
        target: list[str] = [],
    ) -> None:
        """Initialize indexing tensors from includes using name_to_index.

        Parameters
        ----------
        prognostic : list[str]
            List of prognostic variable.
        diagnostic : list[str]
            List of diagnostic variable.
        forcing : list[str]
            List of forcing variable.
        includes : list[str]
            Variables to include in the indexing.
        name_to_index : dict[str, int]
            Dictionary mapping variable names to their index in the Tensor.
        target : optional, list[str]
            List of target variable.
        """
        self.name_to_index = name_to_index
        self.includes = sorted(includes)
        assert set(includes).issubset(
            self.name_to_index.keys(),
        ), f"Data indexing has invalid entries {[var for var in includes if var not in self.name_to_index]}, not in dataset."
        self.prognostic = self._build_idx_from_list(prognostic)
        self.diagnostic = self._build_idx_from_list(diagnostic)
        self.forcing = self._build_idx_from_list(forcing)
        self.target = self._build_idx_from_list(target)
        self.full = self._build_idx_from_list(includes)
        self.excludes = sorted(list(set(self.name_to_index.keys()) - set(self.includes)))

    def _build_idx_from_list(self, var_list):
        sorted_variables = torch.Tensor(sorted(i for name, i in self.name_to_index.items() if name in var_list)).to(
            torch.int
        )
        return sorted_variables

    def __len__(self) -> int:
        return len(self.full)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(includes={self.includes}, excludes={self.excludes}, full={self.full})"

    def __eq__(self, other):
        if not isinstance(other, BaseTensorIndex):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (
            torch.allclose(self.prognostic, other.prognostic)
            and torch.allclose(self.diagnostic, other.diagnostic)
            and torch.allclose(self.forcing, other.forcing)
            and torch.allclose(self.target, other.target)
            and torch.allclose(self.full, other.full)
            and self.includes == other.includes
            and self.excludes == other.excludes
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "prognostic": self.prognostic,
            "diagnostic": self.diagnostic,
            "forcing": self.forcing,
            "target": self.target,
            "full": self.full,
        }

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar(f"!{data.__class__.__name__}", repr(data))


class InputTensorIndex(BaseTensorIndex):
    """Indexing for input variables."""

    def __init__(
        self,
        *,
        prognostic: list[str],
        diagnostic: list[str],
        forcing: list[str],
        includes: list[str],
        name_to_index: dict[str, int],
        target: list[str] = [],
    ) -> None:
        super().__init__(
            prognostic=prognostic,
            diagnostic=diagnostic,
            forcing=forcing,
            target=target,
            includes=includes,
            name_to_index=name_to_index,
        )

    def __len__(self) -> int:
        return len(self.prognostic) + len(self.forcing)


class OutputTensorIndex(BaseTensorIndex):
    """Indexing for output variables."""

    def __init__(
        self,
        *,
        prognostic: list[str],
        diagnostic: list[str],
        forcing: list[str],
        includes: list[str],
        name_to_index: dict[str, int],
        target: list[str] = [],
        **kwargs,
    ) -> None:
        super().__init__(
            prognostic=prognostic,
            diagnostic=diagnostic,
            forcing=forcing,
            target=target,
            includes=includes,
            name_to_index=name_to_index,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.full)
