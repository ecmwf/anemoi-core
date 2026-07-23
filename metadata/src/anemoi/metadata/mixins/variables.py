# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Mixin for variable selection and categorisation."""

from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from ..base import MetadataContract


class VariablesMixin:
    """Mixin providing variable selection and categorisation methods.

    This mixin requires the class to have a ``_raw`` attribute that is a
    :class:`~anemoi.metadata.base.MetadataContract` instance.  All inference data
    is accessed through the version-agnostic contract methods defined on
    ``MetadataContract``, so this mixin works with any schema version.
    """

    _raw: "MetadataContract"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @cached_property
    def variables(self) -> list[str]:
        """List of all variable names in the dataset (input and output).

        Names are taken from the union of input and output index mappings,
        ordered by their input tensor index first, then output-only variables
        appended in their output index order.

        Returns
        -------
        list[str]
            All variable names from both input and output index mappings.
        """
        input_vars = self._raw.get_variable_indices()
        output_vars = self._raw.get_output_variable_indices()
        # Start with input vars in order
        result = list(input_vars.keys())
        # Append output-only vars in their output index order
        output_only = {k: v for k, v in output_vars.items() if k not in input_vars}
        result.extend(k for k, _ in sorted(output_only.items(), key=lambda x: x[1]))
        return result

    @cached_property
    def num_variables(self) -> int:
        """Number of variables in the dataset.

        Returns
        -------
        int
            Count of variables.
        """
        return len(self.variables)

    def variable_categories(
        self,
        dataset_name: str | None = None,
        *,
        per_variable: bool = False,
    ) -> dict[str, list[str]]:
        """Categorize variables by their role in the model.

        In addition to the base categories returned by the contract
        (``"forcing"``, ``"prognostic"``, ``"diagnostic"``, ``"target"``),
        three derived categories are computed and merged in:

        * ``"computed"``  — variables flagged as computed forcings via
          :meth:`~anemoi.metadata.base.MetadataContract.get_computed_forcings`.
        * ``"accumulation"`` — variables whose ``process`` is ``"accumulated"``
          via :meth:`~anemoi.metadata.base.MetadataContract.get_accumulations`.
        * ``"constant"`` — variables whose ``constant_in_time`` flag is ``True``
          in the per-variable metadata returned by
          :meth:`~anemoi.metadata.base.MetadataContract.get_variables_metadata`.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.
        per_variable : bool, optional
            If ``True``, return a variable-keyed dict where each variable name
            maps to its sorted list of category strings.  If ``False`` (default),
            return a category-keyed dict where each category maps to a list of
            variable names belonging to it.

        Returns
        -------
        dict[str, list[str]]
            Category mapping.  Format depends on *per_variable*:

            * ``per_variable=False`` (default) — ``{category: [var, ...]}``.
              Only categories that contain at least one variable are present,
              except for the four base categories which are always included
              (possibly empty) for backward compatibility.
            * ``per_variable=True`` — ``{variable: [category, ...]}``.
              Categories are sorted alphabetically.

        Examples
        --------
        >>> cats = metadata.variable_categories()
        >>> cats["prognostic"]
        ['u', 'v', 't', 'q']

        >>> cats = metadata.variable_categories(per_variable=True)
        >>> cats["u"]
        ['prognostic', 'target']
        >>> cats["cos_latitude"]
        ['computed', 'constant', 'forcing']
        """
        # Base categories from the contract (category -> [variables]).
        # Always returns the four canonical keys, possibly with empty lists.
        base = self._raw.get_variable_types(dataset_name)

        if not per_variable:
            # Start from a shallow copy so we never mutate the contract's data.
            result: dict[str, list[str]] = {k: list(v) for k, v in base.items()}

            # -- computed forcings -------------------------------------------
            existing_computed: set[str] = set(result.setdefault("computed", []))
            for name in self._raw.get_computed_forcings(dataset_name):
                if name not in existing_computed:
                    result["computed"].append(name)
                    existing_computed.add(name)

            # -- accumulations -----------------------------------------------
            existing_accum: set[str] = set(result.setdefault("accumulation", []))
            for name in self._raw.get_accumulations(dataset_name):
                if name not in existing_accum:
                    result["accumulation"].append(name)
                    existing_accum.add(name)

            # -- constant-in-time --------------------------------------------
            existing_const: set[str] = set(result.setdefault("constant", []))
            for name, meta in self._raw.get_variables_metadata(dataset_name).items():
                if meta.get("constant_in_time") and name not in existing_const:
                    result["constant"].append(name)
                    existing_const.add(name)

            # Drop augmented keys that ended up empty to keep the output tidy
            # (the four base keys are always kept for backward compatibility).
            for aug_key in ("computed", "accumulation", "constant"):
                if not result[aug_key]:
                    del result[aug_key]

            return result

        # -- per_variable=True: invert to variable -> sorted[categories] -----
        inverted: defaultdict[str, set[str]] = defaultdict(set)

        for cat, names in base.items():
            for name in names:
                inverted[name].add(cat)

        for name in self._raw.get_computed_forcings(dataset_name):
            inverted[name].add("computed")

        for name in self._raw.get_accumulations(dataset_name):
            inverted[name].add("accumulation")

        for name, meta in self._raw.get_variables_metadata(dataset_name).items():
            if meta.get("constant_in_time"):
                inverted[name].add("constant")

        return {name: sorted(cats) for name, cats in inverted.items()}

    def select_variables(
        self,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        dataset_name: str | None = None,
    ) -> list[str]:
        """Select variables by category expression or substring pattern.

        Each entry in *include* / *exclude* is first parsed as a category
        expression:

        * A single category name (e.g. ``"prognostic"``) matches all variables
          belonging to that category.
        * A compound expression with ``+`` (e.g. ``"prognostic+forcing"``)
          matches variables belonging to **all** of the listed categories
          (conjunction / intersection).

        If an entry does not match any known category expression it is treated
        as a substring pattern matched against variable names (backward-compat).

        Parameters
        ----------
        include : list[str] | None, optional
            If provided, only variables matching at least one of these
            expressions are kept.
        exclude : list[str] | None, optional
            If provided, variables matching any of these expressions are removed.
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        list[str]
            Filtered list of variable names, in original order.

        Raises
        ------
        ValueError
            If a compound expression (containing ``+``) references one or more
            unknown category names.

        Examples
        --------
        >>> metadata.select_variables(include=["prognostic"])
        ['u', 'v', 't', 'q']

        >>> metadata.select_variables(include=["prognostic+forcing"])
        []  # no variable is both prognostic and forcing

        >>> metadata.select_variables(include=["computed+constant"])
        ['cos_latitude', 'sin_latitude']
        """
        # Per-variable mapping used for compound (intersection) matching.
        per_var = self.variable_categories(dataset_name, per_variable=True)
        # Category-keyed mapping used to enumerate known category names.
        cat_keyed = self.variable_categories(dataset_name)

        # Use the dataset-specific variable list when a dataset is specified,
        # falling back to the cached first-dataset list otherwise.
        if dataset_name is not None:
            all_variables = list(self._raw.get_variable_indices(dataset_name).keys())
        else:
            all_variables = list(self.variables)

        known_categories: set[str] = set(cat_keyed.keys()) | {
            "computed",
            "accumulation",
            "constant",
        }

        def _parse_expression(expr: str) -> frozenset[str] | None:
            """Parse a category expression into a frozenset of category names.

            Returns a frozenset when *all* ``+``-separated parts are known
            category names, or ``None`` when the expression is a single token
            that does not match any category (substring fallback).

            Raises ``ValueError`` for compound expressions that contain at
            least one unknown category name.
            """
            parts = expr.split("+")
            if all(p in known_categories for p in parts):
                return frozenset(parts)
            if len(parts) == 1:
                # Single unrecognised token — fall back to substring matching.
                return None
            # Compound expression with at least one unknown part: raise.
            unknown = [p for p in parts if p not in known_categories]
            raise ValueError(
                f"Unknown category(ies) {unknown!r} in compound expression "
                f"{expr!r}. Known: {sorted(known_categories)}"
            )

        def _matches(variable: str, expressions: list[str]) -> bool:
            """Return True if *variable* matches any expression in the list."""
            var_cats = set(per_var.get(variable, []))
            for expr in expressions:
                parsed = _parse_expression(expr)
                if parsed is not None:
                    # Category expression: variable must belong to ALL parts.
                    if parsed.issubset(var_cats):
                        return True
                else:
                    # Substring fallback for backward compatibility.
                    if expr in variable:
                        return True
            return False

        result = list(all_variables)

        if include is not None:
            result = [v for v in result if _matches(v, include)]

        if exclude is not None:
            result = [v for v in result if not _matches(v, exclude)]

        return result

    def typed_variables(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Return strongly-typed Variable objects for each variable.

        Each variable is wrapped in an :class:`anemoi.transform.variables.Variable`
        instance constructed from its per-variable metadata dict.  This gives
        typed access to properties such as ``is_accumulation``,
        ``is_computed_forcing``, ``is_constant_in_time``, etc.

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to query.  Defaults to the first dataset.

        Returns
        -------
        dict[str, Variable]
            Mapping of variable names to ``Variable`` instances.

        Raises
        ------
        ImportError
            If ``anemoi-transform`` is not installed.

        Examples
        --------
        >>> tvars = metadata.typed_variables()
        >>> tvars["cos_latitude"].is_computed_forcing
        True
        >>> tvars["cos_latitude"].is_constant_in_time
        True
        """
        try:
            from anemoi.transform.variables import Variable
        except ImportError as exc:
            raise ImportError(
                "typed_variables() requires anemoi-transform. " "Install it with: pip install anemoi-transform"
            ) from exc

        var_meta = self._raw.get_variables_metadata(dataset_name)
        if dataset_name is not None:
            input_vars = self._raw.get_variable_indices(dataset_name)
            output_vars = self._raw.get_output_variable_indices(dataset_name)
            # Union of input and output variables
            all_variables = list(input_vars.keys())
            all_variables.extend(k for k in output_vars if k not in input_vars)
        else:
            all_variables = list(self.variables)

        return {name: Variable.from_dict(name, var_meta.get(name, {})) for name in all_variables}
