# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

from typing import Any


class NotSet:
    pass


NULL_VALUE = NotSet()


def get_closest_key(dictionary: dict[int, Any], key: int) -> int | None:
    """
    Get the closest int key in a dictionary to a given key.

    Where the closest key is the one with the smallest absolute difference
    and the key is less than or equal to the given key.

    If no lower key is found, returns None.

    Parameters
    ----------
    dictionary : dict[int, Any]
        Dictionary to search.
    key : int
        Key to search for.

    Returns
    -------
    int | None
        Closest key in the dictionary.
    """
    if len(dictionary) == 0:
        return None

    lowest_key = min(dictionary.keys(), key=lambda x: abs(x - key) if x <= key else float("inf"))
    if key < lowest_key:
        return None
    return lowest_key


def get_value_from_closest_key(dictionary: dict[int, Any], key: int, default: Any = NULL_VALUE) -> int:
    """Get value from dictionary with the closest key."""
    closest_key = get_closest_key(dictionary, key)
    if not isinstance(default, NotSet):
        return dictionary.get(closest_key, default)
    return dictionary.get(closest_key)
