# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Caching utilities for stateless preprocessors."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def cached_parameters(key_fn: Callable[..., tuple]) -> Callable:
    """Decorator that caches method results on the instance.

    Caches the return value of a method in ``self._param_cache``, keyed by the
    result of ``key_fn(*args, **kwargs)``. The decorated method is only called
    on cache misses.

    The owning class must initialise ``self._param_cache = {}`` (typically in
    ``__init__``). Call ``self.reset_cache()`` to invalidate all entries.

    Parameters
    ----------
    key_fn : Callable[..., tuple]
        A function that receives the same positional and keyword arguments as
        the decorated method (excluding ``self``) and returns a hashable cache
        key. Typical keys combine the variable set and device.

    Example
    -------
    >>> def _cache_key(statistics, name_to_index, device):
    ...     return (tuple(sorted(name_to_index.items())), str(device))
    ...
    >>> class MyPreprocessor(BasePreprocessor):
    ...     def __init__(self, ...):
    ...         ...
    ...         self._param_cache = {}
    ...
    ...     @cached_parameters(key_fn=_cache_key)
    ...     def get_params(self, statistics, name_to_index, device):
    ...         # expensive computation
    ...         return build_params(statistics, name_to_index, device)
    """

    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = key_fn(*args, **kwargs)
            cached = self._param_cache.get(key)
            if cached is not None:
                return cached
            result = method(self, *args, **kwargs)
            self._param_cache[key] = result
            return result

        return wrapper

    return decorator
