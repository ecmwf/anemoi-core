# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import logging

import torch
import torch_geometric

LOG = logging.getLogger(__name__)


# we need to have ConditionalCompile and _ConditionalCompile, to be able to pass arguments to the compile wrapper
# so like @ConditionalCompile(dynamic=False)
# We can then pass these arguments to torch.compile
def ConditionalCompile(_func=None, *, dynamic=False):
    def decorator(method):
        class _ConditionalCompile:
            def __init__(self, method):
                self.method = method
                self.compiled_methods = {}
                self.dynamic = dynamic

            def __get__(self, instance, owner):
                if instance is None:
                    return self.method

                @functools.wraps(self.method)
                def wrapper(*args, **kwargs):
                    if instance.compile:
                        req = torch.__version__ >= "2.6" and torch_geometric.__version__ >= "2.6"
                        if not req:
                            LOG.debug(
                                f"Requirements not met to conditionally compile {instance.__class__.__name__}, returning original function."
                            )
                            return self.method.__get__(instance, owner)(*args, **kwargs)
                        if instance not in self.compiled_methods:
                            method_name = f"{owner.__name__}.{self.method.__name__}"
                            LOG.info(f"Compiling {method_name} using conditional compile (dynamic={self.dynamic})")
                            compiled = torch.compile(self.method.__get__(instance, owner), dynamic=self.dynamic)
                            self.compiled_methods[instance] = compiled
                        return self.compiled_methods[instance](*args, **kwargs)
                    else:
                        return self.method.__get__(instance, owner)(*args, **kwargs)

                return wrapper

        return _ConditionalCompile(method)

    if _func is None:
        return decorator
    else:
        return decorator(_func)
