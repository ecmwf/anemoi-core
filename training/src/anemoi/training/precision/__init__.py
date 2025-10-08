# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Custom precision plugins for Anemoi training."""

from anemoi.training.precision.bf16_precision import BF16FP32OptPrecision

__all__ = ["BF16FP32OptPrecision"]
