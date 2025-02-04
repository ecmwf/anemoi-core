# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError
from pydantic_core import ErrorDetails


class BaseModel(PydanticBaseModel):
    class Config:
        """Pydantic BaseModel configuration."""

        use_attribute_docstrings = True
        use_enum_values = True
        validate_assignment = True
        validate_default = True
        extra = "ignore"


CUSTOM_MESSAGES = {
    "missing": "A config entry seems to be missing. If not please check for any typos.",
    "extra_forbidden": "Extra entries in the config are forebidden. Please check for typos.",
}


def convert_errors(e: ValidationError, custom_messages: dict[str, str]) -> list[ErrorDetails]:
    new_errors: list[ErrorDetails] = []
    for error in e.errors():
        custom_message = custom_messages.get(error["type"])

        if custom_message:

            ctx = error.get("ctx")
            error["msg"] = custom_message.format(**ctx) if ctx else custom_message
        new_errors.append(error)
    return new_errors


class ValidationError(Exception):
    pass


def allowed_values(v: Any, values: list[Any]) -> Any:
    if v not in values:
        msg = {f"Value {v} not in {values}"}
        raise ValidationError(msg)
    return v
