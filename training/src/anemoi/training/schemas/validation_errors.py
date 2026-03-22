from __future__ import annotations

from typing import Any

from pydantic import ValidationError


class ConfigValidationError(ValueError):
    """Readable wrapper around strict schema validation failures."""


def _format_error_location(location: tuple[Any, ...]) -> str:
    """Turn Pydantic error locations into a dotted config path."""
    parts = [str(part) for part in location if isinstance(part, str | int)]
    return ".".join(parts) if parts else "<root>"


def _normalise_discriminator_name(value: Any) -> str | None:
    """Clean Pydantic's quoted discriminator name for display."""
    if not isinstance(value, str):
        return None
    return value.strip("'")


def _format_error_value(value: Any, max_length: int = 80) -> str:
    """Shorten the displayed input value so errors stay readable."""
    rendered = repr(value)
    if len(rendered) <= max_length:
        return rendered
    return f"{rendered[: max_length - 3]}..."


def format_validation_error(error: ValidationError) -> str:
    """Build a short, config-focused summary from a Pydantic error."""
    lines = ["Config validation failed."]

    for entry in error.errors():
        location_parts = tuple(entry.get("loc", ()))
        location = _format_error_location(location_parts)
        message = entry.get("msg", "Invalid value.")
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")

        error_type = entry.get("type")
        context = entry.get("ctx", {})
        discriminator = _normalise_discriminator_name(context.get("discriminator"))

        if error_type == "union_tag_invalid":
            expected_tags = context.get("expected_tags", "")
            if discriminator is not None:
                location = _format_error_location((*location_parts, discriminator))
            message = f"Invalid discriminator value. Expected one of: {expected_tags}."
        elif error_type == "union_tag_not_found":
            if discriminator is not None:
                location = _format_error_location((*location_parts, discriminator))
            message = "Missing discriminator value."
        elif error_type == "extra_forbidden":
            message = "Unknown field."
        elif error_type == "missing":
            message = "Missing required value."

        line = f"- `{location}`: {message}"
        if "input" in entry and error_type not in {"missing", "extra_forbidden", "union_tag_not_found"}:
            line = f"{line} Got {_format_error_value(entry['input'])}."
        lines.append(line)

    lines.append("Set `config_validation=False` to skip strict schema checks.")
    return "\n".join(lines)
