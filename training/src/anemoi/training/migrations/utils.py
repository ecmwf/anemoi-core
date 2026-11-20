# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
import inspect

_MIGRATION_TEMPLATE = """\
{% for import in imports %}
{{import}}
{% endfor %}

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "{{migration_version}}",
        "anemoi-training": "%NEXT_ANEMOI_TRAINING_VERSION%",
    },
    {% if final %}
    final=True,
    {% endif %}
)
# <-- END DO NOT CHANGE
{% if not final %}


def migrate(config: Config) -> Config:
    \"""Migrate the config.

    Parameters
    ----------
    config : Config
        The config object to migrated.

    Returns
    -------
    Config
        The migrated config.
    \"""
    config.add_summary("\n".join([
        "Add here a summary of the changes to the config, or give context",
        "of the changes introduced in your PR. This summary will be displayed",
        "at the top of the users migrated config."
    ]))
    return config
{% endif %}
"""

_LLM_PROMPT_GUIDELINES = """\
You are an expert Python developer tasked with writing a **configuration migration script** for a Python 3.11+ project.
Follow these **strict guidelines** to ensure the code is maintainable, type-safe, and idiomatic:

#### **General Best Practices**
- **Typing**:
  - Use native Python types (e.g., `list`, `dict`, `set`) instead of their `typing` module equivalents (e.g., `List`,
  `Dict`, `Set`).
  - Prefer `collections.abc` (e.g., `Sequence`, `Mapping`, `Callable`) over `typing` for abstract base classes.
  - Use union types with `|` (e.g., `int | str`) instead of `Union[int, str]`.

- **Code Style**:
  - Follow [PEP 8](https://peps.python.org/pep-0008/) strictly (e.g., 4-space indents, snake_case for
  variables/functions, CamelCase for classes).
  - Use f-strings for string formatting.
  - Avoid redundant comments. Only add comments to explain **why** (not **what**), or for complex logic.

- **Error Handling**:
  - Use custom exceptions for domain-specific errors.
  - Prefer `raise ... from ...` for chained exceptions.
  - Prefer testing with if blocks than using `try/except` blocks extensively.
"""

_LLM_MIGRATION_GUIDELINES = """
Here is the migration template that you have to edit:
```python
{{migration_template}}
```

Please update the `migrate` method to reflect the changes made (see section 4. Git diff for
the actual changes to the code).

The migration should focus on structural changes to the config, not on changing default values.

Please add a general introduction about the purpose of the migration is `config.add_summary`.

Always use `config.has_key` before altering a key.

The input config is assumed to be a config working before the given changes and the returned
config is the config migrated to accomodate for the changes made.
This config is a config dump done by Hydra on all the configuration files.

Use the API given in section 2. to help you write the migration.
"""


def extract_class_api(module_name: str, class_name: str) -> str:
    """Extract the API (signature + docstring) of a class and its methods.

    Parameters
    ----------
        module_name : str
            Name of the module (e.g., "migration.system").
        class_name : str
            Name of the class (e.g., "SelectionError").

    Returns
    -------
        Formatted string with class and method details.
    """
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    api_lines = []

    api_lines.append(f"### Class: {class_name}")
    try:
        sig = inspect.signature(cls)
        api_lines.append(f"Signature: {sig}")
    except (ValueError, TypeError):
        api_lines.append("Signature: (unavailable)")

    doc = inspect.getdoc(cls)
    if doc:
        api_lines.append(f"Docstring: {doc}")
    api_lines.append("")

    api_lines.append("#### Methods:")
    methods = inspect.getmembers(cls, inspect.isfunction)
    for method_name, method in methods:
        # Skip private methods and inherited methods
        if method_name.startswith("_") and method.__qualname__.split(".")[0] != class_name:
            continue

        api_lines.append(f"--- {method_name}")

        try:
            sig = inspect.signature(method)
            api_lines.append(f"Signature: {sig}")
        except (ValueError, TypeError):
            api_lines.append("Signature: (unavailable)")

        doc = inspect.getdoc(method)
        if doc:
            api_lines.append(f"Docstring: {doc}")
        api_lines.append("")

    return "\n".join(api_lines)


def extract_api_from_files(selected_modules: list[tuple[str, str]]) -> str:
    """Extract API from a list of Python files."""
    api_parts = []
    for module_name, class_name in selected_modules:
        try:
            api_parts.append(f"--- API from {module_name}.{class_name} ---")
            api_parts.append(extract_class_api(module_name, class_name))
        except ImportError as e:
            api_parts.append(f"Failed to import {module_name}: {e}")
    return "\n".join(api_parts)
