# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to edit checkpoint metadata in an external editor."""

import json
import logging
import os
import shlex
import subprocess
import tempfile
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from anemoi.utils.cli import Command

LOG = logging.getLogger(__name__)

# Some editors require extra flags to block until the file is closed.
_EDITOR_WAIT_FLAGS: dict[str, list[str]] = {"code": ["--wait"]}


class EditCommand(Command):
    """Edit checkpoint metadata in an editor."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")
        command_parser.add_argument(
            "--editor",
            default=os.environ.get("EDITOR", "vi"),
            help=("Editor command to use. Defaults to $EDITOR if set, otherwise 'vi'."),
        )
        command_parser.add_argument(
            "--yaml",
            action="store_true",
            help="Edit in YAML format (requires PyYAML).",
        )
        command_parser.add_argument(
            "--json",
            action="store_true",
            help="Edit in JSON format (default).",
        )

    def run(self, args: Namespace) -> None:
        """Execute the edit command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from ..checkpoint import extract_metadata_dict
        from ..checkpoint import replace_metadata

        use_yaml = args.yaml
        ext = "yaml" if use_yaml else "json"

        def _serialise(data: dict[str, Any], fh: Any) -> None:
            if use_yaml:
                yaml.dump(data, fh, default_flow_style=False)
            else:
                json.dump(data, fh, indent=4, sort_keys=True)

        def _deserialise(fh: Any) -> dict[str, Any]:
            if use_yaml:
                return yaml.safe_load(fh)
            return json.load(fh)

        metadata = extract_metadata_dict(args.path)

        # Serialise original metadata to compare later.
        original_serialised = _serialise_to_string(metadata, use_yaml)

        with TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, f"checkpoint.{ext}")

            with open(tmp_file, "w") as fh:
                _serialise(metadata, fh)

            cmd_parts = shlex.split(args.editor)
            extra_flags = _EDITOR_WAIT_FLAGS.get(os.path.basename(cmd_parts[0]), [])

            try:
                subprocess.check_call([*cmd_parts, *extra_flags, tmp_file])
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                # Editor failed or wasn't found. Preserve the edited file.
                preserved_path = _preserve_edited_file(tmp_file, ext, original_serialised)
                if preserved_path:
                    LOG.error(
                        "Editor command failed: %s\nEdited file preserved at: %s",
                        exc,
                        preserved_path,
                    )
                else:
                    LOG.error("Editor command failed: %s", exc)
                raise SystemExit(1) from exc

            with open(tmp_file) as fh:
                edited = _deserialise(fh)

            if edited != metadata:
                replace_metadata(Path(args.path), edited)
                LOG.info("Metadata updated in %s", args.path)
            else:
                LOG.info("No changes detected; checkpoint left unmodified.")


def _serialise_to_string(data: dict[str, Any], use_yaml: bool) -> str:
    """Serialise metadata to a string for comparison.

    Parameters
    ----------
    data : dict[str, Any]
        Metadata dict to serialise.
    use_yaml : bool
        Whether to serialise as YAML or JSON.

    Returns
    -------
    str
        Serialised metadata string.
    """
    if use_yaml:
        return yaml.dump(data, default_flow_style=False)
    return json.dumps(data, indent=4, sort_keys=True)


def _preserve_edited_file(
    tmp_file: str,
    ext: str,
    original_serialised: str,
) -> str | None:
    """Preserve edited temporary file to a persistent location if it differs from original.

    Parameters
    ----------
    tmp_file : str
        Path to the temporary edited file.
    ext : str
        File extension (json or yaml).
    original_serialised : str
        Original metadata serialised to string for comparison.

    Returns
    -------
    str | None
        Path to the preserved file, or None if file doesn't exist or is unchanged.
    """
    if not os.path.exists(tmp_file):
        return None

    try:
        with open(tmp_file) as f:
            edited_content = f.read()
    except Exception:
        # If we can't read the file, don't preserve it.
        return None

    # Only preserve if content differs from original.
    if edited_content.strip() == original_serialised.strip():
        return None

    # Create a persistent temporary file.
    fd, preserved_path = tempfile.mkstemp(
        prefix="anemoi-metadata-edit-",
        suffix=f".{ext}",
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(edited_content)
        return preserved_path
    except Exception:
        # Clean up on error.
        try:
            os.unlink(preserved_path)
        except Exception:
            pass
        return None


command = EditCommand
