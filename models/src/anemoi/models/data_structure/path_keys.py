# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

from boltons.iterutils import research as _research  # noqa: F401

SEPARATOR = "__"

_MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS = {
    # "Xbackslash_": "\\",
    # "Xbar_": "|",
    "Xcolon_": ":",
    # "Xequal_": "=",
    "Xgreater_": "<",
    "Xless_": ">",
    "Xminus_": "-",
    "Xplus_": "+",
    # "Xquestion_": "?",
    # "Xslash_": "/",
    "Xstar_": "*",
    # "Xtilde_": "~",
}
_ALLOWED_CHARACTERS_IN_DICT_KEYS = set(_MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS.values())


def _check_no_conflict(dic):
    for key in dic:
        conflicts = {k: v for k, v in dic.items() if k.startswith(key)}
        if len(conflicts) > 1:
            raise ValueError(f"Conflict in allowed characters mapping: {conflicts}")


_check_no_conflict(_MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS)

_RESERVED_WORDS_FOR_TORCH_MODULE = [  # from torch.nn.Module
    "dump_patches",
    "call_super_init",
    "forward",
    "register_buffer",
    "register_parameter",
    "add_module",
    "register_module",
    "get_submodule",
    "set_submodule",
    "get_parameter",
    "get_buffer",
    "get_extra_state",
    "set_extra_state",
    "apply",
    "cuda",
    "ipu",
    "xpu",
    "mtia",
    "cpu",
    "type",
    "float",
    "double",
    "half",
    "bfloat16",
    "to_empty",
    "to",
    "register_full_backward_pre_hook",
    "register_backward_hook",
    "register_full_backward_hook",
    "register_forward_pre_hook",
    "register_forward_hook",
    "register_state_dict_post_hook",
    "register_state_dict_pre_hook",
    "T_destination",
    "state_dict",
    "register_load_state_dict_pre_hook",
    "register_load_state_dict_post_hook",
    "load_state_dict",
    "parameters",
    "named_parameters",
    "buffers",
    "named_buffers",
    "children",
    "named_children",
    "modules",
    "named_modules",
    "train",
    "eval",
    "requires_grad_",
    "zero_grad",
    "share_memory",
    "extra_repr",
    "compile",
]


def is_decoded(path):
    if not isinstance(path, str):
        raise TypeError(f"Expected string for path, got {type(path)}: {path}")
    if SEPARATOR in path:  # Found separator, this only happens in encoded paths
        return False
    if not path.lower() == path:  # there is a capital letter, this only happens in encoded paths
        return False
    return True


def is_encoded(path):
    if not isinstance(path, str):
        raise TypeError(f"Expected string for path, got {type(path)}: {path}")
    if "." in path:  # Found dot, this only happens in human readable paths
        return False
    if any(c.isupper() for c in path):  # there is a capital letter, this only happens in encoded paths
        return True
    if any(c in _ALLOWED_CHARACTERS_IN_DICT_KEYS for c in path):
        return False
    return True


def encode_path_if_needed(path):
    if isinstance(path, (list, tuple)):
        assert False, "Should not happen anymore"
        return SEPARATOR.join(encode_path_if_needed(x) for x in path)

    if is_encoded(path):
        return path

    if path.startswith("."):
        raise KeyError(f"Path starting with {SEPARATOR} is not allowed. Got {path}")

    # here we may want to have some generic handling of special characters
    # but for now we only handle the ones in MAPPING_OF_ALLOWED_CHARACTERS_IN
    for k, v in _MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS.items():
        if v in path:
            path = path.replace(v, k)

    path = path.replace(".", SEPARATOR)
    check_encoded_path(path)
    return path


def check_encoded_path(path):
    def _check_encoded_character(c, path):
        if c.islower():
            return
        if c.isdigit():
            return
        if c in ["_", "X"]:
            return
        raise KeyError(f"Cannot contain '{c}', got {path}")

    for c in path:
        _check_encoded_character(c, path)

    if "___" in path:
        raise KeyError(f"Cannot contain '___' sequence, got {path}")


def decode_path_if_needed(path):
    if is_decoded(path):
        return path

    if not is_encoded(path):
        raise KeyError(f"Path is neither encoded nor decoded, cannot decode: {path}")

    for k, v in _MAPPING_OF_ALLOWED_CHARACTERS_IN_DICT_KEYS.items():
        if k in path:
            path = path.replace(k, v)

    path = path.replace(SEPARATOR, ".")
    assert is_decoded(path), path
    return path


def check_dictionary_key(k):
    # some of these checks are duplicates, but it's safer to have them all here
    # speed optimisation is not critical here

    if not isinstance(k, str):
        raise TypeError(f"Expected string for dictionary key, got {type(k)}: {k}")

    if k.lower() != k:
        # we could allow capital letters and encode the case in the mapping
        # but the encoded keys would be hard to read
        # and this would need careful handling
        raise ValueError(f"Keys in dictionary must be lowercase, got: {k}.")

    if "." in k:
        raise ValueError(f"Keys in dictionary must not contain '.', got: {k}")

    if SEPARATOR in k:
        # this may be ok in some cases, but for now we disallow it
        raise ValueError(f"Keys in dictionary must not contain '{SEPARATOR}', got: {k}")

    if k.startswith("_"):
        raise ValueError(f"Keys in dictionary must not start with '_', got: {k}")

    if k.endswith("_"):
        # this is not needed and should be removed, when all this file has been refactored
        raise ValueError(f"Keys in dictionary must not end with '_', got: {k}")

    for c in k:
        if not c.isalnum() and c not in _ALLOWED_CHARACTERS_IN_DICT_KEYS and c not in ["_", "X"]:
            raise ValueError(
                f"Keys in dictionary must only contain {_ALLOWED_CHARACTERS_IN_DICT_KEYS} characters and _, got: '{c}' in '{k}'",
            )

    # if k[0].isdigit():
    # this is to keep open the possibility to handle lists/tuples in the future
    #    raise ValueError(f"Keys in dictionary must not start with a digit for now, got: {k}")

    if k in _RESERVED_WORDS_FOR_TORCH_MODULE:
        raise ValueError(f"Keys in dictionary must not be a reserved word for torch.nn.Module '{k}'.")

    assert is_decoded(k), k

    return k


def join_paths(path1, path2):
    if not path1:
        return path2
    if not path2:
        return path1
    return SEPARATOR.join([path1, path2])


def path_as_tuple(path):
    path = decode_path_if_needed(path)
    return tuple(path.split("."))


def tuple_to_path(t):
    assert isinstance(t, (list, tuple)), t
    return SEPARATOR.join(t)
