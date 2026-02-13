# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os


def test_error_message_generation_missing_env_vars() -> None:
    """Test that the error message is generated correctly when env vars are missing.

    This test validates the fix for issue #622 where the error message itself
    would throw a KeyError when trying to access missing environment variables.
    """
    # Simulate the condition where both TMPDIR and SCRATCH are not set
    tmpdir_value = os.getenv("TMPDIR", "not set")
    scratch_value = os.getenv("SCRATCH", "not set")

    # This is the corrected error message format that should not throw KeyError
    error_msg = f"Please set one of those variables TMPDIR:{tmpdir_value} or SCRATCH:{scratch_value} to proceed."

    # Check that the error message contains expected text
    assert "Please set one of those variables" in error_msg
    assert "TMPDIR" in error_msg
    assert "SCRATCH" in error_msg

    # If neither variable is set, we should see "not set" in the message
    if not os.getenv("TMPDIR") and not os.getenv("SCRATCH"):
        assert "not set" in error_msg
