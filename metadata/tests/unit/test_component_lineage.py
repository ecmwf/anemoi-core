# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests of the lineage component and of the version gate, on the golden fixtures."""

from collections.abc import Callable
from typing import Any

import pytest

from anemoi.metadata import ComponentRegistry
from anemoi.metadata import Metadata
from anemoi.metadata import VersionError
from anemoi.metadata import requires
from anemoi.metadata.base import CheckpointAdapter
from anemoi.metadata.components.base import ComponentAdapter
from anemoi.metadata.components.lineage import LineageV0_1
from anemoi.metadata.components.lineage import LineageV0_2
from anemoi.metadata.components.lineage import LineageV1_0

PARENT = "0dc0e962-102b-47dd-9f35-2e3dbd285b9a"


# ---------------------------------------------------------------------------
# Version detection, by what the file records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "version", "adapter"),
    [
        ("f0_ai_models", "0.1", LineageV0_1),  # nothing recorded
        ("f1_null_diagnostic", "0.1", LineageV0_1),
        ("f1_ensemble", "0.2", LineageV0_2),  # uuid only
        ("f1_single_dataset", "0.2", LineageV0_2),  # uuid and parent_uuid, no lineage list
        ("f5_offsets", "0.2", LineageV0_2),
        ("f6_main", "1.0", LineageV1_0),  # lineage list, empty
        ("f6_warm_start", "1.0", LineageV1_0),
        ("f8_richer_batch", "1.0", LineageV1_0),
    ],
)
def test_version_follows_what_the_file_records(
    golden_loader: Callable[[str], dict[str, Any]],
    name: str,
    version: str,
    adapter: type,
) -> None:
    checkpoint = CheckpointAdapter(golden_loader(name))
    assert checkpoint.component_versions["lineage"] == version
    assert type(checkpoint.lineage) is adapter


def test_stamp_wins_over_content(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    raw = golden_loader("f6_warm_start")
    raw["component_versions"] = {"lineage": "0.2"}
    lineage = Metadata.from_dict(raw).lineage
    assert lineage.parent_uuid == PARENT
    with pytest.raises(VersionError):
        lineage.entries


# ---------------------------------------------------------------------------
# Values and the gate
# ---------------------------------------------------------------------------


def test_nothing_recorded_raises_instead_of_returning_none(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    lineage = Metadata.from_dict(golden_loader("f0_ai_models")).lineage
    for name in ("uuid", "parent_uuid", "entries"):
        with pytest.raises(VersionError, match=rf"lineage\.{name} requires lineage version >= .*version 0\.1"):
            getattr(lineage, name)


def test_uuid_only_file_has_no_entries(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    """F1 and F2 to F5 record a uuid (and a warm-start parent), not the lineage list."""
    ensemble = Metadata.from_dict(golden_loader("f1_ensemble")).lineage
    assert ensemble.uuid
    assert ensemble.parent_uuid is None
    single = Metadata.from_dict(golden_loader("f1_single_dataset")).lineage
    assert single.parent_uuid == "28ecd615-30ab-46c9-b785-23d6ed40f245"
    with pytest.raises(VersionError, match=r"lineage\.entries requires lineage version >= 1\.0, but .* version 0\.2"):
        single.entries


def test_fresh_run_means_no_parent_not_unknown(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    lineage = Metadata.from_dict(golden_loader("f6_main")).lineage
    assert lineage.uuid == "11f0410c-d9d2-4255-8232-8cc6602dd1ab"
    assert lineage.parent_uuid is None
    assert lineage.entries == []


def test_warm_start(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    lineage = Metadata.from_dict(golden_loader("f6_warm_start")).lineage
    assert lineage.parent_uuid == PARENT
    assert lineage.entries == [{"loading_strategy": "warm_start", "parent_uuid": PARENT}]


def test_missing_parent_uuid_falls_back_to_the_last_entry(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    """Reading is lenient: the entries say who the parent is when the scalar is missing."""
    raw = golden_loader("f6_warm_start")
    del raw["parent_uuid"]
    assert Metadata.from_dict(raw).lineage.parent_uuid == PARENT
    raw["checkpoint_lineage"].append({"loading_strategy": "cold_start", "parent_uuid": None})
    assert Metadata.from_dict(raw).lineage.parent_uuid is None


def test_uuid_and_lineage_are_not_in_the_sections(golden_loader: Callable[[str], dict[str, Any]]) -> None:
    """They moved to the lineage component; the raw keys stay reachable as stored."""
    metadata = Metadata.from_dict(golden_loader("f6_warm_start"))
    assert not hasattr(metadata.sections, "uuid")
    assert metadata.sections.get("uuid") == metadata.lineage.uuid
    assert metadata["checkpoint_lineage"] == metadata.lineage.entries


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.fixture
def probe(registry_state: None) -> tuple[type, type]:
    """A test-only component with a value that its second version adds and older ones cannot give."""
    del registry_state  # Requested for its teardown, which restores the registry.

    @ComponentRegistry.register("probe", "1.0")
    class OldProbe(ComponentAdapter):
        @property
        @requires("1.1")
        def added_in_1_1(self) -> str:
            return "new"

        @requires("1.0")
        def since_1_0(self, suffix: str) -> str:
            return f"old{suffix}"

    @ComponentRegistry.register("probe", "1.1")
    class NewProbe(OldProbe):
        pass

    return OldProbe, NewProbe


def test_gate_raises_on_an_older_component_version_naming_everything(probe: tuple[type, type]) -> None:
    old, _ = probe
    with pytest.raises(VersionError) as info:
        old({}).added_in_1_1
    message = str(info.value)
    assert "probe.added_in_1_1" in message
    assert "version >= 1.1" in message
    assert "probe version 1.0" in message


def test_gate_passes_at_the_required_version_and_above(probe: tuple[type, type]) -> None:
    old, new = probe
    assert new({}).added_in_1_1 == "new"
    assert old({}).since_1_0("!") == "old!"
    assert new({}).since_1_0("?") == "old?"


PROBE_VERSIONS = ("1.0", "1.1", "1.9", "1.10", "2.0", "2.1")


@pytest.fixture
def ranged_probe(registry_state: None) -> dict[str, type]:
    """A test-only component with gated methods, one adapter class per version."""
    del registry_state  # Requested for its teardown, which restores the registry.

    class Base(ComponentAdapter):
        @property
        @requires(max_version="2.0")
        def up_to_2_0(self) -> str:
            """Docstring of up_to_2_0."""
            return "max"

        @property
        @requires("1.1", "2.0")
        def from_1_1_to_2_0(self) -> str:
            return "range"

        @requires("1.1", max_version="2.0")
        def method(self, left: str, *, right: str = "") -> str:
            """Docstring of method."""
            return left + right

        @requires("1.9")
        def from_1_9(self) -> str:
            return "min"

        @requires("1.10")
        def from_1_10(self) -> str:
            return "min"

        @requires(max_version="1.9")
        def up_to_1_9(self) -> str:
            return "max"

    adapters: dict[str, type] = {}
    for version in PROBE_VERSIONS:
        adapters[version] = ComponentRegistry.register("probe", version)(type(f"Probe{version}", (Base,), {}))
    return adapters


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.9", "1.10", "2.0"])
def test_gate_max_version_passes_up_to_and_including_the_bound(ranged_probe: dict[str, type], version: str) -> None:
    assert ranged_probe[version]({}).up_to_2_0 == "max"


def test_gate_max_version_raises_above_the_bound(ranged_probe: dict[str, type]) -> None:
    with pytest.raises(VersionError) as info:
        ranged_probe["2.1"]({}).up_to_2_0
    message = str(info.value)
    assert "probe.up_to_2_0" in message
    assert "version <= 2.0" in message
    assert "probe version 2.1" in message
    assert ">=" not in message


@pytest.mark.parametrize("version", ["1.1", "1.9", "1.10", "2.0"])
def test_gate_min_and_max_pass_inside_the_range_inclusive(ranged_probe: dict[str, type], version: str) -> None:
    assert ranged_probe[version]({}).from_1_1_to_2_0 == "range"


@pytest.mark.parametrize("version", ["1.0", "2.1"])
def test_gate_min_and_max_raise_outside_the_range(ranged_probe: dict[str, type], version: str) -> None:
    with pytest.raises(VersionError) as info:
        ranged_probe[version]({}).from_1_1_to_2_0
    message = str(info.value)
    assert "probe.from_1_1_to_2_0" in message
    assert "version >= 1.1 <= 2.0" in message
    assert f"probe version {version}" in message


def test_gate_with_a_bound_on_each_side_names_both_in_the_message(ranged_probe: dict[str, type]) -> None:
    with pytest.raises(VersionError, match=r">= 1\.1.*<= 2\.0"):
        ranged_probe["2.1"]({}).from_1_1_to_2_0


def test_gate_with_a_max_keeps_the_method_name_docstring_and_arguments(ranged_probe: dict[str, type]) -> None:
    adapter = ranged_probe["2.0"]
    assert adapter.method.__name__ == "method"
    assert adapter.method.__doc__ == "Docstring of method."
    assert adapter.up_to_2_0.fget.__name__ == "up_to_2_0"
    assert adapter.up_to_2_0.__doc__ == "Docstring of up_to_2_0."
    assert adapter({}).method("a", right="b") == "ab"
    with pytest.raises(VersionError, match=r"probe\.method requires probe version >= 1\.1 <= 2\.0"):
        ranged_probe["2.1"]({}).method("a", right="b")


def test_gate_raises_when_the_adapter_has_no_version() -> None:
    class Unversioned(ComponentAdapter):
        @requires(max_version="2.0")
        def only_max(self) -> str:
            return "value"

        @requires("1.0")
        def only_min(self) -> str:
            return "value"

    assert getattr(Unversioned({}), "version", None) is None
    with pytest.raises(VersionError, match=r"but this checkpoint has .* version None"):
        Unversioned({}).only_max()
    with pytest.raises(VersionError, match=r"but this checkpoint has .* version None"):
        Unversioned({}).only_min()


def test_gate_compares_versions_as_pep_440_not_as_strings(ranged_probe: dict[str, type]) -> None:
    # As strings "1.10" < "1.9", so a string comparison would get both of these wrong.
    assert ranged_probe["1.10"]({}).from_1_9() == "min"
    with pytest.raises(VersionError):
        ranged_probe["1.9"]({}).from_1_10()
    with pytest.raises(VersionError):
        ranged_probe["1.10"]({}).up_to_1_9()
    assert ranged_probe["1.9"]({}).up_to_1_9() == "max"
