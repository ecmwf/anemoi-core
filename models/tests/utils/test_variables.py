# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.utils.variables import parse_feature_name


class TestParseFeatureName:
    def test_pressure_level_variable(self):
        assert parse_feature_name("q_850") == ("q", 850, True)

    def test_no_level_variable(self):
        assert parse_feature_name("lsm") == ("lsm", 0, False)

    def test_near_surface_diagnostic_has_no_level(self):
        """'10u'/'2t' are metres-above-ground diagnostics, not a pressure-level position -
        must not collide with an unrelated pressure level sharing the same number (e.g. a
        hypothetical 't_2' would otherwise be grouped with '2t' as if both were 'level 2')."""
        assert parse_feature_name("10u") == ("u", 0, False)
        assert parse_feature_name("2t") == ("t", 0, False)

    def test_variable_only_unaffected_by_has_level(self):
        """variable_only=True groups by physical variable regardless of level type - used by
        JointVariableNormalizer, which intentionally pools statistics across '2t' and 't_850'
        as the same physical variable 't'."""
        assert parse_feature_name("10u", variable_only=True) == "u"
        assert parse_feature_name("2t", variable_only=True) == "t"
        assert parse_feature_name("q_850", variable_only=True) == "q"
        assert parse_feature_name("lsm", variable_only=True) == "lsm"
