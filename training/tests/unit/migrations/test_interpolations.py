from omegaconf import OmegaConf

from anemoi.training.migrations.interpolations import get_interpolation_tree
from anemoi.training.migrations.interpolations import get_interpolations
from anemoi.training.migrations.interpolations import replace_interpolation


def test_get_interpolations():
    assert len(get_interpolations("no interpolations")) == 0
    assert get_interpolations("there is ${a} interpolation(s)") == ["a"]
    assert get_interpolations("${a.b} is ${a} interpolation(s)") == ["a.b", "a"]
    assert get_interpolations("${oc.env:TEST} is ${a} interpolation(s)") == ["a"]


def test_interpolation_tree():
    tree = get_interpolation_tree(OmegaConf.create({"a": 0, "b": "${a}"}))
    assert tree == {"a": {"b"}}

    tree = get_interpolation_tree(OmegaConf.create({"a": 0, "b": "${a} and ${a}", "c": "${a}"}))
    assert tree == {"a": {"b", "c"}}

    tree = get_interpolation_tree(OmegaConf.create({"a": [0], "b": "${a[0]} and ${a}"}))
    assert tree == {"a": {"b"}, "a[0]": {"b"}}

    tree = get_interpolation_tree(OmegaConf.create({"a": 0, "b": "${a}", "c": "${b}+${a}"}))
    assert tree == {"a": {"b", "c"}, "b": {"c"}}

    tree = get_interpolation_tree(OmegaConf.create({"a": 0, "b": "${a }"}))
    assert tree == {"a": {"b"}}


def test_replace_interpolation():
    assert replace_interpolation("${a} and ${b }", "b", "c.d") == "${a} and ${c.d}"
