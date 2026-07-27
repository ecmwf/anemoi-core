import re
from collections import defaultdict

from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import Node
from omegaconf import OmegaConf
from omegaconf.grammar_parser import parse
from omegaconf.grammar_visitor import GrammarVisitor

INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]*)\}", flags=re.ASCII)


def get_interpolations(value: str) -> list[str]:
    interpolations: list[str] = []

    def node_interpolation_callback(inter_key: str, _: set[int] | None) -> Node | None:
        interpolations.append(inter_key)

    def resolver_interpolation_callback(*args, **kwargs) -> None:
        pass

    visitor = GrammarVisitor(node_interpolation_callback, resolver_interpolation_callback, None)
    parse_tree = parse(value)
    visitor.visit(parse_tree)
    return interpolations


def get_interpolation_tree(
    cfg: DictConfig | ListConfig, key_prefix: list[str] | None = None, interpolations=None
) -> dict[str, set[str]]:
    if key_prefix is None:
        key_prefix = []
    if interpolations is None:
        interpolations: defaultdict[str, set[str]] = defaultdict(set)
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    if raw_cfg is None:
        return interpolations
    if isinstance(cfg, DictConfig):
        iterator = cfg.items()
    else:
        iterator = enumerate(cfg)
    for k, val in iterator:
        if isinstance(val, (DictConfig, ListConfig)):
            get_interpolation_tree(val, [*key_prefix, str(k)], interpolations)
        elif OmegaConf.is_interpolation(cfg, k):
            for interpo in get_interpolations(raw_cfg[k]):
                interpolations[interpo].add(".".join([*key_prefix, str(k)]))
    return interpolations


def replace_interpolation(value: str, interpo: str, replace: str) -> str:
    for match in INTERPOLATION_PATTERN.finditer(value):
        if match.group(1).strip() == interpo:
            start, end = match.span()
            value = value[:start] + f"${{{replace}}}" + value[end:]
    return value
