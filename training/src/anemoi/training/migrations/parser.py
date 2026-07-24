from omegaconf import Node
from omegaconf.grammar_parser import parse
from omegaconf.grammar_visitor import GrammarVisitor


def get_interpolations(value: str) -> list[str]:
    interpolations: list[str] = []

    def node_interpolation_callback(inter_key: str, _: set[int] | None) -> Node | None:
        interpolations.append(inter_key)

    visitor = GrammarVisitor(node_interpolation_callback, None, None)
    parse_tree = parse(value)
    visitor.visit(parse_tree)
    return interpolations
