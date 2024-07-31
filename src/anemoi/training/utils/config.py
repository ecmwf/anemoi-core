from anemoi.training.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class DotConfig(dict):
    """A Config dictionary that allows access to its keys as attributes."""

    def __init__(self, *args, **kwargs) -> None:
        for a in args:
            self.update(a)
        self.update(kwargs)

    def __getattr__(self, name):
        if name in self:
            x = self[name]
            if isinstance(x, dict):
                return DotConfig(x)
            return x
        raise AttributeError(name)
