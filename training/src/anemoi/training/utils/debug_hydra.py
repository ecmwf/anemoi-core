from hydra.utils import get_class

def instantiate_debug(config, *args, **kwargs):

    _cls = get_class(config._target_)

    return _cls(
        *args,
        **{k:v for k,v in kwargs.items() if k not in ["_recursive_", "_convert_"]},
        **{k:v for k,v in config.items() if k not in ["_target_", "_recursive_", "_convert_"]}
    )
