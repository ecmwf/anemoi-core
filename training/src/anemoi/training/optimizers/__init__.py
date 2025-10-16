from .AdEMAMix import AdEMAMix

# optionally, you can keep a registry of available custom optimizers
CUSTOM_OPTIMIZERS = {
    "AdEMAMix": AdEMAMix,
}

def get_custom_optimizer_class(name: str):
    """Return a custom optimizer class by name, or None if not found."""
    return CUSTOM_OPTIMIZERS.get(name, None)