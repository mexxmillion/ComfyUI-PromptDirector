__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    if __package__:
        from . import nodes
    else:
        import nodes
    return getattr(nodes, name)
