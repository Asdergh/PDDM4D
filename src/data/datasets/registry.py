from dataclasses import is_dataclass
__DATASETS__ = dict()
def register_dataset(name: str, config):
    def decorator(cls):
        if name not in __DATASETS__:
            __DATASETS__[name] = dict(cls=cls, config=config)
        else:
            raise RuntimeError(f"{name} dataset was already registered")
        return cls 
    return decorator

def get_dataset(source: str | object):
    if isinstance(source, str):
        info = __DATASETS__[source]
        return info["cls"](info["config"]())
    elif hasattr(source, "name"):
        name = getattr(source, "name")
        assert name in __DATASETS__
        info = __DATASETS__[name]
        if isinstance(source, info["config"]):
            return info["cls"](source)
    else:
        raise ValueError(f"unknown type or dataset: {source}")
