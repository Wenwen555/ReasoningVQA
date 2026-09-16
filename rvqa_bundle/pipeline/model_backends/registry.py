from . import qwen_vl
from .adapter import BackendAdapter


BACKENDS = {
    adapter.name: adapter
    for adapter in (BackendAdapter.from_module(qwen_vl),)
}


def get_backend(name):
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown backend {name!r}; available={sorted(BACKENDS)}") from exc
