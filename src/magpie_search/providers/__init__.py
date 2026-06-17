"""providers — registry + discovery for federated-search backends.

Built-in providers (ship in the core wheel) are registered here lazily so
importing this package doesn't drag in every backend's heavy deps. Third-party
/ operator-local providers are discovered via the ``magpie_search.providers``
entry-point group — install a package that advertises one and reference its type
name in `sources`, with zero core changes.
"""
from __future__ import annotations

from typing import Any

from .base import Hit, Provider, TrustTier, DEFAULT_TRUST_WEIGHTS, TIER_RANK

# type-name -> "module:ClassName" for built-ins. Lazy: only imported when used.
_BUILTINS: dict[str, str] = {
    "transcripts": "magpie_search.providers.transcripts:TranscriptsProvider",
    "files": "magpie_search.providers.files:FilesProvider",
    "vector": "magpie_search.providers.vector:VectorProvider",
    "kg": "magpie_search.providers.kg:KGProvider",
    "web": "magpie_search.providers.web:WebProvider",
    "youtube": "magpie_search.providers.youtube:YoutubeProvider",
}

_CLASS_CACHE: dict[str, type[Provider]] = {}


def _import_class(path: str) -> type[Provider]:
    mod_name, _, cls_name = path.partition(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _entry_point_classes() -> dict[str, str]:
    """Discover provider classes advertised via the entry-point group.

    Returns {type_name: "module:Class"}; tolerant of older importlib.metadata.
    """
    out: dict[str, str] = {}
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover
        return out
    try:
        eps = entry_points()
        # Python 3.10+: entry_points() returns a SelectableGroups / dict-like.
        if hasattr(eps, "select"):
            group = eps.select(group="magpie_search.providers")
        else:  # pragma: no cover - very old API
            group = eps.get("magpie_search.providers", [])
        for ep in group:
            out[ep.name] = ep.value
    except Exception:  # pragma: no cover - never let discovery break search
        return out
    return out


def provider_class(type_name: str) -> type[Provider]:
    """Resolve a provider TYPE name to its class. Built-ins win; then plugins."""
    if type_name in _CLASS_CACHE:
        return _CLASS_CACHE[type_name]
    path = _BUILTINS.get(type_name) or _entry_point_classes().get(type_name)
    if not path:
        raise KeyError(f"unknown provider type: {type_name!r} "
                       f"(known: {sorted(available_types())})")
    cls = _import_class(path)
    _CLASS_CACHE[type_name] = cls
    return cls


def available_types() -> list[str]:
    return sorted(set(_BUILTINS) | set(_entry_point_classes()))


def make_provider(spec: "Provider | dict[str, Any] | str") -> Provider:
    """Build a Provider from a spec.

    A spec is one of:
      - a Provider instance (returned as-is)
      - a str type name, e.g. "transcripts" (defaults for name/trust/config)
      - a dict: {"type": <name>, "name"?: ..., "trust"?: ..., **config}
    """
    if isinstance(spec, Provider):
        return spec
    if isinstance(spec, str):
        return provider_class(spec)(name=spec)
    if isinstance(spec, dict):
        d = dict(spec)
        type_name = d.pop("type", None) or d.pop("category", None)
        if not type_name:
            raise ValueError(f"source spec missing 'type': {spec!r}")
        name = d.pop("name", None) or type_name
        trust = d.pop("trust", None)
        return provider_class(type_name)(name=name, trust=trust, **d)
    raise TypeError(f"unsupported source spec: {spec!r}")


__all__ = [
    "Hit", "Provider", "TrustTier", "DEFAULT_TRUST_WEIGHTS", "TIER_RANK",
    "provider_class", "available_types", "make_provider",
]
