"""Load params.yaml and build swappable components from it."""
from importlib import import_module
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build(spec: dict):
    """Create an object from {"class": "package.module:ClassName", "args": {...}}.

    This is the only place models are chosen, so swapping a model means
    changing one line in params.yaml.
    """
    if not spec or not spec.get("class"):
        raise ValueError(f"No class set in spec: {spec!r}")
    module_name, class_name = spec["class"].split(":")
    cls = getattr(import_module(module_name), class_name)
    return cls(**(spec.get("args") or {}))
