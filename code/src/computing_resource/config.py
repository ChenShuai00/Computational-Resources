from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path: str | Path | None = None) -> dict:
    path = resolve_config_path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paths = dict(data.get("paths") or {})
    data["paths"] = {key: _resolve_path(value) for key, value in paths.items()}
    data["_meta"] = {
        "config_path": path,
        "project_root": PROJECT_ROOT,
    }
    return data
