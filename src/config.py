from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return {_normalize_key(key): value for key, value in config.items()}


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to a JSON config file.")
    config_args, remaining = config_parser.parse_known_args()
    defaults = load_config_defaults(config_args.config)
    parser.set_defaults(**defaults)
    parser.add_argument("--config", default=config_args.config, help="Path to a JSON config file.")
    return parser.parse_args(remaining)


def save_resolved_config(path: str | Path, args: argparse.Namespace) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = vars(args).copy()
    data["device"] = str(data["device"])
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")
