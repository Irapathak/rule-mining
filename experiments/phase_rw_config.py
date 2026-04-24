#!/usr/bin/env python3
"""
Shared RW-related config for phase 2/3 experiment scripts (random walk).

Attribute weights resolution matches ``algorithms/all_subgroups_loop.py``:
``DATASETS[<key>].get("ATTRIBUTE_WEIGHTS", {})`` with ``<key>`` = ``CHOSEN_DATASET``
unless overridden, and the same ``ValueError`` if the dataset key is missing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_config_dict(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def treatment_col_from_config(cfg: Dict[str, Any]) -> str:
    return str(cfg["TREATMENT_COL"])


def attribute_weights_for_chosen_dataset(cfg: Dict[str, Any], dataset_key: Optional[str]) -> Dict[str, float]:
    """
    Same as ``ATTRIBUTE_WEIGHTS`` in ``all_subgroups_loop`` after loading ``ds_config``:
    ``ds_config.get("ATTRIBUTE_WEIGHTS", {})`` with ``ds_config = config["DATASETS"][key]``.
    """
    key: str
    if dataset_key is not None:
        key = str(dataset_key)
    else:
        if "CHOSEN_DATASET" not in cfg:
            raise KeyError("config.json must define CHOSEN_DATASET (see algorithms/all_subgroups_loop.py)")
        key = str(cfg["CHOSEN_DATASET"])
    datasets = cfg.get("DATASETS") or {}
    if key not in datasets:
        raise ValueError(f"Dataset '{key}' not found in config.json DATASETS")
    raw = datasets[key].get("ATTRIBUTE_WEIGHTS", {})
    if not raw:
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def load_attribute_weights_json(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"Attribute weights JSON must be an object mapping column names to numbers: {path}")
    return {str(k): float(v) for k, v in raw.items()}


def resolve_attribute_weights(
    config_path: Path,
    *,
    dataset_key: Optional[str] = None,
    uniform: bool = False,
    json_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Default: same dict ``all_subgroups_loop`` passes to RW; optional uniform or JSON override."""
    if json_path is not None:
        return load_attribute_weights_json(json_path)
    if uniform:
        return {}
    cfg = load_config_dict(config_path)
    return attribute_weights_for_chosen_dataset(cfg, dataset_key)
