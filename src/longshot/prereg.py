"""Pre-registered analysis configuration (analysis.yaml).

The committed ``analysis.yaml`` declares the analysis questions and
parameters before any dataset is examined: horizons, binning, bootstrap
size, correction methods, and verdict rules. ``longshot analyze`` and
``longshot correct`` accept ``--config PATH`` and take defaults from it;
explicit CLI flags override the file (a deviation worth reporting).
"""

from __future__ import annotations

from pathlib import Path

import yaml

#: Top-level keys the config may set, mapped to their expected types.
_SCHEMA = {
    "horizons": list,
    "bins": int,
    "min_per_bin": int,
    "bootstrap": int,
    "seed": int,
    "correction": dict,
}

_CORRECTION_SCHEMA = {
    "methods": list,
    "train_frac": (int, float),
    "verdict_rules": dict,
}


class AnalysisConfigError(ValueError):
    """Raised when a pre-registered config is missing or malformed."""


def load_analysis_config(path: str | Path) -> dict:
    """Load and validate an analysis.yaml. Returns {} for no path."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise AnalysisConfigError(
            f"analysis config not found: {path}; the repo ships a "
            "pre-registered example at analysis.yaml"
        )
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise AnalysisConfigError(f"{path}: YAML parse error: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise AnalysisConfigError(f"{path}: expected a mapping at the top level")

    problems: list[str] = []
    for key, value in data.items():
        expected = _SCHEMA.get(key)
        if expected is None:
            problems.append(f"unknown key {key!r} (known: {sorted(_SCHEMA)})")
        elif not isinstance(value, expected) or isinstance(value, bool):
            problems.append(f"{key!r} must be {expected}, got {type(value).__name__}")
    corr = data.get("correction") or {}
    if isinstance(corr, dict):
        for key, value in corr.items():
            expected = _CORRECTION_SCHEMA.get(key)
            if expected is None:
                problems.append(
                    f"correction.{key!r} unknown (known: {sorted(_CORRECTION_SCHEMA)})"
                )
            elif not isinstance(value, expected) or isinstance(value, bool):
                problems.append(
                    f"correction.{key!r} must be {expected}, "
                    f"got {type(value).__name__}"
                )
    horizons = data.get("horizons")
    if isinstance(horizons, list) and not all(isinstance(h, str) for h in horizons):
        problems.append("'horizons' must be a list of strings like '30d'")
    if problems:
        raise AnalysisConfigError(f"{path}: " + "; ".join(problems))
    return data


def pick(cli_value, default, config: dict, key: str):
    """Config-aware default: CLI flag > analysis.yaml > built-in default."""
    if cli_value != default:
        return cli_value
    return config.get(key, default)
