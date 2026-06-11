from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
SIMULATIONS_DIR = OUTPUT_DIR / "simulations"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
BRACKETS_DIR = OUTPUT_DIR / "brackets"
MODE_TO_CONFIG = {
    "default": CONFIG_DIR / "model_params_default.json",
    "v2": CONFIG_DIR / "model_params_v2.json",
    "experimental": CONFIG_DIR / "model_params_experimental.json",
    "test": CONFIG_DIR / "model_params_tuned_validation.json",
}


def config_path_for_mode(mode: str = "default") -> Path:
    if mode not in MODE_TO_CONFIG:
        raise ValueError("mode must be one of: default, v2, experimental, test")
    return MODE_TO_CONFIG[mode]


def load_model_config(mode: str = "default", config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else config_path_for_mode(mode)
    config = json.loads(path.read_text())
    config["parameter_config_path"] = str(path)
    config["runtime_mode"] = mode
    config.setdefault("model_status", mode)
    config.setdefault("random_seed_used_for", ["scoreline_sampling", "tiebreakers"])
    config.setdefault("uses_random_pairing", False)
    config.setdefault("official_bracket", False)
    config.setdefault("bracket_source", "matches.csv")
    return config


def model_suffix(config: dict) -> str:
    status = config.get("model_status", "default")
    return "default" if status == "default" else status


def poisson_parameter_kwargs(config: dict) -> dict:
    keys = [
        "base_total_goals",
        "share_scale",
        "mismatch_total_bonus",
        "mismatch_scale",
        "lambda_min",
        "lambda_max",
    ]
    return {key: config[key] for key in keys if key in config}


def draw_calibration_kwargs(config: dict) -> dict:
    keys = ["draw_boost_max", "draw_boost_scale"]
    return {key: config[key] for key in keys if key in config}


def output_path(base_name: str, config: dict, extension: str = ".csv") -> Path:
    suffix = model_suffix(config)
    if "prediction" in base_name:
        directory = PREDICTIONS_DIR
    else:
        directory = DIAGNOSTICS_DIR
    return directory / f"{base_name}_{suffix}{extension}"


def simulation_output_path(base_name: str, config: dict, extension: str = ".csv") -> Path:
    suffix = model_suffix(config)
    if "bracket" in base_name:
        directory = BRACKETS_DIR
    elif "diagnostics" in base_name or "team_strength" in base_name:
        directory = DIAGNOSTICS_DIR
    else:
        directory = SIMULATIONS_DIR
    return directory / f"{base_name}_{suffix}{extension}"


def metadata_columns(config: dict) -> dict:
    return {
        "model_version": config.get("model_version"),
        "model_status": config.get("model_status"),
        "parameter_config_path": config.get("parameter_config_path"),
        "rating_col": config.get("rating_col"),
        "rating_source_path": config.get("rating_source_path"),
        "base_model_version": config.get("base_model_version"),
        "use_v2_probability_stack": config.get("use_v2_probability_stack", False),
        "player_impact_layers": str(config.get("player_impact_layers", [])),
        "squad_values_file": config.get("squad_values_file"),
        "superstar_features_file": config.get("superstar_features_file"),
        "club_form_features_file": config.get("club_form_features_file"),
        "bracket_source": config.get("bracket_source"),
        "uses_random_pairing": config.get("uses_random_pairing", False),
        "random_seed_used_for": str(config.get("random_seed_used_for", [])),
    }
