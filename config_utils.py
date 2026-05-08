import copy
import json
import os
from pathlib import Path


DEFAULT_CONFIG = {
    "seed": 42,
    "device": "auto",
    "structure_path": "structures.json",
    "label_path": "labels.csv",
    "structure_uid_key": "uid",
    "structure_key": "structure",
    "label_uid_key": "uid",
    "target_key": "total",
    "use_magpie_descriptors": True,
    "extra_atom_descriptor_path": None,
    "extra_atom_descriptor_key": "element",
    "extra_atom_descriptor_columns": None,
    "extra_atom_descriptor_missing_value": 0.0,
    "global_descriptor_path": None,
    "global_descriptor_uid_key": "uid",
    "global_descriptor_columns": None,
    "global_descriptor_missing_value": 0.0,
    "target_transform": {
        "enabled": True,
        "method": "signed_log1p_robust",
        "eps": 1e-6,
    },
    "save_dir": "checkpoints_eatgnn_01",
    "n": None,
    "batch_size": 16,
    "epochs": 100,
    "early_stopping_patience": 20,
    "train_ratio": 0.8,
    "valid_ratio": 0.1,
    "test_ratio": 0.1,
    "radial_cutoff": 5,
    "max_radius": 7,
    "heads": 2,
    "lmax": 3,
    "embedding_dim": 64,
    "irreps_query": "32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+4x3e+4x3o+2x4e+2x4o",
    "irreps_key": "32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+4x3e+4x3o+2x4e+2x4o",
    "irreps_out": "2x0e+2x1o+2x2e",
    "formula": "ij",
    "mul": 32,
    "layers": 2,
    "number_of_basis": 10,
    "pool_nodes": True,
    "lr": 0.005,
    "lr_scheduler_mode": "min",
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 8,
    "lr_scheduler_min_lr": 1e-6,
    "loss_type": "weighted_masked_huber_s_q",
    "loss_ws": 1.0,
    "loss_wq": None,
    "huber_delta": None,
    "huber_delta_diag": None,
}


def _deep_update(base, updates):
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(path_value, base_dir):
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(config_path="config.json"):
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    config = _deep_update(DEFAULT_CONFIG, user_config)
    base_dir = config_path.parent
    config["config_path"] = str(config_path)
    config["structure_path"] = _resolve_path(config["structure_path"], base_dir)
    config["label_path"] = _resolve_path(config["label_path"], base_dir)
    if config.get("extra_atom_descriptor_path"):
        config["extra_atom_descriptor_path"] = _resolve_path(config["extra_atom_descriptor_path"], base_dir)
    if config.get("global_descriptor_path"):
        config["global_descriptor_path"] = _resolve_path(config["global_descriptor_path"], base_dir)
    config["save_dir"] = _resolve_path(config["save_dir"], base_dir)
    config["best_model_path"] = os.path.join(config["save_dir"], "best_model.pt")
    config["last_model_path"] = os.path.join(config["save_dir"], "last_model.pt")
    config["log_csv_path"] = os.path.join(config["save_dir"], "training_log.csv")
    config["test_predictions_path"] = os.path.join(config["save_dir"], "test_predictions_best.json")
    config["test_metrics_path"] = os.path.join(config["save_dir"], "test_metrics_best.json")
    return config
