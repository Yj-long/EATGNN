import math
import json
from pathlib import Path

import ase.neighborlist
import numpy as np
import pandas as pd
import torch
import torch_geometric
from ase.atoms import Atom
from ase.data import atomic_numbers, chemical_symbols
from jarvis.core.specie import Specie
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import DataLoader

from .target_transform import apply_target_transform, build_target_transform

STRUCTURE_FILE_SUFFIXES = {".cif", ".vasp", ".poscar", ".contcar", ".cssr", ".json"}
STRUCTURE_FILE_NAMES = {"POSCAR", "CONTCAR"}


def split(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"train/valid/test ratios must sum to 1.0, got {total}")

    train_data, holdout_data = train_test_split(
        dataset, test_size=(1.0 - train_ratio), random_state=seed, shuffle=True
    )
    if len(holdout_data) == 0:
        return train_data, [], []

    test_share_in_holdout = test_ratio / (valid_ratio + test_ratio)
    valid_data, test_data = train_test_split(
        holdout_data, test_size=test_share_in_holdout, random_state=seed, shuffle=True
    )
    return train_data, valid_data, test_data


def _build_magpie_features():
    try:
        encoder = OneHotEncoder(max_categories=6, sparse=False)
    except TypeError:
        encoder = OneHotEncoder(max_categories=6, sparse_output=False)

    features = [Specie(Atom(i).symbol, source="magpie").get_descrp_arr for i in range(1, 102)]
    return np.asarray(encoder.fit_transform(features), dtype=np.float32)


def _numeric_descriptor_columns(frame, key_column):
    columns = []
    for column in frame.columns:
        if column == key_column:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            columns.append(column)
    return columns


def _element_key_to_z(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            z = int(stripped)
        else:
            z = atomic_numbers.get(stripped)
    else:
        z = int(value)
    if z is None or z < 1 or z > 101:
        raise ValueError(f"Unsupported element descriptor key: {value}")
    return z


def _extra_descriptor_records_from_json(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("descriptors", "data", "records", "elements"):
            if isinstance(data.get(key), list):
                return data[key]
        return [{"element": key, "descriptor": value} for key, value in data.items()]
    raise ValueError("Extra atom descriptor JSON must be a list or a mapping.")


def _load_extra_atom_descriptors(config):
    path_value = config.get("extra_atom_descriptor_path")
    if not path_value:
        return None

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Extra atom descriptor path not found: {path}")

    key_column = config.get("extra_atom_descriptor_key", "element")
    descriptor_columns = config.get("extra_atom_descriptor_columns")
    missing_value = float(config.get("extra_atom_descriptor_missing_value", 0.0))

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if key_column not in frame.columns:
            for fallback in ("element", "symbol", "Z", "atomic_number"):
                if fallback in frame.columns:
                    key_column = fallback
                    break
            else:
                raise ValueError(f"Extra descriptor CSV is missing key column '{key_column}'.")
        if descriptor_columns is None:
            descriptor_columns = _numeric_descriptor_columns(frame, key_column)
        if not descriptor_columns:
            raise ValueError("No numeric descriptor columns found in extra descriptor CSV.")

        table = np.full((101, len(descriptor_columns)), missing_value, dtype=np.float32)
        for _, row in frame.iterrows():
            z = _element_key_to_z(row[key_column])
            values = [pd.to_numeric(row[col], errors="coerce") for col in descriptor_columns]
            table[z - 1] = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=missing_value)
        return table

    if path.suffix.lower() == ".json":
        records = _extra_descriptor_records_from_json(_read_json(path))
        descriptor_values = {}
        descriptor_keys = None
        for record in records:
            if key_column not in record:
                for fallback in ("element", "symbol", "Z", "atomic_number"):
                    if fallback in record:
                        key_column = fallback
                        break
                else:
                    raise ValueError(f"Extra descriptor JSON record is missing key '{key_column}'.")
            z = _element_key_to_z(record[key_column])
            if descriptor_columns is not None:
                values = [record[col] for col in descriptor_columns]
            elif "descriptor" in record:
                values = record["descriptor"]
            else:
                descriptor_keys = [
                    key for key, value in record.items()
                    if key != key_column and isinstance(value, (int, float))
                ]
                values = [record[key] for key in descriptor_keys]
            descriptor_values[z] = values

        width = len(next(iter(descriptor_values.values())))
        table = np.full((101, width), missing_value, dtype=np.float32)
        for z, values in descriptor_values.items():
            table[z - 1] = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=missing_value)
        return table

    raise ValueError(f"Unsupported extra atom descriptor format: {path.suffix}. Expected .csv or .json.")


def build_atom_features(config):
    feature_blocks = []
    if config.get("use_magpie_descriptors", True):
        feature_blocks.append(_build_magpie_features())

    extra_features = _load_extra_atom_descriptors(config)
    if extra_features is not None:
        feature_blocks.append(extra_features)

    if not feature_blocks:
        raise ValueError(
            "No atom features are enabled. Set use_magpie_descriptors=true or provide extra_atom_descriptor_path."
        )

    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def r_cut2D(x, cell):
    structure = AseAtomsAdaptor.get_structure(cell)
    cell = structure.lattice.matrix
    r_cut = max(np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), x)
    return r_cut


def datatransform(crystal, property, features, radial_cutoff, default_dtype, global_descriptor=None):
    r_cut = r_cut2D(radial_cutoff, crystal)
    edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list(
        "ijS", a=crystal, cutoff=r_cut, self_interaction=False
    )

    target_tensor = torch.as_tensor(property, dtype=default_dtype).float()
    if target_tensor.ndim == 1 and target_tensor.numel() == 3:
        target_tensor = torch.diag(target_tensor)
        target_mask = torch.eye(3, dtype=default_dtype)
    elif target_tensor.ndim == 2 and target_tensor.shape == (3, 3):
        target_mask = torch.isfinite(target_tensor).to(default_dtype)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)
    elif target_tensor.ndim == 0:
        target_tensor = target_tensor.unsqueeze(0)
        target_mask = torch.ones_like(target_tensor)
    else:
        raise ValueError(
            f"Unsupported target shape {tuple(target_tensor.shape)}. "
            "Expected [3] (diagonal-only) or [3, 3] (full tensor)."
        )

    target_tensor = target_tensor.unsqueeze(0)
    target_mask = target_mask.unsqueeze(0)

    data_kwargs = {}
    if global_descriptor is not None:
        data_kwargs["global_attr"] = torch.as_tensor(
            global_descriptor, dtype=default_dtype
        ).unsqueeze(0).float()

    data = torch_geometric.data.Data(
        pos=torch.as_tensor(crystal.get_positions(), dtype=default_dtype).float(),
        lattice=torch.as_tensor(crystal.cell.array, dtype=default_dtype).unsqueeze(0).float(),
        x=torch.as_tensor(
            [features[atomic_numbers[atom] - 1] for atom in crystal.symbols],
            dtype=default_dtype,
        ).float(),
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.as_tensor(edge_shift, dtype=default_dtype).float(),
        energy=target_tensor,
        energy_mask=target_mask,
        **data_kwargs,
    )
    return data


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _structure_from_json_value(value):
    if isinstance(value, Structure):
        return value
    if isinstance(value, dict):
        return Structure.from_dict(value)
    raise ValueError("JSON structure entries must be pymatgen Structure dictionaries.")


def _records_from_json(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("structures", "data", "records", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [{"uid": uid, "structure": structure} for uid, structure in data.items()]
    raise ValueError("Structure JSON must be a list, a record container, or a uid-to-structure mapping.")


def load_structures(config):
    structure_path = Path(config["structure_path"])
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure path not found: {structure_path}")

    if structure_path.is_dir():
        structures = {}
        for path in sorted(structure_path.rglob("*")):
            if not path.is_file():
                continue
            if path.name.upper() in STRUCTURE_FILE_NAMES or path.suffix.lower() in STRUCTURE_FILE_SUFFIXES:
                if path.suffix.lower() == ".json":
                    structures.update(_load_json_structures(path, config))
                else:
                    structures[path.stem] = AseAtomsAdaptor.get_atoms(Structure.from_file(str(path)))
        return structures

    if structure_path.suffix.lower() == ".json":
        return _load_json_structures(structure_path, config)

    uid = config.get("single_structure_uid") or structure_path.stem
    return {uid: AseAtomsAdaptor.get_atoms(Structure.from_file(str(structure_path)))}


def _load_json_structures(path, config):
    uid_key = config.get("structure_uid_key", "uid")
    structure_key = config.get("structure_key", "structure")
    records = _records_from_json(_read_json(path))
    structures = {}
    for record in records:
        if uid_key not in record:
            raise ValueError(f"Structure record in {path} is missing uid key '{uid_key}'.")
        if structure_key not in record:
            raise ValueError(f"Structure record {record[uid_key]} is missing structure key '{structure_key}'.")
        structures[str(record[uid_key])] = AseAtomsAdaptor.get_atoms(
            _structure_from_json_value(record[structure_key])
        )
    return structures


def _parse_target_value(value):
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            raise ValueError("Empty target value.")
        return json.loads(value)
    return value


def _load_json_labels(path, config):
    uid_key = config.get("label_uid_key", "uid")
    target_key = config["target_key"]
    data = _read_json(path)

    if isinstance(data, dict):
        for key in ("labels", "data", "records", "samples"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            return {str(uid): _parse_target_value(target) for uid, target in data.items()}

    labels = {}
    for record in data:
        if uid_key not in record:
            raise ValueError(f"Label record in {path} is missing uid key '{uid_key}'.")
        if target_key not in record:
            raise ValueError(f"Label record {record[uid_key]} is missing target key '{target_key}'.")
        labels[str(record[uid_key])] = _parse_target_value(record[target_key])
    return labels


def _load_csv_labels(path, config):
    uid_key = config.get("label_uid_key", "uid")
    target_key = config["target_key"]
    frame = pd.read_csv(path)
    if uid_key not in frame.columns:
        raise ValueError(f"Label CSV is missing uid column '{uid_key}'.")

    if target_key in frame.columns:
        return {
            str(row[uid_key]): _parse_target_value(row[target_key])
            for _, row in frame.iterrows()
        }

    component_cols = config.get("target_component_columns")
    if component_cols:
        missing = [col for col in component_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Label CSV is missing target component columns: {missing}")
        return {
            str(row[uid_key]): [row[col] for col in component_cols]
            for _, row in frame.iterrows()
        }

    matrix_cols = config.get(
        "target_matrix_columns",
        [["xx", "xy", "xz"], ["yx", "yy", "yz"], ["zx", "zy", "zz"]],
    )
    flat_cols = [col for row in matrix_cols for col in row]
    if all(col in frame.columns for col in flat_cols):
        return {
            str(row[uid_key]): [[row[col] for col in col_row] for col_row in matrix_cols]
            for _, row in frame.iterrows()
        }

    raise ValueError(
        f"Label CSV must contain '{target_key}', target_component_columns, or matrix columns {flat_cols}."
    )


def load_labels(config):
    label_path = Path(config["label_path"])
    if not label_path.exists():
        raise FileNotFoundError(f"Label path not found: {label_path}")

    suffix = label_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_labels(label_path, config)
    if suffix == ".json":
        return _load_json_labels(label_path, config)
    raise ValueError(f"Unsupported label file format: {label_path.suffix}. Expected .csv or .json.")


def _global_descriptor_columns(frame, uid_column):
    columns = []
    for column in frame.columns:
        if column == uid_column:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            columns.append(column)
    return columns


def _global_descriptor_records_from_json(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("descriptors", "data", "records", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [{"uid": uid, "descriptor": descriptor} for uid, descriptor in data.items()]
    raise ValueError("Global descriptor JSON must be a list or a mapping.")


def load_global_descriptors(config):
    path_value = config.get("global_descriptor_path")
    if not path_value:
        return {}, 0

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Global descriptor path not found: {path}")

    uid_key = config.get("global_descriptor_uid_key", "uid")
    descriptor_columns = config.get("global_descriptor_columns")
    missing_value = float(config.get("global_descriptor_missing_value", 0.0))

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if uid_key not in frame.columns:
            raise ValueError(f"Global descriptor CSV is missing uid column '{uid_key}'.")
        if descriptor_columns is None:
            descriptor_columns = _global_descriptor_columns(frame, uid_key)
        if not descriptor_columns:
            raise ValueError("No numeric descriptor columns found in global descriptor CSV.")

        descriptors = {}
        for _, row in frame.iterrows():
            values = [pd.to_numeric(row[col], errors="coerce") for col in descriptor_columns]
            descriptors[str(row[uid_key])] = np.nan_to_num(
                np.asarray(values, dtype=np.float32), nan=missing_value
            )
        return descriptors, len(descriptor_columns)

    if path.suffix.lower() == ".json":
        records = _global_descriptor_records_from_json(_read_json(path))
        descriptors = {}
        width = None
        for record in records:
            if uid_key not in record:
                raise ValueError(f"Global descriptor JSON record is missing uid key '{uid_key}'.")
            if descriptor_columns is not None:
                values = [record[col] for col in descriptor_columns]
            elif "descriptor" in record:
                values = record["descriptor"]
            else:
                values = [
                    value for key, value in record.items()
                    if key != uid_key and isinstance(value, (int, float))
                ]
            values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=missing_value)
            width = len(values) if width is None else width
            if len(values) != width:
                raise ValueError("Global descriptor JSON records must have the same descriptor length.")
            descriptors[str(record[uid_key])] = values
        return descriptors, (width or 0)

    raise ValueError(f"Unsupported global descriptor format: {path.suffix}. Expected .csv or .json.")


def align_structures_and_labels(structures, labels):
    structure_uids = set(structures)
    label_uids = set(labels)
    common_uids = sorted(structure_uids & label_uids)
    if not common_uids:
        raise ValueError("No matching uid values found between structures and labels.")

    missing_labels = sorted(structure_uids - label_uids)
    missing_structures = sorted(label_uids - structure_uids)
    if missing_labels:
        print(f"Warning: {len(missing_labels)} structures have no label and will be skipped.")
    if missing_structures:
        print(f"Warning: {len(missing_structures)} labels have no structure and will be skipped.")
    return [(uid, structures[uid], labels[uid]) for uid in common_uids]


def build_dataloaders(config, default_dtype):
    sample_limit = config.get("n")
    structures_by_uid = load_structures(config)
    labels_by_uid = load_labels(config)
    global_descriptors_by_uid, global_descriptor_dim = load_global_descriptors(config)
    aligned_records = align_structures_and_labels(structures_by_uid, labels_by_uid)
    if sample_limit is not None:
        aligned_records = aligned_records[:sample_limit]

    structures = [record[1] for record in aligned_records]
    targets_source = [record[2] for record in aligned_records]
    num_nodes = sum(len(item) for item in structures) / len(structures)

    features = build_atom_features(config)
    feature_dim = len(features[0])
    print(feature_dim)

    missing_global = 0
    dataset = []
    for uid, crystal, target in aligned_records:
        global_descriptor = None
        if global_descriptor_dim > 0:
            global_descriptor = global_descriptors_by_uid.get(uid)
            if global_descriptor is None:
                missing_global += 1
                global_descriptor = np.full(
                    global_descriptor_dim,
                    float(config.get("global_descriptor_missing_value", 0.0)),
                    dtype=np.float32,
                )
        dataset.append(
            datatransform(
                crystal,
                target,
                features=features,
                radial_cutoff=config["radial_cutoff"],
                default_dtype=default_dtype,
                global_descriptor=global_descriptor,
            )
        )
    if missing_global:
        print(f"Warning: {missing_global} samples have no global descriptor and were filled.")

    train_dataset, valid_dataset, test_dataset = split(
        dataset,
        train_ratio=config["train_ratio"],
        valid_ratio=config["valid_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
    )

    target_transform = build_target_transform(config, train_dataset)
    if target_transform is not None:
        apply_target_transform(train_dataset, target_transform)
        apply_target_transform(valid_dataset, target_transform)
        apply_target_transform(test_dataset, target_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"])
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])
    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        feature_dim,
        num_nodes,
        target_transform,
        global_descriptor_dim,
    )
