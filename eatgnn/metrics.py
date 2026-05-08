import math

import torch

from .target_transform import COMPONENTS_6, mask_to_symmetric_component_mask, tensor_to_symmetric_components


COMPONENT_NAMES = ("xx", "yy", "zz", "xy", "yz", "xz")


def _to_json_number(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _masked_values(values, mask):
    return values[mask > 0.5]


def _mse(diff):
    return (diff ** 2).mean()


def _rmse(diff):
    return torch.sqrt(_mse(diff))


def _mae(diff):
    return diff.abs().mean()


def _medae(diff):
    return diff.abs().median()


def _r2(pred, target):
    if pred.numel() == 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    ss_res = ((pred - target) ** 2).sum()
    centered = target - target.mean()
    ss_tot = (centered ** 2).sum()
    if ss_tot <= 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    return 1.0 - ss_res / ss_tot


def compute_original_tensor_metrics(pred, target, mask):
    pred = 0.5 * (pred.detach().cpu().double() + pred.detach().cpu().double().transpose(-1, -2))
    target = 0.5 * (target.detach().cpu().double() + target.detach().cpu().double().transpose(-1, -2))
    mask = mask.detach().cpu()

    pred_components = tensor_to_symmetric_components(pred)
    target_components = tensor_to_symmetric_components(target)
    component_mask = mask_to_symmetric_component_mask(mask)

    component_diff = _masked_values(pred_components - target_components, component_mask)
    metrics = {
        "component_mse": _to_json_number(_mse(component_diff)),
        "component_rmse": _to_json_number(_rmse(component_diff)),
        "component_mae": _to_json_number(_mae(component_diff)),
        "component_medae": _to_json_number(_medae(component_diff)),
    }

    r2_values = []
    for idx, name in enumerate(COMPONENT_NAMES):
        valid = component_mask[..., idx] > 0.5
        pred_i = pred_components[..., idx][valid]
        target_i = target_components[..., idx][valid]
        diff_i = pred_i - target_i
        metrics[f"{name}_mae"] = _to_json_number(_mae(diff_i))
        metrics[f"{name}_rmse"] = _to_json_number(_rmse(diff_i))
        metrics[f"{name}_r2"] = _to_json_number(_r2(pred_i, target_i))
        metrics[f"{name}_medae"] = _to_json_number(_medae(diff_i))
        if metrics[f"{name}_r2"] is not None:
            r2_values.append(metrics[f"{name}_r2"])

    metrics["r2_uniform_average"] = (
        _to_json_number(sum(r2_values) / len(r2_values)) if r2_values else None
    )

    tensor_valid = (component_mask > 0.5).all(dim=-1)
    pred_valid = pred[tensor_valid]
    target_valid = target[tensor_valid]
    tensor_diff = pred_valid - target_valid
    frob_errors = torch.linalg.matrix_norm(tensor_diff, ord="fro", dim=(-2, -1))
    target_frob = torch.linalg.matrix_norm(target_valid, ord="fro", dim=(-2, -1))

    metrics["frobenius_mse"] = _to_json_number((frob_errors ** 2).mean())
    metrics["frobenius_rmse"] = _to_json_number(torch.sqrt((frob_errors ** 2).mean()))
    metrics["frobenius_absolute_error"] = _to_json_number(frob_errors.mean())
    relative = frob_errors / target_frob.clamp_min(1e-12)
    metrics["relative_frobenius_error"] = _to_json_number(relative.mean())

    eigvals = torch.linalg.eigvalsh(pred_valid)
    min_eigs = eigvals[..., 0]
    metrics["positive_definite_rate"] = _to_json_number((min_eigs > 0.0).double().mean())
    metrics["min_predicted_eigenvalue"] = _to_json_number(min_eigs.min())
    metrics["max_symmetry_error"] = _to_json_number((pred_valid - pred_valid.transpose(-1, -2)).abs().max())

    diag_valid = component_mask[..., 0] & component_mask[..., 1] & component_mask[..., 2]
    pred_trace = pred.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[diag_valid]
    target_trace = target.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[diag_valid]
    trace_diff = pred_trace - target_trace
    metrics["trace_mae"] = _to_json_number(_mae(trace_diff))
    metrics["trace_rmse"] = _to_json_number(_rmse(trace_diff))
    metrics["trace_medae"] = _to_json_number(_medae(trace_diff))
    metrics["trace_r2"] = _to_json_number(_r2(pred_trace, target_trace))
    metrics["trace_mean_error"] = _to_json_number(trace_diff.mean())

    return metrics
