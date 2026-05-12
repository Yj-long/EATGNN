import math

import torch


COMPONENT_NAMES = ("xx", "yy", "xy")


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


def _add_tail_trace_metrics(metrics, pred_trace, target_trace, fractions=(0.20, 0.10)):
    if target_trace.numel() == 0:
        for fraction in fractions:
            pct = int(round(fraction * 100))
            metrics[f"trace_tail{pct}_mae"] = None
            metrics[f"trace_tail{pct}_rmse"] = None
            metrics[f"trace_tail{pct}_r2"] = None
            metrics[f"trace_tail{pct}_count"] = 0
        return

    tail_scores = target_trace.abs()
    for fraction in fractions:
        pct = int(round(fraction * 100))
        tail_count = max(1, int(math.ceil(target_trace.numel() * fraction)))
        tail_indices = torch.topk(tail_scores, k=tail_count, largest=True).indices
        pred_tail = pred_trace[tail_indices]
        target_tail = target_trace[tail_indices]
        tail_diff = pred_tail - target_tail
        metrics[f"trace_tail{pct}_mae"] = _to_json_number(_mae(tail_diff))
        metrics[f"trace_tail{pct}_rmse"] = _to_json_number(_rmse(tail_diff))
        metrics[f"trace_tail{pct}_r2"] = _to_json_number(_r2(pred_tail, target_tail))
        metrics[f"trace_tail{pct}_count"] = int(tail_count)


def compute_original_tensor_metrics(pred, target, mask):
    pred = 0.5 * (pred.detach().cpu().double() + pred.detach().cpu().double().transpose(-1, -2))
    target = 0.5 * (target.detach().cpu().double() + target.detach().cpu().double().transpose(-1, -2))
    mask = mask.detach().cpu()

    pred_components = torch.stack([pred[..., 0, 0], pred[..., 1, 1], pred[..., 0, 1]], dim=-1)
    target_components = torch.stack([target[..., 0, 0], target[..., 1, 1], target[..., 0, 1]], dim=-1)
    component_mask = torch.stack(
        [mask[..., 0, 0], mask[..., 1, 1], (mask[..., 0, 1] > 0.5) & (mask[..., 1, 0] > 0.5)],
        dim=-1,
    ).to(mask.dtype)

    component_diff = _masked_values(pred_components - target_components, component_mask)
    if component_diff.numel() == 0:
        return {
            "component_mse": None,
            "component_rmse": None,
            "component_mae": None,
            "component_medae": None,
            "r2_uniform_average": None,
        }

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
    pred_valid = pred[..., :2, :2][tensor_valid]
    target_valid = target[..., :2, :2][tensor_valid]
    if pred_valid.numel() > 0:
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
    else:
        metrics["frobenius_mse"] = None
        metrics["frobenius_rmse"] = None
        metrics["frobenius_absolute_error"] = None
        metrics["relative_frobenius_error"] = None
        metrics["positive_definite_rate"] = None
        metrics["min_predicted_eigenvalue"] = None
        metrics["max_symmetry_error"] = None

    diag_valid = (component_mask[..., 0] > 0.5) & (component_mask[..., 1] > 0.5)
    pred_trace = (pred[..., 0, 0] + pred[..., 1, 1])[diag_valid]
    target_trace = (target[..., 0, 0] + target[..., 1, 1])[diag_valid]
    trace_diff = pred_trace - target_trace
    metrics["trace_mae"] = _to_json_number(_mae(trace_diff)) if trace_diff.numel() > 0 else None
    metrics["trace_rmse"] = _to_json_number(_rmse(trace_diff)) if trace_diff.numel() > 0 else None
    metrics["trace_medae"] = _to_json_number(_medae(trace_diff)) if trace_diff.numel() > 0 else None
    metrics["trace_r2"] = _to_json_number(_r2(pred_trace, target_trace)) if trace_diff.numel() > 0 else None
    metrics["trace_mean_error"] = _to_json_number(trace_diff.mean()) if trace_diff.numel() > 0 else None
    _add_tail_trace_metrics(metrics, pred_trace, target_trace)

    return metrics


def compute_model_basis_diag_mae(pred, target, mask):
    pred = 0.5 * (pred.detach().cpu().double() + pred.detach().cpu().double().transpose(-1, -2))
    target = 0.5 * (target.detach().cpu().double() + target.detach().cpu().double().transpose(-1, -2))
    mask = mask.detach().cpu() > 0.5
    result = {}

    for name, i, j in (("xx", 0, 0), ("yy", 1, 1), ("xy", 0, 1)):
        valid = mask[..., i, j]
        if name == "xy":
            valid = valid & mask[..., 1, 0]
        result[name] = (
            _to_json_number((pred[..., i, j][valid] - target[..., i, j][valid]).abs().mean())
            if valid.any()
            else None
        )
    return result


def estimate_model_basis_diag_normalizers(dataset, eps=1e-6):
    values = {"xx": [], "yy": [], "xy": []}
    for sample in dataset:
        target = sample.energy_raw if hasattr(sample, "energy_raw") else sample.energy
        mask = sample.energy_raw_mask if hasattr(sample, "energy_raw_mask") else sample.energy_mask
        if target.ndim == 2:
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)
        valid = mask > 0.5

        for name, i, j in (("xx", 0, 0), ("yy", 1, 1), ("xy", 0, 1)):
            component_valid = valid[..., i, j]
            if name == "xy":
                component_valid = component_valid & valid[..., 1, 0]
            if component_valid.any():
                values[name].append(target[..., i, j][component_valid].detach().cpu().float())

    normalizers = {}
    for name, chunks in values.items():
        if not chunks:
            normalizers[name] = 1.0
            continue
        vals = torch.cat(chunks, dim=0)
        median = vals.median()
        scale = (vals - median).abs().mean().clamp_min(eps)
        normalizers[name] = float(scale.item())
    return normalizers


def compute_diag_score(diag_mae, normalizers, weights):
    score = 0.0
    total_weight = 0.0
    for name in COMPONENT_NAMES:
        value = diag_mae.get(name)
        if value is None:
            continue
        weight = float(weights.get(name, 0.0))
        score += weight * (float(value) / max(float(normalizers.get(name, 1.0)), 1e-12))
        total_weight += weight
    if total_weight == 0.0:
        return None
    return score
