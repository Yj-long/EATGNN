import torch
import torch.nn.functional as F

from .target_transform import (
    MODEL_COMPONENT_NAMES,
    tensor_to_inplane_components,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_tensor_metrics(pred, target):
    diff = pred - target
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    max_abs = diff.abs().max().item()
    return mae, rmse, max_abs


def _component_delta(delta, idx, device, dtype):
    if isinstance(delta, dict):
        return float(delta.get(MODEL_COMPONENT_NAMES[idx], 1.0))
    if isinstance(delta, (list, tuple)):
        return float(delta[idx])
    if torch.is_tensor(delta):
        delta = delta.to(device=device, dtype=dtype)
        if delta.numel() == 1:
            return float(delta.item())
        return float(delta[idx].item())
    return float(delta)


def tensor_basis_huber_frobenius_loss(
    pred,
    target,
    mask,
    component_weights=None,
    delta=1.0,
    lambda_tensor=0.0,
):
    if component_weights is None:
        component_weights = {name: 1.0 for name in MODEL_COMPONENT_NAMES}

    pred_components = tensor_to_inplane_components(pred)
    target_components = tensor_to_inplane_components(target)
    component_mask = tensor_to_inplane_components(mask)

    total_loss = pred.sum() * 0.0
    for idx, name in enumerate(MODEL_COMPONENT_NAMES):
        valid = component_mask[..., idx] > 0.5
        weight = float(component_weights.get(name, 1.0))
        if valid.any() and weight != 0.0:
            total_loss = total_loss + weight * F.huber_loss(
                pred_components[..., idx][valid],
                target_components[..., idx][valid],
                reduction="mean",
                delta=_component_delta(delta, idx, pred.device, pred.dtype),
            )

    if float(lambda_tensor) != 0.0:
        tensor_valid = (component_mask > 0.5).all(dim=-1)
        if tensor_valid.any():
            pred_2x2 = pred[..., :2, :2][tensor_valid]
            target_2x2 = target[..., :2, :2][tensor_valid]
            frob = torch.linalg.matrix_norm(pred_2x2 - target_2x2, ord="fro", dim=(-2, -1))
            total_loss = total_loss + float(lambda_tensor) * frob.mean()

    return total_loss


def estimate_model_component_huber_delta_from_trainset(train_dataset):
    values = [[] for _ in range(3)]
    for sample in train_dataset:
        target = sample.energy
        mask = sample.energy_mask
        if target.ndim == 2:
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)
        components = tensor_to_inplane_components(target)
        component_mask = tensor_to_inplane_components(mask)
        for idx in range(3):
            valid = component_mask[..., idx] > 0.5
            if valid.any():
                values[idx].append(components[..., idx][valid].detach().cpu().float())

    deltas = []
    for idx in range(3):
        if not values[idx]:
            deltas.append(1.0)
            continue
        vals = torch.cat(values[idx], dim=0)
        med = vals.median()
        mad = (vals - med).abs().median().clamp_min(1e-6)
        robust_sigma = 1.4826 * mad
        delta_i = float((1.345 * robust_sigma).item())
        deltas.append(max(delta_i, 1e-3))
    return deltas


def evaluate_masked_tensor_metrics(pred, target, mask):
    valid = mask > 0.5
    if not valid.any():
        return 0.0, 0.0, 0.0
    diff = pred[valid] - target[valid]
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    max_abs = diff.abs().max().item()
    return mae, rmse, max_abs


