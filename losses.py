import math

import torch
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_tensor_metrics(pred, target):
    diff = pred - target
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    max_abs = diff.abs().max().item()
    return mae, rmse, max_abs


def tensor_to_scalar_q_components(tensor_3x3: torch.Tensor):
    """Decompose a (possibly batched) 3x3 tensor into scalar s and five q components."""
    sym = 0.5 * (tensor_3x3 + tensor_3x3.transpose(-1, -2))
    xx, yy, zz = sym[..., 0, 0], sym[..., 1, 1], sym[..., 2, 2]
    xy, yz, xz = sym[..., 0, 1], sym[..., 1, 2], sym[..., 0, 2]

    s = (xx + yy + zz) / 3.0
    q1 = 0.5 * (xx - yy)
    q2 = (2.0 * zz - xx - yy) / (2.0 * math.sqrt(3.0))
    q3 = xy
    q4 = yz
    q5 = xz
    q = torch.stack([q1, q2, q3, q4, q5], dim=-1)
    return s, q


def mask_to_component_mask(mask_3x3: torch.Tensor):
    """Map element-wise 3x3 supervision mask to [s, q1..q5] supervision masks."""
    valid = mask_3x3 > 0.5
    m_xx, m_yy, m_zz = valid[..., 0, 0], valid[..., 1, 1], valid[..., 2, 2]
    m_xy = valid[..., 0, 1] & valid[..., 1, 0]
    m_yz = valid[..., 1, 2] & valid[..., 2, 1]
    m_xz = valid[..., 0, 2] & valid[..., 2, 0]

    # s, q1, q2 all depend on the three diagonal terms
    m_diag = m_xx & m_yy & m_zz
    m_s = m_diag
    m_q = torch.stack([m_diag, m_diag, m_xy, m_yz, m_xz], dim=-1)
    return m_s, m_q


def weighted_masked_huber_loss(
    pred,
    target,
    mask,
    ws: float = 1.0,
    wq = (1.0, 1.0, 1.0, 1.0, 1.0),
    delta = 1.0,
):
    pred_s, pred_q = tensor_to_scalar_q_components(pred)
    target_s, target_q = tensor_to_scalar_q_components(target)
    m_s, m_q = mask_to_component_mask(mask)

    if isinstance(delta, (list, tuple)):
        delta = torch.tensor(delta, dtype=pred.dtype, device=pred.device)
    elif torch.is_tensor(delta):
        delta = delta.to(device=pred.device, dtype=pred.dtype)

    if torch.is_tensor(delta) and delta.numel() == 3:
        delta_xx, delta_yy, delta_zz = delta[0], delta[1], delta[2]
        delta_s = float(((delta_xx + delta_yy + delta_zz) / 3.0).item())
        delta_q1 = float(((delta_xx + delta_yy) / 2.0).item())
        delta_q2 = float(((delta_xx + delta_yy + delta_zz) / 3.0).item())
        delta_q3 = float(((delta_xx + delta_yy + delta_zz) / 3.0).item())
        delta_q4 = float(((delta_xx + delta_yy + delta_zz) / 3.0).item())
        delta_q5 = float(((delta_xx + delta_yy + delta_zz) / 3.0).item())
        q_deltas = [delta_q1, delta_q2, delta_q3, delta_q4, delta_q5]
    else:
        delta_s = float(delta)
        q_deltas = [float(delta)] * 5

    total_loss = pred.sum() * 0.0
    total_weight = 0.0

    if m_s.any():
        l_s = F.huber_loss(pred_s[m_s], target_s[m_s], reduction="mean", delta=delta_s)
        total_loss = total_loss + ws * l_s
        total_weight += ws

    for k in range(5):
        mk = m_q[..., k]
        wk = float(wq[k])
        if mk.any():
            l_qk = F.huber_loss(
                pred_q[..., k][mk],
                target_q[..., k][mk],
                reduction="mean",
                delta=q_deltas[k],
            )
            total_loss = total_loss + wk * l_qk
            total_weight += wk

    if total_weight == 0.0:
        return total_loss
    return total_loss / total_weight


def estimate_loss_hparams_from_trainset(train_dataset):
    """
    Estimate loss_wq and huber_delta from supervised targets in the training set.
    - loss_wq: inverse per-component scale (normalized to mean=1 on observed components)
    - huber_delta: robust global scale estimated from supervised [s, q1..q5]
    """
    q_values = [[] for _ in range(5)]
    sq_values = []

    for sample in train_dataset:
        target = sample.energy
        mask = sample.energy_mask
        if target.ndim == 2:
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)

        s, q = tensor_to_scalar_q_components(target)
        m_s, m_q = mask_to_component_mask(mask)

        if m_s.any():
            s_vals = s[m_s].detach().cpu()
            sq_values.append(s_vals)

        for k in range(5):
            mk = m_q[..., k]
            if mk.any():
                qk_vals = q[..., k][mk].detach().cpu()
                q_values[k].append(qk_vals)
                sq_values.append(qk_vals)

    wq = torch.ones(5, dtype=torch.float32)
    observed = []
    for k in range(5):
        if len(q_values[k]) == 0:
            wq[k] = 0.0
            continue
        vals = torch.cat(q_values[k], dim=0).float()
        scale = vals.std(unbiased=False).clamp_min(1e-6)
        wq[k] = 1.0 / scale
        observed.append(k)

    if len(observed) > 0:
        wq_mean = wq[observed].mean().clamp_min(1e-6)
        wq[observed] = wq[observed] / wq_mean

    if len(sq_values) == 0:
        huber_delta = 1.0
    else:
        all_vals = torch.cat(sq_values, dim=0).float()
        med = all_vals.median()
        mad = (all_vals - med).abs().median().clamp_min(1e-6)
        robust_sigma = 1.4826 * mad
        huber_delta = float((1.345 * robust_sigma).item())
        huber_delta = max(huber_delta, 1e-3)

    return wq.tolist(), huber_delta


def estimate_diag_huber_delta_from_trainset(train_dataset):
    """
    Estimate per-diagonal huber deltas [delta_xx, delta_yy, delta_zz] from supervised
    diagonal entries only. Formula is unchanged:
      robust_sigma = 1.4826 * MAD
      delta = max(1.345 * robust_sigma, 1e-3)
    """
    diag_values = [[] for _ in range(3)]

    for sample in train_dataset:
        target = sample.energy
        mask = sample.energy_mask
        if target.ndim == 2:
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)

        valid = mask > 0.5
        for i in range(3):
            mi = valid[..., i, i]
            if mi.any():
                vals = target[..., i, i][mi].detach().cpu().float()
                diag_values[i].append(vals)

    deltas = []
    for i in range(3):
        if len(diag_values[i]) == 0:
            deltas.append(1.0)
            continue
        all_vals = torch.cat(diag_values[i], dim=0)
        med = all_vals.median()
        mad = (all_vals - med).abs().median().clamp_min(1e-6)
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


