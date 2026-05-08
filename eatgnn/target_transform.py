import math

import torch


COMPONENTS_6 = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
SYMMETRIC_PAIRS = {
    (0, 1): (1, 0),
    (1, 2): (2, 1),
    (0, 2): (2, 0),
}


def tensor_to_symmetric_components(tensor):
    sym = 0.5 * (tensor + tensor.transpose(-1, -2))
    return torch.stack([sym[..., i, j] for i, j in COMPONENTS_6], dim=-1)


def symmetric_components_to_tensor(components):
    shape = components.shape[:-1] + (3, 3)
    tensor = components.new_zeros(shape)
    for idx, (i, j) in enumerate(COMPONENTS_6):
        tensor[..., i, j] = components[..., idx]
        if (i, j) in SYMMETRIC_PAIRS:
            ii, jj = SYMMETRIC_PAIRS[(i, j)]
            tensor[..., ii, jj] = components[..., idx]
    return tensor


def mask_to_symmetric_component_mask(mask):
    valid = mask > 0.5
    masks = []
    for i, j in COMPONENTS_6:
        if (i, j) in SYMMETRIC_PAIRS:
            ii, jj = SYMMETRIC_PAIRS[(i, j)]
            masks.append(valid[..., i, j] & valid[..., ii, jj])
        else:
            masks.append(valid[..., i, j])
    return torch.stack(masks, dim=-1)


def symmetric_component_mask_to_tensor(mask_components):
    mask = mask_components.new_zeros(mask_components.shape[:-1] + (3, 3))
    for idx, (i, j) in enumerate(COMPONENTS_6):
        mask[..., i, j] = mask_components[..., idx]
        if (i, j) in SYMMETRIC_PAIRS:
            ii, jj = SYMMETRIC_PAIRS[(i, j)]
            mask[..., ii, jj] = mask_components[..., idx]
    return mask


def _masked_quantile(values, q):
    if values.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.quantile(values.float(), q)


class SymmetricTensorTargetTransform:
    def __init__(self, method="signed_log1p_robust", eps=1e-6):
        self.method = method
        self.eps = eps
        self.tail_scale = torch.ones(6, dtype=torch.float32)
        self.center = torch.zeros(6, dtype=torch.float32)
        self.scale = torch.ones(6, dtype=torch.float32)
        self.fitted = False

    def fit(self, dataset):
        components = []
        masks = []
        for sample in dataset:
            target = sample.energy
            mask = sample.energy_mask
            if target.ndim == 2:
                target = target.unsqueeze(0)
                mask = mask.unsqueeze(0)
            components.append(tensor_to_symmetric_components(target).detach().cpu())
            masks.append(mask_to_symmetric_component_mask(mask).detach().cpu())

        components = torch.cat(components, dim=0).float()
        masks = torch.cat(masks, dim=0) > 0.5

        tail_scale = torch.ones(6, dtype=torch.float32)
        center = torch.zeros(6, dtype=torch.float32)
        scale = torch.ones(6, dtype=torch.float32)
        for idx in range(6):
            vals = components[..., idx][masks[..., idx]]
            if vals.numel() == 0:
                continue

            abs_median = vals.abs().median().clamp_min(self.eps)
            tail_scale[idx] = abs_median
            transformed = self._tail_transform(vals, tail_scale[idx])
            median = transformed.median()
            q75 = _masked_quantile(transformed, 0.75)
            q25 = _masked_quantile(transformed, 0.25)
            robust_scale = ((q75 - q25) / 1.349).abs().clamp_min(self.eps)
            center[idx] = median
            scale[idx] = robust_scale

        self.tail_scale = tail_scale
        self.center = center
        self.scale = scale
        self.fitted = True
        return self

    def _tail_transform(self, values, tail_scale):
        if self.method == "none":
            return values
        if self.method == "signed_log1p_robust":
            return torch.sign(values) * torch.log1p(values.abs() / tail_scale.clamp_min(self.eps))
        raise ValueError(f"Unsupported target transform method: {self.method}")

    def _inverse_tail_transform(self, values, tail_scale):
        if self.method == "none":
            return values
        if self.method == "signed_log1p_robust":
            return torch.sign(values) * torch.expm1(values.abs()) * tail_scale.clamp_min(self.eps)
        raise ValueError(f"Unsupported target transform method: {self.method}")

    def transform_tensor(self, tensor):
        components = tensor_to_symmetric_components(tensor)
        tail_scale = self.tail_scale.to(device=tensor.device, dtype=tensor.dtype)
        center = self.center.to(device=tensor.device, dtype=tensor.dtype)
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        transformed = self._tail_transform(components, tail_scale)
        transformed = (transformed - center) / scale
        return symmetric_components_to_tensor(transformed)

    def inverse_tensor(self, tensor):
        components = tensor_to_symmetric_components(tensor)
        tail_scale = self.tail_scale.to(device=tensor.device, dtype=tensor.dtype)
        center = self.center.to(device=tensor.device, dtype=tensor.dtype)
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        original = components * scale + center
        original = self._inverse_tail_transform(original, tail_scale)
        return symmetric_components_to_tensor(original)

    def state_dict(self):
        return {
            "method": self.method,
            "eps": self.eps,
            "tail_scale": self.tail_scale.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "components": ["xx", "yy", "zz", "xy", "yz", "xz"],
        }


def apply_target_transform(dataset, transform):
    for sample in dataset:
        sample.energy_raw = sample.energy.clone()
        sample.energy = transform.transform_tensor(sample.energy)
    return dataset


def build_target_transform(config, train_dataset):
    target_config = config.get("target_transform", {})
    if not target_config.get("enabled", False):
        return None
    transform = SymmetricTensorTargetTransform(
        method=target_config.get("method", "signed_log1p_robust"),
        eps=float(target_config.get("eps", 1e-6)),
    )
    return transform.fit(train_dataset)
