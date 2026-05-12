import torch


COMPONENTS_6 = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
IN_PLANE_COMPONENTS = ((0, 0), (1, 1), (0, 1))
MODEL_COMPONENT_NAMES = ("xx", "yy", "xy")
INPLANE_IRREP_COMPONENT_NAMES = ("trace", "anisotropy", "shear")
RAW_COMPONENT_NAMES = ("xx", "yy", "zz", "xy", "yz", "xz")
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


def tensor_to_inplane_components(tensor):
    sym = 0.5 * (tensor + tensor.transpose(-1, -2))
    return torch.stack([sym[..., i, j] for i, j in IN_PLANE_COMPONENTS], dim=-1)


def inplane_components_to_tensor(components):
    shape = components.shape[:-1] + (3, 3)
    tensor = components.new_zeros(shape)
    tensor[..., 0, 0] = components[..., 0]
    tensor[..., 1, 1] = components[..., 1]
    tensor[..., 0, 1] = components[..., 2]
    tensor[..., 1, 0] = components[..., 2]
    return tensor


def inplane_components_to_irreps(components):
    sqrt2 = torch.sqrt(components.new_tensor(2.0))
    xx = components[..., 0]
    yy = components[..., 1]
    xy = components[..., 2]
    return torch.stack(
        [
            (xx + yy) / sqrt2,
            (xx - yy) / sqrt2,
            sqrt2 * xy,
        ],
        dim=-1,
    )


def inplane_irreps_to_components(irreps):
    sqrt2 = torch.sqrt(irreps.new_tensor(2.0))
    trace = irreps[..., 0]
    anisotropy = irreps[..., 1]
    shear = irreps[..., 2]
    return torch.stack(
        [
            (trace + anisotropy) / sqrt2,
            (trace - anisotropy) / sqrt2,
            shear / sqrt2,
        ],
        dim=-1,
    )


def tensor_to_inplane_irreps(tensor):
    return inplane_components_to_irreps(tensor_to_inplane_components(tensor))


def inplane_irreps_to_tensor(irreps):
    return inplane_components_to_tensor(inplane_irreps_to_components(irreps))


def mask_to_inplane_irrep_mask(mask):
    component_mask = tensor_to_inplane_components(mask) > 0.5
    diagonal_valid = component_mask[..., 0] & component_mask[..., 1]
    shear_valid = component_mask[..., 2]
    return torch.stack([diagonal_valid, diagonal_valid, shear_valid], dim=-1)


def project_tensor_to_inplane(tensor):
    return inplane_components_to_tensor(tensor_to_inplane_components(tensor))


def raw_tensor_to_model_tensor(tensor):
    return project_tensor_to_inplane(tensor)


def model_tensor_to_raw_tensor(tensor):
    return project_tensor_to_inplane(tensor)


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


def raw_mask_to_model_mask(mask):
    valid = mask > 0.5
    m_xx, m_yy = valid[..., 0, 0], valid[..., 1, 1]
    m_xy = valid[..., 0, 1] & valid[..., 1, 0]
    model_mask_components = torch.stack([m_xx, m_yy, m_xy], dim=-1)
    return inplane_components_to_tensor(model_mask_components.to(mask.dtype))


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
        self.tail_scale = torch.ones(3, dtype=torch.float32)
        self.center = torch.zeros(3, dtype=torch.float32)
        self.scale = torch.ones(3, dtype=torch.float32)
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
            components.append(tensor_to_inplane_components(target).detach().cpu())
            masks.append(mask_to_symmetric_component_mask(mask).detach().cpu())

        components = torch.cat(components, dim=0).float()
        masks = torch.cat(masks, dim=0)[..., [0, 1, 3]] > 0.5

        tail_scale = torch.ones(3, dtype=torch.float32)
        center = torch.zeros(3, dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        for idx in range(3):
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
        components = tensor_to_inplane_components(tensor)
        tail_scale = self.tail_scale.to(device=tensor.device, dtype=tensor.dtype)
        center = self.center.to(device=tensor.device, dtype=tensor.dtype)
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        transformed = self._tail_transform(components, tail_scale)
        transformed = (transformed - center) / scale
        return inplane_components_to_tensor(transformed)

    def inverse_tensor(self, tensor):
        components = tensor_to_inplane_components(tensor)
        tail_scale = self.tail_scale.to(device=tensor.device, dtype=tensor.dtype)
        center = self.center.to(device=tensor.device, dtype=tensor.dtype)
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        original = components * scale + center
        original = self._inverse_tail_transform(original, tail_scale)
        return inplane_components_to_tensor(original)

    def state_dict(self):
        return {
            "method": self.method,
            "eps": self.eps,
            "tail_scale": self.tail_scale.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "components": list(MODEL_COMPONENT_NAMES),
        }


def apply_target_transform(dataset, transform):
    for sample in dataset:
        if not hasattr(sample, "energy_model_raw"):
            sample.energy_model_raw = sample.energy.clone()
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
