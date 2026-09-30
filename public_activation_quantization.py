"""Package-local inverse-softplus correction for transformer activation QDQ."""

from __future__ import annotations

import torch
from transformer_quant_models import ActQuantTokenwise


def inverse_softplus(scale: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("activation scales must be finite and positive")
    return scale + torch.log(-torch.expm1(-scale))


class CorrectedActQuantTokenwise(ActQuantTokenwise):
    """Retain the original QDQ forward/state schema, fix calibration inverse."""

    def finalize_calib_scale(self):
        if not torch.isfinite(self.calib_min).all() or not torch.isfinite(self.calib_max).all():
            raise RuntimeError("activation calibration observed non-finite channels")
        if (self.calib_min > self.calib_max).any():
            raise RuntimeError("activation calibration min exceeds max")
        super().finalize_calib_scale()
        with torch.no_grad():
            self.log_scale.copy_(inverse_softplus(self.log_scale.exp()))
            self.base_log_scale.copy_(self.log_scale)


def install_local_activation_quantizers(model):
    """Replace fresh quantizer modules before loading user-supplied weights."""
    for name, child in list(model.named_children()):
        if isinstance(child, ActQuantTokenwise):
            replacement = CorrectedActQuantTokenwise(
                child.num_channels, init_bits=child.bits,
                scale_min=float(child.scale_min), scale_max=float(child.scale_max),
                unsigned=child.unsigned, cl=child.cl,
            )
            replacement.load_state_dict(child.state_dict(), strict=True)
            replacement.quant_mode = child.quant_mode
            replacement.train(child.training)
            replacement.log_scale.requires_grad_(child.log_scale.requires_grad)
            setattr(model, name, replacement)
        else:
            install_local_activation_quantizers(child)
