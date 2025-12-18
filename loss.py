import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union, List

# Binary cross entropy with logits is stable for segmentation masks.
bce = nn.BCEWithLogitsLoss()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def bce_dice_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Combination of BCEWithLogits and Dice loss."""
    return bce(pred, mask) + dice_loss(pred, mask)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Return Dice, IoU, Precision, and Recall for a batch."""
    pred = (torch.sigmoid(pred) > threshold).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(1)
    union_sum = pred.sum(1) + target.sum(1)

    dice = (2.0 * intersection + 1e-6) / (union_sum + 1e-6)
    iou = (intersection + 1e-6) / (union_sum - intersection + 1e-6)
    precision = (intersection + 1e-6) / (pred.sum(1) + 1e-6)
    recall = (intersection + 1e-6) / (target.sum(1) + 1e-6)

    return (
        dice.mean().item(),
        iou.mean().item(),
        precision.mean().item(),
        recall.mean().item(),
    )


class LearnableDSUncertaintyLoss(nn.Module):
    """
    Deep supervision with homoscedastic uncertainty weighting (Kendall & Gal).
    For each head i with base loss L_i and log variance s_i:
        reg head: 0.5 * exp(-s_i) * L_i + 0.5 * s_i
        cls head: 1.0 * exp(-s_i) * L_i + 0.5 * s_i
    The main head keeps s_main = 0 (not learned).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        n_heads: int = 3,
        main_index: int = -1,
        init_value: float = 0.0,
        loss_types: Sequence[str] = None,
        interp_mode: str = "nearest",
        interp_modes: Sequence[str] = None,
        clamp_s: float = 10.0,
    ):
        super().__init__()
        assert n_heads >= 1
        self.base_loss = base_loss
        self.n_heads = n_heads
        self.main_index = main_index if main_index >= 0 else (n_heads - 1)
        assert 0 <= self.main_index < n_heads

        if loss_types is None:
            loss_types = tuple("cls" for _ in range(n_heads))
        assert len(loss_types) == n_heads
        for lt in loss_types:
            if lt not in ("reg", "cls"):
                raise ValueError("loss_types must be 'reg' or 'cls'")
        self.loss_types: List[str] = list(loss_types)

        self.interp_mode = interp_mode
        self.interp_modes = list(interp_modes) if interp_modes is not None else None
        self.clamp_s = float(clamp_s)

        # Learnable log sigma^2 for auxiliary heads
        n_aux = n_heads - 1
        self.theta = nn.Parameter(torch.full((n_aux,), float(init_value)))

        self.register_buffer(
            "lambdas",
            torch.tensor([0.5 if t == "reg" else 1.0 for t in loss_types], dtype=torch.float32),
            persistent=False,
        )

    def _build_s_full(self, device, dtype) -> torch.Tensor:
        s_full = torch.zeros(self.n_heads, device=device, dtype=dtype)
        j = 0
        for i in range(self.n_heads):
            if i == self.main_index:
                s_full[i] = 0.0
            else:
                s_full[i] = self.theta[j]
                j += 1
        return torch.clamp(s_full, -self.clamp_s, self.clamp_s)

    @torch.inference_mode()
    def current_weights(self) -> torch.Tensor:
        device, dtype = self.theta.device, self.theta.dtype
        s_full = self._build_s_full(device, dtype)
        lambdas = self.lambdas.to(device=device, dtype=dtype)
        raw = lambdas * torch.exp(-s_full)
        return raw / (raw.sum() + 1e-12)

    def _resize_target(self, t: torch.Tensor, size, mode: str) -> torch.Tensor:
        if t.shape[2:] == size:
            return t
        is_linear = mode in ("linear", "bilinear", "bicubic", "trilinear")
        kwargs = dict(mode=mode, align_corners=False) if is_linear else dict(mode=mode)
        if t.dim() in (3, 4, 5):
            return F.interpolate(t, size=size, **kwargs)
        raise ValueError(f"Unsupported tensor dimension: {t.dim()}")

    def forward(self, inputs: Union[torch.Tensor, Sequence[torch.Tensor]], target: torch.Tensor):
        if not isinstance(inputs, (list, tuple)):
            t = target.float() if target.dtype != torch.float32 else target
            return self.base_loss(inputs, t)

        if len(inputs) != self.n_heads:
            raise ValueError(f"Expected {self.n_heads} heads, got {len(inputs)}")

        device = inputs[0].device
        dtype = inputs[0].dtype
        target_float = target.float() if target.dtype != torch.float32 else target

        losses = []
        for i, x in enumerate(inputs):
            t = target_float
            if t.shape[2:] != x.shape[2:]:
                mode_i = self.interp_modes[i] if self.interp_modes is not None else self.interp_mode
                t = self._resize_target(t, size=x.shape[2:], mode=mode_i)
            losses.append(self.base_loss(x, t))

        L = torch.stack(losses, dim=0)
        s_full = self._build_s_full(device, dtype)
        lambdas = self.lambdas.to(device=device, dtype=dtype)
        return torch.sum(lambdas * torch.exp(-s_full) * L + 0.5 * s_full)
