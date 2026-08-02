"""Loss reduction helpers.

The masked-loss reduction here mirrors musubi-tuner's ``apply_loss_mask`` (batch-global
denominator) at ``musubi-tuner/src/musubi_tuner/training/losses.py``. Fizgig's per-element
loss tensor is always 4-D ``[B, C, h, w]``, so we only handle the 4-D case plus a 3-D mask
shortcut for convenience.
"""

from __future__ import annotations

from typing import Optional

import torch


def apply_loss_mask(
    per_elem: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reduce a per-element loss to a scalar with optional mask weighting.

    ``mask`` is a soft weight in ``[0, 1]``: 1 means "include this element fully in the
    loss", 0 means "ignore it". The denominator is the mask's mean over the whole batch
    so the result is the average loss weighted by mask coverage. A fully-masked-out batch
    (``denom == 0``) falls back to the plain mean to avoid a NaN — call sites shouldn't
    rely on this, but it shouldn't crash training either.

    Returns ``(loss, metrics)`` where metrics carries ``mask_active`` (mean mask value,
    i.e. the fraction of the loss grid that's contributing), ``loss_unmasked`` (what
    ``.mean()`` would have given), and ``loss_masked`` (the returned scalar).
    """
    if mask is None:
        return per_elem.mean(), {}

    mask = mask.to(device=per_elem.device)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B, h, w] -> [B, 1, h, w]
    # Already [B, 1, h, w] or broadcastable; trust the caller otherwise.

    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    metrics: dict[str, float] = {
        "mask_active": float(denom.detach().float().item()),
        "loss_unmasked": float(per_elem.detach().float().mean().item()),
    }
    if denom.item() == 0:
        loss = per_elem.mean()
    else:
        loss = (per_elem * mask_f).div(denom).mean()
    metrics["loss_masked"] = float(loss.detach().float().item())
    return loss, metrics
