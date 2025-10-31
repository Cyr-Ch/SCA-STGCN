import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def signer_adversarial_loss(signer_logits: torch.Tensor, signer_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(signer_logits, signer_ids)


def info_nce(
    z: torch.Tensor,
    pos_idx: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Simple InfoNCE between pairs (z_i, z_pos_i), all others as negatives.
    z: (B, D)
    pos_idx: (B,) index of positive partner for each i
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.t()  # (B,B)
    logits = sim / temperature
    loss = F.cross_entropy(logits, pos_idx)
    return loss


def attention_regularization(
    H_att: torch.Tensor, sparsity_weight: float = 0.0, smooth_weight: float = 0.0
) -> torch.Tensor:
    """
    Example reg on attended features:
    - sparsity: L1 on features to encourage focused channels
    - smoothness: temporal TV-loss
    H_att: (B, T, D)
    """
    loss = torch.tensor(0.0, device=H_att.device)
    if sparsity_weight > 0:
        loss = loss + sparsity_weight * H_att.abs().mean()
    if smooth_weight > 0:
        tv = (H_att[:, 1:] - H_att[:, :-1]).abs().mean()
        loss = loss + smooth_weight * tv
    return loss


