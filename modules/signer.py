import torch
import torch.nn as nn


class SignerEncoder(nn.Module):
    """
    Encodes signer/style statistics into an embedding.
    Input: simple per-window stats (e.g., mean, std of keypoints) or learned from early features.
    Here: we expect pooled pose stats: (B, F)
    """
    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 2 * emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * emb_dim, emb_dim),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats)  # (B, emb_dim)


