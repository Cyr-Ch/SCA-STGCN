import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasAwareAttention(nn.Module):
    """
    Feature gating conditioned on signer embedding.
    Input:
      H: (B, T, D)  -- temporally pooled node features
      s: (B, d)     -- signer/style embedding
    Output:
      H': (B, T, D)
    """
    def __init__(self, feat_dim: int, signer_dim: int, num_heads: int = 4, attn_dropout: float = 0.0):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dk = feat_dim // num_heads

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim + signer_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, h: torch.Tensor, signer_emb: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D), signer_emb: (B, d)
        B, T, D = h.shape
        s = signer_emb.unsqueeze(1).expand(B, T, signer_emb.size(-1))  # (B, T, d)
        k_in = torch.cat([h, s], dim=-1)  # (B, T, D + d)

        q = self.q_proj(h).view(B, T, self.num_heads, self.dk).transpose(1, 2)  # (B,H,T,dk)
        k = self.k_proj(k_in).view(B, T, self.num_heads, self.dk).transpose(1, 2)  # (B,H,T,dk)
        v = self.v_proj(h).view(B, T, self.num_heads, self.dk).transpose(1, 2)  # (B,H,T,dk)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,H,T,T)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # (B,H,T,dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)  # (B, T, D)
        return out


