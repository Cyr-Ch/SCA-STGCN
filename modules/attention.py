import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasAwareAttention(nn.Module):
    """
    Feature gating conditioned on signer embedding (additive concatenation).
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


class FiLMSignerAttention(nn.Module):
    """
    Multiplicative signer-conditioned attention using Feature-wise Linear Modulation (FiLM).
    
    Instead of concatenating signer embeddings, uses multiplicative gating:
    - Keys: k = (W_k h) ⊙ (1 + scale_k(s)) + shift_k(s)
    - Values: v = (W_v h) ⊙ (1 + scale_v(s)) + shift_v(s)
    
    This is more expressive than additive concatenation and allows signer-specific
    feature scaling and shifting.
    
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
        self.feat_dim = feat_dim
        self.signer_dim = signer_dim

        # Standard attention projections
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.drop = nn.Dropout(attn_dropout)

        # FiLM-style modulation for keys: scale and shift
        self.k_scale_net = nn.Sequential(
            nn.Linear(signer_dim, feat_dim),
            nn.Tanh()  # Bounded scaling
        )
        self.k_shift_net = nn.Linear(signer_dim, feat_dim)

        # FiLM-style modulation for values: scale and shift
        self.v_scale_net = nn.Sequential(
            nn.Linear(signer_dim, feat_dim),
            nn.Tanh()  # Bounded scaling
        )
        self.v_shift_net = nn.Linear(signer_dim, feat_dim)

    def forward(self, h: torch.Tensor, signer_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, D) - input features
            signer_emb: (B, d) - signer embedding
        Returns:
            out: (B, T, D) - modulated features
        """
        B, T, D = h.shape

        # Compute FiLM parameters from signer embedding
        k_scale = self.k_scale_net(signer_emb)  # (B, D)
        k_shift = self.k_shift_net(signer_emb)   # (B, D)
        v_scale = self.v_scale_net(signer_emb)   # (B, D)
        v_shift = self.v_shift_net(signer_emb)   # (B, D)

        # Expand to temporal dimension
        k_scale = k_scale.unsqueeze(1).expand(B, T, D)  # (B, T, D)
        k_shift = k_shift.unsqueeze(1).expand(B, T, D)  # (B, T, D)
        v_scale = v_scale.unsqueeze(1).expand(B, T, D)  # (B, T, D)
        v_shift = v_shift.unsqueeze(1).expand(B, T, D)  # (B, T, D)

        # Project queries (no signer conditioning)
        q = self.q_proj(h).view(B, T, self.num_heads, self.dk).transpose(1, 2)  # (B, H, T, dk)

        # Project and modulate keys: k = (W_k h) ⊙ (1 + scale) + shift
        k_base = self.k_proj(h)  # (B, T, D)
        k_modulated = k_base * (1.0 + k_scale) + k_shift  # (B, T, D)
        k = k_modulated.view(B, T, self.num_heads, self.dk).transpose(1, 2)  # (B, H, T, dk)

        # Project and modulate values: v = (W_v h) ⊙ (1 + scale) + shift
        v_base = self.v_proj(h)  # (B, T, D)
        v_modulated = v_base * (1.0 + v_scale) + v_shift  # (B, T, D)
        v = v_modulated.view(B, T, self.num_heads, self.dk).transpose(1, 2)  # (B, H, T, dk)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, T, dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.out_proj(out)  # (B, T, D)

        return out


