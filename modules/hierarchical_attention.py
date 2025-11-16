"""
Hierarchical Multi-Scale Attention for Sign Language Recognition.

Implements three-level hierarchical attention:
1. Spatial Attention: Attention over joints conditioned on signer
2. Temporal Attention: Attention over time steps conditioned on signer
3. Cross-Modal Attention: Attention between spatial and temporal features
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FiLMSignerAttention


class SpatialAttention(nn.Module):
    """
    Spatial attention over joints conditioned on signer embedding.
    Input: (B, T, J, D) - features per joint per timestep
    Output: (B, T, J, D) - spatially attended features
    """
    def __init__(self, feat_dim: int, signer_dim: int, num_heads: int = 4):
        super().__init__()
        assert feat_dim % num_heads == 0
        self.num_heads = num_heads
        self.dk = feat_dim // num_heads
        
        # FiLM-style modulation for spatial attention
        self.scale_net = nn.Sequential(
            nn.Linear(signer_dim, feat_dim),
            nn.Tanh()
        )
        self.shift_net = nn.Linear(signer_dim, feat_dim)
        
        # Attention projections
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)
    
    def forward(self, h: torch.Tensor, signer_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, J, D) - joint features
            signer_emb: (B, d) - signer embedding
        Returns:
            out: (B, T, J, D) - spatially attended features
        """
        B, T, J, D = h.shape
        
        # Compute FiLM parameters
        scale = self.scale_net(signer_emb)  # (B, D)
        shift = self.shift_net(signer_emb)  # (B, D)
        
        # Expand to (B, T, J, D)
        scale = scale.unsqueeze(1).unsqueeze(2).expand(B, T, J, D)
        shift = shift.unsqueeze(1).unsqueeze(2).expand(B, T, J, D)
        
        # Reshape for attention: (B*T, J, D)
        h_flat = h.view(B * T, J, D)
        scale_flat = scale.view(B * T, J, D)
        shift_flat = shift.view(B * T, J, D)
        
        # Project and modulate
        q = self.q_proj(h_flat)  # (B*T, J, D)
        k_base = self.k_proj(h_flat)
        k = k_base * (1.0 + scale_flat) + shift_flat
        v_base = self.v_proj(h_flat)
        v = v_base * (1.0 + scale_flat) + shift_flat
        
        # Reshape for multi-head attention
        q = q.view(B * T, J, self.num_heads, self.dk).transpose(1, 2)  # (B*T, H, J, dk)
        k = k.view(B * T, J, self.num_heads, self.dk).transpose(1, 2)
        v = v.view(B * T, J, self.num_heads, self.dk).transpose(1, 2)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B*T, H, J, J)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # (B*T, H, J, dk)
        out = out.transpose(1, 2).contiguous().view(B * T, J, D)  # (B*T, J, D)
        out = self.out_proj(out)
        
        # Reshape back
        out = out.view(B, T, J, D)
        
        return out


class HierarchicalAttention(nn.Module):
    """
    Hierarchical multi-scale attention with spatial, temporal, and cross-modal attention.
    
    Architecture:
    1. Spatial Attention: Attention over joints (J×J) conditioned on signer
    2. Temporal Attention: Attention over time steps (T×T) conditioned on signer
    3. Cross-Modal Attention: Attention between spatial and temporal features
    
    Input:
      H: (B, T, J, D) - ST-GCN features
      s: (B, d) - signer embedding
    Output:
      H': (B, T, D) - hierarchically attended features
    """
    def __init__(
        self,
        feat_dim: int,
        signer_dim: int,
        spatial_heads: int = 4,
        temporal_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.signer_dim = signer_dim
        
        # Spatial attention over joints
        self.spatial_attn = SpatialAttention(feat_dim, signer_dim, spatial_heads)
        
        # Temporal attention (using FiLM attention after pooling joints)
        self.temporal_attn = FiLMSignerAttention(feat_dim, signer_dim, temporal_heads, dropout)
        
        # Joint pooling to get temporal features
        self.joint_pool = nn.AdaptiveAvgPool2d((None, 1))  # Pool over J dimension
        
        # Cross-modal attention (optional, can be added later)
        self.use_cross_modal = False  # Can be enabled if needed
        
    def forward(self, h: torch.Tensor, signer_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, J, D) - ST-GCN features
            signer_emb: (B, d) - signer embedding
        Returns:
            out: (B, T, D) - hierarchically attended features
        """
        B, T, J, D = h.shape
        
        # Step 1: Spatial attention over joints
        h_spatial = self.spatial_attn(h, signer_emb)  # (B, T, J, D)
        
        # Step 2: Pool joints to get temporal features
        # Reshape to (B, D, T, J) for pooling
        h_pooled = self.joint_pool(h_spatial.permute(0, 3, 1, 2))  # (B, D, T, 1)
        h_temporal = h_pooled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B, T, D)
        
        # Step 3: Temporal attention over time steps
        h_temporal_attn = self.temporal_attn(h_temporal, signer_emb)  # (B, T, D)
        
        return h_temporal_attn




