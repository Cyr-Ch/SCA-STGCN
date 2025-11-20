"""
Adaptive Signer Embedding from Learned Features.

Instead of using simple statistics (mean/std), extracts signer embedding
from early ST-GCN features, allowing the signer encoder to learn what's
important for signer identification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSignerEncoder(nn.Module):
    """
    Adaptive signer encoder that extracts signer embedding from ST-GCN features.
    
    Architecture:
    1. Extract features from early ST-GCN layer
    2. Pool over spatial and temporal dimensions
    3. Encode into signer embedding
    
    This is more expressive than using hand-crafted statistics.
    """
    def __init__(
        self,
        feat_dim: int,
        emb_dim: int = 64,
        pool_type: str = 'attention',  # 'attention', 'avg', 'max'
        hidden_dim: int = 128
    ):
        """
        Args:
            feat_dim: Dimension of ST-GCN features (B, T, J, D)
            emb_dim: Dimension of signer embedding
            pool_type: Type of pooling ('attention', 'avg', 'max')
            hidden_dim: Hidden dimension for encoder network
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.pool_type = pool_type
        
        # Attention pooling over spatial and temporal dimensions
        if pool_type == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract signer embedding from ST-GCN features.
        
        Args:
            features: (B, T, J, D) - ST-GCN features from early layer
        
        Returns:
            signer_emb: (B, emb_dim) - signer embedding
        """
        B, T, J, D = features.shape
        
        if self.pool_type == 'attention':
            # Attention pooling over spatial and temporal dimensions
            # Flatten spatial and temporal: (B, T*J, D)
            features_flat = features.view(B, T * J, D)
            
            # Compute attention weights
            attn_weights = self.attn_pool(features_flat)  # (B, T*J, 1)
            attn_weights = F.softmax(attn_weights, dim=1)  # (B, T*J, 1)
            
            # Weighted sum
            pooled = (features_flat * attn_weights).sum(dim=1)  # (B, D)
        
        elif self.pool_type == 'avg':
            # Average pooling over spatial and temporal
            pooled = features.mean(dim=(1, 2))  # (B, D)
        
        elif self.pool_type == 'max':
            # Max pooling over spatial and temporal
            pooled = features.max(dim=1)[0].max(dim=1)[0]  # (B, D)
        
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Encode to signer embedding
        signer_emb = self.encoder(pooled)  # (B, emb_dim)
        
        return signer_emb


class HybridSignerEncoder(nn.Module):
    """
    Hybrid signer encoder that combines statistics and learned features.
    
    Uses both:
    1. Hand-crafted statistics (mean/std) - for stability
    2. Learned features from ST-GCN - for expressivity
    
    Combines them with a learned fusion mechanism.
    """
    def __init__(
        self,
        stats_dim: int,
        feat_dim: int,
        emb_dim: int = 64,
        hidden_dim: int = 128
    ):
        """
        Args:
            stats_dim: Dimension of hand-crafted statistics
            feat_dim: Dimension of ST-GCN features
            emb_dim: Dimension of signer embedding
            hidden_dim: Hidden dimension for networks
        """
        super().__init__()
        
        # Statistics encoder
        self.stats_encoder = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim // 2),
        )
        
        # Feature encoder (adaptive)
        self.feat_encoder = AdaptiveSignerEncoder(
            feat_dim=feat_dim,
            emb_dim=emb_dim // 2,
            pool_type='attention',
            hidden_dim=hidden_dim
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
        )
    
    def forward(
        self,
        stats: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            stats: (B, stats_dim) - hand-crafted statistics
            features: (B, T, J, D) - ST-GCN features
        
        Returns:
            signer_emb: (B, emb_dim) - fused signer embedding
        """
        # Encode statistics
        stats_emb = self.stats_encoder(stats)  # (B, emb_dim // 2)
        
        # Encode features
        feat_emb = self.feat_encoder(features)  # (B, emb_dim // 2)
        
        # Concatenate and fuse
        combined = torch.cat([stats_emb, feat_emb], dim=-1)  # (B, emb_dim)
        signer_emb = self.fusion(combined)  # (B, emb_dim)
        
        return signer_emb






