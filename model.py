from __future__ import annotations
from typing import Tuple, Dict
import torch
import torch.nn as nn

# Absolute imports (GC_training scripts already push repo root onto sys.path)
from layers import STGCNBackbone
from modules import SignerEncoder, BiasAwareAttention, FiLMSignerAttention, TemporalAggregator, ClassifierHead, SignerHead


class SignSTGCNModel(nn.Module):
    """
    End-to-end:
      Input:  pose keypoints X of shape (B, T, J, C)
              A: adjacency (J,J) pre-normalized
              pose_stats: (B, S) simple per-window stats for signer encoder
      Output: dict with logits and optional auxiliary outputs
    """
    def __init__(
        self,
        num_joints: int,
        in_coords: int = 2,
        stgcn_channels: Tuple[int, ...] = (64, 128, 128),
        stgcn_kernel: int = 3,
        stgcn_dilations: Tuple[int, ...] = (1, 2, 3),
        temporal_hidden: int = 256,
        num_classes: int = 100,
        signer_stats_dim: int = 16,
        signer_emb_dim: int = 64,
        use_signer_head: bool = True,
        num_signers: int = 50,
        lambda_grl: float = 0.5,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        use_film_attention: bool = False
    ):
        super().__init__()
        self.J = num_joints
        self.C_in = in_coords

        # backbone
        self.backbone = STGCNBackbone(
            in_ch=in_coords,
            channels=stgcn_channels,
            kernel_size=stgcn_kernel,
            dilations=stgcn_dilations,
            dropout=dropout
        )
        feat_dim = stgcn_channels[-1]

        # joint pooling -> temporal features (B,T,D)
        self.joint_pool = nn.AdaptiveAvgPool2d((None, 1))  # pool over J

        # signer encoder + bias-aware attention
        self.signer_enc = SignerEncoder(signer_stats_dim, signer_emb_dim)
        # Use FiLM-style multiplicative attention if requested, otherwise use additive concatenation
        if use_film_attention:
            self.attn = FiLMSignerAttention(feat_dim=feat_dim, signer_dim=signer_emb_dim, num_heads=attn_heads, attn_dropout=attn_dropout)
        else:
            self.attn = BiasAwareAttention(feat_dim=feat_dim, signer_dim=signer_emb_dim, num_heads=attn_heads, attn_dropout=attn_dropout)

        # temporal aggregator
        self.temporal = TemporalAggregator(in_dim=feat_dim, hidden=temporal_hidden, num_layers=1, bidirectional=True, dropout=dropout)
        agg_dim = self.temporal.out_dim

        # heads
        self.cls_head = ClassifierHead(agg_dim, num_classes=num_classes, dropout=dropout)
        self.use_signer_head = use_signer_head
        if use_signer_head:
            self.signer_head = SignerHead(agg_dim, num_signers=num_signers, lambda_grl=lambda_grl, dropout=dropout)

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        pose_stats: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        X: (B, T, J, C)
        A: (J, J)
        pose_stats: (B, S)
        """
        assert X.dim() == 4 and X.size(2) == self.J and X.size(3) == self.C_in, "Bad input shape"
        # ST-GCN
        H = self.backbone(X, A)  # (B, T, J, D)
        # pool joints -> (B, T, D)
        H_pooled = self.joint_pool(H.permute(0,3,1,2)).squeeze(-1).permute(0,2,1).contiguous()
        # signer embedding
        s = self.signer_enc(pose_stats)  # (B, signer_emb_dim)
        # bias-aware attention gated features
        H_att = self.attn(H_pooled, s)   # (B, T, D)
        # temporal aggregation
        Z = self.temporal(H_att)         # (B, D*)
        # classification
        logits = self.cls_head(Z)        # (B, num_classes)
        out = {"logits": logits, "embedding": Z}
        # optional signer adversarial head
        if self.use_signer_head:
            signer_logits = self.signer_head(Z)  # (B, num_signers)
            out["signer_logits"] = signer_logits
        if return_features:
            out.update({"H": H, "H_att": H_att, "signer_emb": s})
        return out


