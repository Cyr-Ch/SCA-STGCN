# sign_gcn_model.py
# PyTorch implementation for real-time sign language detection with:
# - ST-GCN backbone (spatio-temporal graph convs)
# - Temporal aggregator (BiGRU by default)
# - Bias-aware attention (conditioned on signer embedding)
# - Optional domain-adversarial signer head (via Gradient Reversal Layer)
# - Optional contrastive loss helpers (InfoNCE)
# Author: (you)

import torch
from typing import Tuple

from .model import SignSTGCNModel
from .utils import build_hand_body_adjacency
from .losses import classification_loss, signer_adversarial_loss, info_nce, attention_regularization


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hypothetical skeleton (e.g., 33 joints incl. hands subset). Replace with your mapping.
    J = 33
    C_in = 2
    B, T = 8, 32
    NUM_CLASSES = 120

    # Dummy edges (replace with your actual graph: MediaPipe/OpenPose skeleton + detailed hands)
    hand_edges = tuple((i, i+1) for i in range(0, 20) if i+1 < 21)
    body_edges = ((21,22),(22,23),(11,12),(12,13),(13,14),(11,15),(15,16),(16,17))

    A = build_hand_body_adjacency(J, hand_edges, body_edges).to(device)

    model = SignSTGCNModel(
        num_joints=J,
        in_coords=C_in,
        stgcn_channels=(64,128,128),
        stgcn_kernel=3,
        stgcn_dilations=(1,2,3),
        temporal_hidden=256,
        num_classes=NUM_CLASSES,
        signer_stats_dim=16,
        signer_emb_dim=64,
        use_signer_head=True,
        num_signers=30,
        lambda_grl=0.5,
        attn_heads=4,
        dropout=0.1
    ).to(device)

    X = torch.randn(B, T, J, C_in, device=device)
    pose_stats = torch.randn(B, 16, device=device)
    targets = torch.randint(0, NUM_CLASSES, (B,), device=device)
    signer_ids = torch.randint(0, 30, (B,), device=device)

    out = model(X, A, pose_stats, return_features=True)
    logits = out["logits"]
    cls_loss = classification_loss(logits, targets)

    total_loss = cls_loss
    if "signer_logits" in out:
        dom_loss = signer_adversarial_loss(out["signer_logits"], signer_ids)
        total_loss = total_loss + 0.5 * dom_loss

    att_reg = attention_regularization(out["H_att"], sparsity_weight=0.0, smooth_weight=0.0)
    total_loss = total_loss + att_reg

    total_loss.backward()
    print("Forward/backward OK. Loss:", float(total_loss.detach().cpu()))
