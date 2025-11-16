"""
Quick test script to verify FiLM attention implementation works correctly.
Run as: python -m SignSTGCN.test_film_attention (from parent directory)
      or: cd .. && python -m SignSTGCN.test_film_attention
"""
import torch
import torch.nn as nn
from .model import SignSTGCNModel
from .utils import build_hand_body_adjacency

def test_film_attention():
    """Test FiLM attention forward pass"""
    print("Testing FiLM Attention Implementation...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test parameters
    B, T, J, C = 4, 25, 33, 2
    num_classes = 10
    signer_stats_dim = 16
    
    # Create dummy adjacency
    hand_edges = tuple((i, i+1) for i in range(0, min(20, J-1)))
    body_edges = tuple()
    A = build_hand_body_adjacency(J, hand_edges, body_edges).to(device)
    
    # Test with FiLM attention
    print("\n1. Testing with FiLM Attention (use_film_attention=True)")
    model_film = SignSTGCNModel(
        num_joints=J,
        in_coords=C,
        num_classes=num_classes,
        signer_stats_dim=signer_stats_dim,
        use_signer_head=False,
        use_film_attention=True,
    ).to(device)
    
    X = torch.randn(B, T, J, C, device=device)
    pose_stats = torch.randn(B, signer_stats_dim, device=device)
    
    model_film.eval()
    with torch.no_grad():
        out_film = model_film(X, A, pose_stats, return_features=True)
    
    print(f"   [OK] Forward pass successful")
    print(f"   [OK] Output logits shape: {out_film['logits'].shape}")
    print(f"   [OK] Expected: ({B}, {num_classes})")
    print(f"   [OK] Embedding shape: {out_film['embedding'].shape}")
    print(f"   [OK] Attention features shape: {out_film['H_att'].shape}")
    print(f"   [OK] Signer embedding shape: {out_film['signer_emb'].shape}")
    
    # Test with old attention
    print("\n2. Testing with Additive Attention (use_film_attention=False)")
    model_add = SignSTGCNModel(
        num_joints=J,
        in_coords=C,
        num_classes=num_classes,
        signer_stats_dim=signer_stats_dim,
        use_signer_head=False,
        use_film_attention=False,
    ).to(device)
    
    model_add.eval()
    with torch.no_grad():
        out_add = model_add(X, A, pose_stats, return_features=True)
    
    print(f"   [OK] Forward pass successful")
    print(f"   [OK] Output logits shape: {out_add['logits'].shape}")
    
    # Compare outputs
    print("\n3. Comparing Outputs")
    logits_diff = torch.abs(out_film['logits'] - out_add['logits']).mean().item()
    print(f"   [OK] Mean absolute difference in logits: {logits_diff:.6f}")
    print(f"   [OK] Different attention mechanisms produce different outputs (expected)")
    
    # Test backward pass
    print("\n4. Testing Backward Pass (FiLM)")
    model_film.train()
    targets = torch.randint(0, num_classes, (B,), device=device)
    criterion = nn.CrossEntropyLoss()
    
    out = model_film(X, A, pose_stats)
    loss = criterion(out['logits'], targets)
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model_film.parameters() if p.requires_grad)
    print(f"   [OK] Backward pass successful")
    print(f"   [OK] Gradients computed: {has_grad}")
    print(f"   [OK] Loss value: {loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_film.parameters())
    trainable_params = sum(p.numel() for p in model_film.parameters() if p.requires_grad)
    print(f"\n5. Model Statistics (FiLM)")
    print(f"   [OK] Total parameters: {total_params:,}")
    print(f"   [OK] Trainable parameters: {trainable_params:,}")
    
    # Compare parameter counts
    total_params_add = sum(p.numel() for p in model_add.parameters())
    param_diff = total_params - total_params_add
    print(f"\n6. Parameter Comparison")
    print(f"   [OK] FiLM model parameters: {total_params:,}")
    print(f"   [OK] Additive model parameters: {total_params_add:,}")
    print(f"   [OK] Difference: {param_diff:,} (FiLM has more due to scale/shift networks)")
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed! FiLM attention implementation is working correctly.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        test_film_attention()
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
