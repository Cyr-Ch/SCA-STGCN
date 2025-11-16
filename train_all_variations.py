"""
Script to train all model variations efficiently by loading dataset once
and training all models in the same loop.

Usage:
    python train_all_variations.py --data /path/to/landmarks/folder [options]
"""

import os
import sys
import argparse
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset, random_split

# Absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from model import SignSTGCNModel
from datasets import NpzLandmarksDataset
from utils import build_hand_body_adjacency
from losses import classification_loss, signer_adversarial_loss, info_nce, CLUBEstimator, mutual_information_loss, mutual_information_loss_onehot, focal_loss

try:
    from sklearn.metrics import average_precision_score, f1_score
except Exception:
    average_precision_score = None
    f1_score = None


def kmeans_gpu(data: torch.Tensor, k: int, n_init: int = 10, max_iters: int = 300, 
                tol: float = 1e-4, random_state: int = 0, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated KMeans clustering using PyTorch.
    
    Args:
        data: (N, D) tensor of data points
        k: number of clusters
        n_init: number of initializations (best result is returned)
        max_iters: maximum iterations per initialization
        tol: tolerance for convergence
        random_state: random seed
        device: device to run on (uses data.device if None)
    
    Returns:
        centroids: (k, D) tensor of cluster centroids
        labels: (N,) tensor of cluster assignments
    """
    if device is None:
        device = data.device
    
    data = data.to(device)
    N, D = data.shape
    
    if N < k:
        raise ValueError(f"Number of samples ({N}) must be >= number of clusters ({k})")
    
    best_centroids = None
    best_labels = None
    best_inertia = float('inf')
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    for init_idx in range(n_init):
        # Initialize centroids randomly
        indices = torch.randperm(N, device=device)[:k]
        centroids = data[indices].clone()  # (k, D)
        
        prev_centroids = centroids.clone()
        
        for iteration in range(max_iters):
            # Compute distances: (N, k) = pairwise distances between all points and centroids
            distances = torch.cdist(data, centroids, p=2)  # (N, k) - GPU-accelerated
            
            # Assign to nearest centroid
            labels = torch.argmin(distances, dim=1)  # (N,)
            
            # Update centroids (vectorized for GPU efficiency)
            for i in range(k):
                mask = (labels == i)
                if mask.any():
                    centroids[i] = data[mask].mean(dim=0)
                else:
                    # If cluster is empty, reinitialize randomly
                    centroids[i] = data[torch.randint(0, N, (1,), device=device)].squeeze(0)
            
            # Check convergence
            centroid_shift = torch.cdist(centroids, prev_centroids, p=2).max().item()
            if centroid_shift < tol:
                break
            
            prev_centroids = centroids.clone()
        
        # Compute inertia (sum of squared distances to nearest centroid)
        final_distances = torch.cdist(data, centroids, p=2)  # (N, k)
        final_labels = torch.argmin(final_distances, dim=1)  # (N,)
        inertia = final_distances[torch.arange(N, device=device), final_labels].pow(2).sum().item()
        
        # Keep best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.clone()
            best_labels = final_labels.clone()
    
    return best_centroids, best_labels


def collate_pad(batch):
    """Collate function for DataLoader"""
    Xs, stats, ys, metas = zip(*batch)
    X = torch.stack(Xs, dim=0)
    stats = torch.stack(stats, dim=0)
    y = torch.stack(ys, dim=0)
    return X, stats, y, metas


class ModelConfig:
    """Configuration for a single model architecture"""
    def __init__(self, name, out_dir, **kwargs):
        self.name = name
        self.out_dir = out_dir
        self.binary = kwargs.get('binary', False)
        self.use_pseudo_signers = kwargs.get('use_pseudo_signers', False)
        self.use_supcon = kwargs.get('use_supcon', False)
        self.use_film_attention = kwargs.get('use_film_attention', False)
        self.use_mi_minimization = kwargs.get('use_mi_minimization', False)
        self.num_pseudo_signers = kwargs.get('num_pseudo_signers', 4)
        self.signer_loss_weight = kwargs.get('signer_loss_weight', 0.5)
        self.supcon_weight = kwargs.get('supcon_weight', 0.1)
        self.supcon_temp = kwargs.get('supcon_temp', 0.07)
        self.mi_weight = kwargs.get('mi_weight', 0.1)
        self.mi_hidden_dim = kwargs.get('mi_hidden_dim', 64)
        self.best_metric = kwargs.get('best_metric', 'acc')
        self.signing_labels = kwargs.get('signing_labels', 'sign,signing,gesture')
        
        # Will be set during initialization
        self.model = None
        self.optimizer = None
        self.mi_estimator = None
        self.best_val_metric = None
        self.pseudo_cluster_map = None
        self.pseudo_centroids = None
        
        # Pre-compute binary label mapping for efficiency
        if self.binary:
            self.sign_set = set([s.strip().lower() for s in self.signing_labels.split(',') if s.strip()])
        else:
            self.sign_set = None


def main():
    parser = argparse.ArgumentParser(
        description="Train all model variations efficiently in a single loop"
    )
    parser.add_argument('--data', required=True, help='Path to landmarks folder')
    parser.add_argument('--max-files', type=int, default=3, help='Max .npz files to use')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--window', type=int, default=25, help='Temporal window size')
    parser.add_argument('--stride', type=int, default=1, help='Window stride')
    parser.add_argument('--coords', type=int, default=2, help='Number of coordinates')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes (default: 3 = signing, speaking, other)')
    parser.add_argument('--num-signers', type=int, default=30, help='Number of signers')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers (0=single-threaded, recommended for Colab/memory-constrained environments)')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch factor (number of batches to prefetch per worker, reduce if memory constrained)')
    parser.add_argument('--base-out', type=str, default='runs/all_architectures', help='Base output directory')
    parser.add_argument('--include-pose', action='store_true', help='Include pose landmarks')
    parser.add_argument('--include-hands', action='store_true', help='Include hand landmarks')
    parser.add_argument('--include-face', action='store_false', help='Include face landmarks')
    parser.add_argument('--skip-basic', action='store_true', help='Skip basic training')
    parser.add_argument('--skip-binary', action='store_true', help='Skip binary classification')
    parser.add_argument('--skip-pseudo', action='store_true', help='Skip pseudo-signer clustering')
    parser.add_argument('--skip-supcon', action='store_true', help='Skip supervised contrastive')
    parser.add_argument('--skip-film', action='store_true', help='Skip FiLM attention')
    parser.add_argument('--skip-mi', action='store_true', help='Skip MI minimization')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--print-every', type=int, default=50, help='Print every N batches')
    parser.add_argument('--show-diagnostics', action='store_true', help='Show diagnostic label distribution checks (disabled by default)')
    # Focal loss options (used for all models)
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha parameter (balancing factor, default: 0.25)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma parameter (focusing parameter, default: 2.0)')
    parser.add_argument('--use-cross-entropy', action='store_true', help='Use standard cross-entropy loss instead of focal loss')
    # Label mapping options
    parser.add_argument('--map-unknown-to-n', action='store_true', help='Map unknown labels (?) to not-signing (n)')
    parser.add_argument('--preprocessed', type=str, default=None, help='Path to preprocessed segments NPZ file (created with create_segments.py)')
    parser.add_argument('--preprocessed-dir', type=str, default=None, help='Path to directory with train/val/test subdirectories containing per-video NPZ files')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Resolve data path
    if not os.path.isabs(args.data):
        resolved_path = os.path.abspath(os.path.expanduser(args.data))
        if os.path.exists(resolved_path):
            args.data = resolved_path
        else:
            alt_path = os.path.abspath(os.path.join(script_dir, args.data))
            if os.path.exists(alt_path):
                args.data = alt_path
            else:
                args.data = resolved_path
    
    print(f"Data path: {args.data}")
    
    # Check if using preprocessed data
    use_preprocessed_dir = args.preprocessed_dir is not None
    use_preprocessed_file = args.preprocessed is not None
    
    # Only load groundtruth and process files if NOT using preprocessed data
    if not use_preprocessed_dir and not use_preprocessed_file:
        # Load dataset once
        gt_path = os.path.join(args.data, 'groundtruth.txt') if os.path.exists(os.path.join(args.data, 'groundtruth.txt')) else os.path.join(args.data, 'groundtruth')
        
        # Limit files if specified
        allowed_ids = None
        if args.max_files:
            npz_files = sorted(Path(args.data).glob('*.npz'))
            if len(npz_files) == 0:
                print(f"[ERROR] No .npz files found in {args.data}")
                sys.exit(1)
            limited_files = npz_files[:args.max_files]
            allowed_ids = set([f.stem for f in limited_files])
            print(f"Using {len(allowed_ids)} files: {', '.join(list(allowed_ids)[:5])}...")
    else:
        # Not needed when using preprocessed data
        gt_path = None
        allowed_ids = None
    
    # Determine label mapping based on num_classes
    # Default: 3 classes (S=signing, P=speaking, n/?=other)
    # 2 classes: S=signing, everything else=other
    # 4 classes: S, P, n, ? all separate
    print(f"\nLabel mapping configuration:")
    print(f"  num_classes: {args.num_classes}")
    print(f"  map_unknown_to_n: {args.map_unknown_to_n}")
    
    if args.num_classes == 2:
        print("  Mapping: S -> 0 (signing), P/n/? -> 1 (other)")
    elif args.num_classes == 3:
        print("  Mapping: S -> 0 (signing), P -> 1 (speaking), n/? -> 2 (other)")
    elif args.num_classes == 4:
        print("  Mapping: S -> 0, P -> 1, n -> 2, ? -> 3")
    else:
        print(f"  Mapping: Dynamic (will map up to {args.num_classes} classes)")
    
    # Create dataset
    print("\n" + "="*70)
    print("Dataset Preprocessing")
    print("="*70)
    
    if use_preprocessed_dir:
        print(f"Using preprocessed segments from directory: {args.preprocessed_dir}")
        print("Creating train/val datasets from preprocessed splits...")
        
        # Create train and val datasets from preprocessed directory
        # Note: gt_path, allowed_ids, and other processing params are not needed
        # when loading from preprocessed_dir - segments are already computed
        print("Loading segments directly from train/val folders (no groundtruth processing needed)...")
        train_ds = NpzLandmarksDataset(
            root=args.data,
            gt_path=None,  # Not needed for preprocessed data
            window=args.window,  # Only used for compatibility check
            stride=args.stride,  # Only used for compatibility check
            in_coords=args.coords,  # Only used for compatibility check
            include_pose=args.include_pose,  # Only used for compatibility check
            include_hands=args.include_hands,  # Only used for compatibility check
            include_face=args.include_face,  # Only used for compatibility check
            allowed_ids=None,  # Not needed for preprocessed data
            map_unknown_to_n=args.map_unknown_to_n,
            num_classes=args.num_classes,
            preprocessed_dir=args.preprocessed_dir,
            split='train',
        )
        
        val_ds = NpzLandmarksDataset(
            root=args.data,
            gt_path=None,  # Not needed for preprocessed data
            window=args.window,  # Only used for compatibility check
            stride=args.stride,  # Only used for compatibility check
            in_coords=args.coords,  # Only used for compatibility check
            include_pose=args.include_pose,  # Only used for compatibility check
            include_hands=args.include_hands,  # Only used for compatibility check
            include_face=args.include_face,  # Only used for compatibility check
            allowed_ids=None,  # Not needed for preprocessed data
            map_unknown_to_n=args.map_unknown_to_n,
            num_classes=args.num_classes,
            preprocessed_dir=args.preprocessed_dir,
            split='val',
        )
        
        if len(train_ds) == 0:
            print("[ERROR] Train dataset is empty")
            sys.exit(1)
        if len(val_ds) == 0:
            print("[ERROR] Val dataset is empty")
            sys.exit(1)
        
        print(f"\n✓ Train dataset loaded: {len(train_ds)} samples")
        print(f"✓ Val dataset loaded: {len(val_ds)} samples")
        print("="*70)
        
        # Skip the random_split section below
        skip_split = True
    elif use_preprocessed_file:
        print(f"Using preprocessed segments from: {args.preprocessed}")
        print("Loading segments directly from preprocessed file (no groundtruth processing needed)...")
        ds = NpzLandmarksDataset(
            root=args.data,
            gt_path=None,  # Not needed for preprocessed data
            window=args.window,
            stride=args.stride,
            in_coords=args.coords,
            include_pose=args.include_pose,
            include_hands=args.include_hands,
            include_face=args.include_face,
            allowed_ids=None,  # Not needed for preprocessed data
            map_unknown_to_n=args.map_unknown_to_n,
            num_classes=args.num_classes,
            preprocessed_file=args.preprocessed,
        )
        
        if len(ds) == 0:
            print("[ERROR] Dataset is empty")
            sys.exit(1)
        
        print(f"\n✓ Dataset loaded: {len(ds)} samples")
        print("="*70)
        skip_split = False
    else:
        # On-the-fly processing: need groundtruth and file processing
        print(f"Loading dataset from: {args.data}")
        print(f"Processing segments on-the-fly (this may take time)...")
        print(f"Configuration:")
        print(f"  Window size: {args.window}")
        print(f"  Stride: {args.stride}")
        print(f"  Include pose: {args.include_pose}")
        print(f"  Include hands: {args.include_hands}")
        print(f"  Include face: {args.include_face}")
        print(f"  Coordinates: {args.coords}")
        
        ds = NpzLandmarksDataset(
            root=args.data,
            gt_path=gt_path,
            window=args.window,
            stride=args.stride,
            in_coords=args.coords,
            include_pose=args.include_pose,
            include_hands=args.include_hands,
            include_face=args.include_face,
            allowed_ids=allowed_ids,
            map_unknown_to_n=args.map_unknown_to_n,
            num_classes=args.num_classes,
        )
        
        if len(ds) == 0:
            print("[ERROR] Dataset is empty")
            sys.exit(1)
        
        print(f"\n✓ Dataset loaded: {len(ds)} samples")
        print("="*70)
        skip_split = False
    
    # DIAGNOSTIC: Check label distribution (optional, disabled by default)
    if args.show_diagnostics:
        print("\n" + "="*70)
        print("Dataset Label Diagnostics")
        print("="*70)
        label_counts = {}
        label_str_counts = {}
        for i in range(min(1000, len(ds))):  # Check first 1000 samples
            _, _, y, meta = ds[i]
            label_str = meta.get('label', 'unknown')
            label_str_counts[label_str] = label_str_counts.get(label_str, 0) + 1
            label_counts[y.item()] = label_counts.get(y.item(), 0) + 1
        
        print(f"Label string distribution (first {min(1000, len(ds))} samples):")
        for label_str, count in sorted(label_str_counts.items()):
            print(f"  '{label_str}': {count} samples")
        
        print(f"\nLabel integer distribution (first {min(1000, len(ds))} samples):")
        for label_int, count in sorted(label_counts.items()):
            print(f"  Class {label_int}: {count} samples")
        
        print(f"\nTotal unique label strings: {len(label_str_counts)}")
        print(f"Total unique label integers: {len(label_counts)}")
        
        if len(label_counts) == 1:
            print("\n[WARNING] Only one class found in dataset! This will result in 100% accuracy.")
            print("Possible causes:")
            print("  1. All samples have the same label in groundtruth")
            print("  2. Dataset is too small")
            print("  3. Label mapping issue")
        elif len(label_counts) < args.num_classes:
            print(f"\n[INFO] Found {len(label_counts)} classes, but --num-classes={args.num_classes}")
            print("This is OK - model will use {len(label_counts)} classes.")
        print("="*70 + "\n")
    
    # Limit samples and split dataset (only if not using preprocessed_dir)
    if not skip_split:
        # Limit samples if specified
        if args.max_samples and args.max_samples < len(ds):
            indices = list(range(min(args.max_samples, len(ds))))
            ds = Subset(ds, indices)
            print(f"Limited to {len(ds)} samples")
        
        # Split dataset (static/deterministic split with fixed seed)
        train_size = int(0.9 * len(ds))
        val_size = len(ds) - train_size
        split_generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=split_generator)
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)} (static split with seed=42)")
    
    # DIAGNOSTIC: Check train/val label distribution (optional, disabled by default)
    if args.show_diagnostics:
        print("\n" + "="*70)
        print("Train/Val Split Label Distribution")
        print("="*70)
        train_label_counts = {}
        val_label_counts = {}
        for i in range(min(500, len(train_ds))):
            _, _, y, meta = train_ds[i]
            label_str = meta.get('label', 'unknown')
            train_label_counts[label_str] = train_label_counts.get(label_str, 0) + 1
        for i in range(min(500, len(val_ds))):
            _, _, y, meta = val_ds[i]
            label_str = meta.get('label', 'unknown')
            val_label_counts[label_str] = val_label_counts.get(label_str, 0) + 1
        
        print("Train set labels:")
        for label_str, count in sorted(train_label_counts.items()):
            print(f"  '{label_str}': {count} samples")
        print("Val set labels:")
        for label_str, count in sorted(val_label_counts.items()):
            print(f"  '{label_str}': {count} samples")
        print("="*70 + "\n")
    
    # Create data loaders with pin_memory for faster CPU→GPU transfers
    # Adjust pin_memory based on batch size (large batches + pin_memory can use lots of RAM)
    # For very large batches, pin_memory benefits are minimal and memory cost is high
    use_pin_memory = torch.cuda.is_available() and args.batch <= 128
    
    print(f"\nCreating data loaders...")
    print(f"  Batch size: {args.batch}")
    print(f"  Workers: {args.workers}")
    print(f"  Pin memory: {use_pin_memory} ({'enabled' if use_pin_memory else 'disabled (large batch size)'})")
    if args.workers > 0:
        print(f"  Prefetch factor: {args.prefetch_factor}")
    
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, 
                          collate_fn=collate_pad, pin_memory=use_pin_memory, 
                          prefetch_factor=args.prefetch_factor if args.workers > 0 else None)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, 
                        collate_fn=collate_pad, pin_memory=use_pin_memory,
                        prefetch_factor=args.prefetch_factor if args.workers > 0 else None)
    print(f"  Train batches: {len(train_dl)}")
    print(f"  Val batches: {len(val_dl)}")
    
    # Get sample to infer dimensions
    sample_X, sample_stats, _, _ = train_ds[0]
    T, J, C = sample_X.shape
    print(f"Sample shapes -> X: {sample_X.shape}, stats: {sample_stats.shape}")
    
    # Verify label mapping (optional, disabled by default)
    if args.show_diagnostics:
        print("\nVerifying label mapping...")
        actual_classes = set()
        label_str_to_class = {}
        scan_samples = min(10000, len(train_ds))  # Scan up to 10k samples
        for i in range(scan_samples):
            _, _, y, meta = train_ds[i]
            actual_classes.add(y.item())
            label_str = meta.get('label', 'unknown')
            if label_str not in label_str_to_class:
                label_str_to_class[label_str] = y.item()
        
        actual_num_classes = len(actual_classes)
        print(f"Found {actual_num_classes} unique classes in dataset")
        print("Label to class mapping:")
        for label_str, class_id in sorted(label_str_to_class.items()):
            print(f"  '{label_str}' -> class {class_id}")
        
        # Verify num_classes matches
        if actual_num_classes != args.num_classes:
            print(f"\nWARNING: Dataset has {actual_num_classes} classes, but --num-classes={args.num_classes}")
            print(f"Using {args.num_classes} classes as specified (model will have {args.num_classes} outputs)")
        else:
            print(f"\n✓ num_classes={args.num_classes} matches dataset")
    
    # Build adjacency
    hand_edges = tuple((i, i+1) for i in range(0, max(0, J-1)))
    body_edges = tuple()
    A = build_hand_body_adjacency(J, hand_edges, body_edges).to(device)
    
    # Define all model configurations
    configs = []
    base_out = args.base_out
    
    if not args.skip_basic:
        configs.append(ModelConfig(
            "Basic Training",
            os.path.join(base_out, 'basic'),
        ))
    
    # COMMENTED OUT: Only training baseline for now
    # if not args.skip_binary:
    #     configs.append(ModelConfig(
    #         "Binary Classification",
    #         os.path.join(base_out, 'binary'),
    #         binary=True,
    #         best_metric='pr_auc',
    #     ))
    
    # if not args.skip_pseudo:
    #     configs.append(ModelConfig(
    #         "Pseudo-Signer Clustering",
    #         os.path.join(base_out, 'pseudo_signers'),
    #         use_pseudo_signers=True,
    #         num_pseudo_signers=4,
    #         signer_loss_weight=0.5,
    #     ))
    
    # if not args.skip_supcon:
    #     configs.append(ModelConfig(
    #         "Supervised Contrastive Learning",
    #         os.path.join(base_out, 'supcon'),
    #         use_supcon=True,
    #         supcon_weight=0.1,
    #         supcon_temp=0.07,
    #     ))
    
    # if not args.skip_film:
    #     configs.append(ModelConfig(
    #         "FiLM Attention",
    #         os.path.join(base_out, 'film_attention'),
    #         use_film_attention=True,
    #     ))
    
    # if not args.skip_mi:
    #     configs.append(ModelConfig(
    #         "MI Minimization",
    #         os.path.join(base_out, 'mi_minimization'),
    #         use_mi_minimization=True,
    #         mi_weight=0.1,
    #         mi_hidden_dim=64,
    #     ))
    
    # if not args.skip_film and not args.skip_pseudo:
    #     configs.append(ModelConfig(
    #         "FiLM Attention + Pseudo-Signers",
    #         os.path.join(base_out, 'film_pseudo'),
    #         use_film_attention=True,
    #         use_pseudo_signers=True,
    #         num_pseudo_signers=4,
    #         signer_loss_weight=0.5,
    #     ))
    
    # if not args.skip_supcon and not args.skip_film:
    #     configs.append(ModelConfig(
    #         "Supervised Contrastive + FiLM",
    #         os.path.join(base_out, 'supcon_film'),
    #         use_supcon=True,
    #         use_film_attention=True,
    #         supcon_weight=0.1,
    #         supcon_temp=0.07,
    #     ))
    
    if len(configs) == 0:
        print("[ERROR] No configurations selected")
        sys.exit(1)
    
    print(f"\nWill train {len(configs)} configurations:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.name}")
    
    # Print loss function info
    if args.use_cross_entropy:
        print(f"\nLoss function: Cross-Entropy Loss")
    else:
        print(f"\nLoss function: Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    
    # Pre-compute binary label mapping (required for binary models, not just statistics)
    if args.show_diagnostics:
        print("\nPre-computing binary label mappings...")
    for cfg in configs:
        if cfg.binary:
            # Pre-compute mapping from label strings to binary class IDs
            cfg.binary_label_map = {}
            all_labels = set()
            scan_samples = min(5000, len(train_ds))  # Scan samples to find all labels
            for i in range(scan_samples):
                _, _, _, meta = train_ds[i]
                label_str = meta.get('label', 'unknown')
                all_labels.add(label_str)
            
            # Build mapping: signing labels -> 1, others -> 0
            for label_str in all_labels:
                if str(label_str).lower() in cfg.sign_set:
                    cfg.binary_label_map[label_str] = 1
                else:
                    cfg.binary_label_map[label_str] = 0
            
            if args.show_diagnostics:
                print(f"  {cfg.name}: Pre-computed mapping for {len(cfg.binary_label_map)} labels")
                print(f"    Signing labels ({cfg.signing_labels}): {sum(1 for v in cfg.binary_label_map.values() if v == 1)}")
                print(f"    Other labels: {sum(1 for v in cfg.binary_label_map.values() if v == 0)}")
    
    # Initialize all models
    print("\nInitializing models...")
    for cfg in configs:
        os.makedirs(cfg.out_dir, exist_ok=True)
        
        num_signers = cfg.num_pseudo_signers if cfg.use_pseudo_signers else args.num_signers
        use_signer_head = cfg.use_pseudo_signers
        
        cfg.model = SignSTGCNModel(
            num_joints=J,
            in_coords=C,
            num_classes=(2 if cfg.binary else args.num_classes),
            num_signers=num_signers,
            signer_stats_dim=sample_stats.numel(),
            use_signer_head=use_signer_head,
            use_film_attention=cfg.use_film_attention,
        ).to(device)
        
        # Create MI estimator if needed
        if cfg.use_mi_minimization:
            with torch.no_grad():
                dummy_X = torch.randn(1, T, J, C, device=device)
                dummy_stats = torch.randn(1, sample_stats.numel(), device=device)
                dummy_out = cfg.model(dummy_X, A, dummy_stats, return_features=True)
                embedding_dim = dummy_out['embedding'].shape[-1]
                signer_emb_dim = dummy_out['signer_emb'].shape[-1]
            
            if cfg.use_pseudo_signers:
                cfg.mi_estimator = CLUBEstimator(x_dim=embedding_dim, y_dim=num_signers, hidden_dim=cfg.mi_hidden_dim).to(device)
            else:
                cfg.mi_estimator = CLUBEstimator(x_dim=embedding_dim, y_dim=signer_emb_dim, hidden_dim=cfg.mi_hidden_dim).to(device)
        
        # Create optimizer
        params = list(cfg.model.parameters())
        if cfg.mi_estimator is not None:
            params += list(cfg.mi_estimator.parameters())
        cfg.optimizer = torch.optim.Adam(params, lr=args.lr)
        
        # Initialize CSV logging
        csv_path = os.path.join(cfg.out_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch','train_loss','train_acc','val_loss','val_acc','val_pr_auc','val_f1'])
        
        if args.debug:
            total_params = sum(p.numel() for p in cfg.model.parameters())
            print(f"  {cfg.name}: {total_params:,} parameters")
    
    # Compute pseudo-signer clusters for configs that need them (GPU-accelerated)
    print("\nComputing pseudo-signer clusters (GPU-accelerated)...")
    for cfg in configs:
        if cfg.use_pseudo_signers:
            k = max(2, cfg.num_pseudo_signers)
            cache_file = os.path.join(cfg.out_dir, 'pseudo_signers_cache.pt')
            cache_meta_file = os.path.join(cfg.out_dir, 'pseudo_signers_cache_meta.json')
            
            # Check if cache exists and is valid
            cache_valid = False
            if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
                try:
                    with open(cache_meta_file, 'r') as f:
                        cache_meta = json.load(f)
                    
                    # Verify cache is valid (same dataset size, k, etc.)
                    if (cache_meta.get('num_samples') == len(train_ds) and
                        cache_meta.get('k') == k and
                        cache_meta.get('num_pseudo_signers') == cfg.num_pseudo_signers):
                        cache_valid = True
                        print(f"  {cfg.name}: Found valid cache, loading pseudo-signer clusters...")
                        
                        # Load from cache
                        cache_data = torch.load(cache_file, map_location=device)
                        cfg.pseudo_centroids = cache_data['centroids'].to(device)  # (k, D)
                        cfg.pseudo_cluster_map = cache_data['cluster_map']
                        
                        print(f"  {cfg.name}: Loaded {k} clusters from cache (centroids on {device})")
                    else:
                        print(f"  {cfg.name}: Cache invalid (dataset size or k changed), recomputing...")
                except Exception as e:
                    print(f"  {cfg.name}: Error loading cache: {e}, recomputing...")
            
            if not cache_valid:
                # Cache miss or invalid: compute clusters
                train_stats_list = []
                train_keys = []
                def sample_key(meta_rec):
                    return f"{meta_rec.get('video','')}::{int(meta_rec.get('start',0))}"
                
                # Collect stats on GPU - OPTIMIZED: use stats-only loading (skips X to save memory/I/O)
                print(f"  {cfg.name}: Collecting stats from {len(train_ds)} samples (stats-only, no segment data)...")
                with torch.no_grad():
                    # Iterate through dataset directly (not DataLoader) to use stats-only method
                    # This avoids loading full segment data (X) which we don't need for clustering
                    for i in range(len(train_ds)):
                        stats, meta = train_ds.get_stats_only(i)
                        # Keep stats on GPU
                        train_stats_list.append(stats.unsqueeze(0).to(device))  # (1, D) -> (1, D)
                        train_keys.append(sample_key(meta))
                        
                        # Progress indicator for large datasets
                        if (i + 1) % 10000 == 0:
                            print(f"    Collected stats from {i+1}/{len(train_ds)} samples...")
                
                # Concatenate on GPU
                train_stats = torch.cat(train_stats_list, dim=0)  # (N, D) - already on GPU
                
                print(f"  {cfg.name}: Running GPU KMeans on {train_stats.shape[0]} samples, {train_stats.shape[1]} features, k={k}...")
                centroids, train_clusters = kmeans_gpu(
                    train_stats, 
                    k=k, 
                    n_init=10, 
                    random_state=0, 
                    device=device
                )
                
                # Store centroids on GPU (no CPU conversion)
                cfg.pseudo_centroids = centroids  # (k, D) tensor on GPU
                cfg.pseudo_cluster_map = {key: int(cid.item()) for key, cid in zip(train_keys, train_clusters)}
                print(f"  {cfg.name}: {k} clusters computed (centroids on {device})")
                
                # Save to cache
                print(f"  {cfg.name}: Saving clusters to cache...")
                os.makedirs(cfg.out_dir, exist_ok=True)
                torch.save({
                    'centroids': centroids.cpu(),  # Save on CPU for portability
                    'cluster_map': cfg.pseudo_cluster_map,
                }, cache_file)
                
                # Save metadata
                with open(cache_meta_file, 'w') as f:
                    json.dump({
                        'num_samples': len(train_ds),
                        'k': k,
                        'num_pseudo_signers': cfg.num_pseudo_signers,
                        'stats_dim': train_stats.shape[1],
                    }, f, indent=2)
                
                print(f"  {cfg.name}: Cache saved to {cache_file}")
    
    # Training loop - all models in same loop
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")
    print(f"Training {len(configs)} models for {args.epochs} epochs")
    print(f"Total train batches per epoch: {len(train_dl)}")
    print(f"Total val batches per epoch: {len(val_dl)}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Training phase
        for cfg in configs:
            cfg.model.train()
        
        train_stats = {cfg.name: {'total': 0, 'correct': 0, 'loss_sum': 0.0} for cfg in configs}
        
        for bi, (X, stats, y, meta) in enumerate(train_dl, start=1):
            # Non-blocking transfer for faster CPU→GPU (works with pin_memory=True)
            # Note: X, stats, y are from DataLoader - they'll be released after loop iteration
            X, stats = X.to(device, non_blocking=True), stats.to(device, non_blocking=True)
            
            # Train each model on this batch
            for cfg in configs:
                # Prepare labels
                if cfg.binary:
                    # Use pre-computed binary label mapping (no string operations)
                    # Create tensor directly on GPU for better performance
                    y_batch = torch.tensor([cfg.binary_label_map.get(m.get('label', 'unknown'), 0) for m in meta], 
                                         device=device, dtype=torch.long)
                else:
                    y_batch = y.to(device, non_blocking=True)
                
                cfg.optimizer.zero_grad()
                
                # Forward pass
                need_features = cfg.use_supcon or cfg.use_mi_minimization
                out = cfg.model(X, A, stats, return_features=need_features)
                
                # Compute loss
                if args.use_cross_entropy:
                    loss = classification_loss(out['logits'], y_batch)
                else:
                    # Use focal loss for all models (handles class imbalance better)
                    loss = focal_loss(out['logits'], y_batch, alpha=args.focal_alpha, gamma=args.focal_gamma)
                
                # Add MI minimization loss
                if cfg.use_mi_minimization and cfg.mi_estimator is not None and 'embedding' in out:
                    z = out['embedding']
                    if cfg.use_pseudo_signers:
                        # GPU-accelerated: use nearest centroid instead of dictionary lookup
                        centroids = cfg.pseudo_centroids  # (K, D) - already on GPU
                        dists = torch.cdist(stats, centroids, p=2)  # (B, K) - GPU-accelerated
                        signer_ids = torch.argmin(dists, dim=1)  # (B,) - GPU operation
                        mi_loss = mutual_information_loss_onehot(z, signer_ids, cfg.num_pseudo_signers, cfg.mi_estimator)
                    else:
                        if 'signer_emb' in out:
                            s = out['signer_emb']
                            mi_loss = mutual_information_loss(z, s, cfg.mi_estimator)
                        else:
                            mi_loss = torch.tensor(0.0, device=device)
                    loss = loss + cfg.mi_weight * mi_loss
                
                # Add adversarial signer loss
                if cfg.use_pseudo_signers and 'signer_logits' in out and not cfg.use_mi_minimization:
                    # GPU-accelerated: use nearest centroid instead of dictionary lookup
                    centroids = cfg.pseudo_centroids  # (K, D) - already on GPU
                    dists = torch.cdist(stats, centroids, p=2)  # (B, K) - GPU-accelerated
                    signer_ids = torch.argmin(dists, dim=1)  # (B,) - GPU operation
                    loss = loss + cfg.signer_loss_weight * signer_adversarial_loss(out['signer_logits'], signer_ids)
                
                # Add supervised contrastive loss (GPU-optimized)
                if cfg.use_supcon and 'embedding' in out:
                    z = out['embedding']
                    B = z.size(0)
                    # GPU-optimized: find positive pairs without CPU transfer
                    # Initialize with self (fallback)
                    pos_idx = torch.arange(B, device=device, dtype=torch.long)
                    
                    # For each unique class, find positive pairs
                    unique_classes = torch.unique(y_batch)
                    for cls in unique_classes:
                        cls_mask = (y_batch == cls)
                        cls_indices = torch.where(cls_mask)[0]  # GPU operation
                        if len(cls_indices) > 1:
                            # Circular shift: each sample pairs with the next one in the same class
                            shifted = torch.roll(cls_indices, shifts=-1)
                            pos_idx[cls_indices] = shifted
                    
                    loss = loss + cfg.supcon_weight * info_nce(z, pos_idx, temperature=cfg.supcon_temp)
                
                # Backward pass
                loss.backward()
                cfg.optimizer.step()
                
                # Update stats
                train_stats[cfg.name]['loss_sum'] += float(loss.detach().cpu()) * y_batch.size(0)
                pred = out['logits'].argmax(dim=1)
                correct = (pred == y_batch).sum().item()
                train_stats[cfg.name]['correct'] += int(correct)
                train_stats[cfg.name]['total'] += y_batch.size(0)
                
                # DIAGNOSTIC: Print first batch details for debugging (before deleting out)
                if args.debug and bi == 1 and epoch == 1:
                    pred_debug = out['logits'].argmax(dim=1).clone()
                    print(f"\n[DEBUG] {cfg.name} - First batch:")
                    print(f"  Predictions: {pred_debug.cpu().tolist()}")
                    print(f"  Labels:      {y_batch.cpu().tolist()}")
                    print(f"  Correct:     {correct}/{y_batch.size(0)}")
                    print(f"  Unique predictions: {len(torch.unique(pred_debug))}")
                    print(f"  Unique labels: {len(torch.unique(y_batch))}")
                    del pred_debug
                
                # Explicitly free intermediate tensors to prevent memory accumulation
                del loss, pred
                # Free output dict items
                if 'z' in out:
                    del out['z']
                if 'signer_emb' in out:
                    del out['signer_emb']
                del out
            
            # Progress indicator (print every N batches or at start of epoch)
            if bi == 1 or bi % args.print_every == 0:
                print(f"  [Train] Batch {bi}/{len(train_dl)} ({100.0*bi/len(train_dl):.1f}%)")
            
            # Explicitly free batch tensors after processing (outside model loop)
            # Note: y_batch is defined inside the model loop, so we can't delete it here
            # But X, stats, y are shared across models, so delete after all models process them
            if bi % 50 == 0:  # More frequent GC for memory-constrained environments
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Validation phase
        for cfg in configs:
            cfg.model.eval()
        
        val_stats = {cfg.name: {'total': 0, 'correct': 0, 'loss_sum': 0.0, 'targets': [], 'scores': []} for cfg in configs}
        
        with torch.no_grad():
            for bi, (X, stats, y, meta) in enumerate(val_dl, start=1):
                # Non-blocking transfer for faster CPU→GPU (works with pin_memory=True)
                X, stats = X.to(device, non_blocking=True), stats.to(device, non_blocking=True)
                
                for cfg in configs:
                    # Prepare labels
                    if cfg.binary:
                        # Use pre-computed binary label mapping (no string operations)
                        # Create tensor directly on GPU for better performance
                        y_batch = torch.tensor([cfg.binary_label_map.get(m.get('label', 'unknown'), 0) for m in meta], 
                                             device=device, dtype=torch.long)
                    else:
                        y_batch = y.to(device, non_blocking=True)
                    
                    # Forward pass
                    need_features = cfg.use_supcon or cfg.use_mi_minimization
                    out = cfg.model(X, A, stats, return_features=need_features)
                    
                    # Compute loss
                    if args.use_cross_entropy:
                        loss = classification_loss(out['logits'], y_batch)
                    else:
                        # Use focal loss for all models (handles class imbalance better)
                        loss = focal_loss(out['logits'], y_batch, alpha=args.focal_alpha, gamma=args.focal_gamma)
                    
                    # Add MI minimization loss
                    if cfg.use_mi_minimization and cfg.mi_estimator is not None and 'embedding' in out:
                        z = out['embedding']
                        if cfg.use_pseudo_signers:
                            # Centroids already on GPU (no conversion needed)
                            centroids = cfg.pseudo_centroids  # (K, D) - already on GPU
                            # Compute distances: (B, K) = sum over D of (stats - centroids)^2
                            dists = torch.cdist(stats, centroids, p=2)  # (B, K) - GPU-accelerated
                            signer_ids = torch.argmin(dists, dim=1)  # (B,) - GPU operation
                            mi_loss = mutual_information_loss_onehot(z, signer_ids, cfg.num_pseudo_signers, cfg.mi_estimator)
                        else:
                            if 'signer_emb' in out:
                                s = out['signer_emb']
                                mi_loss = mutual_information_loss(z, s, cfg.mi_estimator)
                            else:
                                mi_loss = torch.tensor(0.0, device=device)
                        loss = loss + cfg.mi_weight * mi_loss
                    
                    # Add adversarial signer loss
                    if cfg.use_pseudo_signers and 'signer_logits' in out and not cfg.use_mi_minimization:
                        # Centroids already on GPU (no conversion needed)
                        centroids = cfg.pseudo_centroids  # (K, D) - already on GPU
                        dists = torch.cdist(stats, centroids, p=2)  # (B, K) - GPU-accelerated
                        signer_ids = torch.argmin(dists, dim=1)  # (B,) - GPU operation
                        loss = loss + cfg.signer_loss_weight * signer_adversarial_loss(out['signer_logits'], signer_ids)
                    
                    # Add supervised contrastive loss (GPU-optimized)
                    if cfg.use_supcon and 'embedding' in out:
                        z = out['embedding']
                        B = z.size(0)
                        # GPU-optimized: find positive pairs without CPU transfer
                        # Initialize with self (fallback)
                        pos_idx = torch.arange(B, device=device, dtype=torch.long)
                        
                        # For each unique class, find positive pairs
                        unique_classes = torch.unique(y_batch)
                        for cls in unique_classes:
                            cls_mask = (y_batch == cls)
                            cls_indices = torch.where(cls_mask)[0]  # GPU operation
                            if len(cls_indices) > 1:
                                # Circular shift: each sample pairs with the next one in the same class
                                shifted = torch.roll(cls_indices, shifts=-1)
                                pos_idx[cls_indices] = shifted
                        
                        loss = loss + cfg.supcon_weight * info_nce(z, pos_idx, temperature=cfg.supcon_temp)
                    
                    # Update stats
                    val_stats[cfg.name]['loss_sum'] += float(loss.detach().cpu()) * y_batch.size(0)
                    pred = out['logits'].argmax(dim=1)
                    val_stats[cfg.name]['correct'] += int((pred == y_batch).sum().item())
                    val_stats[cfg.name]['total'] += y_batch.size(0)
                    
                    # Collect scores for binary metrics
                    if cfg.binary:
                        probs = torch.softmax(out['logits'], dim=1)[:, 1].detach().cpu().numpy()
                        val_stats[cfg.name]['scores'].extend(probs.tolist())
                        val_stats[cfg.name]['targets'].extend(y_batch.detach().cpu().tolist())
        
        # Compute metrics and save checkpoints
        for cfg in configs:
            train_acc = 100.0 * train_stats[cfg.name]['correct'] / max(1, train_stats[cfg.name]['total'])
            train_loss = train_stats[cfg.name]['loss_sum'] / max(1, train_stats[cfg.name]['total'])
            val_acc = 100.0 * val_stats[cfg.name]['correct'] / max(1, val_stats[cfg.name]['total'])
            val_loss = val_stats[cfg.name]['loss_sum'] / max(1, val_stats[cfg.name]['total'])
            
            # Compute PR-AUC and F1 for binary
            val_pr_auc = float('nan')
            val_f1 = float('nan')
            if cfg.binary and average_precision_score and f1_score and len(val_stats[cfg.name]['targets']) > 0:
                try:
                    val_pr_auc = float(average_precision_score(val_stats[cfg.name]['targets'], val_stats[cfg.name]['scores']))
                    preds_05 = [1 if s >= 0.5 else 0 for s in val_stats[cfg.name]['scores']]
                    val_f1 = float(f1_score(val_stats[cfg.name]['targets'], preds_05))
                except Exception:
                    pass
            
            # Choose metric based on config
            metric_map = {'acc': val_acc, 'loss': val_loss, 'pr_auc': val_pr_auc, 'f1': val_f1}
            current_metric = metric_map.get(cfg.best_metric, val_acc)
            
            # Save checkpoint
            last_ckpt = os.path.join(cfg.out_dir, 'last.pt')
            torch.save({
                'model': cfg.model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, last_ckpt)
            
            # Save best checkpoint
            is_better = False
            if cfg.best_metric == 'loss':
                is_better = (cfg.best_val_metric is None) or (current_metric < cfg.best_val_metric)
            else:
                is_better = (cfg.best_val_metric is None) or (current_metric > cfg.best_val_metric)
            
            if is_better:
                cfg.best_val_metric = current_metric
                best_ckpt = os.path.join(cfg.out_dir, 'best.pt')
                torch.save({
                    'model': cfg.model.state_dict(),
                    'epoch': epoch,
                    'metric': current_metric,
                    'metric_name': cfg.best_metric
                }, best_ckpt)
            
            # Log to CSV
            csv_path = os.path.join(cfg.out_dir, 'metrics.csv')
            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.2f}", f"{val_loss:.6f}", f"{val_acc:.2f}", f"{val_pr_auc:.6f}", f"{val_f1:.6f}"])
            
            # Print metrics
            print(f"\n{cfg.name}:")
            print(f"  Train: loss={train_loss:.4f} acc={train_acc:.2f}%")
            print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.2f}% pr_auc={val_pr_auc:.4f} f1={val_f1:.4f}")
            if is_better:
                print(f"  [BEST] {cfg.best_metric}={current_metric:.4f}")
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"Results saved to: {args.base_out}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
