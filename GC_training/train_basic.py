"""
Training script for Basic Model (baseline) on Google Cloud.

Usage:
    python train_basic.py \
        --gcs-data gs://your-bucket/segments \
        --output-dir ./runs/basic \
        --epochs 50 \
        --batch 64
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from datasets import NpzLandmarksDataset
from utils import build_hand_body_adjacency
from train_base import train_model, setup_gcs_path, collate_pad


def main():
    parser = argparse.ArgumentParser(description="Train basic ST-GCN model on Google Cloud")
    
    # Data arguments
    parser.add_argument('--gcs-data', type=str, required=True, 
                       help='GCS path to preprocessed segments (gs://bucket/path or local path)')
    parser.add_argument('--data', type=str, default=None,
                       help='Local path to landmarks folder (for groundtruth, if not using preprocessed)')
    parser.add_argument('--window', type=int, default=25, help='Temporal window size')
    parser.add_argument('--stride', type=int, default=1, help='Window stride')
    parser.add_argument('--coords', type=int, default=2, help='Number of coordinates')
    
    # Model arguments
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--num-signers', type=int, default=30, help='Number of signers (not used for basic)')
    parser.add_argument('--map-unknown-to-n', action='store_true', help='Map unknown labels (?) to not-signing (n)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch factor')
    parser.add_argument('--print-every', type=int, default=50, help='Print every N batches')
    parser.add_argument('--limit-train', type=int, default=None,
                        help='If set, randomly limit the training dataset to this many samples')
    parser.add_argument('--limit-val', type=int, default=None,
                        help='If set, randomly limit the validation dataset to this many samples')
    parser.add_argument('--train-segment-list', type=str, default=None,
                        help='Optional text file listing train segment NPZ paths to load')
    parser.add_argument('--val-segment-list', type=str, default=None,
                        help='Optional text file listing val segment NPZ paths to load')
    
    # Feature arguments
    parser.add_argument('--include-pose', action='store_true', help='Include pose landmarks')
    parser.add_argument('--include-hands', action='store_true', help='Include hand landmarks')
    parser.add_argument('--include-face', action='store_false', help='Include face landmarks')
    
    # Loss arguments
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--use-cross-entropy', action='store_true', help='Use cross-entropy instead of focal loss')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./runs/basic', help='Output directory for checkpoints')
    parser.add_argument('--gcs-output', type=str, default=None, 
                       help='GCS path to upload checkpoints (gs://bucket/path, optional)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Handle GCS paths
    data_path = setup_gcs_path(args.gcs_data)
    if args.data:
        data_path = setup_gcs_path(args.data)
    
    print(f"Data path: {data_path}")
    
    # Load datasets
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    
    # Load datasets - now supports GCS paths directly
    train_ds = NpzLandmarksDataset(
        root=data_path if not data_path.startswith('gs://') else '/tmp/gcs_data',
        gt_path=None,
        window=args.window,
        stride=args.stride,
        in_coords=args.coords,
        include_pose=args.include_pose,
        include_hands=args.include_hands,
        include_face=args.include_face,
        allowed_ids=None,
        map_unknown_to_n=args.map_unknown_to_n,
        num_classes=args.num_classes,
        preprocessed_dir=data_path,  # Pass GCS path directly - dataset will handle it
        split='train',
        segment_list_file=args.train_segment_list,
    )
    
    val_ds = NpzLandmarksDataset(
        root=data_path if not data_path.startswith('gs://') else '/tmp/gcs_data',
        gt_path=None,
        window=args.window,
        stride=args.stride,
        in_coords=args.coords,
        include_pose=args.include_pose,
        include_hands=args.include_hands,
        include_face=args.include_face,
        allowed_ids=None,
        map_unknown_to_n=args.map_unknown_to_n,
        num_classes=args.num_classes,
        preprocessed_dir=data_path,  # Pass GCS path directly - dataset will handle it
        split='val',
        segment_list_file=args.val_segment_list,
    )
    
    def maybe_limit(ds, limit, name):
        if limit is not None and len(ds) > limit:
            idx = torch.randperm(len(ds))[:limit].tolist()
            print(f"Limiting {name} dataset from {len(ds)} to {limit} samples")
            return Subset(ds, idx)
        return ds

    train_ds = maybe_limit(train_ds, args.limit_train, "train")
    val_ds = maybe_limit(val_ds, args.limit_val, "val")

    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")
    
    # Get sample dimensions
    sample_X, sample_stats, _, _ = train_ds[0]
    T, J, C = sample_X.shape
    print(f"Sample shapes -> X: {sample_X.shape}, stats: {sample_stats.shape}")
    
    # Build adjacency
    hand_edges = tuple((i, i+1) for i in range(0, max(0, J-1)))
    body_edges = tuple()
    A = build_hand_body_adjacency(J, hand_edges, body_edges).to(device)
    
    # Model configuration
    model_config = {
        'binary': False,
        'use_pseudo_signers': False,
        'use_supcon': False,
        'use_film_attention': False,
        'use_mi_minimization': False,
        'best_metric': 'acc',
    }
    
    # Train model
    train_model(
        model_name='basic',
        model_config=model_config,
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        args=args,
        A=A,
        sample_stats=sample_stats,
        T=T, J=J, C=C
    )


if __name__ == "__main__":
    main()

