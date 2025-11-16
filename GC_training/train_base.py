"""
Base training script for Google Cloud training.
Supports loading data from GCS buckets and training a single model.

This is a base class that individual model scripts can use.
"""

import os
import sys
import argparse
import csv
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset, random_split

# Absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import SignSTGCNModel
from datasets import NpzLandmarksDataset
from utils import build_hand_body_adjacency
from losses import (
    classification_loss, signer_adversarial_loss, info_nce, 
    CLUBEstimator, mutual_information_loss, mutual_information_loss_onehot, focal_loss
)

try:
    from sklearn.metrics import average_precision_score, f1_score
except Exception:
    average_precision_score = None
    f1_score = None


def setup_gcs_path(path: str, temp_dir: str = '/tmp/gcs_cache') -> str:
    """
    Handle GCS bucket paths (gs://bucket/path) by downloading to local temp directory.
    Returns local path that can be used normally.
    
    Args:
        path: Path that may be gs://bucket/path or local path
        temp_dir: Local directory to cache GCS files
    
    Returns:
        Local path to use
    """
    if path.startswith('gs://'):
        try:
            import gcsfs
            # Parse GCS path
            path_parts = path[5:].split('/', 1)  # Remove 'gs://' prefix
            bucket = path_parts[0]
            gcs_path = path_parts[1] if len(path_parts) > 1 else ''
            
            # Create local cache directory
            os.makedirs(temp_dir, exist_ok=True)
            local_path = os.path.join(temp_dir, bucket, gcs_path)
            
            # Check if already cached
            if os.path.exists(local_path):
                print(f"Using cached GCS path: {local_path}")
                return local_path
            
            # Download from GCS (for directories, this is more complex - we'll handle it in dataset)
            print(f"GCS path detected: {path}")
            print(f"Note: Dataset will load files directly from GCS during training")
            print(f"Cache directory: {temp_dir}")
            
            # For now, return the GCS path - dataset will handle it
            # In production, you might want to download all files first
            return path
        except ImportError:
            print("[WARNING] gcsfs not installed. Install with: pip install gcsfs")
            print("Falling back to treating GCS path as local (may fail)")
            return path
    return path


def kmeans_gpu(data: torch.Tensor, k: int, n_init: int = 10, max_iters: int = 300, 
                tol: float = 1e-4, random_state: int = 0, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated KMeans clustering using PyTorch."""
    if device is None:
        device = data.device
    
    data = data.to(device)
    N, D = data.shape
    
    if N < k:
        raise ValueError(f"Number of samples ({N}) must be >= number of clusters ({k})")
    
    best_centroids = None
    best_labels = None
    best_inertia = float('inf')
    
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    for init_idx in range(n_init):
        indices = torch.randperm(N, device=device)[:k]
        centroids = data[indices].clone()
        prev_centroids = centroids.clone()
        
        for iteration in range(max_iters):
            distances = torch.cdist(data, centroids, p=2)
            labels = torch.argmin(distances, dim=1)
            
            for i in range(k):
                mask = (labels == i)
                if mask.any():
                    centroids[i] = data[mask].mean(dim=0)
                else:
                    centroids[i] = data[torch.randint(0, N, (1,), device=device)].squeeze(0)
            
            centroid_shift = torch.cdist(centroids, prev_centroids, p=2).max().item()
            if centroid_shift < tol:
                break
            prev_centroids = centroids.clone()
        
        final_distances = torch.cdist(data, centroids, p=2)
        final_labels = torch.argmin(final_distances, dim=1)
        inertia = final_distances[torch.arange(N, device=device), final_labels].pow(2).sum().item()
        
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


def save_checkpoint_to_gcs(checkpoint_path: str, checkpoint_data: dict, gcs_output: str = None):
    """Save checkpoint locally, and optionally upload to GCS."""
    # Save locally first
    torch.save(checkpoint_data, checkpoint_path)
    
    # Upload to GCS if specified
    if gcs_output and checkpoint_path.startswith('/') and os.path.exists(checkpoint_path):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            
            # Parse GCS path
            if gcs_output.startswith('gs://'):
                gcs_path = gcs_output[5:]  # Remove 'gs://'
            else:
                gcs_path = gcs_output
            
            # Upload file
            gcs_checkpoint_path = f"{gcs_path}/{os.path.basename(checkpoint_path)}"
            fs.put(checkpoint_path, gcs_checkpoint_path)
            print(f"  Uploaded checkpoint to gs://{gcs_checkpoint_path}")
        except ImportError:
            print(f"  [WARNING] gcsfs not installed, skipping GCS upload")
        except Exception as e:
            print(f"  [WARNING] Failed to upload to GCS: {e}")


def train_model(
    model_name: str,
    model_config: dict,
    train_ds,
    val_ds,
    device,
    args,
    A,
    sample_stats,
    T, J, C
):
    """
    Train a single model configuration.
    
    Args:
        model_name: Name of the model (for logging)
        model_config: Dictionary with model configuration:
            - binary: bool
            - use_pseudo_signers: bool
            - use_supcon: bool
            - use_film_attention: bool
            - use_mi_minimization: bool
            - num_pseudo_signers: int
            - signer_loss_weight: float
            - supcon_weight: float
            - supcon_temp: float
            - mi_weight: float
            - mi_hidden_dim: int
            - best_metric: str
        train_ds, val_ds: Training and validation datasets
        device: torch device
        args: Command line arguments
        A: Adjacency matrix
        sample_stats: Sample stats tensor for dimension inference
        T, J, C: Temporal, joint, coordinate dimensions
    """
    # Create output directory
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize model
    num_signers = model_config.get('num_pseudo_signers', args.num_signers) if model_config.get('use_pseudo_signers', False) else args.num_signers
    use_signer_head = model_config.get('use_pseudo_signers', False)
    
    model = SignSTGCNModel(
        num_joints=J,
        in_coords=C,
        num_classes=(2 if model_config.get('binary', False) else args.num_classes),
        num_signers=num_signers,
        signer_stats_dim=sample_stats.numel(),
        use_signer_head=use_signer_head,
        use_film_attention=model_config.get('use_film_attention', False),
    ).to(device)
    
    # Create MI estimator if needed
    mi_estimator = None
    if model_config.get('use_mi_minimization', False):
        with torch.no_grad():
            dummy_X = torch.randn(1, T, J, C, device=device)
            dummy_stats = torch.randn(1, sample_stats.numel(), device=device)
            dummy_out = model(dummy_X, A, dummy_stats, return_features=True)
            embedding_dim = dummy_out['embedding'].shape[-1]
            signer_emb_dim = dummy_out['signer_emb'].shape[-1]
        
        if model_config.get('use_pseudo_signers', False):
            mi_estimator = CLUBEstimator(x_dim=embedding_dim, y_dim=num_signers, hidden_dim=model_config.get('mi_hidden_dim', 64)).to(device)
        else:
            mi_estimator = CLUBEstimator(x_dim=embedding_dim, y_dim=signer_emb_dim, hidden_dim=model_config.get('mi_hidden_dim', 64)).to(device)
    
    # Create optimizer
    params = list(model.parameters())
    if mi_estimator is not None:
        params += list(mi_estimator.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # Initialize CSV logging
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch','train_loss','train_acc','val_loss','val_acc','val_pr_auc','val_f1'])
    
    # Pre-compute binary label mapping if needed
    binary_label_map = {}
    if model_config.get('binary', False):
        signing_labels = set(['s', 'sign', 'signing', 'gesture'])
        scan_samples = min(5000, len(train_ds))
        all_labels = set()
        for i in range(scan_samples):
            _, _, _, meta = train_ds[i]
            label_str = meta.get('label', 'unknown')
            all_labels.add(label_str)
        
        for label_str in all_labels:
            if str(label_str).lower() in signing_labels:
                binary_label_map[label_str] = 1
            else:
                binary_label_map[label_str] = 0
    
    # Compute pseudo-signer clusters if needed
    pseudo_centroids = None
    if model_config.get('use_pseudo_signers', False):
        cache_file = os.path.join(out_dir, 'pseudo_signers_cache.pt')
        cache_meta_file = os.path.join(out_dir, 'pseudo_signers_cache_meta.json')
        
        cache_valid = False
        if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
            try:
                with open(cache_meta_file, 'r') as f:
                    cache_meta = json.load(f)
                
                k = max(2, model_config.get('num_pseudo_signers', 4))
                if (cache_meta.get('num_samples') == len(train_ds) and
                    cache_meta.get('k') == k and
                    cache_meta.get('num_pseudo_signers') == model_config.get('num_pseudo_signers', 4)):
                    cache_valid = True
                    print(f"  Loading pseudo-signer clusters from cache...")
                    cache_data = torch.load(cache_file, map_location=device)
                    pseudo_centroids = cache_data['centroids'].to(device)
            except Exception as e:
                print(f"  Error loading cache: {e}, recomputing...")
        
        if not cache_valid:
            print(f"  Computing pseudo-signer clusters...")
            train_stats_list = []
            for i in range(len(train_ds)):
                stats, _ = train_ds.get_stats_only(i)
                train_stats_list.append(stats.unsqueeze(0).to(device))
                if (i + 1) % 10000 == 0:
                    print(f"    Collected stats from {i+1}/{len(train_ds)} samples...")
            
            train_stats = torch.cat(train_stats_list, dim=0)
            k = max(2, model_config.get('num_pseudo_signers', 4))
            centroids, _ = kmeans_gpu(train_stats, k=k, n_init=10, random_state=0, device=device)
            pseudo_centroids = centroids
            
            # Save cache
            torch.save({'centroids': centroids.cpu()}, cache_file)
            with open(cache_meta_file, 'w') as f:
                json.dump({
                    'num_samples': len(train_ds),
                    'k': k,
                    'num_pseudo_signers': model_config.get('num_pseudo_signers', 4),
                }, f)
    
    # Create data loaders
    use_pin_memory = torch.cuda.is_available() and args.batch <= 128
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          collate_fn=collate_pad, pin_memory=use_pin_memory,
                          prefetch_factor=args.prefetch_factor if args.workers > 0 else None)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_pad, pin_memory=use_pin_memory,
                        prefetch_factor=args.prefetch_factor if args.workers > 0 else None)
    
    best_val_metric = None
    best_metric_name = model_config.get('best_metric', 'acc')
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    print(f"Training for {args.epochs} epochs")
    print(f"Train batches per epoch: {len(train_dl)}")
    print(f"Val batches per epoch: {len(val_dl)}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Training phase
        model.train()
        train_total = 0
        train_correct = 0
        train_loss_sum = 0.0
        
        for bi, (X, stats, y, meta) in enumerate(train_dl, start=1):
            X, stats = X.to(device, non_blocking=True), stats.to(device, non_blocking=True)
            
            # Prepare labels
            if model_config.get('binary', False):
                y_batch = torch.tensor([binary_label_map.get(m.get('label', 'unknown'), 0) for m in meta],
                                     device=device, dtype=torch.long)
            else:
                y_batch = y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            need_features = model_config.get('use_supcon', False) or model_config.get('use_mi_minimization', False)
            out = model(X, A, stats, return_features=need_features)
            
            # Compute loss
            if args.use_cross_entropy:
                loss = classification_loss(out['logits'], y_batch)
            else:
                loss = focal_loss(out['logits'], y_batch, alpha=args.focal_alpha, gamma=args.focal_gamma)
            
            # Add MI minimization loss
            if model_config.get('use_mi_minimization', False) and mi_estimator is not None and 'embedding' in out:
                z = out['embedding']
                if model_config.get('use_pseudo_signers', False):
                    dists = torch.cdist(stats, pseudo_centroids, p=2)
                    signer_ids = torch.argmin(dists, dim=1)
                    mi_loss = mutual_information_loss_onehot(z, signer_ids, model_config.get('num_pseudo_signers', 4), mi_estimator)
                else:
                    if 'signer_emb' in out:
                        s = out['signer_emb']
                        mi_loss = mutual_information_loss(z, s, mi_estimator)
                    else:
                        mi_loss = torch.tensor(0.0, device=device)
                loss = loss + model_config.get('mi_weight', 0.1) * mi_loss
            
            # Add adversarial signer loss
            if model_config.get('use_pseudo_signers', False) and 'signer_logits' in out and not model_config.get('use_mi_minimization', False):
                dists = torch.cdist(stats, pseudo_centroids, p=2)
                signer_ids = torch.argmin(dists, dim=1)
                loss = loss + model_config.get('signer_loss_weight', 0.5) * signer_adversarial_loss(out['signer_logits'], signer_ids)
            
            # Add supervised contrastive loss
            if model_config.get('use_supcon', False) and 'embedding' in out:
                z = out['embedding']
                B = z.size(0)
                pos_idx = torch.arange(B, device=device, dtype=torch.long)
                unique_classes = torch.unique(y_batch)
                for cls in unique_classes:
                    cls_mask = (y_batch == cls)
                    cls_indices = torch.where(cls_mask)[0]
                    if len(cls_indices) > 1:
                        shifted = torch.roll(cls_indices, shifts=-1)
                        pos_idx[cls_indices] = shifted
                loss = loss + model_config.get('supcon_weight', 0.1) * info_nce(z, pos_idx, temperature=model_config.get('supcon_temp', 0.07))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update stats
            train_loss_sum += float(loss.detach().cpu()) * y_batch.size(0)
            pred = out['logits'].argmax(dim=1)
            train_correct += int((pred == y_batch).sum().item())
            train_total += y_batch.size(0)
            
            # Cleanup
            del loss, pred, out
            if bi % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if bi == 1 or bi % args.print_every == 0:
                print(f"  [Train] Batch {bi}/{len(train_dl)} ({100.0*bi/len(train_dl):.1f}%)")
        
        # Validation phase
        model.eval()
        val_total = 0
        val_correct = 0
        val_loss_sum = 0.0
        val_targets = []
        val_scores = []
        
        with torch.no_grad():
            for bi, (X, stats, y, meta) in enumerate(val_dl, start=1):
                X, stats = X.to(device, non_blocking=True), stats.to(device, non_blocking=True)
                
                # Prepare labels
                if model_config.get('binary', False):
                    y_batch = torch.tensor([binary_label_map.get(m.get('label', 'unknown'), 0) for m in meta],
                                         device=device, dtype=torch.long)
                else:
                    y_batch = y.to(device, non_blocking=True)
                
                # Forward pass
                need_features = model_config.get('use_supcon', False) or model_config.get('use_mi_minimization', False)
                out = model(X, A, stats, return_features=need_features)
                
                # Compute loss
                if args.use_cross_entropy:
                    loss = classification_loss(out['logits'], y_batch)
                else:
                    loss = focal_loss(out['logits'], y_batch, alpha=args.focal_alpha, gamma=args.focal_gamma)
                
                # Add MI minimization loss
                if model_config.get('use_mi_minimization', False) and mi_estimator is not None and 'embedding' in out:
                    z = out['embedding']
                    if model_config.get('use_pseudo_signers', False):
                        dists = torch.cdist(stats, pseudo_centroids, p=2)
                        signer_ids = torch.argmin(dists, dim=1)
                        mi_loss = mutual_information_loss_onehot(z, signer_ids, model_config.get('num_pseudo_signers', 4), mi_estimator)
                    else:
                        if 'signer_emb' in out:
                            s = out['signer_emb']
                            mi_loss = mutual_information_loss(z, s, mi_estimator)
                        else:
                            mi_loss = torch.tensor(0.0, device=device)
                    loss = loss + model_config.get('mi_weight', 0.1) * mi_loss
                
                # Add adversarial signer loss
                if model_config.get('use_pseudo_signers', False) and 'signer_logits' in out and not model_config.get('use_mi_minimization', False):
                    dists = torch.cdist(stats, pseudo_centroids, p=2)
                    signer_ids = torch.argmin(dists, dim=1)
                    loss = loss + model_config.get('signer_loss_weight', 0.5) * signer_adversarial_loss(out['signer_logits'], signer_ids)
                
                # Add supervised contrastive loss
                if model_config.get('use_supcon', False) and 'embedding' in out:
                    z = out['embedding']
                    B = z.size(0)
                    pos_idx = torch.arange(B, device=device, dtype=torch.long)
                    unique_classes = torch.unique(y_batch)
                    for cls in unique_classes:
                        cls_mask = (y_batch == cls)
                        cls_indices = torch.where(cls_mask)[0]
                        if len(cls_indices) > 1:
                            shifted = torch.roll(cls_indices, shifts=-1)
                            pos_idx[cls_indices] = shifted
                    loss = loss + model_config.get('supcon_weight', 0.1) * info_nce(z, pos_idx, temperature=model_config.get('supcon_temp', 0.07))
                
                # Update stats
                val_loss_sum += float(loss.detach().cpu()) * y_batch.size(0)
                pred = out['logits'].argmax(dim=1)
                val_correct += int((pred == y_batch).sum().item())
                val_total += y_batch.size(0)
                
                # Collect scores for binary metrics
                if model_config.get('binary', False):
                    probs = torch.softmax(out['logits'], dim=1)[:, 1].detach().cpu().numpy()
                    val_scores.extend(probs.tolist())
                    val_targets.extend(y_batch.detach().cpu().tolist())
        
        # Compute metrics
        train_acc = 100.0 * train_correct / max(1, train_total)
        train_loss = train_loss_sum / max(1, train_total)
        val_acc = 100.0 * val_correct / max(1, val_total)
        val_loss = val_loss_sum / max(1, val_total)
        
        val_pr_auc = float('nan')
        val_f1 = float('nan')
        if model_config.get('binary', False) and average_precision_score and f1_score and len(val_targets) > 0:
            try:
                val_pr_auc = float(average_precision_score(val_targets, val_scores))
                preds_05 = [1 if s >= 0.5 else 0 for s in val_scores]
                val_f1 = float(f1_score(val_targets, preds_05))
            except Exception:
                pass
        
        # Choose metric based on config
        metric_map = {'acc': val_acc, 'loss': val_loss, 'pr_auc': val_pr_auc, 'f1': val_f1}
        current_metric = metric_map.get(best_metric_name, val_acc)
        
        # Save checkpoint
        last_ckpt = os.path.join(out_dir, 'last.pt')
        checkpoint_data = {
            'model': model.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'val_loss': val_loss,
        }
        torch.save(checkpoint_data, last_ckpt)
        
        # Save to GCS if specified
        if args.gcs_output:
            save_checkpoint_to_gcs(last_ckpt, checkpoint_data, args.gcs_output)
        
        # Save best checkpoint
        is_better = False
        if best_metric_name == 'loss':
            is_better = (best_val_metric is None) or (current_metric < best_val_metric)
        else:
            is_better = (best_val_metric is None) or (current_metric > best_val_metric)
        
        if is_better:
            best_val_metric = current_metric
            best_ckpt = os.path.join(out_dir, 'best.pt')
            checkpoint_data = {
                'model': model.state_dict(),
                'epoch': epoch,
                'metric': current_metric,
                'metric_name': best_metric_name
            }
            torch.save(checkpoint_data, best_ckpt)
            
            # Save to GCS if specified
            if args.gcs_output:
                save_checkpoint_to_gcs(best_ckpt, checkpoint_data, args.gcs_output)
        
        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.2f}", f"{val_loss:.6f}", f"{val_acc:.2f}", f"{val_pr_auc:.6f}", f"{val_f1:.6f}"])
        
        # Print metrics
        print(f"\n{model_name}:")
        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.2f}% pr_auc={val_pr_auc:.4f} f1={val_f1:.4f}")
        if is_better:
            print(f"  [BEST] {best_metric_name}={current_metric:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Training completed: {model_name}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*70}")

