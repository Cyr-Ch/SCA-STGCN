from __future__ import annotations
import os
import csv
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
try:
    from sklearn.metrics import average_precision_score, f1_score
except Exception:
    average_precision_score = None
    f1_score = None

from .model import SignSTGCNModel
from .datasets import NpzLandmarksDataset
from .utils import build_hand_body_adjacency
from .losses import classification_loss, signer_adversarial_loss, info_nce


def collate_pad(batch):
    # X: (T,J,C); make (B,T,J,C)
    Xs, stats, ys, metas = zip(*batch)
    X = torch.stack(Xs, dim=0)
    stats = torch.stack(stats, dim=0)
    y = torch.stack(ys, dim=0)
    return X, stats, y, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to landmarks folder containing .npz and groundtruth')
    ap.add_argument('--window', type=int, default=25)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--coords', type=int, default=2)
    ap.add_argument('--include-pose', action='store_true', help='Use pose landmarks (default on)')
    ap.add_argument('--no-include-pose', dest='include_pose', action='store_false')
    ap.set_defaults(include_pose=True)
    ap.add_argument('--include-hands', action='store_true')
    ap.add_argument('--include-face', action='store_true')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-classes', type=int, default=100)
    ap.add_argument('--num-signers', type=int, default=30)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--binary', action='store_true', help='Train binary classifier: signing vs not-signing')
    ap.add_argument('--signing-labels', type=str, default='sign,signing,gesture', help='Comma-separated labels that count as signing')
    ap.add_argument('--out', type=str, default='runs/signgcn', help='Output directory for logs and checkpoints')
    ap.add_argument('--save-best', action='store_true', help='Save best model by selected metric')
    ap.add_argument('--best-metric', type=str, default='acc', choices=['acc','loss','pr_auc','f1'], help='Metric to select best checkpoint')
    ap.add_argument('--log-csv', action='store_true', help='Write per-epoch metrics to CSV')
    ap.add_argument('--debug', action='store_true', help='Verbose debug prints')
    ap.add_argument('--print-every', type=int, default=50, help='Batch interval for progress prints (debug)')
    ap.add_argument('--first-only', action='store_true', help='Use only the first .npz file to quickly test pipeline')
    ap.add_argument('--id-list', type=str, default=None, help='Path to txt with video IDs to include (one per line)')
    ap.add_argument('--splits-json', type=str, default=None, help='Path to JSON with video-based splits (keys: splits.train/val/test.videos)')
    ap.add_argument('--stats-max-batches', type=int, default=0, help='Limit class-stats pre-scan to this many batches (0=all)')
    # Pseudo-signer clustering options
    ap.add_argument('--use-pseudo-signers', action='store_true', help='Cluster pose_stats to define pseudo signer IDs')
    ap.add_argument('--num-pseudo-signers', type=int, default=8, help='Number of pseudo signer clusters (K)')
    ap.add_argument('--pseudo-seed', type=int, default=0, help='Random state for KMeans')
    ap.add_argument('--signer-loss-weight', type=float, default=0.5, help='Weight for adversarial signer loss')
    # Supervised contrastive (ID-free)
    ap.add_argument('--use-supcon', action='store_true', help='Add supervised contrastive loss on embeddings using action labels')
    ap.add_argument('--supcon-weight', type=float, default=0.1, help='Weight for supervised contrastive loss')
    ap.add_argument('--supcon-temp', type=float, default=0.07, help='InfoNCE temperature for supervised contrastive')
    # Batch-interval checkpointing
    ap.add_argument('--save-interval-batches', type=int, default=0, help='If >0, evaluate running metric and save best checkpoint every N batches')
    ap.add_argument('--batch-save-metric', type=str, default='acc', choices=['acc','loss','f1'], help='Metric to track for batch-interval best checkpoint')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.debug:
        print('Args:', vars(args))
        print('Device:', device)

    gt_path = os.path.join(args.data, 'groundtruth.txt') if os.path.exists(os.path.join(args.data, 'groundtruth.txt')) else os.path.join(args.data, 'groundtruth')
    # Option A: use splits JSON if provided
    if args.splits_json and os.path.exists(args.splits_json):
        import json
        with open(args.splits_json, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        train_ids = set(splits.get('splits', {}).get('train', {}).get('videos', []))
        val_ids = set(splits.get('splits', {}).get('val', {}).get('videos', []))
        if args.debug:
            print('Loaded splits JSON:', args.splits_json, 'train_ids=', len(train_ids), 'val_ids=', len(val_ids))
        train_ds = NpzLandmarksDataset(
            root=args.data,
            gt_path=gt_path,
            window=args.window,
            stride=args.stride,
            in_coords=args.coords,
            include_pose=bool(args.include_pose),
            include_hands=bool(args.include_hands),
            include_face=bool(args.include_face),
            max_files=(1 if args.first_only else None),
            allowed_ids=train_ids,
        )
        val_ds = NpzLandmarksDataset(
            root=args.data,
            gt_path=gt_path,
            window=args.window,
            stride=args.stride,
            in_coords=args.coords,
            include_pose=bool(args.include_pose),
            include_hands=bool(args.include_hands),
            include_face=bool(args.include_face),
            max_files=None,
            allowed_ids=val_ids,
        )
    else:
        # Option B: use single id-list or default list, then random split
        allowed_ids = None
        default_list = os.path.join(args.data, 'videos_with_S_and_npz.txt')
        id_list_path = args.id_list if args.id_list else (default_list if os.path.exists(default_list) else None)
        if id_list_path and os.path.exists(id_list_path):
            with open(id_list_path, 'r', encoding='utf-8') as f:
                allowed_ids = set([line.strip() for line in f if line.strip()])
            if args.debug:
                print('Loaded allowed IDs from:', id_list_path, 'count=', len(allowed_ids))

        ds = NpzLandmarksDataset(
            root=args.data,
            gt_path=gt_path,
            window=args.window,
            stride=args.stride,
            in_coords=args.coords,
            include_pose=bool(args.include_pose),
            include_hands=bool(args.include_hands),
            include_face=bool(args.include_face),
            max_files=(1 if args.first_only else None),
            allowed_ids=allowed_ids,
        )
        if args.debug:
            print('Total windows in dataset:', len(ds))

        # split train/val
        val_size = max(1, int(0.1 * len(ds)))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
        if args.debug:
            print('Split sizes -> train:', len(train_ds), 'val:', len(val_ds))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_pad)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_pad)
    if args.debug:
        try:
            print('Train batches per epoch:', len(train_dl), 'Val batches:', len(val_dl))
        except Exception:
            pass

    # infer joints from a sample
    # Get a sample from train set
    sample_X, sample_stats, _, _ = train_ds[0] if hasattr(train_ds, '__getitem__') else ds[0]
    T, J, C = sample_X.shape
    if args.debug:
        print('Sample shapes -> X:', tuple(sample_X.shape), 'pose_stats:', tuple(sample_stats.shape))

    # simple adjacency: chain or placeholder; user should replace with their skeleton graph
    hand_edges = tuple((i, i+1) for i in range(0, max(0, J-1)))
    body_edges = tuple()
    A = build_hand_body_adjacency(J, hand_edges, body_edges).to(device)
    if args.debug:
        print('Adjacency built for J=', J)

    # Configure signer head based on pseudo-signer usage
    use_signer_head = bool(args.use_pseudo_signers)
    num_signers = (args.num_pseudo_signers if args.use_pseudo_signers else args.num_signers)

    if args.debug:
        print('Signer head config -> use_signer_head:', use_signer_head, 'num_signers_effective:', num_signers)

    model = SignSTGCNModel(
        num_joints=J,
        in_coords=C,
        num_classes=(2 if args.binary else args.num_classes),
        num_signers=num_signers,
        signer_stats_dim=sample_stats.numel(),
        use_signer_head=use_signer_head,
    ).to(device)
    if args.debug:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model params -> total:', total_params, 'trainable:', trainable_params)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Prepare output dir and CSV
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'metrics.csv')
    if args.log_csv and not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch','train_loss','train_acc','val_loss','val_acc','val_pr_auc','val_f1'])
        if args.debug:
            print('Created CSV log at:', csv_path)

    best_val_metric = None

    # Class distribution stats per split (binary if requested)
    def compute_split_stats(dloader, max_batches=0):
        pos, neg, total = 0, 0, 0
        with torch.no_grad():
            for bi, (_, _, y, meta) in enumerate(dloader, start=1):
                if args.binary:
                    sign_set = set([s.strip().lower() for s in args.signing_labels.split(',') if s.strip()])
                    labels = [m.get('label', '') for m in meta]
                    mapped = [1 if str(lbl).lower() in sign_set else 0 for lbl in labels]
                else:
                    # treat any non-unknown as pos (for summary only)
                    mapped = [1 for _ in range(len(y))]
                pos += sum(1 for v in mapped if v == 1)
                neg += sum(1 for v in mapped if v == 0)
                total += len(mapped)
                if max_batches and bi >= max_batches:
                    break
        return pos, neg, total

    if args.debug:
        print('Computing class stats (capped batches =', int(args.stats_max_batches), ') ...')
    tr_pos, tr_neg, tr_tot = compute_split_stats(train_dl, max_batches=int(args.stats_max_batches))
    va_pos, va_neg, va_tot = compute_split_stats(val_dl, max_batches=int(args.stats_max_batches))
    print(f"Class stats (train): signing={tr_pos} not_signing={tr_neg} total={tr_tot}")
    print(f"Class stats (val):   signing={va_pos} not_signing={va_neg} total={va_tot}")

    # Class weights for binary imbalance (weight order: [nonS, S])
    class_weight_cpu = None
    if args.binary and tr_pos > 0:
        w_s = float(tr_neg) / float(tr_pos)
        class_weight_cpu = torch.tensor([1.0, w_s], dtype=torch.float32)
        if args.debug:
            print('Using class-weighted loss (nonS=1.0, S=%.4f)' % w_s)

    # Pseudo-signer clustering using KMeans over pose_stats
    pseudo_cluster_map = None  # key -> cluster id
    pseudo_centroids = None
    def sample_key(meta_rec):
        return f"{meta_rec.get('video','')}::{int(meta_rec.get('start',0))}"

    if args.use_pseudo_signers:
        try:
            from sklearn.cluster import KMeans
        except Exception:
            print('scikit-learn not installed; install with: pip install scikit-learn')
            raise

        if args.debug:
            print('Computing pseudo-signer clusters (KMeans)...')
        # Collect pose_stats and keys from train split
        train_stats_list = []
        train_keys = []
        with torch.no_grad():
            for X, stats, y, meta in train_dl:
                # stats: (B, S)
                s_np = stats.cpu().numpy()
                train_stats_list.append(s_np)
                train_keys += [sample_key(m) for m in meta]
        train_stats_np = np.concatenate(train_stats_list, axis=0) if len(train_stats_list) > 0 else np.zeros((0, sample_stats.numel()), dtype=np.float32)
        if args.debug:
            print('KMeans fit on', train_stats_np.shape[0], 'windows; feature dim=', train_stats_np.shape[1] if train_stats_np.size else 0)
        if train_stats_np.shape[0] == 0:
            raise RuntimeError('No training stats available for clustering.')
        k = max(2, int(args.num_pseudo_signers))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=int(args.pseudo_seed))
        kmeans.fit(train_stats_np)
        pseudo_centroids = kmeans.cluster_centers_.astype(np.float32)
        train_clusters = kmeans.labels_.astype(int)
        pseudo_cluster_map = {key: int(cid) for key, cid in zip(train_keys, train_clusters)}
        # Assign val via nearest centroid
        if args.debug:
            print('Assigning pseudo-signers for val via nearest centroid...')
        # compute val cluster distribution
        def dist_to_centroids(s):
            # s: (S,), centroids: (K, S)
            d = np.linalg.norm(pseudo_centroids - s[None, :], axis=1)
            return int(np.argmin(d))
        # Stats per split
        tr_counts = np.bincount(train_clusters, minlength=k)
        if args.debug:
            print('Pseudo-signer train cluster counts:', tr_counts.tolist())

    best_batch_metric = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for bi, (X, stats, y, meta) in enumerate(train_dl, start=1):
            X, stats, y = X.to(device), stats.to(device), y.to(device)
            if args.binary:
                sign_set = set([s.strip().lower() for s in args.signing_labels.split(',') if s.strip()])
                # remap using raw label strings
                labels = [m.get('label', '') for m in meta]
                y = torch.tensor([1 if str(lbl).lower() in sign_set else 0 for lbl in labels], device=y.device, dtype=torch.long)
            opt.zero_grad()
            out = model(X, A, stats, return_features=bool(args.use_supcon))
            if args.binary and class_weight_cpu is not None:
                loss = F.cross_entropy(out['logits'], y, weight=class_weight_cpu.to(y.device))
            else:
                loss = classification_loss(out['logits'], y)
            # adversarial signer loss using pseudo clusters
            if args.use_pseudo_signers and 'signer_logits' in out:
                # build signer ids for this batch
                signer_ids = []
                if pseudo_cluster_map is not None:
                    for m in meta:
                        key = sample_key(m)
                        if key in pseudo_cluster_map:
                            signer_ids.append(pseudo_cluster_map[key])
                        else:
                            # assign by nearest centroid
                            s_np = stats.cpu().numpy()
                            # align by index
                            # Fallback: assign zero
                            signer_ids.append(0)
                else:
                    signer_ids = [0 for _ in range(len(meta))]
                signer_ids = torch.tensor(signer_ids, device=y.device, dtype=torch.long)
                loss = loss + float(args.signer_loss_weight) * signer_adversarial_loss(out['signer_logits'], signer_ids)
            # supervised contrastive (batch-wise) using labels
            if args.use_supcon and 'embedding' in out:
                z = out['embedding']  # (B, D)
                B = z.size(0)
                y_cpu = y.detach().cpu().tolist()
                indices_by_class = {}
                for idx, cls in enumerate(y_cpu):
                    indices_by_class.setdefault(int(cls), []).append(idx)
                pos_idx = list(range(B))
                for cls, idxs in indices_by_class.items():
                    if len(idxs) > 1:
                        for k in range(len(idxs)):
                            pos_idx[idxs[k]] = idxs[(k + 1) % len(idxs)]
                    else:
                        pos_idx[idxs[0]] = idxs[0]
                pos_idx = torch.tensor(pos_idx, device=z.device, dtype=torch.long)
                loss = loss + float(args.supcon_weight) * info_nce(z, pos_idx, temperature=float(args.supcon_temp))

            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu()) * y.size(0)
            pred = out['logits'].argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
            if args.debug and (bi % max(1, args.print_every) == 0 or bi == 1):
                running_acc = 100.0 * correct / max(1, total)
                print(f"[Epoch {epoch}] Batch {bi}: loss={float(loss.detach().cpu()):.4f} running_acc={running_acc:.2f}%")
            # Batch-interval best checkpointing
            if int(args.save_interval_batches) > 0 and (bi % int(args.save_interval_batches) == 0):
                running_acc = 100.0 * correct / max(1, total)
                running_loss = loss_sum / max(1, total)
                if args.batch_save_metric == 'acc':
                    current_metric = running_acc
                    better = (best_batch_metric is None) or (current_metric > best_batch_metric)
                else:
                    current_metric = running_loss
                    better = (best_batch_metric is None) or (current_metric < best_batch_metric)
                if better:
                    best_batch_metric = current_metric
                    best_ckpt = os.path.join(out_dir, 'best_batch.pt')
                    torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch': bi, 'metric': current_metric, 'metric_name': args.batch_save_metric}, best_ckpt)
                    if args.debug:
                        print(f"Saved best_batch checkpoint at epoch {epoch} batch {bi} metric {args.batch_save_metric}={current_metric:.4f}")
        train_acc = 100.0 * correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        val_targets, val_scores = [], []  # for PR-AUC/F1 when binary
        with torch.no_grad():
            for bi, (X, stats, y, meta) in enumerate(val_dl, start=1):
                X, stats, y = X.to(device), stats.to(device), y.to(device)
                if args.binary:
                    sign_set = set([s.strip().lower() for s in args.signing_labels.split(',') if s.strip()])
                    labels = [m.get('label', '') for m in meta]
                    y = torch.tensor([1 if str(lbl).lower() in sign_set else 0 for lbl in labels], device=y.device, dtype=torch.long)
                out = model(X, A, stats, return_features=bool(args.use_supcon))
                if args.binary and class_weight_cpu is not None:
                    loss = F.cross_entropy(out['logits'], y, weight=class_weight_cpu.to(y.device))
                else:
                    loss = classification_loss(out['logits'], y)
                if args.use_pseudo_signers and 'signer_logits' in out and pseudo_centroids is not None:
                    # assign val signer ids by nearest centroid
                    s_np = stats.cpu().numpy()  # (B,S)
                    # compute nearest centroid per row
                    dists = ((s_np[:, None, :] - pseudo_centroids[None, :, :]) ** 2).sum(axis=2) ** 0.5
                    signer_ids = np.argmin(dists, axis=1).astype(int)
                    signer_ids = torch.from_numpy(signer_ids).to(y.device)
                    loss = loss + float(args.signer_loss_weight) * signer_adversarial_loss(out['signer_logits'], signer_ids)
                if args.use_supcon and 'embedding' in out:
                    z = out['embedding']
                    B = z.size(0)
                    y_cpu = y.detach().cpu().tolist()
                    indices_by_class = {}
                    for idx, cls in enumerate(y_cpu):
                        indices_by_class.setdefault(int(cls), []).append(idx)
                    pos_idx = list(range(B))
                    for cls, idxs in indices_by_class.items():
                        if len(idxs) > 1:
                            for k in range(len(idxs)):
                                pos_idx[idxs[k]] = idxs[(k + 1) % len(idxs)]
                        else:
                            pos_idx[idxs[0]] = idxs[0]
                    pos_idx = torch.tensor(pos_idx, device=z.device, dtype=torch.long)
                    loss = loss + float(args.supcon_weight) * info_nce(z, pos_idx, temperature=float(args.supcon_temp))

                v_loss_sum += float(loss.detach().cpu()) * y.size(0)
                pred = out['logits'].argmax(dim=1)
                v_correct += int((pred == y).sum().item())
                v_total += y.size(0)
                # collect scores for PR-AUC/F1 if binary
                if args.binary:
                    probs = torch.softmax(out['logits'], dim=1)[:, 1].detach().cpu().numpy()
                    val_scores.extend(probs.tolist())
                    val_targets.extend(y.detach().cpu().tolist())
                if args.debug and (bi % max(1, args.print_every) == 0 or bi == 1):
                    running_val_acc = 100.0 * v_correct / max(1, v_total)
                    print(f"[Epoch {epoch}] Val batch {bi}: loss={float(loss.detach().cpu()):.4f} running_val_acc={running_val_acc:.2f}%")
        val_acc = 100.0 * v_correct / max(1, v_total)
        val_loss = v_loss_sum / max(1, v_total)
        # compute PR-AUC and F1 for binary
        val_pr_auc = float('nan')
        val_f1 = float('nan')
        if args.binary and average_precision_score is not None and f1_score is not None and len(val_targets) > 0:
            try:
                val_pr_auc = float(average_precision_score(val_targets, val_scores))
                # F1 at 0.5 threshold
                preds_05 = [1 if s >= 0.5 else 0 for s in val_scores]
                val_f1 = float(f1_score(val_targets, preds_05))
            except Exception:
                pass

        # Logging
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2f}% val_loss={val_loss:.4f} val_acc={val_acc:.2f}% val_pr_auc={val_pr_auc:.4f} val_f1={val_f1:.4f}")
        if args.log_csv:
            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.2f}", f"{val_loss:.6f}", f"{val_acc:.2f}", f"{val_pr_auc:.6f}", f"{val_f1:.6f}"])
            if args.debug:
                print('Appended metrics to CSV.')

        # Checkpoints
        last_ckpt = os.path.join(out_dir, 'last.pt')
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, last_ckpt)
        if args.debug:
            print('Saved last checkpoint to:', last_ckpt)
        # Choose current metric based on flag
        metric_map = {
            'acc': val_acc,
            'loss': val_loss,
            'pr_auc': val_pr_auc,
            'f1': val_f1,
        }
        current_metric = metric_map.get(args.best_metric, val_acc)
        is_better = False
        if args.best_metric == 'loss':
            is_better = (best_val_metric is None) or (current_metric < best_val_metric)
        else:
            is_better = (best_val_metric is None) or (current_metric > best_val_metric)

        if args.save_best and is_better:
            best_val_metric = current_metric
            best_ckpt = os.path.join(out_dir, 'best.pt')
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'metric': current_metric, 'metric_name': args.best_metric}, best_ckpt)
            if args.debug:
                print(f"Saved best checkpoint to: {best_ckpt} by {args.best_metric}={current_metric:.4f}")


if __name__ == '__main__':
    main()


