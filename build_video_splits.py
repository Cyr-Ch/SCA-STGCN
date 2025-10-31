from __future__ import annotations
import os
import json
import glob
import argparse
import numpy as np


def read_groundtruth(gt_path: str) -> dict[tuple[str, int], str]:
    gt: dict[tuple[str, int], str] = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            vid, frame_str, label = parts[0], parts[1], parts[2]
            try:
                frame = int(frame_str)
            except Exception:
                continue
            gt[(vid, frame)] = label
    return gt


def get_video_length(npz_path: str) -> int:
    with np.load(npz_path, allow_pickle=False) as npz:
        if 'pose' in npz:
            return int(npz['pose'].shape[0])
        for k in npz.files:
            arr = npz[k]
            if hasattr(arr, 'shape') and len(arr.shape) > 0:
                return int(arr.shape[0])
    return 0


def count_windows_for_video(vid: str, T: int, gt: dict[tuple[str, int], str], window: int, stride: int, s_label: str, rule: str) -> tuple[int, int, int]:
    # Build per-frame boolean array for S label
    is_s = np.zeros((T,), dtype=np.int32)
    for t in range(T):
        lab = gt.get((vid, t), None)
        if lab is not None and str(lab).upper() == s_label.upper():
            is_s[t] = 1
    if T < window:
        return T, 0, 0
    # sliding window counts via convolution
    kernel = np.ones((window,), dtype=np.int32)
    conv = np.convolve(is_s, kernel, mode='valid')  # length T-window+1
    if rule == 'any':
        s_windows_bool = conv >= 1
    else:  # majority
        s_windows_bool = conv >= int(np.ceil(window / 2))
    # apply stride by slicing
    s_windows_bool = s_windows_bool[::stride]
    s_count = int(s_windows_bool.sum())
    total_w = int(((T - window) // stride) + 1)
    non_s = max(0, total_w - s_count)
    return T, s_count, non_s


def split_videos(vids_with_meta: list[tuple[str, dict]], ratios: tuple[float, float, float]) -> dict[str, list[str]]:
    # Separate videos with any S windows vs none
    pos, neg = [], []
    for vid, meta in vids_with_meta:
        if meta['S_windows'] > 0:
            pos.append((vid, meta))
        else:
            neg.append((vid, meta))
    # deterministic order
    pos.sort(key=lambda x: x[0])
    neg.sort(key=lambda x: x[0])

    def ratio_split(lst: list[tuple[str, dict]]):
        n = len(lst)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = max(0, n - n_train - n_val)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    pos_tr, pos_va, pos_te = ratio_split(pos)
    neg_tr, neg_va, neg_te = ratio_split(neg)

    train = [v for v, _ in pos_tr + neg_tr]
    val = [v for v, _ in pos_va + neg_va]
    test = [v for v, _ in pos_te + neg_te]

    # ensure each split has at least one positive if available
    for split_name in ('train', 'val', 'test'):
        split = {'train': train, 'val': val, 'test': test}[split_name]
        if len(pos) > 0:
            if not any(v in split for v, _ in pos):
                # move one positive from the largest pos split
                candidates = [('train', pos_tr), ('val', pos_va), ('test', pos_te)]
                candidates.sort(key=lambda x: len(x[1]), reverse=True)
                for name, bucket in candidates:
                    if len(bucket) > 1 and split_name != name:
                        vid_move = bucket.pop()[0]
                        split.append(vid_move)
                        break
    return {'train': train, 'val': val, 'test': test}


def main():
    ap = argparse.ArgumentParser(description='Build video-based train/val/test splits with S/non-S estimates')
    ap.add_argument('--data', required=True, help='Path to landmarks folder (.npz) and groundtruth')
    ap.add_argument('--window', type=int, default=25)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--s-label', type=str, default='S', help='Label string indicating signing')
    ap.add_argument('--rule', type=str, default='any', choices=['any', 'majority'], help='Window positive rule')
    ap.add_argument('--train', type=float, default=0.8)
    ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--test', type=float, default=0.1)
    ap.add_argument('--out', type=str, default=None, help='Output JSON path (default: data/splits.json)')
    args = ap.parse_args()

    data = os.path.abspath(args.data)
    gt_txt = os.path.join(data, 'groundtruth.txt')
    gt_path = gt_txt if os.path.exists(gt_txt) else os.path.join(data, 'groundtruth')
    if not os.path.exists(gt_path):
        raise FileNotFoundError('groundtruth file not found under data directory')
    gt = read_groundtruth(gt_path)

    npz_files = sorted(glob.glob(os.path.join(data, '*.npz')))
    vids = [(os.path.splitext(os.path.basename(p))[0], p) for p in npz_files]

    per_video: dict[str, dict] = {}
    vids_with_meta: list[tuple[str, dict]] = []

    for vid, path in vids:
        T = get_video_length(path)
        frames, s_w, n_s_w = count_windows_for_video(vid, T, gt, args.window, args.stride, args.s_label, args.rule)
        meta = {
            'frames': frames,
            'windows': int(((max(frames, 0) - args.window) // args.stride) + 1) if frames >= args.window else 0,
            'S_windows': int(s_w),
            'nonS_windows': int(n_s_w),
        }
        per_video[vid] = meta
        vids_with_meta.append((vid, meta))

    # Normalize/validate ratios
    total_ratio = args.train + args.val + args.test
    if total_ratio <= 0:
        raise ValueError('Invalid split ratios')
    ratios = (args.train / total_ratio, args.val / total_ratio, args.test / total_ratio)
    splits = split_videos(vids_with_meta, ratios)

    def sum_counts(ids: list[str]):
        s = sum(per_video[v]['S_windows'] for v in ids)
        n = sum(per_video[v]['nonS_windows'] for v in ids)
        return int(s), int(n)

    train_ids, val_ids, test_ids = splits['train'], splits['val'], splits['test']
    train_s, train_n = sum_counts(train_ids)
    val_s, val_n = sum_counts(val_ids)
    test_s, test_n = sum_counts(test_ids)

    out = {
        'meta': {
            'data': data,
            'groundtruth': gt_path,
            'window': int(args.window),
            'stride': int(args.stride),
            's_label': args.s_label,
            'rule': args.rule,
            'ratios': {'train': ratios[0], 'val': ratios[1], 'test': ratios[2]},
        },
        'splits': {
            'train': {'videos': train_ids, 'counts': {'S': train_s, 'nonS': train_n}},
            'val':   {'videos': val_ids,   'counts': {'S': val_s,   'nonS': val_n}},
            'test':  {'videos': test_ids,  'counts': {'S': test_s,  'nonS': test_n}},
        },
        'per_video': per_video,
    }

    out_path = args.out if args.out else os.path.join(data, 'splits.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Saved splits to {out_path}")


if __name__ == '__main__':
    main()


