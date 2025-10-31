from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def read_groundtruth(gt_path: str) -> Dict[Tuple[str, int], int]:
    """
    Expect lines like: <video_id> <frame_idx> <class_label>
    class_label can be str; we map to integer IDs on-the-fly outside dataset if needed.
    """
    mapping: Dict[Tuple[str, int], int] = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            vid, frame_str, label = parts[0], parts[1], parts[2]
            try:
                frame = int(frame_str)
            except Exception:
                continue
            mapping[(vid, frame)] = label
    return mapping


def compute_pose_stats(window: np.ndarray) -> np.ndarray:
    """
    window: (T, J, C) numpy array
    Returns simple stats vector, e.g., mean and std per coord over joints/time -> (2*C,)
    """
    # Flatten over (T,J)
    feat = window.reshape(-1, window.shape[-1])  # (T*J, C)
    mean = feat.mean(axis=0)
    std = feat.std(axis=0) + 1e-6
    return np.concatenate([mean, std], axis=0)


class NpzLandmarksDataset(Dataset):
    def __init__(
        self,
        root: str,
        gt_path: Optional[str] = None,
        window: int = 25,
        stride: int = 1,
        in_coords: int = 2,
        include_pose: bool = True,
        include_hands: bool = True,
        include_face: bool = False,
        max_files: Optional[int] = None,
        allowed_ids: Optional[Set[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.root = root
        self.window = window
        self.stride = max(1, int(stride))
        self.in_coords = in_coords
        self.include_pose = include_pose
        self.include_hands = include_hands
        self.include_face = include_face
        self.max_files = max_files
        self.label_map = label_map or {}
        self.allowed_ids = set(allowed_ids) if allowed_ids else None

        self.files = sorted(glob.glob(os.path.join(root, '*.npz')))
        if self.allowed_ids is not None:
            self.files = [p for p in self.files if os.path.splitext(os.path.basename(p))[0] in self.allowed_ids]
        if self.max_files is not None:
            try:
                k = int(self.max_files)
                if k >= 0:
                    self.files = self.files[:k]
            except Exception:
                pass
        self.gt = read_groundtruth(gt_path) if gt_path and os.path.exists(gt_path) else {}

        self.index: List[Tuple[int, int]] = []  # (file_idx, start_frame)
        self.meta: List[Dict] = []
        # Precompute available windows by reading npz headers quickly
        for fi, path in enumerate(self.files):
            with np.load(path) as npz:
                # Prefer MediaPipe keys; fall back to any array in file
                T = 0
                for key in ('pose', 'left_hand', 'right_hand', 'face'):
                    if key in npz:
                        T = max(T, npz[key].shape[0])
                if T == 0:
                    # generic fallback
                    key = list(npz.keys())[0]
                    T = npz[key].shape[0]
                for s in range(0, max(0, T - window + 1), self.stride):
                    self.index.append((fi, s))
                    self.meta.append({'file': os.path.basename(path), 'start': s, 'len': window})

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fi, s = self.index[i]
        path = self.files[fi]
        vid = os.path.splitext(os.path.basename(path))[0]
        with np.load(path) as npz:
            # Build (T, J, C) by concatenating selected parts
            parts = []
            T = 0
            if self.include_pose and 'pose' in npz:
                T = max(T, npz['pose'].shape[0])
            if self.include_hands:
                if 'left_hand' in npz:
                    T = max(T, npz['left_hand'].shape[0])
                if 'right_hand' in npz:
                    T = max(T, npz['right_hand'].shape[0])
            if self.include_face and 'face' in npz:
                T = max(T, npz['face'].shape[0])

            def pick_coords(x: np.ndarray) -> np.ndarray:
                # x: (..., C) with C>=self.in_coords; pose may have visibility at index 3
                if x.shape[-1] >= self.in_coords:
                    return x[..., :self.in_coords]
                # pad with zeros if fewer coords
                pad = np.zeros(list(x.shape[:-1]) + [self.in_coords - x.shape[-1]], dtype=x.dtype)
                return np.concatenate([x, pad], axis=-1)

            if self.include_pose:
                if 'pose' in npz:
                    pose = npz['pose']  # (T,33,4)
                    pose = pose[:, :, :max(1, self.in_coords)]  # drop visibility
                else:
                    pose = np.full((T, 33, self.in_coords), np.nan, dtype=np.float32)
                parts.append(pick_coords(pose))
            if self.include_hands:
                if 'left_hand' in npz:
                    lh = pick_coords(npz['left_hand'])
                else:
                    lh = np.full((T, 21, self.in_coords), np.nan, dtype=np.float32)
                if 'right_hand' in npz:
                    rh = pick_coords(npz['right_hand'])
                else:
                    rh = np.full((T, 21, self.in_coords), np.nan, dtype=np.float32)
                parts += [lh, rh]
            if self.include_face:
                if 'face' in npz:
                    face = pick_coords(npz['face'])
                else:
                    face = np.full((T, 478, self.in_coords), np.nan, dtype=np.float32)
                parts.append(face)

            if not parts:
                # fallback: use any last array
                arr = npz[list(npz.keys())[-1]]
                arr = pick_coords(arr)
            else:
                arr = np.concatenate(parts, axis=1)

        window_np = arr[s:s + self.window]
        # Replace NaNs/Infs from missing detections with zeros to avoid NaN loss
        window_np = np.nan_to_num(window_np, nan=0.0, posinf=0.0, neginf=0.0)
        # build label: majority or last frame label based on gt
        labels = []
        for t in range(self.window):
            frame_id = s + t
            lab = self.gt.get((vid, frame_id), None)
            if lab is not None:
                labels.append(lab)
        if len(labels) == 0:
            y_str = 'unknown'
        else:
            # pick most frequent
            vals, counts = np.unique(np.array(labels), return_counts=True)
            y_str = vals[np.argmax(counts)].item() if hasattr(vals[0], 'item') else vals[np.argmax(counts)]

        if y_str not in self.label_map:
            self.label_map[y_str] = len(self.label_map)
        y = self.label_map[y_str]

        # stats for signer encoder
        stats_np = compute_pose_stats(window_np)

        X = torch.from_numpy(window_np).float()           # (T, J, C)
        pose_stats = torch.from_numpy(stats_np).float()   # (2*C,)
        y = torch.tensor(y, dtype=torch.long)
        return X, pose_stats, y, {
            'video': vid,
            'start': s,
            'label': y_str,
        }


