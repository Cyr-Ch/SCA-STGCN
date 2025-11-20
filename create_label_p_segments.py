"""
Extract per-segment NPZ files from raw landmark videos that are all labeled as "P" (speaking).

For each input video (.npz) the script:
  * builds sliding windows (window/stride) over the concatenated pose/hands/face landmarks,
  * optionally limits how many segments are exported per video,
  * saves every segment as its own NPZ under <output>/<video_id>/<video_id>_segXXXXXX.npz.

The resulting files follow the same structure used by the training pipeline, so they can
be mixed with existing segment folders or uploaded to a bucket.
"""

from __future__ import annotations

import argparse
import os
import pickle
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm  # type: ignore

from datasets.landmarks_npz import (
    compute_pose_stats,
    create_label_mapping,
    read_groundtruth,
)

METADATA_DTYPE = np.dtype(
    [
        ("video", "U128"),
        ("start", np.int32),
        ("label", "U16"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-segment NPZ files for P-labeled videos."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory that contains per-video landmark NPZ files (e.g., landmarks_p).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-segment folders will be written.",
    )
    parser.add_argument(
        "--groundtruth",
        required=True,
        help="Path to groundtruth txt file (video frame label per line).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Number of frames per segment window (default: 25).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between consecutive windows (default: 1).",
    )
    parser.add_argument(
        "--in-coords",
        type=int,
        default=2,
        help="How many coordinate channels to keep (default: 2 -> x,y).",
    )
    parser.add_argument(
        "--include-face",
        action="store_true",
        help="Include face landmarks (disabled by default).",
    )
    parser.add_argument(
        "--exclude-hands",
        action="store_true",
        help="Skip hand landmarks (pose is always included).",
    )
    parser.add_argument(
        "--max-segments-per-video",
        type=int,
        default=None,
        help="Cap the number of exported segments per video (default: unlimited).",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="When max segments is set, either keep the first ones or sample randomly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed (used only if sample-mode=random).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for *.npz files inside input-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of videos assigned to train split (default: 0.8).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of videos assigned to validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of videos assigned to test split (default: 0.1).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2024,
        help="Random seed used to shuffle videos before split assignment.",
    )
    parser.add_argument(
        "--target-label",
        default="P",
        help="Segment label to keep (default: 'P').",
    )
    parser.add_argument(
        "--map-unknown-to-n",
        action="store_true",
        help="Map '?' labels to 'n' before voting (default: False).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Label map size to embed inside config (default: 3).",
    )
    return parser.parse_args()


def list_input_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(root.glob(pattern))


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def assign_splits(
    files: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {train_ratio}+{val_ratio}+{test_ratio}={total})"
        )
    if len(files) == 0:
        return {}

    rng = np.random.default_rng(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_train = min(n, max(0, n_train))
    n_val = min(n - n_train, max(0, n_val))
    n_test = n - n_train - n_val

    assignments: dict[str, str] = {}
    for path in shuffled[:n_train]:
        assignments[str(path)] = "train"
    for path in shuffled[n_train : n_train + n_val]:
        assignments[str(path)] = "val"
    for path in shuffled[n_train + n_val :]:
        assignments[str(path)] = "test"
    return assignments


def build_feature_array(
    npz_obj,
    in_coords: int,
    include_hands: bool,
    include_face: bool,
) -> np.ndarray | None:
    T = 0
    if "pose" in npz_obj:
        T = max(T, npz_obj["pose"].shape[0])
    if include_hands:
        if "left_hand" in npz_obj:
            T = max(T, npz_obj["left_hand"].shape[0])
        if "right_hand" in npz_obj:
            T = max(T, npz_obj["right_hand"].shape[0])
    if include_face and "face" in npz_obj:
        T = max(T, npz_obj["face"].shape[0])
    if T == 0:
        return None

    def pick_coords(x: np.ndarray) -> np.ndarray:
        if x.shape[-1] >= in_coords:
            return x[..., :in_coords]
        pad = np.zeros(list(x.shape[:-1]) + [in_coords - x.shape[-1]], dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)

    parts: List[np.ndarray] = []
    if "pose" in npz_obj:
        pose = npz_obj["pose"]
        pose = pose[:, :, : max(1, in_coords)]
    else:
        pose = np.full((T, 33, in_coords), np.nan, dtype=np.float32)
    parts.append(pick_coords(pose))

    if include_hands:
        if "left_hand" in npz_obj:
            lh = pick_coords(npz_obj["left_hand"])
        else:
            lh = np.full((T, 21, in_coords), np.nan, dtype=np.float32)
        if "right_hand" in npz_obj:
            rh = pick_coords(npz_obj["right_hand"])
        else:
            rh = np.full((T, 21, in_coords), np.nan, dtype=np.float32)
        parts.extend([lh, rh])

    if include_face:
        if "face" in npz_obj:
            face = pick_coords(npz_obj["face"])
        else:
            face = np.full((T, 478, in_coords), np.nan, dtype=np.float32)
        parts.append(face)

    arr = np.concatenate(parts, axis=1)
    return arr


def order_start_positions(
    total_frames: int,
    window: int,
    stride: int,
    mode: str,
    rng: np.random.Generator | None,
) -> List[int]:
    if total_frames < window:
        return []
    starts = list(range(0, total_frames - window + 1, stride))
    if mode == "first":
        return starts
    assert rng is not None
    rng.shuffle(starts)
    return starts


def save_segment_file(
    base_output: str,
    split: str,
    video_id: str,
    seg_idx: int,
    start_frame: int,
    segment: np.ndarray,
    stats: np.ndarray,
    label_id: int,
    label_str: str,
    config_bytes: bytes,
    fs=None,
):
    metadata = np.array([(video_id, int(start_frame), label_str)], dtype=METADATA_DTYPE)
    labels_arr = np.array([label_id], dtype=np.int64)
    if is_gcs_path(base_output):
        try:
            import gcsfs  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires optional dep
            raise ImportError("gcsfs is required when writing to GCS paths.") from exc
        fs = fs or gcsfs.GCSFileSystem()
        base = base_output.rstrip("/")
        dest_dir = f"{base}/{split}/segments/{video_id}"
        dest_path = f"{dest_dir}/{video_id}_seg{seg_idx:06d}.npz"
        with fs.open(dest_path, "wb") as handle:
            np.savez_compressed(
                handle,
                X=segment[np.newaxis, ...].astype(np.float32),
                stats=stats[np.newaxis, ...].astype(np.float32),
                labels=labels_arr,
                metadata=metadata,
                config=config_bytes,
            )
    else:
        base_path = Path(base_output).expanduser()
        dest_dir_path = base_path / split / "segments" / video_id
        dest_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir_path / f"{video_id}_seg{seg_idx:06d}.npz"
        np.savez_compressed(
            out_path,
            X=segment[np.newaxis, ...].astype(np.float32),
            stats=stats[np.newaxis, ...].astype(np.float32),
            labels=labels_arr,
            metadata=metadata,
            config=config_bytes,
        )


def process_video(args) -> Tuple[str, int, bool, str | None]:
    (
        video_path,
        split_name,
        output_root,
        window,
        stride,
        in_coords,
        include_hands,
        include_face,
        max_segments,
        sample_mode,
        target_label,
        label_id,
        map_unknown_to_n,
        config_bytes,
        seed,
        groundtruth,
    ) = args

    video_path = Path(video_path)
    video_id = video_path.stem
    rng = np.random.default_rng(seed + (abs(hash(video_id)) % (2**31)))
    gcs_fs = None
    if is_gcs_path(output_root):
        try:
            import gcsfs  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("gcsfs is required when writing to GCS paths.") from exc
        gcs_fs = gcsfs.GCSFileSystem()
    try:
        with np.load(video_path, allow_pickle=True) as npz_obj:
            arr = build_feature_array(npz_obj, in_coords, include_hands, include_face)
    except Exception as exc:
        return video_id, 0, False, f"failed to load: {exc}"

    if arr is None:
        return video_id, 0, False, "no usable landmarks"

    starts = order_start_positions(
        total_frames=arr.shape[0],
        window=window,
        stride=stride,
        mode=sample_mode,
        rng=rng if sample_mode == "random" else None,
    )
    if not starts:
        return video_id, 0, True, None

    saved = 0
    for order_idx, start in enumerate(starts):
        if max_segments is not None and saved >= max_segments:
            break
        segment = np.nan_to_num(arr[start : start + window], nan=0.0, posinf=0.0, neginf=0.0)
        stats = compute_pose_stats(segment)
        frame_labels = []
        for t in range(window):
            frame_id = start + t
            lab = groundtruth.get((video_id, frame_id))
            if lab is None:
                continue
            lab = str(lab)
            if map_unknown_to_n and lab == "?":
                lab = "n"
            frame_labels.append(lab)
        if not frame_labels:
            label_str = "unknown"
        else:
            vals, counts = np.unique(np.array(frame_labels), return_counts=True)
            label_str = vals[np.argmax(counts)]
        if label_str != target_label:
            continue

        save_segment_file(
            base_output=output_root,
            split=split_name,
            video_id=video_id,
            seg_idx=saved,
            start_frame=start,
            segment=segment,
            stats=stats,
            label_id=label_id,
            label_str=label_str,
            config_bytes=config_bytes,
            fs=gcs_fs,
        )
        saved += 1

    return video_id, saved, True, None


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_arg = args.output_dir.rstrip("/")
    if is_gcs_path(output_arg):
        output_root = output_arg
    else:
        output_path = Path(output_arg).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        output_root = str(output_path)

    include_hands = not args.exclude_hands

    files = list_input_files(input_root, recursive=args.recursive)
    if not files:
        raise FileNotFoundError(f"No NPZ files found under {input_root}")

    split_assignments = assign_splits(
        files, args.train_ratio, args.val_ratio, args.test_ratio, args.split_seed
    )
    split_video_counts = Counter(split_assignments.values())

    label_map = create_label_mapping(args.num_classes, map_unknown_to_n=args.map_unknown_to_n)
    if args.target_label not in label_map:
        label_map[args.target_label] = len(label_map)
    target_label_id = label_map[args.target_label]

    gt_path = Path(args.groundtruth).expanduser()
    if not gt_path.exists():
        raise FileNotFoundError(f"Groundtruth file not found: {gt_path}")
    groundtruth = read_groundtruth(str(gt_path))

    config = {
        "window": args.window,
        "stride": args.stride,
        "coords": args.in_coords,
        "num_classes": args.num_classes,
        "label_map": label_map,
        "include_pose": True,
        "include_hands": include_hands,
        "include_face": args.include_face,
        "source": str(input_root),
        "note": "Auto-generated from P-labeled landmarks",
    }
    config_bytes = pickle.dumps(config)

    worker_args = []
    for path in files:
        split_name = split_assignments[str(path)]
        worker_args.append(
            (
                str(path),
                split_name,
                output_root,
                args.window,
                args.stride,
                args.in_coords,
                include_hands,
                args.include_face,
                args.max_segments_per_video,
                args.sample_mode,
                args.target_label,
                target_label_id,
                args.map_unknown_to_n,
                config_bytes,
                args.seed,
                groundtruth,
            )
        )

    print(f"Found {len(worker_args)} videos to process.")
    print(
        "Split assignment (videos): "
        + ", ".join(f"{split}={count}" for split, count in split_video_counts.items())
    )
    total_segments = 0
    failures = 0
    split_segment_totals: Counter[str] = Counter()

    if args.workers <= 1:
        iterator: Iterable = tqdm(worker_args, desc="Videos")
        for job in iterator:
            split_name = job[1]
            _, created, ok, err = process_video(job)
            total_segments += created
            split_segment_totals[split_name] += created
            if not ok and err:
                failures += 1
                print(f"[ERROR] {job[0]}: {err}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_video, job): (job[0], job[1]) for job in worker_args
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Videos"):
                video_file, split_name = futures[future]
                try:
                    _, created, ok, err = future.result()
                except Exception as exc:
                    failures += 1
                    print(f"[ERROR] {video_file}: {exc}")
                    continue
                total_segments += created
                split_segment_totals[split_name] += created
                if not ok and err:
                    failures += 1
                    print(f"[ERROR] {video_file}: {err}")

    print("\n========== Summary ==========")
    print(f"Videos processed: {len(worker_args)}")
    print(f"Segments saved:   {total_segments}")
    if split_segment_totals:
        print(
            "Segments per split: "
            + ", ".join(f"{split}={count}" for split, count in split_segment_totals.items())
        )
    if failures:
        print(f"Videos with errors: {failures}")
    print(f"Output directory: {output_root}")
    print("=============================")


if __name__ == "__main__":
    main()

