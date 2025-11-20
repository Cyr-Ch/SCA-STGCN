"""
Convert per-video segment NPZ files into per-segment NPZ files.

Each original video file (e.g., train/video123.npz) contains many segments in
arrays X, stats, labels, metadata. This script splits those into individual
NPZ files (one segment per file) and saves them under:

    <output_dir>/<split>/segments/<video_id>/<video_id>_segXXXXXX.npz

Works with local paths or GCS buckets (gs://...).
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Ensure repo root is on path so we can import helpers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets.landmarks_npz import load_npz  # noqa: E402


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def list_npz_files(base_path: str, split: str, fs=None):
    """Return sorted list of NPZ files for a split."""
    split_path = f"{base_path.rstrip('/')}/{split}"
    if is_gcs_path(base_path):
        if fs is None:
            import gcsfs  # lazy import

            fs = gcsfs.GCSFileSystem()
        if not fs.exists(split_path):
            raise FileNotFoundError(f"Split directory not found in GCS: {split_path}")
        files = [
            f if f.startswith("gs://") else f"gs://{f}"
            for f in fs.ls(split_path)
            if f.endswith(".npz")
        ]
        return sorted(files), fs
    # Local path
    local_split = os.path.join(base_path, split)
    if not os.path.exists(local_split):
        raise FileNotFoundError(f"Split directory not found: {local_split}")
    files = sorted(
        str(Path(local_split) / fname)
        for fname in os.listdir(local_split)
        if fname.endswith(".npz")
    )
    return files, fs


def make_output_path(base_output: str, split: str, video_id: str, seg_idx: int) -> str:
    segment_root = f"{base_output.rstrip('/')}/{split}/segments/{video_id}"
    filename = f"{video_id}_seg{seg_idx:06d}.npz"
    return f"{segment_root}/{filename}"


def save_segment_npz(
    output_path: str,
    X_seg,
    stats_seg,
    label_seg,
    metadata_seg,
    config_bytes,
    fs=None,
    overwrite=False,
):
    if is_gcs_path(output_path):
        if fs is None:
            import gcsfs

            fs = gcsfs.GCSFileSystem()
        if not overwrite and fs.exists(output_path):
            return False
        with fs.open(output_path, "wb") as f:
            np_savez_segment(f, X_seg, stats_seg, label_seg, metadata_seg, config_bytes)
        return True

    # Local path
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    if not overwrite and os.path.exists(output_path):
        return False
    with open(output_path, "wb") as f:
        np_savez_segment(f, X_seg, stats_seg, label_seg, metadata_seg, config_bytes)
    return True


def np_savez_segment(handle, X_seg, stats_seg, label_seg, metadata_seg, config_bytes):
    import numpy as np

    # Ensure shapes include leading dimension of 1 (compatibility with loaders)
    X_arr = X_seg[np.newaxis, ...]
    stats_arr = stats_seg[np.newaxis, ...]
    labels_arr = np.asarray([label_seg], dtype=np.int64)
    metadata_arr = metadata_seg.reshape(1)  # already structured dtype

    np.savez_compressed(
        handle,
        X=X_arr,
        stats=stats_arr,
        labels=labels_arr,
        metadata=metadata_arr,
        config=config_bytes,
    )


def process_video_file(args):
    """Worker helper to process a single video file."""
    (
        video_file,
        split,
        base_output,
        overwrite,
    ) = args

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    segments_created = 0

    try:
        with load_npz(video_file) as npz:
            X = npz["X"]
            stats = npz["stats"]
            labels = npz["labels"]
            metadata = npz["metadata"]
            config_bytes = npz["config"]
    except Exception as exc:
        return video_file, segments_created, False, str(exc)

    num_segments = len(X)
    for seg_idx in range(num_segments):
        out_path = make_output_path(base_output, split, video_name, seg_idx)
        saved = save_segment_npz(
            out_path,
            X[seg_idx],
            stats[seg_idx],
            labels[seg_idx],
            metadata[seg_idx : seg_idx + 1],
            config_bytes,
            overwrite=overwrite,
        )
        if saved:
            segments_created += 1

    return video_file, segments_created, True, None


def main():
    parser = argparse.ArgumentParser(
        description="Split per-video segment NPZ files into per-segment NPZ files."
    )
    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        help="Path or GCS bucket containing per-video NPZ files (train/val/test subdirs).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination root (local path or gs://bucket) where per-segment files will be saved.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process (default: train val test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-segment files if they already exist.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Optional: limit number of videos per split (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (processes) to use per split.",
    )
    args = parser.parse_args()

    base_input = args.preprocessed_dir
    base_output = args.output_dir

    total_segments = 0

    for split in args.splits:
        try:
            video_files, _ = list_npz_files(base_input, split, fs=None)
        except FileNotFoundError:
            print(f"[WARNING] Split '{split}' missing in {base_input}, skipping.")
            continue

        if not video_files:
            print(f"[INFO] No video files for split '{split}', skipping.")
            continue

        if args.limit_videos:
            video_files = video_files[: args.limit_videos]

        print(f"\n[{split.upper()}] Processing {len(video_files)} videos...")

        split_created = 0
        failures = 0

        worker_args = [
            (video_file, split, base_output, args.overwrite)
            for video_file in video_files
        ]

        if args.workers <= 1:
            iterator = tqdm(worker_args, desc=f"{split} videos")
            for job in iterator:
                _, created, ok, err = process_video_file(job)
                split_created += created
                if not ok:
                    failures += 1
                    print(f"[ERROR] Failed to process {job[0]}: {err}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(process_video_file, job): job[0] for job in worker_args
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{split} videos",
                ):
                    video_file = futures[future]
                    try:
                        _, created, ok, err = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        failures += 1
                        print(f"[ERROR] Failed to process {video_file}: {exc}")
                        continue

                    split_created += created
                    if not ok:
                        failures += 1
                        print(f"[ERROR] Failed to process {video_file}: {err}")

        total_segments += split_created
        print(
            f"[{split.upper()}] Finished processing {len(video_files)} videos. "
            f"Segments saved: {split_created}"
        )
        if failures:
            print(f"[{split.upper()}] Videos with errors: {failures}")

    print("\n=============================================")
    print(f"Finished! Total per-segment files created: {total_segments}")
    print(f"Output root: {base_output}")
    print("=============================================")


if __name__ == "__main__":
    main()

