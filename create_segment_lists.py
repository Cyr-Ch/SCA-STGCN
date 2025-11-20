"""
Build per-video segment lists with label quotas.

For each split we scan <segments_root>/<split>/segments/<video_id>/*.npz,
select up to 2,000 S/P segments and 500 n segments per video, and write the
kept segment paths to:

    <segments_root>/<split>/<lists_subdir>/<video_id>.txt

These per-video files can later be merged into dataset-level manifests.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

from tqdm import tqdm  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets.landmarks_npz import load_npz  # noqa: E402

SP_LIMIT = 2000
N_LIMIT = 500


def is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def list_video_dirs_local(base_segments: Path) -> Iterator[Tuple[str, List[str]]]:
    if not base_segments.exists():
        raise FileNotFoundError(f"Segments directory missing: {base_segments}")
    for video_dir in sorted(p for p in base_segments.iterdir() if p.is_dir()):
        seg_files = sorted(str(p) for p in video_dir.glob("*.npz"))
        if seg_files:
            yield video_dir.name, seg_files


def list_video_dirs_gcs(base_path: str, fs) -> Iterator[Tuple[str, List[str]]]:
    if not fs.exists(base_path):
        raise FileNotFoundError(f"Segments directory missing in GCS: {base_path}")
    entries = fs.ls(base_path, detail=True)
    for entry in entries:
        if entry.get("type") != "directory":
            continue
        video_dir = entry["name"]
        video_id = video_dir.rstrip("/").split("/")[-1]
        seg_files = sorted(
            path if path.startswith("gs://") else f"gs://{path}"
            for path in fs.ls(video_dir)
            if path.endswith(".npz")
        )
        if seg_files:
            yield video_id, seg_files


def normalize_label(label: str) -> str:
    return label.strip().lower()


def process_video_segments(video_id: str, segment_paths: Sequence[str]) -> Tuple[str, List[str], int, int]:
    counts = {"sp": 0, "n": 0}
    selected: List[str] = []
    for seg_path in segment_paths:
        with load_npz(seg_path) as npz:
            metadata = npz["metadata"][0]
            label_text = str(metadata["label"])
        norm = normalize_label(label_text)
        if norm in {"s", "sign", "signing"} or norm in {"p", "speak", "speaking"}:
            if counts["sp"] < SP_LIMIT:
                counts["sp"] += 1
                selected.append(seg_path)
        elif norm in {"n", "no", "none"}:
            if counts["n"] < N_LIMIT:
                counts["n"] += 1
                selected.append(seg_path)
    return video_id, selected, counts["sp"], counts["n"]


def write_list(output_path: str, lines: List[str], fs=None) -> None:
    if output_path.startswith("gs://"):
        import gcsfs  # type: ignore

        fs = fs or gcsfs.GCSFileSystem()
        with fs.open(output_path, "w") as f:
            f.write("\n".join(lines))
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_per_video_lists(
    segments_root: str, splits: Iterable[str], lists_subdir: str, workers: int
) -> None:
    gcs_fs = None
    if is_gcs(segments_root):
        import gcsfs  # type: ignore

        gcs_fs = gcsfs.GCSFileSystem()

    for split in splits:
        base_segments = f"{segments_root.rstrip('/')}/{split}/segments"
        try:
            if is_gcs(segments_root):
                video_entries = list(list_video_dirs_gcs(base_segments, gcs_fs))
            else:
                video_entries = list(list_video_dirs_local(Path(base_segments)))
        except FileNotFoundError:
            print(f"[WARNING] Split '{split}' missing; skipping.")
            continue

        if not video_entries:
            print(f"[INFO] Split '{split}' has no videos; skipping.")
            continue

        progress_bar = tqdm(total=len(video_entries), desc=f"{split} videos", unit="video")

        results: List[Tuple[str, List[str], int, int]] = []
        if workers <= 1:
            for video_id, seg_paths in video_entries:
                results.append(process_video_segments(video_id, seg_paths))
                progress_bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(process_video_segments, video_id, seg_paths): video_id
                    for video_id, seg_paths in video_entries
                }
                for future in as_completed(futures):
                    video_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        print(f"[ERROR] Failed to process {video_id}: {exc}")
                    finally:
                        progress_bar.update(1)
        progress_bar.close()

        split_sp = 0
        split_n = 0
        written = 0

        for video_id, selected_paths, sp_count, n_count in results:
            if not selected_paths:
                continue
            split_sp += sp_count
            split_n += n_count
            written += 1
            if is_gcs(segments_root):
                fs = gcs_fs
                output_path = (
                    f"{segments_root.rstrip('/')}/{split}/{lists_subdir}/{video_id}.txt"
                )
            else:
                fs = None
                output_path = str(
                    Path(segments_root) / split / lists_subdir / f"{video_id}.txt"
                )
            write_list(output_path, selected_paths, fs=fs)

        print(
            f"[{split}] Wrote {written} per-video files "
            f"(total S/P segments: {split_sp}, total n segments: {split_n})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-video segment lists with label quotas."
    )
    parser.add_argument(
        "--segments-root",
        required=True,
        help="Root directory or gs:// bucket with <split>/segments/<video_id> folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test).",
    )
    parser.add_argument(
        "--lists-subdir",
        default="segment_lists",
        help="Subdirectory under each split to store per-video list files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (per video parallelism).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_per_video_lists(args.segments_root, args.splits, args.lists_subdir, args.workers)


if __name__ == "__main__":
    main()

