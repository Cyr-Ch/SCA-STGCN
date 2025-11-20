"""
Summarize labels for every per-segment NPZ across all splits/videos.

The script walks each split's `segments/<video_id>/...npz` directory (local or GCS),
reads the stored metadata, and writes a CSV file listing the label for every
segment. Useful for auditing label distribution per video.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets.landmarks_npz import load_npz  # noqa: E402


def is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def list_segment_files(
    base_root: str,
    split: str,
    *,
    fs=None,
) -> Iterator[str]:
    """
    Yield absolute paths to each per-segment NPZ file for a split.
    """
    base_segments = f"{base_root.rstrip('/')}/{split}/segments"
    if is_gcs(base_root):
        if fs is None:
            import gcsfs  # lazy import

            fs = gcsfs.GCSFileSystem()
        if not fs.exists(base_segments):
            raise FileNotFoundError(f"Split segments not found in GCS: {base_segments}")
        for path in fs.find(base_segments):
            if not path.endswith(".npz"):
                continue
            yield path if path.startswith("gs://") else f"gs://{path}"
    else:
        base_path = Path(base_segments)
        if not base_path.exists():
            raise FileNotFoundError(f"Split segments not found: {base_segments}")
        for seg_file in sorted(base_path.glob("*/*.npz")):
            if seg_file.is_file():
                yield str(seg_file)


def summarize_segments(
    segments_root: str,
    splits: Iterable[str],
    output_csv: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    gcs_fs = None
    if is_gcs(segments_root):
        import gcsfs  # lazy import

        gcs_fs = gcsfs.GCSFileSystem()

    total_segments = 0
    per_split_counts = {}

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["split", "video", "segment_file", "start", "label_text", "label_id"]
        )

        for split in splits:
            try:
                segment_files = list_segment_files(
                    segments_root, split, fs=gcs_fs
                )
            except FileNotFoundError:
                print(f"[WARNING] Split '{split}' missing under {segments_root}, skipping.")
                continue

            split_count = 0
            for seg_path in segment_files:
                fs = gcs_fs if is_gcs(seg_path) else None
                with load_npz(seg_path, fs=fs) as npz:
                    label_id = int(npz["labels"][0])
                    meta = npz["metadata"][0]
                    segment_name = os.path.basename(seg_path)
                    writer.writerow(
                        [
                            split,
                            str(meta["video"]),
                            segment_name,
                            int(meta["start"]),
                            str(meta["label"]),
                            label_id,
                        ]
                    )
                split_count += 1

            total_segments += split_count
            per_split_counts[split] = split_count
            print(f"[{split}] Recorded labels for {split_count} segments.")

    print("\nSummary file written to:", output_csv)
    for split, count in per_split_counts.items():
        print(f"  {split}: {count} segments")
    print(f"Total segments processed: {total_segments}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CSV listing labels for every per-segment NPZ."
    )
    parser.add_argument(
        "--segments-root",
        required=True,
        help="Root directory or gs://bucket containing <split>/segments/<video_id> folders.",
    )
    parser.add_argument(
        "--output-file",
        default="segment_labels_summary.csv",
        help="Destination CSV file (local path).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to scan (default: train val test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summarize_segments(args.segments_root, args.splits, args.output_file)


if __name__ == "__main__":
    main()

