"""
Assemble dataset-level segment lists from per-video selections.

This script reads the text files produced by create_segment_lists.py, which store
per-video segment paths under:

    <segments_root>/<split>/<lists_subdir>/<video_id>.txt

and concatenates them into split-level manifests:

    <segments_root>/<split>_segments_list.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Iterator, List

from tqdm import tqdm  # type: ignore


def is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def list_per_video_files_local(base_dir: Path) -> Iterator[Path]:
    if not base_dir.exists():
        return iter([])
    return (entry for entry in sorted(base_dir.glob("*.txt")) if entry.is_file())


def list_per_video_files_gcs(base_dir: str, fs) -> List[str]:
    if not fs.exists(base_dir):
        return []
    files = [
        path if path.startswith("gs://") else f"gs://{path}"
        for path in fs.ls(base_dir)
        if path.endswith(".txt")
    ]
    return sorted(files)


def read_lines(path: str, fs=None) -> List[str]:
    if path.startswith("gs://"):
        import gcsfs  # type: ignore

        fs = fs or gcsfs.GCSFileSystem()
        with fs.open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_output(path: str, lines: List[str], fs=None) -> None:
    if path.startswith("gs://"):
        import gcsfs  # type: ignore

        fs = fs or gcsfs.GCSFileSystem()
        with fs.open(path, "w") as f:
            f.write("\n".join(lines))
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def assemble_lists(segments_root: str, splits: Iterable[str], lists_subdir: str) -> None:
    gcs_fs = None
    if is_gcs(segments_root):
        import gcsfs  # type: ignore

        gcs_fs = gcsfs.GCSFileSystem()

    for split in splits:
        if is_gcs(segments_root):
            per_video_dir = f"{segments_root.rstrip('/')}/{split}/{lists_subdir}"
            files = list_per_video_files_gcs(per_video_dir, gcs_fs)
        else:
            per_video_dir = Path(segments_root) / split / lists_subdir
            files = [str(p) for p in list_per_video_files_local(per_video_dir)]

        if not files:
            print(f"[WARNING] No per-video lists found for split '{split}', skipping.")
            continue

        combined: List[str] = []
        for file_path in tqdm(files, desc=f"{split} files", unit="file"):
            combined.extend(read_lines(file_path, fs=gcs_fs))

        output_path = (
            f"{segments_root.rstrip('/')}/{split}_segments_list.txt"
            if is_gcs(segments_root)
            else str(Path(segments_root) / f"{split}_segments_list.txt")
        )
        write_output(output_path, combined, fs=gcs_fs)
        print(f"[{split}] Wrote {len(combined)} segment paths to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-video segment lists into dataset-level manifests."
    )
    parser.add_argument(
        "--segments-root",
        required=True,
        help="Root directory or gs:// bucket containing per-video list files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to assemble (default: train val test).",
    )
    parser.add_argument(
        "--lists-subdir",
        default="segment_lists",
        help="Subdirectory under each split containing per-video text files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assemble_lists(args.segments_root, args.splits, args.lists_subdir)


if __name__ == "__main__":
    main()

