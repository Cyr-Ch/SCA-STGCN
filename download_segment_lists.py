"""
Download assembled segment lists from GCS to the VM.

Usage:
    python download_segment_lists.py --gcs-root gs://segments-v0 --splits train val
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def download_file(fs, gcs_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with fs.open(gcs_path, "rb") as src, open(local_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def main():
    parser = argparse.ArgumentParser(description="Download segment list manifests from GCS.")
    parser.add_argument("--gcs-root", required=True, help="gs:// bucket root containing *_segments_list.txt files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to download.")
    parser.add_argument("--output-dir", default="segment_lists_downloaded", help="Local directory to store the files.")
    args = parser.parse_args()

    import gcsfs  # type: ignore

    fs = gcsfs.GCSFileSystem()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        gcs_path = f"{args.gcs_root.rstrip('/')}/{split}_segments_list.txt"
        local_path = out_dir / f"{split}_segments_list.txt"
        print(f"Downloading {gcs_path} -> {local_path}")
        download_file(fs, gcs_path, local_path)
        print(f"[DONE] {split}: saved to {local_path}")


if __name__ == "__main__":
    main()

