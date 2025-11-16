"""
Quick diagnostic script to check .npz files in the landmarks directory.
Run as: python check_npz_files.py --data .\landmarks\
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path


def check_npz_files(data_dir, window_size=25):
    """Check .npz files and report their structure"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return
    
    npz_files = sorted(data_path.glob('*.npz'))
    
    if len(npz_files) == 0:
        print(f"[ERROR] No .npz files found in {data_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files")
    print(f"Window size: {window_size}")
    print("=" * 70)
    
    total_windows = 0
    valid_files = 0
    invalid_files = []
    
    for i, fpath in enumerate(npz_files[:10]):  # Check first 10 files
        fname = fpath.name
        vid_id = fpath.stem
        
        try:
            with np.load(fpath) as npz:
                keys = list(npz.keys())
                
                # Find the maximum T (time dimension)
                T = 0
                found_keys = []
                
                for key in ('pose', 'left_hand', 'right_hand', 'face'):
                    if key in npz:
                        shape = npz[key].shape
                        T = max(T, shape[0])
                        found_keys.append(f"{key}: {shape}")
                
                if T == 0 and len(keys) > 0:
                    # Fallback: use first key
                    first_key = keys[0]
                    T = npz[first_key].shape[0]
                    found_keys.append(f"{first_key}: {npz[first_key].shape}")
                
                # Calculate number of windows
                num_windows = max(0, T - window_size + 1) if T >= window_size else 0
                total_windows += num_windows
                
                status = "[OK]" if num_windows > 0 else "[FAIL]"
                if num_windows > 0:
                    valid_files += 1
                else:
                    invalid_files.append((fname, T, window_size))
                
                print(f"\n{status} {fname} (ID: {vid_id})")
                print(f"  Keys: {keys}")
                print(f"  Found: {', '.join(found_keys) if found_keys else 'None'}")
                print(f"  Max frames (T): {T}")
                print(f"  Windows possible: {num_windows}")
                if T < window_size:
                    print(f"  ⚠️  Too short! Need {window_size} frames, have {T}")
                    
        except Exception as e:
            print(f"\n[ERROR] {fname}: ERROR - {e}")
            invalid_files.append((fname, "ERROR", str(e)))
    
    if len(npz_files) > 10:
        print(f"\n... and {len(npz_files) - 10} more files")
    
    print("\n" + "=" * 70)
    print(f"Summary:")
    print(f"  Total files: {len(npz_files)}")
    print(f"  Valid files (>={window_size} frames): {valid_files}")
    print(f"  Invalid files: {len(invalid_files)}")
    print(f"  Total windows (first 10 files): {total_windows}")
    
    if invalid_files:
        print(f"\nInvalid files:")
        for fname, T, reason in invalid_files:
            print(f"  - {fname}: {reason}")
    
    # Check groundtruth file
    gt_path = data_path / 'groundtruth.txt'
    if not gt_path.exists():
        gt_path = data_path / 'groundtruth'
    
    if gt_path.exists():
        print(f"\n[OK] Groundtruth file found: {gt_path.name}")
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            print(f"  Lines: {len(lines)}")
            if lines:
                print(f"  First line: {lines[0].strip()}")
    else:
        print(f"\n[FAIL] Groundtruth file not found (checked groundtruth.txt and groundtruth)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check .npz files in landmarks directory")
    parser.add_argument('--data', required=True, help='Path to landmarks folder')
    parser.add_argument('--window', type=int, default=25, help='Window size (default: 25)')
    
    args = parser.parse_args()
    check_npz_files(args.data, args.window)

