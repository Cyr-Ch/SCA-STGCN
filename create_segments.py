"""
Script to pre-compute segments from landmark data organized by video splits.
Each video's segments are saved in a separate NPZ file, organized by split (train/val/test).

Usage:
    # First, create video splits:
    python build_video_splits.py --data /path/to/landmarks --out splits.json
    
    # Then, create segments:
    python create_segments.py --data /path/to/landmarks --splits splits.json --output-dir segments
"""

import os
import sys
import json
import argparse
import pickle
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from datasets.landmarks_npz import read_groundtruth, compute_pose_stats, create_label_mapping


def process_file(
    path: str,
    vid: str,
    gt: dict,
    window: int,
    stride: int,
    in_coords: int,
    include_pose: bool,
    include_hands: bool,
    include_face: bool,
    label_map: dict,
    map_unknown_to_n: bool,
    num_classes: int,
) -> tuple[list, list, list, list]:
    """
    Process a single NPZ file and extract all segments for one video.
    
    Returns:
        segments: List of (T, J, C) arrays
        stats: List of (D,) stat arrays
        labels: List of integer labels
        metadata: List of dicts with video, start, label_str
    """
    segments = []
    stats_list = []
    labels_list = []
    metadata_list = []
    
    with np.load(path) as npz:
        # Determine T (max length across all parts)
        T = 0
        if include_pose and 'pose' in npz:
            T = max(T, npz['pose'].shape[0])
        if include_hands:
            if 'left_hand' in npz:
                T = max(T, npz['left_hand'].shape[0])
            if 'right_hand' in npz:
                T = max(T, npz['right_hand'].shape[0])
        if include_face and 'face' in npz:
            T = max(T, npz['face'].shape[0])
        
        if T == 0:
            return segments, stats_list, labels_list, metadata_list
        
        def pick_coords(x: np.ndarray) -> np.ndarray:
            if x.shape[-1] >= in_coords:
                return x[..., :in_coords]
            pad = np.zeros(list(x.shape[:-1]) + [in_coords - x.shape[-1]], dtype=x.dtype)
            return np.concatenate([x, pad], axis=-1)
        
        # Build concatenated array
        parts = []
        if include_pose:
            if 'pose' in npz:
                pose = npz['pose']
                pose = pose[:, :, :max(1, in_coords)]
            else:
                pose = np.full((T, 33, in_coords), np.nan, dtype=np.float32)
            parts.append(pick_coords(pose))
        
        if include_hands:
            if 'left_hand' in npz:
                lh = pick_coords(npz['left_hand'])
            else:
                lh = np.full((T, 21, in_coords), np.nan, dtype=np.float32)
            if 'right_hand' in npz:
                rh = pick_coords(npz['right_hand'])
            else:
                rh = np.full((T, 21, in_coords), np.nan, dtype=np.float32)
            parts += [lh, rh]
        
        if include_face:
            if 'face' in npz:
                face = pick_coords(npz['face'])
            else:
                face = np.full((T, 478, in_coords), np.nan, dtype=np.float32)
            parts.append(face)
        
        if not parts:
            return segments, stats_list, labels_list, metadata_list
        
        arr = np.concatenate(parts, axis=1)  # (T, J, C)
        # Free parts memory (no longer needed after concatenation)
        del parts
        
        # Create sliding windows
        num_windows = max(0, T - window + 1)
        for s in range(0, num_windows, stride):
            window_np = arr[s:s + window].copy()  # (window, J, C)
            
            # Replace NaNs/Infs with zeros
            window_np = np.nan_to_num(window_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get label from groundtruth (majority voting)
            labels = []
            for t in range(window):
                frame_id = s + t
                lab = gt.get((vid, frame_id), None)
                if lab is not None:
                    if map_unknown_to_n and lab == '?':
                        lab = 'n'
                    labels.append(lab)
            
            if len(labels) == 0:
                y_str = 'unknown'
            else:
                vals, counts = np.unique(np.array(labels), return_counts=True)
                y_str = vals[np.argmax(counts)].item() if hasattr(vals[0], 'item') else vals[np.argmax(counts)]
            
            # Map label to class ID
            if y_str in label_map:
                y = label_map[y_str]
            else:
                # Handle unknown labels
                if num_classes is not None:
                    # Map to "other" if available
                    if 'other' not in label_map:
                        label_map['other'] = len(label_map)
                    y = label_map.get('other', len(label_map) - 1)
                else:
                    # Dynamic mapping
                    label_map[y_str] = len(label_map)
                    y = label_map[y_str]
            
            # Compute stats
            stats_np = compute_pose_stats(window_np)
            
            segments.append(window_np)
            stats_list.append(stats_np)
            labels_list.append(y)
            metadata_list.append({
                'video': vid,
                'start': s,
                'label': y_str,
            })
    
    # Free the large concatenated array before returning
    del arr
    
    return segments, stats_list, labels_list, metadata_list


def save_video_segments(
    output_path: str,
    segments: list,
    stats_list: list,
    labels_list: list,
    metadata_list: list,
    config: dict,
):
    """Save segments for a single video to an NPZ file."""
    if len(segments) == 0:
        return False
    
    # Convert to numpy arrays
    X = np.array(segments, dtype=np.float32)  # (N, T, J, C)
    stats = np.array(stats_list, dtype=np.float32)  # (N, D)
    labels = np.array(labels_list, dtype=np.int64)  # (N,)
    
    # Save metadata as structured array
    metadata_dtype = [
        ('video', 'U100'),  # Unicode string, max 100 chars
        ('start', np.int32),
        ('label', 'U20'),  # Unicode string, max 20 chars
    ]
    metadata_array = np.array(
        [(m['video'], m['start'], m['label']) for m in metadata_list],
        dtype=metadata_dtype
    )
    
    # Save config as pickled bytes
    config_bytes = pickle.dumps(config)
    
    # Save to NPZ
    np.savez_compressed(
        output_path,
        X=X,
        stats=stats,
        labels=labels,
        metadata=metadata_array,
        config=config_bytes,
    )
    
    # Explicitly free memory after saving
    del X, stats, labels, metadata_array, config_bytes
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute segments from landmark data organized by video splits"
    )
    parser.add_argument('--data', required=True, help='Path to landmarks folder')
    parser.add_argument('--splits', required=True, help='Path to JSON file with video splits (from build_video_splits.py)')
    parser.add_argument('--output-dir', required=True, help='Output directory for segment files (will create train/val/test subdirectories)')
    parser.add_argument('--window', type=int, default=25, help='Temporal window size')
    parser.add_argument('--stride', type=int, default=1, help='Window stride')
    parser.add_argument('--coords', type=int, default=2, help='Number of coordinates')
    parser.add_argument('--include-pose', action='store_true', help='Include pose landmarks')
    parser.add_argument('--include-hands', action='store_true', help='Include hand landmarks')
    parser.add_argument('--include-face', action='store_false', help='Include face landmarks')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--map-unknown-to-n', action='store_true', help='Map ? to n')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    
    # Load splits JSON
    if not os.path.exists(args.splits):
        print(f"[ERROR] Splits file not found: {args.splits}")
        print("First run: python build_video_splits.py --data <path> --out splits.json")
        sys.exit(1)
    
    with open(args.splits, 'r') as f:
        splits_data = json.load(f)
    
    splits = splits_data.get('splits', {})
    train_videos = splits.get('train', {}).get('videos', [])
    val_videos = splits.get('val', {}).get('videos', [])
    test_videos = splits.get('test', {}).get('videos', [])
    
    print(f"Loaded splits:")
    print(f"  Train: {len(train_videos)} videos")
    print(f"  Val: {len(val_videos)} videos")
    print(f"  Test: {len(test_videos)} videos")
    
    # Find groundtruth file
    gt_path = os.path.join(args.data, 'groundtruth.txt')
    if not os.path.exists(gt_path):
        gt_path = os.path.join(args.data, 'groundtruth')
        if not os.path.exists(gt_path):
            print(f"[WARNING] No groundtruth file found. Labels will be 'unknown'.")
            gt = {}
        else:
            gt = read_groundtruth(gt_path)
    else:
        gt = read_groundtruth(gt_path)
    
    print(f"Loaded {len(gt)} groundtruth entries")
    
    # Create label mapping
    label_map = create_label_mapping(args.num_classes, args.map_unknown_to_n)
    print(f"Label mapping: {label_map}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directories:")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    print(f"  Test: {test_dir}")
    
    # Configuration for saving
    config = {
        'window': args.window,
        'stride': args.stride,
        'coords': args.coords,
        'include_pose': args.include_pose,
        'include_hands': args.include_hands,
        'include_face': args.include_face,
        'num_classes': args.num_classes,
        'map_unknown_to_n': args.map_unknown_to_n,
        'label_map': label_map,
    }
    
    print(f"\nConfiguration:")
    print(f"  Window: {args.window}")
    print(f"  Stride: {args.stride}")
    print(f"  Coords: {args.coords}")
    print(f"  Include pose: {args.include_pose}")
    print(f"  Include hands: {args.include_hands}")
    print(f"  Include face: {args.include_face}")
    print(f"  Num classes: {args.num_classes}")
    print(f"  Map ? to n: {args.map_unknown_to_n}")
    
    # Process each split
    split_configs = [
        ('train', train_videos, train_dir),
        ('val', val_videos, val_dir),
        ('test', test_videos, test_dir),
    ]
    
    total_segments = 0
    total_videos_processed = 0
    
    for split_name, video_list, output_subdir in split_configs:
        if len(video_list) == 0:
            print(f"\n[{split_name.upper()}] No videos in this split, skipping...")
            continue
        
        print(f"\n[{split_name.upper()}] Processing {len(video_list)} videos...")
        split_segments = 0
        skipped_count = 0
        
        for vid in tqdm(video_list, desc=f"Processing {split_name}"):
            # Save to NPZ file (one file per video) - check early to skip if exists
            output_file = output_subdir / f"{vid}.npz"
            
            # Check if file exists - skip if it does (unless overwrite is requested)
            if output_file.exists() and not args.overwrite:
                # Skip existing file (default behavior - don't recreate segments)
                skipped_count += 1
                continue  # Skip this video, don't recreate segments
            
            # Find corresponding NPZ file
            npz_path = Path(args.data) / f"{vid}.npz"
            if not npz_path.exists():
                print(f"\n[WARNING] Video {vid} not found: {npz_path}")
                continue
            
            try:
                # Process video
                segments, stats_list, labels_list, metadata_list = process_file(
                    str(npz_path),
                    vid,
                    gt,
                    args.window,
                    args.stride,
                    args.coords,
                    args.include_pose,
                    args.include_hands,
                    args.include_face,
                    label_map,
                    args.map_unknown_to_n,
                    args.num_classes,
                )
                
                if len(segments) == 0:
                    print(f"\n[WARNING] No segments created for video {vid}")
                    continue
                
                # Save segments (only if file doesn't exist or overwrite is True)
                success = save_video_segments(
                    str(output_file),
                    segments,
                    stats_list,
                    labels_list,
                    metadata_list,
                    config,
                )
                
                if success:
                    split_segments += len(segments)
                    total_videos_processed += 1
                
                # Explicitly free memory after saving
                del segments, stats_list, labels_list, metadata_list
                
                # Force garbage collection every N videos to free memory
                if total_videos_processed % 10 == 0:
                    gc.collect()
            except KeyboardInterrupt:
                # User interrupted - save progress and exit
                print(f"\n\n[INTERRUPTED] Processing stopped by user at video {vid}")
                print(f"Progress saved: {total_videos_processed} videos processed so far")
                print(f"To resume, run the script again (it will skip already processed videos)")
                raise
            except Exception as e:
                # Log error but continue with next video
                print(f"\n[ERROR] Failed to process video {vid}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        videos_created = len([v for v in video_list if (output_subdir / f'{v}.npz').exists()])
        print(f"[{split_name.upper()}] Created {split_segments} segments from {videos_created} videos")
        if skipped_count > 0:
            print(f"  (Skipped {skipped_count} videos that already exist)")
        total_segments += split_segments
    
    # Save summary
    summary = {
        'config': config,
        'splits': {
            'train': {'videos': len(train_videos), 'files_created': len(list(train_dir.glob('*.npz')))},
            'val': {'videos': len(val_videos), 'files_created': len(list(val_dir.glob('*.npz')))},
            'test': {'videos': len(test_videos), 'files_created': len(list(test_dir.glob('*.npz')))},
        },
        'total_segments': total_segments,
        'total_videos_processed': total_videos_processed,
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total segments created: {total_segments}")
    print(f"Total videos processed: {total_videos_processed}")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nTo use in training, specify:")
    print(f"  --preprocessed-dir {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
