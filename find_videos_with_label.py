"""
Script to find videos with a specific label in the groundtruth file.
Usage: python find_videos_with_label.py [label]
Default: finds videos with label 'S'
"""

import sys
from collections import defaultdict

def find_videos_with_label(gt_path, target_label='S'):
    """Find all videos containing a specific label"""
    video_frames = defaultdict(list)
    
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                vid, frame_str, label = parts[0], parts[1], parts[2]
                if label == target_label:
                    try:
                        frame = int(frame_str)
                        video_frames[vid].append(frame)
                    except ValueError:
                        continue
    
    return video_frames

def main():
    gt_path = 'landmarks/groundtruth.txt'
    target_label = sys.argv[1] if len(sys.argv) > 1 else 'S'
    
    print(f"Finding videos with label '{target_label}'...")
    print(f"Reading from: {gt_path}\n")
    
    video_frames = find_videos_with_label(gt_path, target_label)
    
    if len(video_frames) == 0:
        print(f"No videos found with label '{target_label}'")
        return
    
    # Sort by number of frames (descending)
    sorted_videos = sorted(video_frames.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"Found {len(sorted_videos)} videos with label '{target_label}'\n")
    print("="*70)
    print(f"Videos with label '{target_label}' (sorted by frame count):")
    print("="*70)
    
    # Show top 50 videos with most frames
    for vid, frames in sorted_videos[:50]:
        frames_sorted = sorted(frames)
        frame_range = f"{frames_sorted[0]}-{frames_sorted[-1]}" if len(frames_sorted) > 1 else str(frames_sorted[0])
        print(f"  {vid}: {len(frames)} frames (range: {frame_range})")
    
    if len(sorted_videos) > 50:
        print(f"\n  ... and {len(sorted_videos) - 50} more videos")
    
    # Statistics
    frame_counts = [len(frames) for _, frames in sorted_videos]
    print("\n" + "="*70)
    print("Statistics:")
    print("="*70)
    print(f"  Total videos: {len(sorted_videos)}")
    print(f"  Total frames with '{target_label}': {sum(frame_counts)}")
    print(f"  Average frames per video: {sum(frame_counts) / len(frame_counts):.2f}")
    print(f"  Min frames: {min(frame_counts)}")
    print(f"  Max frames: {max(frame_counts)}")
    
    # Save to file
    output_file = f'videos_with_label_{target_label}.txt'
    with open(output_file, 'w') as f:
        f.write(f"Videos with label '{target_label}'\n")
        f.write("="*70 + "\n\n")
        for vid, frames in sorted_videos:
            frames_sorted = sorted(frames)
            frame_range = f"{frames_sorted[0]}-{frames_sorted[-1]}" if len(frames_sorted) > 1 else str(frames_sorted[0])
            f.write(f"{vid}: {len(frames)} frames (range: {frame_range})\n")
    
    print(f"\nFull list saved to: {output_file}")

if __name__ == "__main__":
    main()

