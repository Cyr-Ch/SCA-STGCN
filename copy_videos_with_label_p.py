"""
Script to copy .npz files for videos with label 'P' that are NOT in landmarks_s folder.
"""

import os
import shutil
from collections import defaultdict

def find_videos_with_label(gt_path, target_label='P'):
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
    source_folder = 'landmarks'
    exclude_folder = 'landmarks_s'
    dest_folder = 'landmarks_p'
    target_label = 'P'
    
    print(f"Finding videos with label '{target_label}' that are NOT in {exclude_folder}/")
    print("="*70)
    
    # Find videos with label P
    gt_path = os.path.join(source_folder, 'groundtruth.txt')
    if not os.path.exists(gt_path):
        gt_path = os.path.join(source_folder, 'groundtruth')
    
    if not os.path.exists(gt_path):
        print(f"Error: Groundtruth file not found in {source_folder}")
        return
    
    video_frames = find_videos_with_label(gt_path, target_label)
    videos_with_p = set(video_frames.keys())
    print(f"Found {len(videos_with_p)} videos with label '{target_label}'")
    
    # Get videos already in landmarks_s
    videos_in_s = set()
    if os.path.exists(exclude_folder):
        npz_files_s = [f for f in os.listdir(exclude_folder) if f.endswith('.npz')]
        videos_in_s = {os.path.splitext(f)[0] for f in npz_files_s}
        print(f"Found {len(videos_in_s)} videos in {exclude_folder}/")
    else:
        print(f"Folder {exclude_folder}/ does not exist, will include all videos with P")
    
    # Find videos with P that are NOT in landmarks_s
    videos_with_p_not_in_s = videos_with_p - videos_in_s
    print(f"\nVideos with '{target_label}' NOT in {exclude_folder}: {len(videos_with_p_not_in_s)}")
    
    if len(videos_with_p_not_in_s) == 0:
        print("No videos to copy!")
        return
    
    # Create destination folder
    os.makedirs(dest_folder, exist_ok=True)
    print(f"Created destination folder: {dest_folder}/")
    
    # Get all .npz files in source folder
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist")
        return
    
    npz_files = [f for f in os.listdir(source_folder) if f.endswith('.npz')]
    print(f"Found {len(npz_files)} .npz files in source folder")
    
    # Find matching files
    matching = []
    for f in npz_files:
        vid_id = os.path.splitext(f)[0]
        if vid_id in videos_with_p_not_in_s:
            matching.append(f)
    
    print(f"Found {len(matching)} matching .npz files to copy")
    
    # Copy files
    copied = 0
    for f in matching:
        src_path = os.path.join(source_folder, f)
        dst_path = os.path.join(dest_folder, f)
        shutil.copy2(src_path, dst_path)
        copied += 1
    
    print(f"Copied {copied} .npz files to {dest_folder}/")
    
    # Check for missing files
    missing = [vid for vid in videos_with_p_not_in_s if f'{vid}.npz' not in npz_files]
    if missing:
        print(f"\nWarning: {len(missing)} videos with label '{target_label}' don't have .npz files")
        print("Sample missing videos:", missing[:10])
    
    # Save list of videos
    list_file = f'videos_with_label_{target_label}_not_in_s.txt'
    with open(list_file, 'w') as f:
        f.write(f"Videos with label '{target_label}' that are NOT in {exclude_folder}/\n")
        f.write("="*70 + "\n\n")
        for vid in sorted(videos_with_p_not_in_s):
            frame_count = len(video_frames[vid])
            frames_sorted = sorted(video_frames[vid])
            frame_range = f"{frames_sorted[0]}-{frames_sorted[-1]}" if len(frames_sorted) > 1 else str(frames_sorted[0])
            f.write(f"{vid}: {frame_count} frames (range: {frame_range})\n")
    
    print(f"\nList of videos saved to: {list_file}")
    
    # Also copy groundtruth file if it exists
    gt_src = os.path.join(source_folder, 'groundtruth.txt')
    gt_alt = os.path.join(source_folder, 'groundtruth')
    gt_dst = os.path.join(dest_folder, 'groundtruth.txt')
    
    if os.path.exists(gt_src):
        shutil.copy2(gt_src, gt_dst)
        print(f"Copied groundtruth.txt to {dest_folder}/")
    elif os.path.exists(gt_alt):
        shutil.copy2(gt_alt, gt_dst)
        print(f"Copied groundtruth file to {dest_folder}/groundtruth.txt")
    
    print("\n" + "="*70)
    print("Done!")

if __name__ == "__main__":
    main()

