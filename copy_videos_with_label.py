"""
Script to copy .npz files for videos with a specific label to a new folder.
Usage: python copy_videos_with_label.py [label] [source_folder] [dest_folder]
"""

import os
import shutil
import sys

def copy_videos_with_label(gt_path, source_folder, dest_folder, target_label='S'):
    """Copy .npz files for videos containing a specific label"""
    
    # Find videos with the target label
    videos_with_label = set()
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                vid, label = parts[0], parts[2]
                if label == target_label:
                    videos_with_label.add(vid)
    
    print(f"Found {len(videos_with_label)} videos with label '{target_label}'")
    
    # Create destination folder
    os.makedirs(dest_folder, exist_ok=True)
    print(f"Created destination folder: {dest_folder}")
    
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
        if vid_id in videos_with_label:
            matching.append(f)
    
    print(f"Found {len(matching)} matching .npz files")
    
    # Copy files
    copied = 0
    for f in matching:
        src_path = os.path.join(source_folder, f)
        dst_path = os.path.join(dest_folder, f)
        shutil.copy2(src_path, dst_path)
        copied += 1
    
    print(f"Copied {copied} .npz files to {dest_folder}/")
    
    # Check for missing files
    missing = [vid for vid in videos_with_label if f'{vid}.npz' not in npz_files]
    if missing:
        print(f"\nWarning: {len(missing)} videos with label '{target_label}' don't have .npz files")
        print("Sample missing videos:", missing[:10])
    
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

def main():
    target_label = sys.argv[1] if len(sys.argv) > 1 else 'S'
    source_folder = sys.argv[2] if len(sys.argv) > 2 else 'landmarks'
    dest_folder = sys.argv[3] if len(sys.argv) > 3 else f'landmarks_{target_label.lower()}'
    
    gt_path = os.path.join(source_folder, 'groundtruth.txt')
    if not os.path.exists(gt_path):
        gt_path = os.path.join(source_folder, 'groundtruth')
    
    if not os.path.exists(gt_path):
        print(f"Error: Groundtruth file not found in {source_folder}")
        return
    
    print(f"Copying videos with label '{target_label}'")
    print(f"From: {source_folder}")
    print(f"To: {dest_folder}")
    print("="*70)
    
    copy_videos_with_label(gt_path, source_folder, dest_folder, target_label)
    
    print("="*70)
    print("Done!")

if __name__ == "__main__":
    main()

