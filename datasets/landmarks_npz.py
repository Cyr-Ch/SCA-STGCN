from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def read_groundtruth(gt_path: str) -> Dict[Tuple[str, int], int]:
    """
    Expect lines like: <video_id> <frame_idx> <class_label>
    class_label can be str; we map to integer IDs on-the-fly outside dataset if needed.
    """
    mapping: Dict[Tuple[str, int], int] = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            vid, frame_str, label = parts[0], parts[1], parts[2]
            try:
                frame = int(frame_str)
            except Exception:
                continue
            mapping[(vid, frame)] = label
    return mapping


def compute_pose_stats(window: np.ndarray) -> np.ndarray:
    """
    window: (T, J, C) numpy array
    Returns simple stats vector, e.g., mean and std per coord over joints/time -> (2*C,)
    """
    # Flatten over (T,J)
    feat = window.reshape(-1, window.shape[-1])  # (T*J, C)
    mean = feat.mean(axis=0)
    std = feat.std(axis=0) + 1e-6
    return np.concatenate([mean, std], axis=0)


def load_npz(file_path: str, fs=None):
    """
    Load NPZ file from local path or GCS.
    
    Args:
        file_path: Local path or GCS path (gs://bucket/path)
        fs: GCSFileSystem instance (if None, will create one for GCS paths)
    
    Returns:
        NpzFile object (use with context manager: with load_npz(...) as npz:)
    """
    if file_path.startswith('gs://'):
        if fs is None:
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
            except ImportError:
                raise ImportError("gcsfs is required for GCS paths. Install with: pip install gcsfs")
        # Open GCS file
        return np.load(fs.open(file_path, 'rb'), allow_pickle=True)
    else:
        # Local file
        return np.load(file_path, allow_pickle=True)


def create_label_mapping(num_classes: int, map_unknown_to_n: bool = False) -> Dict[str, int]:
    """
    Create label mapping based on number of classes.
    
    Rules:
    - 2 classes: S -> 0 (signing), P/n/? -> 1 (other)
    - 3 classes (default): S -> 0 (signing), P -> 1 (speaking), n/? -> 2 (other)
    - 4 classes: S -> 0, P -> 1, n -> 2, ? -> 3
    - >4 classes: All labels separate (dynamic mapping)
    
    Args:
        num_classes: Target number of classes
        map_unknown_to_n: If True, map ? to n before mapping
    
    Returns:
        Dictionary mapping label strings to integer class IDs
    """
    label_map = {}
    
    if num_classes == 2:
        # Binary: signing vs everything else
        label_map['S'] = 0  # signing
        label_map['P'] = 1  # other (speaking, not-signing, unknown)
        label_map['n'] = 1  # other
        label_map['?'] = 1  # other
        label_map['unknown'] = 1  # other
    elif num_classes == 3:
        # Default: signing, speaking, other
        label_map['S'] = 0  # signing
        label_map['P'] = 1  # speaking
        label_map['n'] = 2  # other (not-signing)
        if map_unknown_to_n:
            label_map['?'] = 2  # other (mapped to n)
        else:
            label_map['?'] = 2  # other
        label_map['unknown'] = 2  # other
    elif num_classes == 4:
        # All 4 classes separate
        label_map['S'] = 0  # signing
        label_map['P'] = 1  # speaking
        label_map['n'] = 2  # not-signing
        if map_unknown_to_n:
            label_map['?'] = 2  # mapped to n
        else:
            label_map['?'] = 3  # unknown
        label_map['unknown'] = 3  # unknown
    else:
        # For >4 classes, use dynamic mapping (will be built on-the-fly)
        # But initialize with known labels
        label_map['S'] = 0
        label_map['P'] = 1
        label_map['n'] = 2
        if map_unknown_to_n:
            label_map['?'] = 2
        else:
            label_map['?'] = 3
        label_map['unknown'] = len(label_map) - 1
    
    return label_map


class NpzLandmarksDataset(Dataset):
    def __init__(
        self,
        root: str,
        gt_path: Optional[str] = None,
        window: int = 25,
        stride: int = 1,
        in_coords: int = 2,
        include_pose: bool = True,
        include_hands: bool = True,
        include_face: bool = False,
        max_files: Optional[int] = None,
        allowed_ids: Optional[Set[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
        map_unknown_to_n: bool = False,
        num_classes: Optional[int] = None,
        preprocessed_file: Optional[str] = None,
        preprocessed_dir: Optional[str] = None,
        split: Optional[str] = None,  # 'train', 'val', or 'test'
    ):
        self.root = root
        self.window = window
        self.stride = max(1, int(stride))
        self.in_coords = in_coords
        self.include_pose = include_pose
        self.include_hands = include_hands
        self.include_face = include_face
        self.max_files = max_files
        self.allowed_ids = set(allowed_ids) if allowed_ids else None
        self.map_unknown_to_n = map_unknown_to_n
        self._num_classes = num_classes
        self.preprocessed_file = preprocessed_file
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.is_gcs = preprocessed_dir and preprocessed_dir.startswith('gs://')
        
        # Load from preprocessed directory (per-video NPZ files) - LAZY LOADING
        if preprocessed_dir:
            if split is None:
                raise ValueError("split must be specified when using preprocessed_dir (e.g., 'train', 'val', 'test')")
            
            # Handle GCS paths
            if self.is_gcs:
                try:
                    import gcsfs
                    self.fs = gcsfs.GCSFileSystem()
                except ImportError:
                    raise ImportError("gcsfs is required for GCS paths. Install with: pip install gcsfs")
                
                # Build GCS path: gs://bucket/train or gs://bucket/path/train
                # Remove trailing slash and append split
                split_dir = f"{preprocessed_dir.rstrip('/')}/{split}"
                
                # Check if split directory exists
                if not self.fs.exists(split_dir):
                    raise FileNotFoundError(f"Split directory not found in GCS: {split_dir}")
                
                print(f"Setting up lazy loading from GCS: {split_dir}...")
                import pickle
                
                # List all NPZ files in GCS directory
                try:
                    # fs.ls() returns list of paths
                    all_files = self.fs.ls(split_dir, detail=False)
                    # Filter for .npz files and ensure they're full GCS paths
                    video_files = []
                    for f in all_files:
                        if f.endswith('.npz'):
                            # Ensure full GCS path
                            if f.startswith('gs://'):
                                video_files.append(f)
                            else:
                                # If relative path, construct full path
                                video_files.append(f"gs://{f}")
                    video_files = sorted(video_files)
                except Exception as e:
                    print(f"[WARNING] Error listing GCS files: {e}")
                    # Try alternative: list with detail=True to get full paths
                    try:
                        all_files = self.fs.ls(split_dir, detail=True)
                        video_files = []
                        for item in all_files:
                            if isinstance(item, dict):
                                path = item.get('name', '')
                            else:
                                path = str(item)
                            if path.endswith('.npz'):
                                if path.startswith('gs://'):
                                    video_files.append(path)
                                else:
                                    video_files.append(f"gs://{path}")
                        video_files = sorted(video_files)
                    except Exception as e2:
                        raise RuntimeError(f"Failed to list NPZ files from GCS: {e2}")
            else:
                # Local path
                if not os.path.exists(preprocessed_dir):
                    raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")
                
                split_dir = os.path.join(preprocessed_dir, split)
                if not os.path.exists(split_dir):
                    raise FileNotFoundError(f"Split directory not found: {split_dir}")
                
                print(f"Setting up lazy loading from {split_dir}...")
                import pickle
                
                # Get all video NPZ files from the split directory
                video_files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
            
            if len(video_files) == 0:
                raise ValueError(f"No NPZ files found in {split_dir}")
            
            # Build index mapping: segment_idx -> (video_file_idx, segment_idx_in_video)
            # Also count total segments and load config from first file
            self.preprocessed_video_files = video_files
            self.preprocessed_index = []  # List of (video_idx, segment_idx_in_video) tuples
            self.preprocessed_segment_counts = []  # Number of segments per video file
            
            total_segments = 0
            config_loaded = False
            
            # Try to load full index cache (much faster than rebuilding)
            # For GCS, try to load from GCS first, but fall back to local cache if GCS write is not available
            if self.is_gcs:
                index_cache_file_gcs = f"{split_dir}/_preprocessed_index.json"
                # Use local cache directory (in /tmp or home directory)
                import tempfile
                cache_dir = os.path.join(os.path.expanduser('~'), '.signstgcn_cache')
                os.makedirs(cache_dir, exist_ok=True)
                # Create a safe filename from the GCS path
                cache_name = split_dir.replace('gs://', '').replace('/', '_') + '_preprocessed_index.json'
                index_cache_file = os.path.join(cache_dir, cache_name)
            else:
                index_cache_file = os.path.join(split_dir, '_preprocessed_index.json')
                index_cache_file_gcs = None
            
            index_cache_valid = False
            
            # Try to load cache (check local first for GCS, then GCS if available)
            cache_loaded = False
            if self.is_gcs:
                # First try local cache
                if os.path.exists(index_cache_file):
                    try:
                        import json
                        with open(index_cache_file, 'r') as f:
                            cache_data = json.load(f)
                        cache_loaded = True
                    except Exception as e:
                        pass  # Try GCS cache next
                
                # If local cache not found, try GCS cache
                if not cache_loaded and index_cache_file_gcs and self.fs.exists(index_cache_file_gcs):
                    try:
                        import json
                        with self.fs.open(index_cache_file_gcs, 'r') as f:
                            cache_data = json.load(f)
                        cache_loaded = True
                    except Exception as e:
                        pass  # Will rebuild
            else:
                # Local path - just check if file exists
                if os.path.exists(index_cache_file):
                    try:
                        import json
                        with open(index_cache_file, 'r') as f:
                            cache_data = json.load(f)
                        cache_loaded = True
                    except Exception as e:
                        pass  # Will rebuild
            
            # Validate and use cache if loaded
            if cache_loaded:
                try:
                    cached_video_files = cache_data.get('video_files', [])
                    cached_segment_counts = cache_data.get('segment_counts', [])
                    cached_index = cache_data.get('index', [])
                    
                    # Check if video files list matches
                    if (len(cached_video_files) == len(video_files) and 
                        all(cached_video_files[i] == video_files[i] for i in range(len(video_files)))):
                        # Cache is valid - use it!
                        self.preprocessed_index = [tuple(item) for item in cached_index]  # Convert back to tuples
                        self.preprocessed_segment_counts = cached_segment_counts
                        total_segments = len(self.preprocessed_index)
                        cache_source = "local" if (self.is_gcs and os.path.exists(index_cache_file)) or not self.is_gcs else "GCS"
                        print(f"  ✓ Loaded index cache from {cache_source} ({total_segments} segments)")
                        index_cache_valid = True
                    else:
                        print(f"  [INFO] Index cache invalid (video files changed), rebuilding...")
                except Exception as e:
                    print(f"  [WARNING] Failed to validate index cache: {e}, rebuilding...")
            
            # If cache is invalid or missing, build index from scratch
            if not index_cache_valid:
                # Try to load segment counts from cache file (much faster than opening all NPZ files)
                if self.is_gcs:
                    segment_counts_cache_file_gcs = f"{split_dir}/_segment_counts.json"
                    # Use local cache directory for GCS (same as index cache)
                    import tempfile
                    cache_dir = os.path.join(os.path.expanduser('~'), '.signstgcn_cache')
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_name = split_dir.replace('gs://', '').replace('/', '_') + '_segment_counts.json'
                    segment_counts_cache_file = os.path.join(cache_dir, cache_name)
                    
                    segment_counts_cache = {}
                    # Try local cache first
                    if os.path.exists(segment_counts_cache_file):
                        try:
                            import json
                            with open(segment_counts_cache_file, 'r') as f:
                                segment_counts_cache = json.load(f)
                            print(f"  Loaded segment counts cache from local cache")
                        except Exception as e:
                            pass  # Try GCS next
                    
                    # Try GCS cache if local not found
                    if not segment_counts_cache and self.fs.exists(segment_counts_cache_file_gcs):
                        try:
                            import json
                            with self.fs.open(segment_counts_cache_file_gcs, 'r') as f:
                                segment_counts_cache = json.load(f)
                            print(f"  Loaded segment counts cache from GCS")
                        except Exception as e:
                            print(f"  [WARNING] Failed to load segment counts cache: {e}")
                else:
                    segment_counts_cache_file = os.path.join(split_dir, '_segment_counts.json')
                    segment_counts_cache = {}
                    if os.path.exists(segment_counts_cache_file):
                        try:
                            import json
                            with open(segment_counts_cache_file, 'r') as f:
                                segment_counts_cache = json.load(f)
                            print(f"  Loaded segment counts cache from {segment_counts_cache_file}")
                        except Exception as e:
                            print(f"  [WARNING] Failed to load segment counts cache: {e}")
                
                # Build index - with progress indicator for large datasets
                print(f"  Building index from {len(video_files)} video files...")
                for video_idx, video_file in enumerate(video_files):
                    video_name = os.path.splitext(os.path.basename(video_file))[0]
                    
                    # Check cache first
                    if video_name in segment_counts_cache:
                        num_segments = segment_counts_cache[video_name]
                    else:
                        # Need to open file to get segment count
                        with load_npz(video_file, fs=self.fs if self.is_gcs else None) as npz:
                            num_segments = len(npz['X'])
                            # Cache it for next time
                            segment_counts_cache[video_name] = num_segments
                    
                    self.preprocessed_segment_counts.append(num_segments)
                    
                    # Load config from first file only
                    if not config_loaded:
                        with load_npz(video_file, fs=self.fs if self.is_gcs else None) as npz:
                            if 'config' in npz:
                                config_bytes = npz['config']
                                if isinstance(config_bytes, np.ndarray):
                                    config = pickle.loads(config_bytes.tobytes())
                                else:
                                    config = pickle.loads(config_bytes)
                                
                                print(f"Preprocessed config: window={config.get('window')}, stride={config.get('stride')}, "
                                      f"coords={config.get('coords')}, num_classes={config.get('num_classes')}")
                                
                                # Verify compatibility
                                if config.get('window') != window:
                                    print(f"[WARNING] Window mismatch: preprocessed={config.get('window')}, requested={window}")
                                if config.get('stride') != stride:
                                    print(f"[WARNING] Stride mismatch: preprocessed={config.get('stride')}, requested={stride}")
                                if config.get('coords') != in_coords:
                                    print(f"[WARNING] Coords mismatch: preprocessed={config.get('coords')}, requested={in_coords}")
                                
                                # Use label_map from config
                                if 'label_map' in config:
                                    self.label_map = config['label_map']
                                else:
                                    if label_map is not None:
                                        self.label_map = label_map
                                    elif num_classes is not None:
                                        self.label_map = create_label_mapping(num_classes, map_unknown_to_n)
                                    else:
                                        self.label_map = create_label_mapping(3, map_unknown_to_n)
                                
                                config_loaded = True
                    
                    # Build index: for each segment in this video, store (video_idx, segment_idx_in_video)
                    # Use extend for better memory efficiency than repeated append
                    segment_indices = [(video_idx, seg_idx) for seg_idx in range(num_segments)]
                    self.preprocessed_index.extend(segment_indices)
                    total_segments += num_segments
                    
                    # Progress indicator for large datasets
                    if (video_idx + 1) % 20 == 0 or (video_idx + 1) == len(video_files):
                        print(f"    Processed {video_idx + 1}/{len(video_files)} videos ({total_segments} segments so far)...")
                
                # Save segment counts cache for next time
                if segment_counts_cache:
                    try:
                        import json
                        if self.is_gcs:
                            # Try to save to GCS first, but fall back to local if write permission denied
                            try:
                                if segment_counts_cache_file_gcs:
                                    with self.fs.open(segment_counts_cache_file_gcs, 'w') as f:
                                        json.dump(segment_counts_cache, f, indent=2)
                            except Exception as e:
                                # GCS write failed (permission denied), save locally instead
                                if "Forbidden" in str(e) or "not authorized" in str(e).lower():
                                    with open(segment_counts_cache_file, 'w') as f:
                                        json.dump(segment_counts_cache, f, indent=2)
                                else:
                                    raise
                        else:
                            with open(segment_counts_cache_file, 'w') as f:
                                json.dump(segment_counts_cache, f, indent=2)
                    except Exception as e:
                        print(f"  [WARNING] Failed to save segment counts cache: {e}")
                
                # Save full index cache for next time
                try:
                    import json
                    cache_data = {
                        'video_files': video_files,
                        'segment_counts': self.preprocessed_segment_counts,
                        'index': [list(item) for item in self.preprocessed_index]  # Convert tuples to lists for JSON
                    }
                    if self.is_gcs:
                        # Try to save to GCS first, but fall back to local if write permission denied
                        try:
                            if index_cache_file_gcs:
                                with self.fs.open(index_cache_file_gcs, 'w') as f:
                                    json.dump(cache_data, f, indent=2)
                                print(f"  ✓ Saved index cache to GCS: {index_cache_file_gcs}")
                        except Exception as e:
                            # GCS write failed (permission denied), save locally instead
                            if "Forbidden" in str(e) or "not authorized" in str(e).lower():
                                with open(index_cache_file, 'w') as f:
                                    json.dump(cache_data, f, indent=2)
                                print(f"  ✓ Saved index cache locally (GCS write not available): {index_cache_file}")
                            else:
                                raise
                    else:
                        with open(index_cache_file, 'w') as f:
                            json.dump(cache_data, f, indent=2)
                        print(f"  ✓ Saved index cache to {index_cache_file}")
                except Exception as e:
                    print(f"  [WARNING] Failed to save index cache: {e}")
            
            # If we loaded from cache, still need to load config from first file
            if index_cache_valid and not config_loaded:
                first_file = video_files[0]
                with load_npz(first_file, fs=self.fs if self.is_gcs else None) as npz:
                    if 'config' in npz:
                        config_bytes = npz['config']
                        if isinstance(config_bytes, np.ndarray):
                            config = pickle.loads(config_bytes.tobytes())
                        else:
                            config = pickle.loads(config_bytes)
                        
                        print(f"Preprocessed config: window={config.get('window')}, stride={config.get('stride')}, "
                              f"coords={config.get('coords')}, num_classes={config.get('num_classes')}")
                        
                        # Use label_map from config
                        if 'label_map' in config:
                            self.label_map = config['label_map']
                        else:
                            if label_map is not None:
                                self.label_map = label_map
                            elif num_classes is not None:
                                self.label_map = create_label_mapping(num_classes, map_unknown_to_n)
                            else:
                                self.label_map = create_label_mapping(3, map_unknown_to_n)
            
            # Create label_map if not set from config
            if not hasattr(self, 'label_map') or self.label_map is None:
                if label_map is not None:
                    self.label_map = label_map
                elif num_classes is not None:
                    self.label_map = create_label_mapping(num_classes, map_unknown_to_n)
                else:
                    self.label_map = create_label_mapping(3, map_unknown_to_n)
            
            # Optional: cache for recently loaded files (to avoid reloading same file in batches)
            # Cache size: keep last N files in memory
            # Note: With multiple DataLoader workers, each worker has its own cache, so total memory
            # usage = num_workers × cache_size × video_file_size. Use cache_size=1 with workers > 0.
            self.preprocessed_cache = {}  # video_idx -> (X, stats, labels, metadata)
            self.preprocessed_cache_size = 1  # Number of video files to cache (reduced for multi-worker safety)
            self.preprocessed_cache_order = []  # LRU order
            
            print(f"  Lazy loading setup: {total_segments} segments from {len(video_files)} videos")
            print(f"  Using file cache (size={self.preprocessed_cache_size}) to speed up batch loading")
            self.files = []  # Not needed when using preprocessed data
            self.gt = {}  # Not needed when using preprocessed data
            return  # Skip normal initialization
        
        # Load from preprocessed file if provided
        if preprocessed_file and os.path.exists(preprocessed_file):
            print(f"Loading preprocessed segments from {preprocessed_file}...")
            import pickle
            with np.load(preprocessed_file, allow_pickle=True) as npz:
                self.X = npz['X']  # (N, T, J, C)
                self.stats = npz['stats']  # (N, D)
                self.labels = npz['labels']  # (N,)
                self.metadata = npz['metadata']  # (N,) structured array
                
                # Load config and verify compatibility
                if 'config' in npz:
                    config_bytes = npz['config']
                    if isinstance(config_bytes, np.ndarray):
                        config = pickle.loads(config_bytes.tobytes())
                    else:
                        config = pickle.loads(config_bytes)
                    
                    print(f"Preprocessed config: window={config.get('window')}, stride={config.get('stride')}, "
                          f"coords={config.get('coords')}, num_classes={config.get('num_classes')}")
                    
                    # Verify compatibility
                    if config.get('window') != window:
                        print(f"[WARNING] Window mismatch: preprocessed={config.get('window')}, requested={window}")
                    if config.get('stride') != stride:
                        print(f"[WARNING] Stride mismatch: preprocessed={config.get('stride')}, requested={stride}")
                    if config.get('coords') != in_coords:
                        print(f"[WARNING] Coords mismatch: preprocessed={config.get('coords')}, requested={in_coords}")
                    
                    # Use label_map from config
                    if 'label_map' in config:
                        self.label_map = config['label_map']
                    else:
                        # Fall back to creating label_map
                        if label_map is not None:
                            self.label_map = label_map
                        elif num_classes is not None:
                            self.label_map = create_label_mapping(num_classes, map_unknown_to_n)
                        else:
                            self.label_map = create_label_mapping(3, map_unknown_to_n)
                else:
                    # No config, create label_map
                    if label_map is not None:
                        self.label_map = label_map
                    elif num_classes is not None:
                        self.label_map = create_label_mapping(num_classes, map_unknown_to_n)
                    else:
                        self.label_map = create_label_mapping(3, map_unknown_to_n)
                
                print(f"  Loaded {len(self.X)} preprocessed segments")
                self.files = []  # Not needed when using preprocessed data
                self.gt = {}  # Not needed when using preprocessed data
                return  # Skip normal initialization
        
        # Normal initialization (create segments on-the-fly)
        # Initialize label_map based on num_classes if provided
        if label_map is not None:
            self.label_map = label_map
        elif num_classes is not None:
            self.label_map = create_label_mapping(num_classes, map_unknown_to_n)
        else:
            # Default: 3 classes (signing, speaking, other)
            self.label_map = create_label_mapping(3, map_unknown_to_n)

        self.files = sorted(glob.glob(os.path.join(root, '*.npz')))
        if self.allowed_ids is not None:
            before_count = len(self.files)
            self.files = [p for p in self.files if os.path.splitext(os.path.basename(p))[0] in self.allowed_ids]
            # Debug: check if files were filtered out
            if len(self.files) == 0 and before_count > 0:
                # Try to find why files don't match
                sample_ids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(root, '*.npz')))[:5]]
                print(f"[DEBUG] Dataset: Found {before_count} .npz files, but {len(self.files)} match allowed_ids")
                print(f"[DEBUG] Dataset: Sample file IDs: {sample_ids}")
                print(f"[DEBUG] Dataset: Allowed IDs: {list(self.allowed_ids)[:5]}")
        if self.max_files is not None:
            try:
                k = int(self.max_files)
                if k >= 0:
                    self.files = self.files[:k]
            except Exception:
                pass
        self.gt = read_groundtruth(gt_path) if gt_path and os.path.exists(gt_path) else {}

        self.index: List[Tuple[int, int]] = []  # (file_idx, start_frame)
        self.meta: List[Dict] = []
        
        # Debug: print what we're looking for
        if len(self.files) > 0:
            print(f"[DEBUG] Dataset: Processing {len(self.files)} files")
            print(f"[DEBUG] Dataset: include_pose={self.include_pose}, include_hands={self.include_hands}, include_face={self.include_face}")
        
        # Precompute available windows by reading npz headers quickly
        for fi, path in enumerate(self.files):
            try:
                with np.load(path) as npz:
                    # Prefer MediaPipe keys; fall back to any array in file
                    T = 0
                    for key in ('pose', 'left_hand', 'right_hand', 'face'):
                        if key in npz:
                            T = max(T, npz[key].shape[0])
                    if T == 0:
                        # generic fallback
                        if len(npz.keys()) > 0:
                            key = list(npz.keys())[0]
                            T = npz[key].shape[0]
                        else:
                            print(f"[WARNING] Dataset: File {os.path.basename(path)} has no keys")
                            continue
                    
                    # Check if we have the required data based on include flags
                    has_required = True
                    missing_keys = []
                    if self.include_pose and 'pose' not in npz:
                        has_required = False
                        missing_keys.append('pose')
                    if self.include_hands:
                        # For hands, we need at least one hand (left OR right)
                        if 'left_hand' not in npz and 'right_hand' not in npz:
                            has_required = False
                            missing_keys.append('hands (both missing)')
                        elif 'left_hand' not in npz:
                            missing_keys.append('left_hand')
                        elif 'right_hand' not in npz:
                            missing_keys.append('right_hand')
                    if self.include_face and 'face' not in npz:
                        has_required = False
                        missing_keys.append('face')
                    
                    if not has_required:
                        print(f"[WARNING] Dataset: File {os.path.basename(path)} missing required keys: {missing_keys}")
                        continue
                    
                    num_windows = max(0, T - window + 1)
                    if num_windows > 0:
                        for s in range(0, num_windows, self.stride):
                            self.index.append((fi, s))
                            self.meta.append({'file': os.path.basename(path), 'start': s, 'len': window})
                    else:
                        print(f"[WARNING] Dataset: File {os.path.basename(path)} has {T} frames, but window={window}, so num_windows=0")
            except Exception as e:
                print(f"[ERROR] Dataset: Failed to process {os.path.basename(path)}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(self.files) > 0:
            print(f"  Created {len(self.index)} windows from {len(self.files)} files")

    def __len__(self) -> int:
        if hasattr(self, 'X'):
            # Using preprocessed file (all in memory)
            return len(self.X)
        elif hasattr(self, 'preprocessed_index'):
            # Using preprocessed directory (lazy loading)
            return len(self.preprocessed_index)
        return len(self.index)

    def __getitem__(self, i: int):
        # If using preprocessed file (all in memory), return directly
        if hasattr(self, 'X'):
            X = torch.from_numpy(self.X[i]).float()  # (T, J, C)
            stats = torch.from_numpy(self.stats[i]).float()  # (D,)
            y = torch.tensor(self.labels[i], dtype=torch.long)
            meta = self.metadata[i]
            return X, stats, y, {
                'video': str(meta['video']),
                'start': int(meta['start']),
                'label': str(meta['label']),
            }
        
        # If using preprocessed directory (lazy loading)
        if hasattr(self, 'preprocessed_index'):
            video_idx, seg_idx = self.preprocessed_index[i]
            video_file = self.preprocessed_video_files[video_idx]
            
            # Check cache first
            if video_idx in self.preprocessed_cache:
                # Cache hit: use cached data
                cached_data = self.preprocessed_cache[video_idx]
                X_arr, stats_arr, labels_arr, metadata_arr = cached_data
                
                # If X is not in cache (was loaded via get_stats_only), load it now
                if X_arr is None:
                    fs = getattr(self, 'fs', None) if getattr(self, 'is_gcs', False) else None
                    with load_npz(video_file, fs=fs) as npz:
                        X_arr = npz['X'].copy()  # (N, T, J, C)
                        labels_arr = npz['labels'].copy()  # (N,)
                    # Update cache with X and labels
                    self.preprocessed_cache[video_idx] = (X_arr, stats_arr, labels_arr, metadata_arr)
                
                # Move to end of LRU order
                if video_idx in self.preprocessed_cache_order:
                    self.preprocessed_cache_order.remove(video_idx)
                self.preprocessed_cache_order.append(video_idx)
            else:
                # Cache miss: load from file
                fs = getattr(self, 'fs', None) if getattr(self, 'is_gcs', False) else None
                with load_npz(video_file, fs=fs) as npz:
                    # Copy arrays to ensure they persist in memory (not memory-mapped)
                    X_arr = npz['X'].copy()  # (N, T, J, C)
                    stats_arr = npz['stats'].copy()  # (N, D)
                    labels_arr = npz['labels'].copy()  # (N,)
                    metadata_arr = npz['metadata'].copy()  # (N,)
                
                # Add to cache
                self.preprocessed_cache[video_idx] = (X_arr, stats_arr, labels_arr, metadata_arr)
                self.preprocessed_cache_order.append(video_idx)
                
                # Evict oldest if cache is full (ensure we stay at cache_size)
                while len(self.preprocessed_cache) > self.preprocessed_cache_size:
                    oldest_idx = self.preprocessed_cache_order.pop(0)
                    # Explicitly delete cached arrays to free memory immediately
                    if oldest_idx in self.preprocessed_cache:
                        cached_data = self.preprocessed_cache[oldest_idx]
                        del cached_data  # Delete the tuple contents
                        del self.preprocessed_cache[oldest_idx]
            
            # Get the specific segment
            X = torch.from_numpy(X_arr[seg_idx]).float()  # (T, J, C)
            stats = torch.from_numpy(stats_arr[seg_idx]).float()  # (D,)
            y = torch.tensor(labels_arr[seg_idx], dtype=torch.long)
            meta = metadata_arr[seg_idx]
            return X, stats, y, {
                'video': str(meta['video']),
                'start': int(meta['start']),
                'label': str(meta['label']),
            }
    
    def get_stats_only(self, i: int) -> tuple[torch.Tensor, dict]:
        """
        Get only stats and metadata for a sample (used for clustering).
        Skips loading full segment data (X) to save memory and I/O.
        
        Returns:
            stats: (D,) tensor of pose statistics
            meta: dict with 'video', 'start', 'label'
        """
        # If using preprocessed file (all in memory), return directly
        if hasattr(self, 'X'):
            stats = torch.from_numpy(self.stats[i]).float()  # (D,)
            meta = self.metadata[i]
            return stats, {
                'video': str(meta['video']),
                'start': int(meta['start']),
                'label': str(meta['label']),
            }
        
        # If using preprocessed directory (lazy loading)
        if hasattr(self, 'preprocessed_index'):
            video_idx, seg_idx = self.preprocessed_index[i]
            video_file = self.preprocessed_video_files[video_idx]
            
            # Check cache first
            if video_idx in self.preprocessed_cache:
                # Cache hit: use cached data
                cached_data = self.preprocessed_cache[video_idx]
                # Cache might have X loaded (from __getitem__) or not (from previous get_stats_only)
                if cached_data[1] is not None:  # stats_arr exists
                    stats_arr = cached_data[1]
                    metadata_arr = cached_data[3]
                else:
                    # Shouldn't happen, but handle gracefully
                    stats_arr, metadata_arr = None, None
                
                # Move to end of LRU order
                if video_idx in self.preprocessed_cache_order:
                    self.preprocessed_cache_order.remove(video_idx)
                self.preprocessed_cache_order.append(video_idx)
                
                # If stats not in cache, load them
                if stats_arr is None:
                    fs = getattr(self, 'fs', None) if getattr(self, 'is_gcs', False) else None
                    with load_npz(video_file, fs=fs) as npz:
                        stats_arr = npz['stats'].copy()  # (N, D)
                        metadata_arr = npz['metadata'].copy()  # (N,)
                    # Update cache
                    X_arr, _, labels_arr, _ = cached_data
                    self.preprocessed_cache[video_idx] = (X_arr, stats_arr, labels_arr, metadata_arr)
            else:
                # Cache miss: load from file (only stats and metadata, not X!)
                fs = getattr(self, 'fs', None) if getattr(self, 'is_gcs', False) else None
                with load_npz(video_file, fs=fs) as npz:
                    # Only load stats and metadata, skip X to save memory
                    stats_arr = npz['stats'].copy()  # (N, D)
                    metadata_arr = npz['metadata'].copy()  # (N,)
                
                # Add to cache (but we only have stats/metadata, not X)
                # Store None for X and labels to indicate they're not loaded
                self.preprocessed_cache[video_idx] = (None, stats_arr, None, metadata_arr)
                self.preprocessed_cache_order.append(video_idx)
                
                # Evict oldest if cache is full (ensure we stay at cache_size)
                while len(self.preprocessed_cache) > self.preprocessed_cache_size:
                    oldest_idx = self.preprocessed_cache_order.pop(0)
                    # Explicitly delete cached arrays to free memory immediately
                    if oldest_idx in self.preprocessed_cache:
                        cached_data = self.preprocessed_cache[oldest_idx]
                        del cached_data  # Delete the tuple contents
                        del self.preprocessed_cache[oldest_idx]
            
            # Get the specific segment's stats
            stats = torch.from_numpy(stats_arr[seg_idx]).float()  # (D,)
            meta = metadata_arr[seg_idx]
            return stats, {
                'video': str(meta['video']),
                'start': int(meta['start']),
                'label': str(meta['label']),
            }
        
        # Normal on-the-fly processing - would need to compute stats, but this shouldn't be used
        # for clustering (clustering should use preprocessed data)
        raise NotImplementedError("get_stats_only() not supported for on-the-fly processing. Use preprocessed data for clustering.")
        
        # Normal on-the-fly processing
        fi, s = self.index[i]
        path = self.files[fi]
        vid = os.path.splitext(os.path.basename(path))[0]
        with np.load(path) as npz:
            # Build (T, J, C) by concatenating selected parts
            parts = []
            T = 0
            if self.include_pose and 'pose' in npz:
                T = max(T, npz['pose'].shape[0])
            if self.include_hands:
                if 'left_hand' in npz:
                    T = max(T, npz['left_hand'].shape[0])
                if 'right_hand' in npz:
                    T = max(T, npz['right_hand'].shape[0])
            if self.include_face and 'face' in npz:
                T = max(T, npz['face'].shape[0])

            def pick_coords(x: np.ndarray) -> np.ndarray:
                # x: (..., C) with C>=self.in_coords; pose may have visibility at index 3
                if x.shape[-1] >= self.in_coords:
                    return x[..., :self.in_coords]
                # pad with zeros if fewer coords
                pad = np.zeros(list(x.shape[:-1]) + [self.in_coords - x.shape[-1]], dtype=x.dtype)
                return np.concatenate([x, pad], axis=-1)

            if self.include_pose:
                if 'pose' in npz:
                    pose = npz['pose']  # (T,33,4)
                    pose = pose[:, :, :max(1, self.in_coords)]  # drop visibility
                else:
                    pose = np.full((T, 33, self.in_coords), np.nan, dtype=np.float32)
                parts.append(pick_coords(pose))
            if self.include_hands:
                if 'left_hand' in npz:
                    lh = pick_coords(npz['left_hand'])
                else:
                    lh = np.full((T, 21, self.in_coords), np.nan, dtype=np.float32)
                if 'right_hand' in npz:
                    rh = pick_coords(npz['right_hand'])
                else:
                    rh = np.full((T, 21, self.in_coords), np.nan, dtype=np.float32)
                parts += [lh, rh]
            if self.include_face:
                if 'face' in npz:
                    face = pick_coords(npz['face'])
                else:
                    face = np.full((T, 478, self.in_coords), np.nan, dtype=np.float32)
                parts.append(face)

            if not parts:
                # fallback: use any last array
                arr = npz[list(npz.keys())[-1]]
                arr = pick_coords(arr)
            else:
                arr = np.concatenate(parts, axis=1)

        window_np = arr[s:s + self.window]
        # Replace NaNs/Infs from missing detections with zeros to avoid NaN loss
        window_np = np.nan_to_num(window_np, nan=0.0, posinf=0.0, neginf=0.0)
        # build label: majority or last frame label based on gt
        labels = []
        for t in range(self.window):
            frame_id = s + t
            lab = self.gt.get((vid, frame_id), None)
            if lab is not None:
                # Map ? to n if requested
                if self.map_unknown_to_n and lab == '?':
                    lab = 'n'
                labels.append(lab)
        if len(labels) == 0:
            y_str = 'unknown'
        else:
            # pick most frequent
            vals, counts = np.unique(np.array(labels), return_counts=True)
            y_str = vals[np.argmax(counts)].item() if hasattr(vals[0], 'item') else vals[np.argmax(counts)]

        # Map label to class ID
        if y_str in self.label_map:
            y = self.label_map[y_str]
        else:
            # For dynamic mapping (num_classes > 4), add new labels on-the-fly
            # But ensure we don't exceed num_classes if it was specified
            if hasattr(self, '_num_classes') and self._num_classes is not None:
                # If we have a fixed num_classes, map unknown labels to "other"
                if 'other' not in self.label_map:
                    self.label_map['other'] = len(self.label_map)
                y = self.label_map.get('other', len(self.label_map) - 1)
            else:
                # Dynamic mapping: add new label
                self.label_map[y_str] = len(self.label_map)
                y = self.label_map[y_str]

        # stats for signer encoder
        stats_np = compute_pose_stats(window_np)

        X = torch.from_numpy(window_np).float()           # (T, J, C)
        pose_stats = torch.from_numpy(stats_np).float()   # (2*C,)
        y = torch.tensor(y, dtype=torch.long)
        return X, pose_stats, y, {
            'video': vid,
            'start': s,
            'label': y_str,
        }


