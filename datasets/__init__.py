# Support both relative and absolute imports
try:
    from .landmarks_npz import NpzLandmarksDataset
except ImportError:
    # Fallback to absolute imports
    from datasets.landmarks_npz import NpzLandmarksDataset


