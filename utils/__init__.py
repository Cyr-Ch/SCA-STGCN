# Support both relative and absolute imports
try:
    from .graph import normalize_adjacency, build_hand_body_adjacency
except ImportError:
    # Fallback to absolute imports
    from utils.graph import normalize_adjacency, build_hand_body_adjacency


