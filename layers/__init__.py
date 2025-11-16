# Support both relative and absolute imports
try:
    from .graph import GraphConv, STGCNBlock, STGCNBackbone
except ImportError:
    # Fallback to absolute imports
    from layers.graph import GraphConv, STGCNBlock, STGCNBackbone


