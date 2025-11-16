# Support both relative and absolute imports
try:
    from .signer import SignerEncoder
    from .attention import BiasAwareAttention, FiLMSignerAttention
    from .temporal import TemporalAggregator
    from .heads import ClassifierHead, SignerHead
    from .hierarchical_attention import HierarchicalAttention
    from .adaptive_signer import AdaptiveSignerEncoder, HybridSignerEncoder
except ImportError:
    # Fallback to absolute imports
    from modules.signer import SignerEncoder
    from modules.attention import BiasAwareAttention, FiLMSignerAttention
    from modules.temporal import TemporalAggregator
    from modules.heads import ClassifierHead, SignerHead
    from modules.hierarchical_attention import HierarchicalAttention
    from modules.adaptive_signer import AdaptiveSignerEncoder, HybridSignerEncoder


