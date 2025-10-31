import torch
import torch.nn as nn


class TemporalAggregator(nn.Module):
    """
    BiGRU aggregator over time on pooled joint features.
    Input:  (B, T, D_in)
    Output: (B, D_out)
    """
    def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_in)
        out, h_n = self.gru(x)     # out: (B, T, H*D), h_n: (L*D, B, H)
        # take last timestep
        last = out[:, -1, :]       # (B, H*)
        return last


