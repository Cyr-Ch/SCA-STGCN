from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """
    Spatial graph convolution over joints using a fixed adjacency A (J x J).
    Implements: Y = A * X * W, with batch/time dims preserved.
    X: (B, T, J, C_in)
    A: (J, J)
    """
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, C_in)
        xw = self.lin(x)                      # (B, T, J, C_out)
        # batched matmul over joints: (J, J) x (B,T,J,C) -> (B,T,J,C)
        xw = torch.einsum("ij,btjc->btic", A, xw)
        return xw


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal GCN block: GraphConv -> TemporalConv1D (depthwise-separable optional) + BN + ReLU + Residual
    Input:  (B, T, J, C_in)
    Output: (B, T, J, C_out)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.gc = GraphConv(in_ch, out_ch)
        pad = (kernel_size - 1) // 2 * dilation
        self.temporal = nn.Conv2d(  # treat J as "width", apply conv over time only by reshaping
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            dilation=(dilation, 1),
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.do = nn.Dropout(dropout)
        self.use_residual = use_residual
        if use_residual:
            self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, C_in)
        B, T, J, _ = x.shape
        out = self.gc(x, A)  # (B, T, J, C_out)
        # reshape to (B, C, T, J) for temporal conv
        out = out.permute(0, 3, 1, 2).contiguous()
        out = self.temporal(out)
        out = self.bn(out)
        if self.use_residual:
            res = x
            if not isinstance(self.res_proj, nn.Identity):
                res = self.res_proj(res)
            res = res.permute(0, 3, 1, 2).contiguous()
            out = out + res
        out = self.relu(out)
        out = self.do(out)
        # back to (B, T, J, C)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


class STGCNBackbone(nn.Module):
    """
    A small stack of ST-GCN blocks with increasing channels.
    """
    def __init__(
        self,
        in_ch: int,
        channels,
        kernel_size: int = 3,
        dilations=(1, 2, 3),
        dropout: float = 0.1
    ):
        super().__init__()
        assert len(channels) == len(dilations)
        layers = []
        c_in = in_ch
        for c_out, dil in zip(channels, dilations):
            layers.append(
                STGCNBlock(c_in, c_out, kernel_size=kernel_size, dilation=dil, dropout=dropout)
            )
            c_in = c_out
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, C_in)
        for layer in self.layers:
            x = layer(x, A)
        return x  # (B, T, J, C_out)


