import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignerHead(nn.Module):
    def __init__(self, in_dim: int, num_signers: int, lambda_grl: float = 1.0, dropout: float = 0.1):
        super().__init__()
        # Minimal GRL implemented inline to avoid dependency cycles
        class _GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, l):
                ctx.l = l
                return x.view_as(x)
            @staticmethod
            def backward(ctx, g):
                return -ctx.l * g, None
        self._grl = _GRL
        self.lambda_grl = lambda_grl
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_signers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._grl.apply(x, self.lambda_grl)
        return self.net(x)


