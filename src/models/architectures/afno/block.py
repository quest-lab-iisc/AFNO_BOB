"""
AFNO Transformer Block
Combines AFNO2D with MLP and residual connections
"""
import torch
import torch.nn as nn

from .afno2d import AFNO2D
from .mlp import MLP


class AFNOBlock(nn.Module):
    """
    AFNO Block with normalization, Fourier operator, and MLP
    """
    def __init__(self, config, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(config.data.emd_dim)
        self.filter = AFNO2D(config)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(config.data.emd_dim)
        mlp_hidden_dim = int(config.data.emd_dim * config.afno2d.mlp_ratio)
        self.mlp = MLP(config)
        self.double_skip = config.afno2d.double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x
