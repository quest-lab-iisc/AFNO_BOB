"""
AFNO (Adaptive Fourier Neural Operator) Network
Main architecture for ocean surface dynamics prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .patch import PatchEmbed
from .block import AFNOBlock


class AFNONet(nn.Module):
    """
    Adaptive Fourier Neural Operator Network

    Processes spatiotemporal data through patch embedding, Fourier operations,
    and transformer-style blocks for next-timestep prediction.
    """
    def __init__(self, config):
        super().__init__()
        self.img_size = config.data.img_size
        self.patch_size = (config.data.patch_size[0], config.data.patch_size[1])
        self.in_chans = config.data.in_chs
        self.out_chans = config.data.out_chs
        self.num_features = self.embed_dim = config.data.emd_dim
        self.num_blocks = config.afno2d.num_blocks

        # Patch embedding layer
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.data.emd_dim))
        self.pos_drop = nn.Dropout(p=config.data.pos_drop_rate)

        # Spatial dimensions after patching
        self.h = config.data.img_size[0] // self.patch_size[0]
        self.w = config.data.img_size[1] // self.patch_size[1]

        # AFNO transformer blocks
        self.blocks = nn.ModuleList([
            AFNOBlock(config) for i in range(config.afno2d.n_blocks)
        ])

        # Output projection head
        self.head = nn.Linear(
            config.data.emd_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Reshape to spatial grid
        x = x.reshape(B, self.h, self.w, self.embed_dim)

        # Apply AFNO blocks
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C_in, H, W)
        Returns:
            Output tensor (B, C_out, H, W)
        """
        x = self.forward_features(x)
        x = self.head(x)

        # Rearrange patches back to spatial dimensions
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x
