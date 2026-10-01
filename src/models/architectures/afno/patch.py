"""
Patch Embedding Layer
Converts input images into patch embeddings
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, config, atm=None):
        super().__init__()
        num_patches = (config.data.img_size[1] // config.data.patch_size[1]) * \
                     (config.data.img_size[0] // config.data.patch_size[0])

        self.img_size = config.data.img_size
        self.patch_size = config.data.patch_size
        self.num_patches = num_patches
        self.dynamic_size = config.data.dynamic_size

        in_channels = config.data.atm_chs if atm else config.data.in_chs

        self.proj = nn.Conv2d(
            in_channels,
            config.data.emd_dim,
            kernel_size=config.data.patch_size,
            stride=config.data.patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Flexible size checking
        if not self.dynamic_size:
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # Patch embedding
        x = self.proj(x)

        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)

        return x
