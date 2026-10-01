"""
2D Adaptive Fourier Neural Operator Layer
Implements spectral convolution in Fourier space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNO2D(nn.Module):
    """
    2D Adaptive Fourier Neural Operator

    Applies learnable transformations in Fourier space with sparsity regularization.
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.afno2d.hidden_size
        self.sparsity_threshold = config.afno2d.sparsity_threshold
        self.num_blocks = config.afno2d.num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = config.afno2d.hard_thresholding_fraction
        self.hidden_size_factor = config.afno2d.hidden_size_factor
        self.scale = 0.02

        assert self.hidden_size % self.num_blocks == 0, \
            f"hidden_size {self.hidden_size} should be divisible by num_blocks {self.num_blocks}"

        # Learnable weights for Fourier space operations
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size,
                                     self.block_size * self.hidden_size_factor)
        )
        self.b1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor,
                                     self.block_size)
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        # Transform to Fourier space
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        # Initialize output tensors
        o1_real = torch.zeros(
            [B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device
        )
        o1_imag = torch.zeros(
            [B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # Apply hard thresholding
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # First layer: complex-valued MLP in Fourier space
        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo',
                        x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo',
                        x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo',
                        x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo',
                        x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # Second layer
        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo',
                        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo',
                        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo',
                        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo',
                        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        # Combine real and imaginary parts
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)

        # Inverse FFT back to spatial domain
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias
