"""Convolutional LSTM cell for spatiotemporal sequence modelling.

Implements the ConvLSTM cell from Shi et al. (2015), replacing the fully-connected
gates of a standard LSTM with 2-D convolutions so that spatial structure is
preserved across time steps.

Inputs:
    x (torch.Tensor): Current-step feature map (B, in_channels, H, W).
    h_prev (torch.Tensor): Previous hidden state (B, hidden_channels, H, W).
    c_prev (torch.Tensor): Previous cell state (B, hidden_channels, H, W).

Outputs:
    h (torch.Tensor): Updated hidden state (B, hidden_channels, H, W).
    c (torch.Tensor): Updated cell state (B, hidden_channels, H, W).

Example:
    >>> cell = ConvLSTMCell(in_channels=128, hidden_channels=128, kernel_size=3)
    >>> B, H, W = 2, 224, 224
    >>> x = torch.randn(B, 128, H, W)
    >>> h = torch.zeros(B, 128, H, W)
    >>> c = torch.zeros(B, 128, H, W)
    >>> h_new, c_new = cell(x, h, c)
    >>> h_new.shape
    torch.Size([2, 128, 224, 224])
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell computing all four gates in one fused convolution.

    Attributes:
        in_channels (int): Number of input feature channels.
        hidden_channels (int): Number of hidden / cell-state channels.
        kernel_size (int): Convolution kernel size (odd; same padding applied).
        conv (nn.Conv2d): Fused gate convolution: [i, f, g, o] stacked on dim 1.

    Args:
        in_channels (int): Channels in the input feature map.
        hidden_channels (int): Channels in hidden and cell states.
        kernel_size (int): Spatial kernel size (default 3).
        bias (bool): Whether to add a bias term (default True).

    Example:
        >>> cell = ConvLSTMCell(64, 128, kernel_size=3)
        >>> h, c = cell(torch.randn(2, 64, 32, 32),
        ...             torch.zeros(2, 128, 32, 32),
        ...             torch.zeros(2, 128, 32, 32))
        >>> h.shape
        torch.Size([2, 128, 32, 32])
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3, bias: bool = True) -> None:
        """Initialise gates convolution with same-padding."""
        super().__init__()
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # All four gates (i, f, g, o) computed in a single convolution
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor,
                h_prev: torch.Tensor,
                c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one ConvLSTM time step.

        Args:
            x (torch.Tensor): Input features (B, in_channels, H, W).
            h_prev (torch.Tensor): Previous hidden state (B, hidden_channels, H, W).
            c_prev (torch.Tensor): Previous cell state (B, hidden_channels, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (h, c) both (B, hidden_channels, H, W).

        Example:
            >>> cell = ConvLSTMCell(32, 64)
            >>> h, c = cell(torch.randn(1, 32, 8, 8),
            ...             torch.zeros(1, 64, 8, 8),
            ...             torch.zeros(1, 64, 8, 8))
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates    = self.conv(combined)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size: int, height: int, width: int,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised (h, c) for the start of a new sequence.

        Args:
            batch_size (int): Batch size.
            height (int): Spatial height.
            width (int): Spatial width.
            device (torch.device): Target device.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Zero tensors (h, c).

        Example:
            >>> cell = ConvLSTMCell(32, 64)
            >>> h, c = cell.init_hidden(4, 224, 224, torch.device('cpu'))
            >>> h.shape
            torch.Size([4, 64, 224, 224])
        """
        shape = (batch_size, self.hidden_channels, height, width)
        return (torch.zeros(shape, device=device),
                torch.zeros(shape, device=device))
