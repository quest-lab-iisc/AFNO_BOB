"""Multi-layer ConvLSTM network for autoregressive ocean state forecasting.

Accepts the same 11-channel input as AFNONet (6 atmospheric + 5 ocean channels)
and outputs a 5-channel next-step ocean prediction.  A learnable input projection
maps the 11-channel input to ``hidden_channels`` before the ConvLSTM stack.

The hidden state (h, c) is carried across autoregressive rollout steps at
inference time, giving the model temporal memory over the forecast window.  Pass
``hidden_state=None`` to initialise a fresh zero state at the start of each IC.

Inputs:
    x (torch.Tensor): (B, in_channels, H, W) combined atm+ocean input for one step.
    hidden_state (list[tuple] | None): List of (h, c) per layer, or None.

Outputs:
    pred (torch.Tensor): (B, out_channels, H, W) predicted next ocean state.
    new_hidden (list[tuple]): Updated list of (h, c) per layer.

Example:
    >>> model = ConvLSTMNet(in_channels=11, hidden_channels=192,
    ...                     num_layers=3, out_channels=5)
    >>> x = torch.randn(2, 11, 224, 224)
    >>> pred, hidden = model(x)                 # fresh hidden state
    >>> pred2, hidden = model(x, hidden)        # carry hidden state forward
    >>> pred.shape
    torch.Size([2, 5, 224, 224])
"""

import torch
import torch.nn as nn

from .convlstm_cell import ConvLSTMCell


class ConvLSTMNet(nn.Module):
    """Stacked ConvLSTM encoder with convolutional output head.

    Attributes:
        in_channels (int): Input channels (default 11).
        hidden_channels (int): Hidden state channels per layer.
        num_layers (int): Number of stacked ConvLSTM layers.
        out_channels (int): Output channels (default 5).
        input_proj (nn.Conv2d): 1×1 projection from in_channels to hidden_channels.
        cells (nn.ModuleList): ConvLSTM cells, one per layer.
        output_head (nn.Sequential): Conv layers mapping hidden → out_channels.

    Args:
        in_channels (int): Input feature channels (default 11).
        hidden_channels (int): Hidden/cell-state channels per layer (default 192).
        num_layers (int): Number of ConvLSTM layers (default 3).
        out_channels (int): Output channels (default 5).
        kernel_size (int): Convolution kernel size for LSTM cells (default 3).

    Example:
        >>> model = ConvLSTMNet()
        >>> pred, h = model(torch.randn(2, 11, 224, 224))
        >>> pred.shape
        torch.Size([2, 5, 224, 224])
    """

    def __init__(self, in_channels: int = 11, hidden_channels: int = 192,
                 num_layers: int = 3, out_channels: int = 5,
                 kernel_size: int = 3) -> None:
        """Build input projection, ConvLSTM stack, and output head."""
        super().__init__()
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers      = num_layers
        self.out_channels    = out_channels

        self.input_proj = nn.Conv2d(in_channels, hidden_channels,
                                    kernel_size=1, bias=True)

        self.cells = nn.ModuleList([
            ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
            for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def init_hidden(self, batch_size: int, height: int, width: int,
                    device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initialise zero hidden states for all layers.

        Args:
            batch_size (int): Batch size.
            height (int): Spatial height.
            width (int): Spatial width.
            device (torch.device): Target device.

        Returns:
            list[tuple]: List of (h, c) tensors, one tuple per layer.

        Example:
            >>> model = ConvLSTMNet()
            >>> h = model.init_hidden(2, 224, 224, torch.device('cpu'))
            >>> len(h)
            3
        """
        return [cell.init_hidden(batch_size, height, width, device)
                for cell in self.cells]

    def forward(self, x: torch.Tensor,
                hidden_state: list[tuple] | None = None
                ) -> tuple[torch.Tensor, list[tuple]]:
        """Run one autoregressive step.

        Args:
            x (torch.Tensor): Input (B, in_channels, H, W).
            hidden_state (list[tuple] | None): Per-layer (h, c), or None for zeros.

        Returns:
            tuple:
                pred (torch.Tensor): (B, out_channels, H, W) ocean prediction.
                new_hidden (list[tuple]): Updated per-layer hidden states.

        Example:
            >>> model = ConvLSTMNet()
            >>> pred, h = model(torch.randn(1, 11, 224, 224))
            >>> pred.shape
            torch.Size([1, 5, 224, 224])
        """
        B, _, H, W = x.shape
        if hidden_state is None:
            hidden_state = self.init_hidden(B, H, W, x.device)

        feat = self.input_proj(x)

        new_hidden: list[tuple] = []
        for i, cell in enumerate(self.cells):
            h_prev, c_prev = hidden_state[i]
            h, c = cell(feat, h_prev, c_prev)
            new_hidden.append((h, c))
            feat = h

        pred = self.output_head(feat)
        return pred, new_hidden
