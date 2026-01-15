from torch import nn
import torch


class Model(nn.Module):
    """
    Simple 1D CNN for time-series regression.
    Input:  x shape = [B, 1, T]
    Output: y shape = [B, 1, T]  (e.g. appliance power)
    """

    def __init__(self, window_size: int = 1024):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        return self.net(x)
