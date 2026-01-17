from torch import nn
import torch


class Model(nn.Module):
    """
    1D CNN for energy disaggregation (NILM).
    Predicts appliance power from mains (total) power.
    Input:  x shape = [B, 1, T] (mains power)
    Output: y shape = [B, 1, T] (appliance power)
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
