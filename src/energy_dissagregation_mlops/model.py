from torch import nn
import torch
from loguru import logger


class Model(nn.Module):
    """
    1D CNN for energy disaggregation (NILM).
    Predicts appliance power from mains (total) power.
    Input:  x shape = [B, 1, T] (mains power)
    Output: y shape = [B, 1, T] (appliance power)
    """

    def __init__(self, window_size: int = 1024):
        super().__init__()

        logger.debug(f"Initializing Model with window_size={window_size}")

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.debug(f"Model architecture: 1D CNN with {total_params} total params ({trainable_params} trainable)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        return self.net(x)
