"""A small CNN for slice-level hemorrhage classification.

Why a hand-written CNN and not a pretrained ResNet:
    The point of Step 10 is the Dataset -> DataLoader -> model -> metrics
    chain and the experiment record around it, not leaderboard performance.
    A ~200k-parameter CNN trains on CPU in minutes and is easy to reason
    about. Swapping in a pretrained backbone later changes one class and
    nothing else — which is itself the lesson about where the seams are.

Design notes:
    - One input channel: CT slices are grayscale (Step 3).
    - BatchNorm after each conv: the inputs are normalized with train-split
      statistics, and BN keeps activations well-scaled in a small net.
    - Global average pooling instead of a large flatten: far fewer
      parameters, and it makes the model independent of input resolution.
    - A single output logit, paired with BCEWithLogitsLoss, which is
      numerically stabler than sigmoid followed by BCE.
"""

from __future__ import annotations

import torch
from torch import nn


class SliceCNN(nn.Module):
    """Four conv blocks, global pooling, one logit."""

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.3):
        super().__init__()
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (batch,) — no sigmoid here."""
        return self.head(self.pool(self.features(x))).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
