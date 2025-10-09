import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from project_theia.data.data_spec import DataSpec


@dataclass
class SimpleMNISTNetConfig:
    drop_out_rate: int = 0.1


class SimpleMNISTNet(nn.Module):
    def __init__(self, config: SimpleMNISTNetConfig, data_spec: DataSpec, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)