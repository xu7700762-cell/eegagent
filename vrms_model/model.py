# -*- coding: utf-8 -*-
import torch
from torch import nn

class PathMIL(nn.Module):
    """One logit per complete variable-length path, with a shared attention head."""
    def __init__(self, input_dim, center=None, scale=None):
        super().__init__()
        self.register_buffer("center", torch.zeros(input_dim) if center is None else torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(input_dim) if scale is None else torch.as_tensor(scale, dtype=torch.float32))
        self.projection = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Linear(64, 1)

    def forward(self, window_features):
        if window_features.ndim != 2 or not len(window_features):
            raise ValueError("MIL needs at least one valid window, never an empty path")
        z = self.projection(((window_features.float() - self.center) / self.scale).clamp(-20, 20))
        attention = self.attention(z).softmax(dim=0)
        return self.classifier((attention * z).sum(dim=0)).squeeze(-1)
