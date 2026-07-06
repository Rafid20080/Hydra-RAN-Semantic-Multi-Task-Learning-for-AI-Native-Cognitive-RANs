"""
Shared Semantic Encoder
IEEE TCCN Reproducibility Package
"""

import numpy as np
import torch
import torch.nn as nn


class SharedSemanticEncoder(nn.Module):
    """Shared semantic feature extractor for all cognitive tasks."""
    
    def __init__(self, input_dim=64, hidden_dim=128, latent_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)
    
    def encode(self, observation):
        """Public interface for encoding observations."""
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        return self.forward(observation)
