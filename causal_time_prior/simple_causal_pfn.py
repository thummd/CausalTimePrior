"""Simple Causal PFN with GRU encoder for proof-of-concept.

This is a lightweight model for demonstrating that a PFN trained on 
CausalTimePrior can learn to predict interventional outcomes.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SimpleCausalPFN(nn.Module):
    """Simple causal PFN using GRU encoder.
    
    Architecture:
    - GRU encoder processes observational time series
    - Intervention query encoder embeds (target, time, value)
    - Prediction head outputs Gaussian parameters for Y_int_τ
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_nodes: int = 10,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Input dimension (max number of variables).
        hidden_dim : int
            Hidden dimension for GRU.
        num_layers : int
            Number of GRU layers.
        max_nodes : int
            Maximum number of nodes (for embeddings).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_nodes = max_nodes
        
        # GRU encoder for observational time series
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        
        # Intervention query encoder
        # Embeds: target variable index, intervention time, intervention value
        self.target_embedding = nn.Embedding(max_nodes, hidden_dim // 4)
        self.time_encoder = nn.Linear(1, hidden_dim // 4)
        self.value_encoder = nn.Linear(1, hidden_dim // 4)
        
        # Combine intervention query components
        self.intervention_encoder = nn.Sequential(
            nn.Linear(3 * hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Prediction head: combine GRU output + intervention encoding
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # mean, log_std
        )
    
    def forward(
        self,
        X_obs: torch.Tensor,
        targets: torch.Tensor,
        intervention_times: torch.Tensor,
        intervention_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        X_obs : torch.Tensor
            Observational time series, shape (batch, T, input_dim).
        targets : torch.Tensor
            Target variable indices, shape (batch,).
        intervention_times : torch.Tensor
            Intervention times, shape (batch,).
        intervention_values : torch.Tensor
            Intervention values, shape (batch,).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (mean, std) for Gaussian distribution over Y_int_τ.
            Each has shape (batch,).
        """
        batch_size = X_obs.shape[0]
        
        # Encode observational time series with GRU
        # output: (batch, T, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        gru_output, h_n = self.gru(X_obs)
        
        # Use final hidden state from last layer
        # h_final: (batch, hidden_dim)
        h_final = h_n[-1]
        
        # Encode intervention query
        # target_emb: (batch, hidden_dim // 4)
        target_emb = self.target_embedding(targets)
        
        # time_emb: (batch, hidden_dim // 4)
        time_normalized = intervention_times.float().unsqueeze(1) / X_obs.shape[1]  # Normalize by T
        time_emb = self.time_encoder(time_normalized)
        
        # value_emb: (batch, hidden_dim // 4)
        value_emb = self.value_encoder(intervention_values.unsqueeze(1))
        
        # Concatenate and encode intervention query
        # intervention_query: (batch, 3 * hidden_dim // 4)
        intervention_query = torch.cat([target_emb, time_emb, value_emb], dim=1)
        
        # intervention_encoding: (batch, hidden_dim)
        intervention_encoding = self.intervention_encoder(intervention_query)
        
        # Combine GRU output and intervention encoding
        # combined: (batch, hidden_dim * 2)
        combined = torch.cat([h_final, intervention_encoding], dim=1)
        
        # Predict Gaussian parameters
        # output: (batch, 2)
        output = self.predictor(combined)
        
        # Split into mean and log_std
        mean = output[:, 0]
        log_std = output[:, 1]
        std = torch.exp(log_std).clamp(min=1e-6, max=10.0)  # Ensure positive std
        
        return mean, std
    
    def loss(
        self,
        X_obs: torch.Tensor,
        X_int: torch.Tensor,
        targets: torch.Tensor,
        intervention_times: torch.Tensor,
        intervention_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian NLL loss.
        
        Parameters
        ----------
        X_obs : torch.Tensor
            Observational time series, shape (batch, T, input_dim).
        X_int : torch.Tensor
            Interventional time series, shape (batch, T, input_dim).
        targets : torch.Tensor
            Target variable indices, shape (batch,).
        intervention_times : torch.Tensor
            Intervention times, shape (batch,).
        intervention_values : torch.Tensor
            Intervention values, shape (batch,).
            
        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Forward pass
        mean, std = self.forward(X_obs, targets, intervention_times, intervention_values)
        
        # Extract ground-truth outcomes
        # Y_true: (batch,)
        batch_indices = torch.arange(X_int.shape[0], device=X_int.device)
        Y_true = X_int[batch_indices, intervention_times, targets]
        
        # Gaussian NLL loss
        # loss = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
        nll = 0.5 * torch.log(2 * torch.pi * std**2) + (Y_true - mean)**2 / (2 * std**2)
        
        return nll.mean()