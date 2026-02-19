"""SimpleCausalPFN v2: Predicts downstream effects of interventions.

This version properly tests causal understanding by predicting how interventions
on variable i at time t affect variable j at time τ (where j can differ from i).
"""

import torch
import torch.nn as nn
from typing import Tuple


class SimpleCausalPFNV2(nn.Module):
    """
    Simple causal PFN for temporal intervention prediction with downstream queries.
    
    Key difference from v1: Takes separate intervention and query specifications.
    The model must learn how interventions propagate through the causal graph.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_nodes: int = 10,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_nodes = max_nodes
        
        # Temporal encoder (GRU processes observational time series)
        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Intervention encoder (encodes which variable, when, and what value)
        self.intervention_encoder = nn.Sequential(
            nn.Linear(max_nodes + 2, hidden_dim),  # one-hot target + time + value
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Query encoder (encodes which variable to predict and when)
        self.query_encoder = nn.Sequential(
            nn.Linear(max_nodes + 1, hidden_dim),  # one-hot target + time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Prediction head (combines all encodings to predict outcome)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # mean and log_std
        )
    
    def forward(
        self,
        X_obs: torch.Tensor,
        intervention_target: torch.Tensor,
        intervention_time: torch.Tensor,
        intervention_value: torch.Tensor,
        query_target: torch.Tensor,
        query_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        X_obs : torch.Tensor
            Observational time series (batch_size, T, input_dim).
        intervention_target : torch.Tensor
            Which variable is intervened on (batch_size,).
        intervention_time : torch.Tensor
            When the intervention occurs (batch_size,).
        intervention_value : torch.Tensor
            Value of intervention (batch_size,).
        query_target : torch.Tensor
            Which variable to predict (batch_size,).
        query_time : torch.Tensor
            When to predict (batch_size,).
            
        Returns
        -------
        mean : torch.Tensor
            Predicted mean (batch_size,).
        std : torch.Tensor
            Predicted std (batch_size,).
        """
        batch_size = X_obs.shape[0]
        T = X_obs.shape[1]
        
        # 1. Encode observational time series
        _, h_temporal = self.temporal_encoder(X_obs)  # (num_layers, batch, hidden)
        h_temporal = h_temporal[-1]  # (batch, hidden) - take last layer
        
        # 2. Encode intervention specification
        # One-hot encode intervention target
        intervention_target_onehot = torch.zeros(
            batch_size, self.max_nodes, device=X_obs.device
        )
        intervention_target_onehot[torch.arange(batch_size), intervention_target] = 1.0
        
        # Normalize time and value
        intervention_time_norm = intervention_time.float() / T
        intervention_value_norm = intervention_value / 10.0  # Rough normalization
        
        # Concatenate and encode
        intervention_features = torch.cat([
            intervention_target_onehot,
            intervention_time_norm.unsqueeze(1),
            intervention_value_norm.unsqueeze(1),
        ], dim=1)
        h_intervention = self.intervention_encoder(intervention_features)
        
        # 3. Encode query specification
        # One-hot encode query target
        query_target_onehot = torch.zeros(
            batch_size, self.max_nodes, device=X_obs.device
        )
        query_target_onehot[torch.arange(batch_size), query_target] = 1.0
        
        # Normalize time
        query_time_norm = query_time.float() / T
        
        # Concatenate and encode
        query_features = torch.cat([
            query_target_onehot,
            query_time_norm.unsqueeze(1),
        ], dim=1)
        h_query = self.query_encoder(query_features)
        
        # 4. Combine all encodings
        h_combined = torch.cat([h_temporal, h_intervention, h_query], dim=1)
        
        # 5. Predict outcome
        out = self.prediction_head(h_combined)
        mean = out[:, 0]
        log_std = out[:, 1]
        std = torch.exp(log_std).clamp(min=0.01, max=10.0)
        
        return mean, std
    
    def loss(
        self,
        X_obs: torch.Tensor,
        X_int: torch.Tensor,
        intervention_target: torch.Tensor,
        intervention_time: torch.Tensor,
        intervention_value: torch.Tensor,
        query_target: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Parameters
        ----------
        X_obs : torch.Tensor
            Observational time series (batch_size, T, input_dim).
        X_int : torch.Tensor
            Interventional time series (batch_size, T, input_dim).
        intervention_target : torch.Tensor
            Which variable is intervened on (batch_size,).
        intervention_time : torch.Tensor
            When the intervention occurs (batch_size,).
        intervention_value : torch.Tensor
            Value of intervention (batch_size,).
        query_target : torch.Tensor
            Which variable to predict (batch_size,).
        query_time : torch.Tensor
            When to predict (batch_size,).
            
        Returns
        -------
        loss : torch.Tensor
            Negative log-likelihood.
        """
        # Predict
        mean, std = self(
            X_obs,
            intervention_target,
            intervention_time,
            intervention_value,
            query_target,
            query_time,
        )
        
        # Ground truth: value of query_target at query_time in interventional series
        batch_indices = torch.arange(X_int.shape[0], device=X_int.device)
        Y_true = X_int[batch_indices, query_time, query_target]
        
        # Negative log-likelihood (Gaussian)
        nll = 0.5 * torch.log(2 * torch.pi * std**2) + 0.5 * ((Y_true - mean) / std)**2
        
        return nll.mean()