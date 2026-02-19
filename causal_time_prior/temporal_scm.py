"""Temporal SCM with time-stepped forward simulation."""

import torch
from torch import Tensor
from typing import Dict, List, Optional
import networkx as nx
import numpy as np

from causal_time_prior.temporal_graph import TemporalDAG
from causal_time_prior.temporal_mechanism import TemporalMechanism
from causal_time_prior.interventions import InterventionSpec, InterventionType
from causal_time_prior.utils import clip_values, check_divergence
from dopfnprior.utils.sampling import DistributionSampler


class TemporalSCM:
    """
    Temporal Structural Causal Model with time-stepped forward simulation.
    
    Extends Do-PFN's SCM to support temporal dependencies with lags.
    """
    
    def __init__(
        self,
        dag: TemporalDAG,
        mechanisms: Dict[str, TemporalMechanism],
        noise: Dict[str, DistributionSampler],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        dag : TemporalDAG
            Temporal DAG with instantaneous and lagged edges.
        mechanisms : Dict[str, TemporalMechanism]
            Mechanisms for each variable.
        noise : Dict[str, DistributionSampler]
            Noise distributions for each variable.
        device : torch.device
            Device for computation.
        dtype : torch.dtype
            Data type for computation.
        """
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise = noise
        self.device = device
        self.dtype = dtype
        
        # Store topology
        self._topo = dag.topo_order
        self._G_0 = dag.G_0
        self._G_lags = dag.G_lags
        self._K = dag.K
        
        # Parents for each variable
        self._instant_parents = {v: list(self._G_0.predecessors(v)) for v in self._topo}
        self._lagged_parents = self._compute_lagged_parents()
    
    def _compute_lagged_parents(self) -> Dict[str, List[List[str]]]:
        """Compute lagged parents for each variable."""
        lagged_parents = {}
        node_to_idx = {v: i for i, v in enumerate(self._topo)}
        
        for v in self._topo:
            v_idx = node_to_idx[v]
            parents_per_lag = []
            
            for k in range(self._K):
                G_k = self._G_lags[k]
                # Find parents at lag k+1
                parents_k = [self._topo[j] for j in range(len(self._topo)) if G_k[j, v_idx] > 0]
                parents_per_lag.append(parents_k)
            
            lagged_parents[v] = parents_per_lag
        
        return lagged_parents
    
    @torch.no_grad()
    def sample_observational(
        self,
        T: int,
        burn_in: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample observational data from the temporal SCM.
        
        Parameters
        ----------
        T : int
            Length of time series to generate (after burn-in).
        burn_in : int
            Number of burn-in steps to discard.
        generator : torch.Generator, optional
            RNG for reproducibility.
            
        Returns
        -------
        torch.Tensor
            Time series data of shape (T, N) where N is number of variables.
        """
        total_T = T + burn_in
        N = len(self._topo)
        
        # Initialize buffer
        buffer = torch.zeros(total_T, N, device=self.device, dtype=self.dtype)
        
        # Forward simulation
        for t in range(total_T):
            for i, v in enumerate(self._topo):
                # Get instantaneous parent values
                instant_parents_v = self._instant_parents[v]
                parent_values_instant = {p: buffer[t, self._topo.index(p)] for p in instant_parents_v}
                
                # Get lagged parent values
                parent_values_lagged = []
                for k, parents_k in enumerate(self._lagged_parents[v]):
                    if t >= k + 1:  # Only access if lag is available
                        parent_values_k = {p: buffer[t - k - 1, self._topo.index(p)] for p in parents_k}
                    else:
                        parent_values_k = {}
                    parent_values_lagged.append(parent_values_k)
                
                # Sample noise
                eps = self.noise[v].sample(generator=generator)
                eps_tensor = torch.tensor([eps], device=self.device, dtype=self.dtype)
                
                # Apply mechanism
                mech_v = self.mechanisms[v]
                value = mech_v(parent_values_instant, parent_values_lagged, eps_tensor)
                
                # Clip for stability
                value = clip_values(value)
                buffer[t, i] = value.item()
        
        # Check for divergence
        if check_divergence(buffer):
            # If diverged, return zeros (or could resample)
            print("Warning: SCM diverged during simulation. Returning zeros.")
            return torch.zeros(T, N, device=self.device, dtype=self.dtype)
        
        # Return data after burn-in
        return buffer[burn_in:]
    
    @torch.no_grad()
    def sample_interventional(
        self,
        T: int,
        intervention: InterventionSpec,
        burn_in: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample interventional data from the temporal SCM.
        
        Parameters
        ----------
        T : int
            Length of time series to generate (after burn-in).
        intervention : InterventionSpec
            Intervention specification.
        burn_in : int
            Number of burn-in steps to discard.
        generator : torch.Generator, optional
            RNG for reproducibility.
            
        Returns
        -------
        torch.Tensor
            Time series data of shape (T, N) under intervention.
        """
        total_T = T + burn_in
        N = len(self._topo)
        
        # Initialize buffer
        buffer = torch.zeros(total_T, N, device=self.device, dtype=self.dtype)
        
        # Forward simulation with intervention
        for t in range(total_T):
            for i, v in enumerate(self._topo):
                # Check if this variable is intervened on at this time
                is_intervened = (i in intervention.targets) and (t - burn_in in intervention.times)
                
                if is_intervened:
                    if intervention.intervention_type == InterventionType.HARD:
                        # Hard intervention: replace with constant value
                        buffer[t, i] = intervention.values
                        
                    elif intervention.intervention_type == InterventionType.TIME_VARYING:
                        # Time-varying intervention: use function
                        buffer[t, i] = intervention.values(t - burn_in)
                        
                    else:  # SOFT intervention handled below with additive shift
                        # Get parent values and apply mechanism first
                        instant_parents_v = self._instant_parents[v]
                        parent_values_instant = {p: buffer[t, self._topo.index(p)] for p in instant_parents_v}
                        
                        parent_values_lagged = []
                        for k, parents_k in enumerate(self._lagged_parents[v]):
                            if t >= k + 1:
                                parent_values_k = {p: buffer[t - k - 1, self._topo.index(p)] for p in parents_k}
                            else:
                                parent_values_k = {}
                            parent_values_lagged.append(parent_values_k)
                        
                        eps = self.noise[v].sample(generator=generator)
                        eps_tensor = torch.tensor([eps], device=self.device, dtype=self.dtype)
                        
                        mech_v = self.mechanisms[v]
                        value = mech_v(parent_values_instant, parent_values_lagged, eps_tensor)
                        
                        # Add soft intervention shift
                        value = value + intervention.values
                        value = clip_values(value)
                        buffer[t, i] = value.item()
                else:
                    # No intervention: standard mechanism
                    instant_parents_v = self._instant_parents[v]
                    parent_values_instant = {p: buffer[t, self._topo.index(p)] for p in instant_parents_v}
                    
                    parent_values_lagged = []
                    for k, parents_k in enumerate(self._lagged_parents[v]):
                        if t >= k + 1:
                            parent_values_k = {p: buffer[t - k - 1, self._topo.index(p)] for p in parents_k}
                        else:
                            parent_values_k = {}
                        parent_values_lagged.append(parent_values_k)
                    
                    eps = self.noise[v].sample(generator=generator)
                    eps_tensor = torch.tensor([eps], device=self.device, dtype=self.dtype)
                    
                    mech_v = self.mechanisms[v]
                    value = mech_v(parent_values_instant, parent_values_lagged, eps_tensor)
                    value = clip_values(value)
                    buffer[t, i] = value.item()
        
        # Check for divergence
        if check_divergence(buffer):
            print("Warning: SCM diverged during interventional simulation. Returning zeros.")
            return torch.zeros(T, N, device=self.device, dtype=self.dtype)
        
        # Return data after burn-in
        return buffer[burn_in:]