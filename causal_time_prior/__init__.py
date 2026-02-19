"""CausalTimePrior: Synthetic data generator for temporal causal inference.

This package provides a framework for generating synthetic temporal structural causal
models (SCMs) with paired observational and interventional time series data.

Main components:
- CausalTimePrior: Main API for sampling SCMs and generating data
- TemporalSCM: Temporal structural causal model with time-lagged dependencies
- InterventionSpec: Specification for interventions (hard, soft, time-varying)
- Visualization utilities for plotting paired time series

Example usage:
    from causal_time_prior import CausalTimePrior
    
    # Initialize prior
    prior = CausalTimePrior(seed=42)
    
    # Generate paired data
    X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
    
    # Generate dataset
    dataset = prior.generate_dataset(n_scms=1000, T=100)
"""

from causal_time_prior.prior import CausalTimePrior
from causal_time_prior.temporal_scm import TemporalSCM
from causal_time_prior.interventions import InterventionSpec, InterventionType, InterventionSampler
from causal_time_prior.temporal_graph import TemporalDAG, TemporalGraphBuilder
from causal_time_prior.temporal_mechanism import TemporalMechanism
from causal_time_prior.temporal_scm_builder import TemporalSCMBuilder
from causal_time_prior.utils import DEFAULT_CONFIG

from causal_time_prior import visualization

__version__ = "0.1.0"

__all__ = [
    "CausalTimePrior",
    "TemporalSCM",
    "InterventionSpec",
    "InterventionType",
    "InterventionSampler",
    "TemporalDAG",
    "TemporalGraphBuilder",
    "TemporalMechanism",
    "TemporalSCMBuilder",
    "DEFAULT_CONFIG",
    "visualization",
]