# Copyright 2026 Dennis Thumm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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