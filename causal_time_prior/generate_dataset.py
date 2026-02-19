"""Generate a dataset of temporal SCMs with interventions for training.

This script generates 1K paired observational/interventional time series
and saves them to disk for efficient loading during training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import time
from typing import List, Tuple

from causal_time_prior.prior import CausalTimePrior
from causal_time_prior.interventions import InterventionSpec


def pad_to_max_nodes(X: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Pad time series to have max_nodes variables.
    
    Parameters
    ----------
    X : torch.Tensor
        Time series of shape (T, N) where N <= max_nodes.
    max_nodes : int
        Target number of variables.
        
    Returns
    -------
    torch.Tensor
        Padded time series of shape (T, max_nodes).
    """
    T, N = X.shape
    if N < max_nodes:
        padding = torch.zeros(T, max_nodes - N, dtype=X.dtype, device=X.device)
        X_padded = torch.cat([X, padding], dim=1)
    else:
        X_padded = X[:, :max_nodes]  # Truncate if needed
    return X_padded


def generate_dataset(
    n_scms: int,
    T: int,
    max_nodes: int,
    seed: int = 42,
    output_path: str = "data/causal_time_prior_1k.pt",
):
    """Generate a dataset of temporal SCMs with interventions.
    
    Parameters
    ----------
    n_scms : int
        Number of SCMs to generate.
    T : int
        Length of time series.
    max_nodes : int
        Maximum number of nodes (for padding).
    seed : int
        Random seed.
    output_path : str
        Path to save the dataset.
    """
    print("=" * 80)
    print(f"Generating Dataset: {n_scms} SCMs, T={T}, max_nodes={max_nodes}")
    print("=" * 80)
    
    # Initialize prior
    print("\n1. Initializing CausalTimePrior...")
    prior = CausalTimePrior(seed=seed)
    
    # Storage
    X_obs_list = []
    X_int_list = []
    targets_list = []
    intervention_times_list = []
    intervention_values_list = []
    intervention_types_list = []
    num_vars_list = []
    
    # Statistics
    start_time = time.time()
    intervention_type_counts = {}
    diverged_count = 0
    
    print(f"\n2. Generating {n_scms} SCM pairs...")
    for i in range(n_scms):
        X_obs, X_int, intervention, scm = prior.generate_pair(T=T)
        
        # Check for divergence
        if torch.isnan(X_obs).any() or torch.isnan(X_int).any():
            diverged_count += 1
            # Regenerate
            X_obs, X_int, intervention, scm = prior.generate_pair(T=T)
        
        # Pad to max_nodes
        N = X_obs.shape[1]
        X_obs_padded = pad_to_max_nodes(X_obs, max_nodes)
        X_int_padded = pad_to_max_nodes(X_int, max_nodes)
        
        # Store
        X_obs_list.append(X_obs_padded)
        X_int_list.append(X_int_padded)
        
        # Store intervention info (adjusted for padding)
        target = intervention.targets[0] if len(intervention.targets) > 0 else 0
        targets_list.append(target)
        
        # Use mid-point of intervention time as single target time
        int_time = int(np.mean(intervention.times))
        intervention_times_list.append(int_time)
        
        # Store intervention value (scalar for simplicity)
        if callable(intervention.values):
            value = intervention.values(int_time)
        else:
            value = intervention.values
        intervention_values_list.append(float(value))
        
        intervention_types_list.append(intervention.intervention_type.value)
        num_vars_list.append(N)
        
        # Update statistics
        int_type = intervention.intervention_type.value
        intervention_type_counts[int_type] = intervention_type_counts.get(int_type, 0) + 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_scms - i - 1) / rate
            print(f"   Progress: {i + 1}/{n_scms} ({100 * (i + 1) / n_scms:.1f}%) - "
                  f"Rate: {rate:.1f} SCMs/s - ETA: {eta:.1f}s")
    
    elapsed_total = time.time() - start_time
    
    print(f"\n3. Stacking tensors...")
    # Stack into tensors
    X_obs_tensor = torch.stack(X_obs_list)  # (n_scms, T, max_nodes)
    X_int_tensor = torch.stack(X_int_list)  # (n_scms, T, max_nodes)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)
    intervention_times_tensor = torch.tensor(intervention_times_list, dtype=torch.long)
    intervention_values_tensor = torch.tensor(intervention_values_list, dtype=torch.float32)
    num_vars_tensor = torch.tensor(num_vars_list, dtype=torch.long)
    
    # Create dataset dictionary
    dataset = {
        'X_obs': X_obs_tensor,
        'X_int': X_int_tensor,
        'targets': targets_tensor,
        'intervention_times': intervention_times_tensor,
        'intervention_values': intervention_values_tensor,
        'intervention_types': intervention_types_list,
        'num_vars': num_vars_tensor,
        'metadata': {
            'n_scms': n_scms,
            'T': T,
            'max_nodes': max_nodes,
            'seed': seed,
        }
    }
    
    # Save dataset
    print(f"\n4. Saving dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset, output_path)
    
    # Compute final statistics
    print("\n5. Dataset Statistics:")
    print(f"   Total SCMs generated: {n_scms}")
    print(f"   Time elapsed: {elapsed_total:.2f}s ({n_scms / elapsed_total:.2f} SCMs/s)")
    print(f"   Diverged (regenerated): {diverged_count}/{n_scms}")
    print(f"   Dataset shape: {X_obs_tensor.shape}")
    print(f"   Dataset size: {X_obs_tensor.element_size() * X_obs_tensor.nelement() * 2 / (1024**2):.2f} MB")
    print(f"\n   Intervention type distribution:")
    for int_type, count in sorted(intervention_type_counts.items()):
        print(f"     - {int_type}: {count}/{n_scms} ({100 * count / n_scms:.1f}%)")
    print(f"\n   Saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CausalTimePrior dataset")
    parser.add_argument('--n_scms', type=int, default=1000, help='Number of SCMs to generate')
    parser.add_argument('--T', type=int, default=50, help='Length of time series')
    parser.add_argument('--max_nodes', type=int, default=10, help='Maximum number of nodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/causal_time_prior_1k.pt', 
                        help='Output path')
    
    args = parser.parse_args()
    
    generate_dataset(
        n_scms=args.n_scms,
        T=args.T,
        max_nodes=args.max_nodes,
        seed=args.seed,
        output_path=args.output,
    )