"""Generate OOD (out-of-distribution) test set with different hyperparameters.

This tests structural generalization by using:
- Larger graphs: N ∈ [8,10] (vs train mostly smaller)
- More lags: K = 3 (vs train mostly 1-2)
- Denser graphs: edge_prob ∈ [0.3,0.5] (vs train ~0.29 mean)
- Complex mechanisms: sin/cos/square/tanhReLU only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import time

from causal_time_prior.prior import CausalTimePrior
from causal_time_prior.temporal_scm_builder import TemporalSCMBuilder
from dopfnprior.utils.sampling import ShiftedExponentialSampler


def pad_to_max_nodes(X: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Pad time series to have max_nodes variables."""
    T, N = X.shape
    if N < max_nodes:
        padding = torch.zeros(T, max_nodes - N, dtype=X.dtype, device=X.device)
        X_padded = torch.cat([X, padding], dim=1)
    else:
        X_padded = X[:, :max_nodes]
    return X_padded


def generate_ood_test_set(
    n_scms: int = 1000,
    T: int = 50,
    max_nodes: int = 10,
    seed: int = 999,
    output_path: str = "data/test_set_ood_1k.pt",
):
    """Generate OOD test set with different hyperparameters."""
    
    print("=" * 80)
    print(f"Generating OOD Test Set: {n_scms} SCMs")
    print("=" * 80)
    
    # Initialize prior with OOD configuration
    config = {
        'N_max': 10,
        'K_max': 3,
        'alpha': 5,  # Higher alpha for denser graphs
        'beta': 5,   # Beta(5,5) gives mean 0.5
        'gamma': 0.7,
        'sigma_w': 1.0,
        'sigma_b': 0.5,
        'root_mean': 0.0,
        'non_root_mean': 0.0,
        'T': T,
        'burn_in': 20,
        'device': 'cpu',
    }
    
    prior = CausalTimePrior(config=config, seed=seed, chain_prob=0.0)  # No chains for OOD
    
    # Create complex activation functions
    complex_activations = [
        prior._create_sin(),
        prior._create_cos(),
        prior._create_square(),
        prior.activations[3],  # TanhReLU
    ]
    
    data = []
    start_time = time.time()
    
    print("\n1. Generating OOD SCM pairs...")
    print("   OOD settings: N ∈ [8,10], K = 3, denser graphs, complex mechanisms")
    
    for i in range(n_scms):
        # Force N ∈ [8,10]
        N = int(torch.randint(8, 11, (1,), generator=prior.generator).item())
        K = 3  # Always max lag
        
        # Dense graphs
        edge_prob = float(torch.distributions.Beta(5, 5).sample().item())
        edge_prob = max(0.3, edge_prob)  # At least 0.3
        
        scm_builder = TemporalSCMBuilder(
            num_nodes=N,
            max_lag=K,
            edge_prob=edge_prob,
            dropout_prob=0.0,
            gamma=config['gamma'],
            activations=complex_activations,  # Only complex mechanisms
            root_std_dist=ShiftedExponentialSampler(rate=1.0, shift=0.1),
            non_root_std_dist=ShiftedExponentialSampler(rate=10.0, shift=0.01),
            root_mean=config['root_mean'],
            non_root_mean=config['non_root_mean'],
            sigma_w=config['sigma_w'],
            sigma_b=config['sigma_b'],
            device=config['device'],
        )
        
        scm = scm_builder.sample(prior.generator)
        
        # Sample intervention
        from causal_time_prior.interventions import InterventionSampler
        intervention_sampler = InterventionSampler(
            N=N,
            T=T,
            generator=prior.generator,
        )
        intervention = intervention_sampler.sample()
        
        # Generate data
        X_obs = scm.sample_observational(
            T=T,
            burn_in=config['burn_in'],
            generator=prior.generator,
        )
        
        X_int = scm.sample_interventional(
            T=T,
            intervention=intervention,
            burn_in=config['burn_in'],
            generator=prior.generator,
        )
        
        # Skip if diverged
        if torch.isnan(X_obs).any() or torch.isnan(X_int).any():
            continue
        
        data.append((X_obs, X_int, intervention, scm))
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = len(data) / elapsed
            eta = (n_scms - len(data)) / rate
            print(f"   Progress: {len(data)}/{n_scms} ({100 * len(data) / n_scms:.1f}%) - "
                  f"Rate: {rate:.1f} SCMs/s - ETA: {eta:.1f}s")
    
    print(f"\n2. Generated {len(data)} OOD SCM pairs in {time.time() - start_time:.1f}s")
    
    # Convert to threeway format
    from causal_time_prior.generate_dataset_threeway import convert_to_threeway_format
    
    dataset = convert_to_threeway_format(data, max_nodes=max_nodes)
    torch.save(dataset, output_path)
    
    print(f"\n3. Saved to: {output_path}")
    print("=" * 80)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_scms', type=int, default=1000)
    parser.add_argument('--output', type=str, default='data/test_set_ood_1k.pt')
    
    args = parser.parse_args()
    
    generate_ood_test_set(
        n_scms=args.n_scms,
        output_path=args.output,
    )