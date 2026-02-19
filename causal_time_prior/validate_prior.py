"""Validate CausalTimePrior by analyzing sampled SCMs."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from causal_time_prior.prior import CausalTimePrior

def validate_prior(n_samples=1000, seed=42):
    """Generate statistics about the CausalTimePrior."""
    
    print("=" * 80)
    print("Validating CausalTimePrior")
    print("=" * 80)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    prior = CausalTimePrior(
        n_vars_range=(3, 10),
        max_lag=3,
        T=50,
        intervention_prob=0.8,
    )
    
    stats = {
        'n_vars': [],
        'n_edges': [],
        'graph_density': [],
        'max_lag_used': [],
        'n_parents': [],
        'intervention_effect_size': [],
        'is_regime_switching': [],
    }
    
    diverged = 0
    
    print(f"\nSampling {n_samples} SCMs from CausalTimePrior...")
    
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        # Sample SCM
        scm_data = prior.sample()
        
        # Check for divergence
        X_obs = scm_data['X_obs']
        if torch.isnan(X_obs).any() or torch.isinf(X_obs).any():
            diverged += 1
            continue
        
        # Extract statistics
        n_vars = scm_data['n_vars']
        graph = scm_data['graph']
        
        stats['n_vars'].append(n_vars)
        
        # Count edges and compute density
        n_edges = 0
        max_lag_used = 0
        n_parents_list = []
        
        for lag in range(len(graph)):
            if graph[lag] is not None:
                edges_at_lag = graph[lag].sum().item()
                n_edges += edges_at_lag
                if edges_at_lag > 0 and lag > max_lag_used:
                    max_lag_used = lag
        
        stats['n_edges'].append(n_edges)
        
        # Density = edges / possible_edges
        # Possible edges = n_vars * n_vars * (max_lag + 1) for fully connected
        possible_edges = n_vars * n_vars * len(graph)
        density = n_edges / possible_edges if possible_edges > 0 else 0
        stats['graph_density'].append(density)
        stats['max_lag_used'].append(max_lag_used)
        
        # Average parents per node
        for lag in range(len(graph)):
            if graph[lag] is not None:
                parents_per_node = graph[lag].sum(dim=0)  # sum over source nodes
                n_parents_list.extend(parents_per_node.tolist())
        
        if n_parents_list:
            stats['n_parents'].append(np.mean(n_parents_list))
        
        # Intervention effect size
        X_obs = scm_data['X_obs']
        X_int = scm_data['X_int']
        effect = (X_int - X_obs).abs().mean().item()
        stats['intervention_effect_size'].append(effect)
        
        # Check if regime-switching
        is_regime_switching = scm_data.get('is_regime_switching', False)
        stats['is_regime_switching'].append(is_regime_switching)
    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("Prior Validation Results")
    print("=" * 80)
    
    print(f"\nSample Statistics (n={n_samples}):")
    print(f"  Diverged SCMs: {diverged}/{n_samples} ({100*diverged/n_samples:.1f}%)")
    
    for key in ['n_vars', 'n_edges', 'graph_density', 'max_lag_used', 'n_parents', 'intervention_effect_size']:
        values = stats[key]
        if values:
            print(f"  {key}: {np.mean(values):.2f} ± {np.std(values):.2f} (range: [{np.min(values):.2f}, {np.max(values):.2f}])")
    
    # Regime-switching percentage
    regime_switching_pct = 100 * np.mean(stats['is_regime_switching'])
    print(f"  Regime-switching SCMs: {regime_switching_pct:.1f}%")
    
    # Save results
    results = {
        'n_samples': n_samples,
        'diverged': diverged,
        'stats': stats,
        'summary': {
            'n_vars_mean': float(np.mean(stats['n_vars'])),
            'n_vars_std': float(np.std(stats['n_vars'])),
            'graph_density_mean': float(np.mean(stats['graph_density'])),
            'graph_density_std': float(np.std(stats['graph_density'])),
            'max_lag_mean': float(np.mean(stats['max_lag_used'])),
            'intervention_effect_mean': float(np.mean(stats['intervention_effect_size'])),
            'regime_switching_pct': float(regime_switching_pct),
            'divergence_rate': float(100 * diverged / n_samples),
        }
    }
    
    return results


if __name__ == "__main__":
    results = validate_prior(n_samples=1000, seed=42)
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)