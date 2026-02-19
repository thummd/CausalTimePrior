"""Demo script for CausalTimePrior.

This script demonstrates the end-to-end pipeline:
1. Sample temporal SCMs from the prior
2. Generate paired observational/interventional time series
3. Visualize the results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from causal_time_prior.prior import CausalTimePrior
from causal_time_prior.visualization import (
    plot_paired_timeseries,
    plot_intervention_effect,
    plot_temporal_dag,
    plot_all_variables,
)


def main():
    """Run the CausalTimePrior demo."""
    print("=" * 80)
    print("CausalTimePrior Demo")
    print("=" * 80)
    
    # Initialize prior
    print("\n1. Initializing CausalTimePrior...")
    prior = CausalTimePrior(seed=42)
    print(f"   Config: N_max={prior.config['N_max']}, K_max={prior.config['K_max']}, T={prior.config['T']}")
    
    # Generate a single SCM and paired data
    print("\n2. Generating a single temporal SCM with intervention...")
    X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
    
    print(f"   SCM: {len(scm._topo)} variables, max_lag={scm._K}")
    print(f"   Intervention: {intervention.intervention_type.value} on variables {intervention.targets}")
    print(f"   Intervention times: {min(intervention.times)} to {max(intervention.times)}")
    print(f"   X_obs shape: {X_obs.shape}, X_int shape: {X_int.shape}")
    
    # Check for divergence
    if torch.isnan(X_obs).any() or torch.isnan(X_int).any():
        print("   WARNING: NaN detected in generated data. Trying again...")
        X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
    
    # Visualize
    print("\n3. Creating visualizations...")
    
    # Create output directory
    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot paired time series
    print("   - Plotting paired time series...")
    plot_paired_timeseries(
        X_obs, X_int, intervention,
        save_path=os.path.join(output_dir, "paired_timeseries.png")
    )
    
    # Plot intervention effect
    print("   - Plotting intervention effect...")
    plot_intervention_effect(
        X_obs, X_int, intervention,
        save_path=os.path.join(output_dir, "intervention_effect.png")
    )
    
    # Plot temporal DAG
    print("   - Plotting temporal DAG structure...")
    plot_temporal_dag(
        scm.dag,
        save_path=os.path.join(output_dir, "temporal_dag.png")
    )
    
    # Plot all variables
    print("   - Plotting all variables...")
    plot_all_variables(
        X_obs, X_int, intervention,
        save_path=os.path.join(output_dir, "all_variables.png")
    )
    
    # Generate a small dataset
    print("\n4. Generating a small dataset (10 SCMs)...")
    dataset = prior.generate_dataset(n_scms=10, T=100)
    print(f"   Generated {len(dataset)} SCM pairs")
    
    # Compute statistics
    print("\n5. Computing dataset statistics...")
    intervention_types = {}
    for X_obs, X_int, intervention in dataset:
        int_type = intervention.intervention_type.value
        intervention_types[int_type] = intervention_types.get(int_type, 0) + 1
    
    print(f"   Intervention type distribution:")
    for int_type, count in intervention_types.items():
        print(f"     - {int_type}: {count}/{len(dataset)}")
    
    # Check data quality
    nan_count = sum(1 for X_obs, X_int, _ in dataset if torch.isnan(X_obs).any() or torch.isnan(X_int).any())
    print(f"   Data quality: {len(dataset) - nan_count}/{len(dataset)} valid pairs")
    
    print("\n" + "=" * 80)
    print(f"Demo complete! Visualizations saved to '{output_dir}/'")
    print("=" * 80)


if __name__ == "__main__":
    main()