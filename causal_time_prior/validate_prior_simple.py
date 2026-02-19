"""Quick validation of CausalTimePrior - just report basic statistics."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

# Load the existing 100K dataset and compute statistics
dataset_path = 'data/causal_time_prior_full_100k.pt'

print("=" * 80)
print("CausalTimePrior Validation - Analyzing Existing 100K Dataset")
print("=" * 80)

dataset = torch.load(dataset_path, map_location='cpu')

X_obs = dataset['X_obs']  # Shape: (n_samples, T, n_vars)
X_int = dataset['X_int']

n_samples = X_obs.shape[0]
T = X_obs.shape[1]
max_vars = X_obs.shape[2]

print(f"\nDataset Statistics:")
print(f"  Total samples: {n_samples}")
print(f"  Sequence length (T): {T}")
print(f"  Max variables: {max_vars}")

# Intervention effect statistics
intervention_effects = (X_int - X_obs).abs().mean(dim=(1, 2))  # Mean abs effect per sample
print(f"\nIntervention Effect Size:")
print(f"  Mean: {intervention_effects.mean():.2f}")
print(f"  Std: {intervention_effects.std():.2f}")
print(f"  Min: {intervention_effects.min():.2f}")
print(f"  Max: {intervention_effects.max():.2f}")

# Check for divergence
n_diverged_obs = torch.isnan(X_obs).any(dim=(1,2)).sum().item() + torch.isinf(X_obs).any(dim=(1,2)).sum().item()
n_diverged_int = torch.isnan(X_int).any(dim=(1,2)).sum().item() + torch.isinf(X_int).any(dim=(1,2)).sum().item()

print(f"\nDivergence Check:")
print(f"  Diverged observational series: {n_diverged_obs}/{n_samples} ({100*n_diverged_obs/n_samples:.2f}%)")
print(f"  Diverged interventional series: {n_diverged_int}/{n_samples} ({100*n_diverged_int/n_samples:.2f}%)")

# Data range statistics
print(f"\nObservational Data Range:")
print(f"  Mean: {X_obs.mean():.2f}")
print(f"  Std: {X_obs.std():.2f}")
print(f"  Min: {X_obs.min():.2f}")
print(f"  Max: {X_obs.max():.2f}")

print(f"\nInterventional Data Range:")
print(f"  Mean: {X_int.mean():.2f}")
print(f"  Std: {X_int.std():.2f}")
print(f"  Min: {X_int.min():.2f}")
print(f"  Max: {X_int.max():.2f}")

print("\n" + "=" * 80)
print("Validation Complete!")
print("=" * 80)