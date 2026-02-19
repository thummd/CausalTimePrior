"""Generate prior property distribution histograms for the paper appendix."""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter


def plot_prior_distributions(
    dataset_path: str = 'data/causal_time_prior_full_100k.pt',
    output_path: str = 'demo_outputs/pdf/prior_distributions.pdf',
):
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, map_location='cpu')

    num_vars = dataset['num_vars'].numpy()
    intervention_types = dataset['intervention_types']
    intervention_values = dataset['intervention_values'].numpy()
    intervention_times = dataset['intervention_times'].numpy()
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    n_samples = len(num_vars)

    print(f"Dataset: {n_samples} samples")

    # Compute intervention effect sizes at intervention point
    print("Computing intervention effect sizes...")
    effect_sizes = []
    for i in range(min(n_samples, 10000)):  # subsample for speed
        t = intervention_times[i]
        n = num_vars[i]
        target = dataset['intervention_targets'][i].item()
        if t < X_obs.shape[1] and target < n:
            effect = abs(X_int[i, t, target].item() - X_obs[i, t, target].item())
            effect_sizes.append(effect)
    effect_sizes = np.array(effect_sizes)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # 1. Number of variables
    ax = axes[0, 0]
    counts = Counter(num_vars)
    vals = sorted(counts.keys())
    ax.bar(vals, [counts[v] for v in vals], color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Number of variables (N)')
    ax.set_ylabel('Count')
    ax.set_title('(a) Graph size')
    ax.set_xticks(vals)

    # 2. Intervention type distribution
    ax = axes[0, 1]
    type_counts = Counter(intervention_types)
    labels = ['hard', 'soft', 'time_varying']
    display_labels = ['Hard', 'Soft', 'Time-varying']
    colors = ['#DD8452', '#55A868', '#C44E52']
    bar_vals = [type_counts.get(l, 0) for l in labels]
    ax.bar(display_labels, bar_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Count')
    ax.set_title('(b) Intervention type')
    for i_bar, v in enumerate(bar_vals):
        ax.text(i_bar, v + n_samples * 0.01, f'{v/n_samples*100:.0f}%',
                ha='center', va='bottom', fontsize=9)

    # 3. Intervention effect size (log scale)
    ax = axes[0, 2]
    # Clip to reasonable range for visualization
    clipped = effect_sizes[effect_sizes > 1e-4]
    ax.hist(np.log10(clipped), bins=50, color='#8172B3', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('log₁₀(|effect size|)')
    ax.set_ylabel('Count')
    ax.set_title('(c) Intervention effect magnitude')
    ax.axvline(np.log10(np.median(clipped)), color='red', linestyle='--', alpha=0.7, label=f'Median={np.median(clipped):.1f}')
    ax.legend(fontsize=8)

    # 4. Intervention timing
    ax = axes[1, 0]
    ax.hist(intervention_times, bins=range(0, 51), color='#64B5CD', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Intervention start time')
    ax.set_ylabel('Count')
    ax.set_title('(d) Intervention timing')

    # 5. Edge probability (sampled from Beta prior)
    ax = axes[1, 1]
    rng = np.random.RandomState(42)
    edge_probs = rng.beta(2, 5, size=100000)
    ax.hist(edge_probs, bins=50, color='#CCB974', edgecolor='white', linewidth=0.5, density=True)
    ax.set_xlabel('Edge probability p')
    ax.set_ylabel('Density')
    ax.set_title('(e) Edge probability prior (Beta(2,5))')
    ax.axvline(np.mean(edge_probs), color='red', linestyle='--', alpha=0.7, label=f'Mean={np.mean(edge_probs):.2f}')
    ax.legend(fontsize=8)

    # 6. Intervention values distribution
    ax = axes[1, 2]
    # Clip extreme values for visualization
    clipped_vals = np.clip(intervention_values, -10, 10)
    ax.hist(clipped_vals, bins=50, color='#DA8BC3', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Intervention value')
    ax.set_ylabel('Count')
    ax.set_title('(f) Intervention values')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved to {output_path}")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  N variables: mean={np.mean(num_vars):.1f}, range=[{np.min(num_vars)}, {np.max(num_vars)}]")
    print(f"  Intervention types: {dict(type_counts)}")
    print(f"  Effect sizes: median={np.median(effect_sizes):.2f}, mean={np.mean(effect_sizes):.2f}")
    print(f"  Intervention times: mean={np.mean(intervention_times):.1f}")
    print(f"  Intervention values: mean={np.mean(intervention_values):.2f}, std={np.std(intervention_values):.2f}")


if __name__ == "__main__":
    plot_prior_distributions()
