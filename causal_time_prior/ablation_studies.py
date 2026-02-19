"""Ablation studies for CausalTimePrior.

This script runs ablation studies to demonstrate that prior diversity matters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import time
from typing import Dict, List

from causal_time_prior.simple_causal_pfn import SimpleCausalPFN
from causal_time_prior.baselines import evaluate_baseline
from causal_time_prior.train_simple_pfn import main as train_model


def filter_dataset_by_intervention_type(
    dataset_path: str,
    intervention_type: str,
    output_path: str,
    n_samples: int = 10000,
):
    """Filter dataset to only include specific intervention type.
    
    Parameters
    ----------
    dataset_path : str
        Path to original dataset.
    intervention_type : str
        Type to keep: 'hard', 'soft', or 'time_varying'.
    output_path : str
        Path to save filtered dataset.
    n_samples : int
        Target number of samples (will sample uniformly from matching samples).
    """
    print(f"\\nFiltering dataset for {intervention_type} interventions...")
    print(f"Loading from: {dataset_path}")
    
    # Load original dataset
    dataset = torch.load(dataset_path)
    
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    targets = dataset['targets']
    intervention_times = dataset['intervention_times']
    intervention_values = dataset['intervention_values']
    intervention_types = dataset['intervention_types']
    num_vars = dataset['num_vars']
    
    # Find indices with matching intervention type
    matching_indices = [i for i, t in enumerate(intervention_types) if t == intervention_type]
    
    print(f"Found {len(matching_indices)} samples with {intervention_type} interventions")
    
    if len(matching_indices) == 0:
        raise ValueError(f"No samples found with intervention type: {intervention_type}")
    
    # Sample n_samples uniformly
    if len(matching_indices) > n_samples:
        selected_indices = np.random.choice(matching_indices, n_samples, replace=False)
    else:
        selected_indices = matching_indices
    
    print(f"Selected {len(selected_indices)} samples")
    
    # Create filtered dataset
    filtered_dataset = {
        'X_obs': X_obs[selected_indices],
        'X_int': X_int[selected_indices],
        'targets': targets[selected_indices],
        'intervention_times': intervention_times[selected_indices],
        'intervention_values': intervention_values[selected_indices],
        'intervention_types': [intervention_types[i] for i in selected_indices],
        'num_vars': num_vars[selected_indices],
        'metadata': {
            'n_scms': len(selected_indices),
            'T': dataset['metadata']['T'],
            'max_nodes': dataset['metadata']['max_nodes'],
            'seed': dataset['metadata']['seed'],
            'filtered_by': intervention_type,
        }
    }
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(filtered_dataset, output_path)
    print(f"Saved filtered dataset to: {output_path}")
    print(f"Shape: {filtered_dataset['X_obs'].shape}")
    
    return output_path


def run_intervention_type_ablation():
    """Ablation 1: Impact of intervention type diversity in training data."""
    
    print("=" * 80)
    print("Ablation Study 1: Intervention Type Diversity")
    print("=" * 80)
    
    # Use 10K dataset for faster experimentation
    base_dataset = "data/causal_time_prior_10k.pt"
    
    # Filter datasets for each intervention type
    hard_only_path = filter_dataset_by_intervention_type(
        base_dataset, 'hard', 'data/ablation_hard_only.pt', n_samples=5000
    )
    
    soft_only_path = filter_dataset_by_intervention_type(
        base_dataset, 'soft', 'data/ablation_soft_only.pt', n_samples=5000
    )
    
    time_varying_only_path = filter_dataset_by_intervention_type(
        base_dataset, 'time_varying', 'data/ablation_time_varying_only.pt', n_samples=5000
    )
    
    # Training configurations
    configs = [
        ('hard_only', hard_only_path),
        ('soft_only', soft_only_path),
        ('time_varying_only', time_varying_only_path),
    ]
    
    results = {}
    
    # Train models
    for name, data_path in configs:
        print(f"\\nTraining model on {name} data...")
        checkpoint_path = f"checkpoints/ablation_{name}.pt"
        
        # Train for fewer epochs since this is an ablation
        train_model(
            data_path=data_path,
            hidden_dim=128,
            num_layers=3,
            batch_size=64,
            num_epochs=20,
            lr=5e-5,
            device='cpu',
            save_path=checkpoint_path,
        )
        
        results[name] = checkpoint_path
    
    print("\\n" + "=" * 80)
    print("Intervention Type Ablation Complete!")
    print("=" * 80)
    print("\\nTrained models:")
    for name, path in results.items():
        print(f"  - {name}: {path}")
    
    return results


def run_dataset_size_ablation():
    """Ablation 2: Impact of dataset size on performance."""
    
    print("=" * 80)
    print("Ablation Study 2: Dataset Size Scaling")
    print("=" * 80)
    
    # We already have 1K and 100K models
    # Just need to train on 10K with same config as 100K
    
    print("\\nTraining model on 10K dataset...")
    train_model(
        data_path="data/causal_time_prior_10k.pt",
        hidden_dim=256,
        num_layers=4,
        batch_size=64,
        num_epochs=30,  # Fewer epochs than 100K since less data
        lr=5e-5,
        device='cpu',
        save_path="checkpoints/causal_pfn_10k_stable.pt",
    )
    
    print("\\n" + "=" * 80)
    print("Dataset Size Ablation Complete!")
    print("=" * 80)
    print("\\nModels to compare:")
    print("  - 1K:   checkpoints/simple_causal_pfn.pt (57K params)")
    print("  - 10K:  checkpoints/causal_pfn_10k_stable.pt (1.67M params)")
    print("  - 100K: checkpoints/causal_pfn_100k_stable.pt (1.67M params)")
    
    return {
        '1k': 'checkpoints/simple_causal_pfn.pt',
        '10k': 'checkpoints/causal_pfn_10k_stable.pt',
        '100k': 'checkpoints/causal_pfn_100k_stable.pt',
    }


def main(ablation_type: str = 'all'):
    """Run ablation studies.
    
    Parameters
    ----------
    ablation_type : str
        Which ablation to run: 'intervention_type', 'dataset_size', or 'all'.
    """
    print("=" * 80)
    print("CausalTimePrior Ablation Studies")
    print("=" * 80)
    
    results = {}
    
    if ablation_type in ['intervention_type', 'all']:
        try:
            results['intervention_type'] = run_intervention_type_ablation()
        except Exception as e:
            print(f"Intervention type ablation failed: {e}")
    
    if ablation_type in ['dataset_size', 'all']:
        try:
            results['dataset_size'] = run_dataset_size_ablation()
        except Exception as e:
            print(f"Dataset size ablation failed: {e}")
    
    print("\\n" + "=" * 80)
    print("All Ablation Studies Complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument('--type', type=str, default='all',
                        choices=['intervention_type', 'dataset_size', 'all'],
                        help='Which ablation study to run')
    
    args = parser.parse_args()
    
    main(ablation_type=args.type)