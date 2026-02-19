"""Evaluate ablation studies and generate comparison tables."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict, Tuple

from causal_time_prior.simple_causal_pfn import SimpleCausalPFN


def evaluate_model(
    model_path: str,
    dataset_path: str,
    n_test: int = 1000,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """Evaluate a single model on test set.
    
    Returns
    -------
    Tuple[float, float]
        (RMSE, MAE)
    """
    # Load dataset
    dataset = torch.load(dataset_path)
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    targets = dataset['targets']
    intervention_times = dataset['intervention_times']
    intervention_values = dataset['intervention_values']
    
    # Use last n_test samples
    X_obs_test = X_obs[-n_test:]
    X_int_test = X_int[-n_test:]
    targets_test = targets[-n_test:]
    intervention_times_test = intervention_times[-n_test:]
    intervention_values_test = intervention_values[-n_test:]
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    model = SimpleCausalPFN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        max_nodes=config['max_nodes'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in range(n_test):
            X_obs = X_obs_test[i:i+1].to(device)
            target = targets_test[i:i+1].to(device)
            int_time = intervention_times_test[i:i+1].to(device)
            int_value = intervention_values_test[i:i+1].to(device)
            
            # Normalize inputs
            mean = X_obs.mean(dim=(1, 2), keepdim=True)
            std = X_obs.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs = (X_obs - mean) / std
            
            # Predict
            mean_pred, std_pred = model(X_obs, target, int_time, int_value)
            pred = mean_pred.item()
            
            # Get ground truth
            truth = X_int_test[i, intervention_times_test[i], targets_test[i]].item()
            
            predictions.append(pred)
            ground_truths.append(truth)
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae = np.mean(np.abs(predictions - ground_truths))
    
    return rmse, mae


def run_dataset_size_ablation():
    """Evaluate dataset size ablation."""
    
    print("=" * 80)
    print("Dataset Size Ablation Evaluation")
    print("=" * 80)
    
    # Models to evaluate
    models = {
        '1K': {
            'path': 'checkpoints/simple_causal_pfn.pt',
            'dataset': 'data/causal_time_prior_1k.pt',
            'train_size': 900,  # 90% of 1K
            'params': '57K',
            'train_time': '7s',
        },
        '10K': {
            'path': 'checkpoints/causal_pfn_10k_stable.pt',
            'dataset': 'data/causal_time_prior_10k.pt',
            'train_size': 9000,  # 90% of 10K
            'params': '1.67M',
            'train_time': '10 min',
        },
        '100K': {
            'path': 'checkpoints/causal_pfn_100k_stable.pt',
            'dataset': 'data/causal_time_prior_100k.pt',
            'train_size': 90000,  # 90% of 100K
            'params': '1.67M',
            'train_time': '2.8 hrs',
        },
    }
    
    results = {}
    
    # Evaluate each model on its own test set
    for name, config in models.items():
        print(f"\n{name} model:")
        print(f"  Loading from: {config['path']}")
        
        if not os.path.exists(config['path']):
            print(f"  Model not found! Skipping...")
            continue
        
        # Evaluate on 1000 test samples from the model's own dataset
        rmse, mae = evaluate_model(
            config['path'],
            config['dataset'],
            n_test=1000,
            device='cpu',
        )
        
        results[name] = {
            'rmse': rmse,
            'mae': mae,
            'train_size': config['train_size'],
            'params': config['params'],
            'train_time': config['train_time'],
        }
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("Dataset Size Scaling Results")
    print("=" * 80)
    print(f"{'Dataset':<10} {'Train Size':<12} {'Params':<10} {'Train Time':<12} {'RMSE':<10} {'MAE':<10}")
    print("-" * 80)
    
    for name in ['1K', '10K', '100K']:
        if name in results:
            r = results[name]
            print(f"{name:<10} {r['train_size']:<12} {r['params']:<10} {r['train_time']:<12} "
                  f"{r['rmse']:<10.4f} {r['mae']:<10.4f}")
    
    print("=" * 80)
    
    # Save results
    with open('results/ablation_dataset_size.txt', 'w') as f:
        f.write("Dataset Size Scaling Ablation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Dataset':<10} {'Train Size':<12} {'Params':<10} {'Train Time':<12} {'RMSE':<10} {'MAE':<10}\n")
        f.write("-" * 80 + "\n")
        for name in ['1K', '10K', '100K']:
            if name in results:
                r = results[name]
                f.write(f"{name:<10} {r['train_size']:<12} {r['params']:<10} {r['train_time']:<12} "
                       f"{r['rmse']:<10.4f} {r['mae']:<10.4f}\n")
    
    print("\nResults saved to: results/ablation_dataset_size.txt")
    
    return results


if __name__ == "__main__":
    run_dataset_size_ablation()