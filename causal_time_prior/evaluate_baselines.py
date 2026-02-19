"""Evaluate all methods (SimpleCausalPFN + baselines) and compare."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict

from causal_time_prior.simple_causal_pfn import SimpleCausalPFN
from causal_time_prior.baselines import VARBaseline, MeanBaseline, OracleBaseline, evaluate_baseline


def evaluate_simple_causal_pfn(
    model,
    X_obs_list,
    X_int_list,
    targets_list,
    intervention_times_list,
    intervention_values_list,
    device='cpu',
    normalize=True,
):
    """Evaluate SimpleCausalPFN.
    
    Returns
    -------
    Tuple[float, float]
        (RMSE, MAE)
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in range(len(X_obs_list)):
            X_obs = torch.tensor(X_obs_list[i], dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, N)
            target = torch.tensor([targets_list[i]], dtype=torch.long).to(device)
            int_time = torch.tensor([intervention_times_list[i]], dtype=torch.long).to(device)
            int_value = torch.tensor([intervention_values_list[i]], dtype=torch.float32).to(device)
            
            # Normalize inputs (same as training)
            if normalize:
                mean_val = X_obs.mean(dim=(1, 2), keepdim=True)
                std_val = X_obs.std(dim=(1, 2), keepdim=True) + 1e-8
                X_obs = (X_obs - mean_val) / std_val
            
            # Predict
            mean, std = model(X_obs, target, int_time, int_value)
            pred = mean.item()
            
            # Get ground truth
            truth = X_int_list[i][intervention_times_list[i], targets_list[i]]
            
            predictions.append(pred)
            ground_truths.append(truth)
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae = np.mean(np.abs(predictions - ground_truths))
    
    return rmse, mae


def main(
    data_path: str = "data/causal_time_prior_10k.pt",
    model_path: str = "checkpoints/simple_causal_pfn.pt",
    n_test: int = 1000,
    device: str = "cpu",
):
    """Run evaluation of all methods.
    
    Parameters
    ----------
    data_path : str
        Path to dataset.
    model_path : str
        Path to trained SimpleCausalPFN checkpoint.
    n_test : int
        Number of test samples to evaluate on.
    device : str
        Device for model inference.
    """
    print("=" * 80)
    print("Baseline Evaluation for CausalTimePrior")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading dataset from {data_path}...")
    dataset = torch.load(data_path)
    
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    targets = dataset['targets']
    intervention_times = dataset['intervention_times']
    intervention_values = dataset['intervention_values']
    
    print(f"   Dataset shape: {X_obs.shape}")
    
    # Use last n_test samples for evaluation
    n_total = X_obs.shape[0]
    n_test = min(n_test, n_total)
    
    X_obs_test = X_obs[-n_test:]
    X_int_test = X_int[-n_test:]
    targets_test = targets[-n_test:]
    intervention_times_test = intervention_times[-n_test:]
    intervention_values_test = intervention_values[-n_test:]
    
    print(f"   Using last {n_test} samples for evaluation")
    
    # Convert to lists of numpy arrays
    X_obs_list = [X_obs_test[i].numpy() for i in range(n_test)]
    X_int_list = [X_int_test[i].numpy() for i in range(n_test)]
    targets_list = targets_test.numpy().tolist()
    intervention_times_list = intervention_times_test.numpy().tolist()
    intervention_values_list = intervention_values_test.numpy().tolist()
    
    # Evaluate baselines
    print(f"\n2. Evaluating baselines on {n_test} test samples...")
    
    results = {}
    
    # Oracle (upper bound)
    print("   - Oracle (upper bound)...")
    oracle = OracleBaseline()
    rmse, mae = evaluate_baseline(
        oracle, X_obs_list, X_int_list, targets_list,
        intervention_times_list, intervention_values_list, is_oracle=True
    )
    results['Oracle'] = {'RMSE': rmse, 'MAE': mae}
    print(f"     RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Mean baseline
    print("   - Mean baseline...")
    mean_baseline = MeanBaseline()
    rmse, mae = evaluate_baseline(
        mean_baseline, X_obs_list, X_int_list, targets_list,
        intervention_times_list, intervention_values_list
    )
    results['Mean'] = {'RMSE': rmse, 'MAE': mae}
    print(f"     RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # VAR-OLS
    print("   - VAR-OLS...")
    var_baseline = VARBaseline(lag=3)
    rmse, mae = evaluate_baseline(
        var_baseline, X_obs_list, X_int_list, targets_list,
        intervention_times_list, intervention_values_list
    )
    results['VAR-OLS'] = {'RMSE': rmse, 'MAE': mae}
    print(f"     RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # SimpleCausalPFN
    if os.path.exists(model_path):
        print(f"   - SimpleCausalPFN (loading from {model_path})...")
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
        
        rmse, mae = evaluate_simple_causal_pfn(
            model, X_obs_list, X_int_list, targets_list,
            intervention_times_list, intervention_values_list, device, normalize=True
        )
        results['SimpleCausalPFN'] = {'RMSE': rmse, 'MAE': mae}
        print(f"     RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    else:
        print(f"   - SimpleCausalPFN: Model not found at {model_path}, skipping...")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("Comparison Table (sorted by RMSE)")
    print("=" * 80)
    print(f"{'Method':<20} {'RMSE':<15} {'MAE':<15}")
    print("-" * 50)
    
    # Sort by RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])
    
    for method, metrics in sorted_results:
        print(f"{method:<20} {metrics['RMSE']:<15.4f} {metrics['MAE']:<15.4f}")
    
    print("=" * 80)
    
    # Save results
    results_path = "results/baseline_comparison.txt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("Baseline Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Test samples: {n_test}\n\n")
        f.write(f"{'Method':<20} {'RMSE':<15} {'MAE':<15}\n")
        f.write("-" * 50 + "\n")
        for method, metrics in sorted_results:
            f.write(f"{method:<20} {metrics['RMSE']:<15.4f} {metrics['MAE']:<15.4f}\n")
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate baselines")
    parser.add_argument('--data', type=str, default='data/causal_time_prior_10k.pt',
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn.pt',
                        help='Path to SimpleCausalPFN checkpoint')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data,
        model_path=args.model,
        n_test=args.n_test,
        device=args.device,
    )