"""Fair evaluation: All models + baselines on the SAME test set with causal correctness metrics.

This addresses the methodological issue where models were evaluated on different test sets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict, Tuple
from scipy.stats import pearsonr

from causal_time_prior.simple_causal_pfn import SimpleCausalPFN
from causal_time_prior.baselines import VARBaseline, MeanBaseline, OracleBaseline


def evaluate_model_fair(
    model_path: str,
    test_data: Dict,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Evaluate a SimpleCausalPFN model with causal correctness metrics.
    
    Returns
    -------
    Dict[str, float]
        Metrics: rmse, mae, effect_direction_acc, effect_size_corr
    """
    X_obs = test_data['X_obs']
    X_int = test_data['X_int']
    targets = test_data['targets']
    intervention_times = test_data['intervention_times']
    intervention_values = test_data['intervention_values']
    
    n_test = X_obs.shape[0]
    
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
    
    predictions = []
    ground_truths = []
    observational_values = []
    
    with torch.no_grad():
        for i in range(n_test):
            X_obs_i = X_obs[i:i+1].to(device)
            target = targets[i:i+1].to(device)
            int_time = intervention_times[i:i+1].to(device)
            int_value = intervention_values[i:i+1].to(device)
            
            # Normalize inputs
            mean = X_obs_i.mean(dim=(1, 2), keepdim=True)
            std = X_obs_i.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs_norm = (X_obs_i - mean) / std
            
            # Predict
            mean_pred, std_pred = model(X_obs_norm, target, int_time, int_value)
            pred = mean_pred.item()
            
            # Get ground truth
            truth = X_int[i, intervention_times[i], targets[i]].item()
            
            # Get observational value (what would happen without intervention)
            obs_value = X_obs[i, intervention_times[i], targets[i]].item()
            
            predictions.append(pred)
            ground_truths.append(truth)
            observational_values.append(obs_value)
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    observational_values = np.array(observational_values)
    
    # Standard metrics
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae = np.mean(np.abs(predictions - ground_truths))
    
    # Causal correctness metrics
    # 1. Effect direction accuracy: Does sign(Y_pred - Y_obs) match sign(Y_int - Y_obs)?
    pred_effects = predictions - observational_values
    true_effects = ground_truths - observational_values
    
    # Only consider non-zero true effects (where intervention had an effect)
    nonzero_mask = np.abs(true_effects) > 0.01
    if nonzero_mask.sum() > 0:
        direction_correct = np.sign(pred_effects[nonzero_mask]) == np.sign(true_effects[nonzero_mask])
        effect_direction_acc = direction_correct.mean()
    else:
        effect_direction_acc = np.nan
    
    # 2. Effect size correlation: Correlation between predicted and true effect sizes
    if len(true_effects) > 1 and np.std(true_effects) > 0:
        effect_size_corr, _ = pearsonr(pred_effects, true_effects)
    else:
        effect_size_corr = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'effect_direction_acc': effect_direction_acc,
        'effect_size_corr': effect_size_corr,
    }


def evaluate_baseline_fair(
    baseline_name: str,
    test_data: Dict,
) -> Dict[str, float]:
    """Evaluate a baseline method with causal correctness metrics."""
    
    X_obs = test_data['X_obs']
    X_int = test_data['X_int']
    targets = test_data['targets']
    intervention_times = test_data['intervention_times']
    intervention_values = test_data['intervention_values']
    
    n_test = X_obs.shape[0]
    
    # Initialize baseline
    if baseline_name == 'VAR-OLS':
        baseline = VARBaseline(lag=3)
    elif baseline_name == 'Mean':
        baseline = MeanBaseline()
    elif baseline_name == 'Oracle':
        baseline = OracleBaseline()
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    predictions = []
    ground_truths = []
    observational_values = []
    
    for i in range(n_test):
        X_obs_i = X_obs[i].numpy()
        X_int_i = X_int[i].numpy()
        target = targets[i].item()
        int_time = intervention_times[i].item()
        int_value = intervention_values[i].item()
        
        # Predict
        if baseline_name == 'Oracle':
            pred = baseline.predict_interventional(X_int_i, target, int_time)
        else:
            baseline.fit(X_obs_i)
            pred = baseline.predict_interventional(X_obs_i, target, int_time, int_value)
        
        # Get ground truth
        truth = X_int_i[int_time, target]
        
        # Get observational value
        obs_value = X_obs_i[int_time, target]
        
        predictions.append(pred)
        ground_truths.append(truth)
        observational_values.append(obs_value)
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    observational_values = np.array(observational_values)
    
    # Standard metrics
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae = np.mean(np.abs(predictions - ground_truths))
    
    # Causal correctness metrics
    pred_effects = predictions - observational_values
    true_effects = ground_truths - observational_values
    
    nonzero_mask = np.abs(true_effects) > 0.01
    if nonzero_mask.sum() > 0:
        direction_correct = np.sign(pred_effects[nonzero_mask]) == np.sign(true_effects[nonzero_mask])
        effect_direction_acc = direction_correct.mean()
    else:
        effect_direction_acc = np.nan
    
    if len(true_effects) > 1 and np.std(true_effects) > 0:
        effect_size_corr, _ = pearsonr(pred_effects, true_effects)
    else:
        effect_size_corr = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'effect_direction_acc': effect_direction_acc,
        'effect_size_corr': effect_size_corr,
    }


def main(test_path: str = 'data/test_set_1k.pt'):
    """Run fair evaluation on all models and baselines."""
    
    print("=" * 80)
    print("Fair Evaluation: All Models on Same Test Set")
    print("=" * 80)
    
    # Load test data
    print(f"\n1. Loading test set from {test_path}...")
    test_data = torch.load(test_path)
    n_test = test_data['X_obs'].shape[0]
    print(f"   Test set size: {n_test} samples")
    
    # Models to evaluate
    models = {
        'SimpleCausalPFN (1K)': 'checkpoints/simple_causal_pfn.pt',
        'SimpleCausalPFN (10K)': 'checkpoints/causal_pfn_10k_stable.pt',
        'SimpleCausalPFN (100K)': 'checkpoints/causal_pfn_100k_stable.pt',
    }
    
    baselines = ['VAR-OLS', 'Mean', 'Oracle']
    
    results = {}
    
    # Evaluate models
    print("\n2. Evaluating SimpleCausalPFN models...")
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"   {name}: Model not found, skipping...")
            continue
        
        print(f"   {name}...")
        metrics = evaluate_model_fair(path, test_data, device='cpu')
        results[name] = metrics
        print(f"      RMSE: {metrics['rmse']:.4f}, Effect Dir Acc: {metrics['effect_direction_acc']:.1%}")
    
    # Evaluate baselines
    print("\n3. Evaluating baselines...")
    for baseline_name in baselines:
        print(f"   {baseline_name}...")
        metrics = evaluate_baseline_fair(baseline_name, test_data)
        results[baseline_name] = metrics
        print(f"      RMSE: {metrics['rmse']:.4f}, Effect Dir Acc: {metrics['effect_direction_acc']:.1%}")
    
    # Print results table
    print("\n" + "=" * 80)
    print("Fair Evaluation Results (All on Same Test Set)")
    print("=" * 80)
    print(f"{'Method':<30} {'RMSE':<12} {'MAE':<12} {'Effect Dir Acc':<18} {'Effect Size Corr':<18}")
    print("-" * 90)
    
    # Sort by RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
    
    for method, metrics in sorted_results:
        print(f"{method:<30} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} "
              f"{metrics['effect_direction_acc']:<18.1%} {metrics['effect_size_corr']:<18.3f}")
    
    print("=" * 80)
    
    # Save results
    results_path = 'results/fair_evaluation.txt'
    with open(results_path, 'w') as f:
        f.write("Fair Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test set: {test_path}\n")
        f.write(f"Test samples: {n_test}\n\n")
        f.write(f"{'Method':<30} {'RMSE':<12} {'MAE':<12} {'Effect Dir Acc':<18} {'Effect Size Corr':<18}\n")
        f.write("-" * 90 + "\n")
        for method, metrics in sorted_results:
            f.write(f"{method:<30} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} "
                   f"{metrics['effect_direction_acc']:<18.1%} {metrics['effect_size_corr']:<18.3f}\n")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fair evaluation of all models")
    parser.add_argument('--test', type=str, default='data/test_set_1k.pt',
                        help='Path to test set')
    
    args = parser.parse_args()
    
    main(test_path=args.test)