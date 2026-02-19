"""Evaluate downstream effects: Does the model understand causal propagation?

This evaluation properly tests causal understanding by measuring how well models
predict the effect of interventions on DOWNSTREAM variables (not the intervened variable).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict
from scipy.stats import pearsonr

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2
from causal_time_prior.baselines import VARBaseline, MeanBaseline, OracleBaseline


def evaluate_model_downstream(
    model_path: str,
    test_data: Dict,
    device: str = 'cpu',
) -> Dict[str, Dict[str, float]]:
    """Evaluate SimpleCausalPFNV2 on downstream effects.
    
    Returns separate metrics for:
    - Same variable queries (baseline)
    - Downstream queries (tests causal understanding)
    """
    X_obs = test_data['X_obs']
    X_int = test_data['X_int']
    int_targets = test_data['intervention_targets']
    int_times = test_data['intervention_times']
    int_values = test_data['intervention_values']
    query_targets = test_data['query_targets']
    query_times = test_data['query_times']
    is_downstream = test_data['is_downstream']
    
    n_test = X_obs.shape[0]
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    model = SimpleCausalPFNV2(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        max_nodes=config['max_nodes'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    predictions_all = []
    ground_truths_all = []
    is_downstream_list = []
    
    with torch.no_grad():
        for i in range(n_test):
            X_obs_i = X_obs[i:i+1].to(device)
            
            # Normalize
            mean = X_obs_i.mean(dim=(1, 2), keepdim=True)
            std = X_obs_i.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs_norm = (X_obs_i - mean) / std
            
            # Predict
            int_target = int_targets[i:i+1].to(device)
            int_time = int_times[i:i+1].to(device)
            int_value = int_values[i:i+1].to(device)
            query_target = query_targets[i:i+1].to(device)
            query_time = query_times[i:i+1].to(device)
            
            mean_pred, _ = model(X_obs_norm, int_target, int_time, int_value, query_target, query_time)
            pred = mean_pred.item()
            
            # Ground truth
            truth = X_int[i, query_times[i], query_targets[i]].item()
            
            predictions_all.append(pred)
            ground_truths_all.append(truth)
            is_downstream_list.append(is_downstream[i].item())
    
    predictions_all = np.array(predictions_all)
    ground_truths_all = np.array(ground_truths_all)
    is_downstream_arr = np.array(is_downstream_list)
    
    # Split by query type
    downstream_mask = is_downstream_arr
    same_var_mask = ~is_downstream_arr
    
    results = {}
    
    # Overall metrics
    results['overall'] = compute_metrics(predictions_all, ground_truths_all)
    
    # Downstream queries (the key test of causal understanding)
    if downstream_mask.sum() > 0:
        results['downstream'] = compute_metrics(
            predictions_all[downstream_mask],
            ground_truths_all[downstream_mask]
        )
    else:
        results['downstream'] = {'rmse': np.nan, 'mae': np.nan}
    
    # Same variable queries (easier baseline)
    if same_var_mask.sum() > 0:
        results['same_var'] = compute_metrics(
            predictions_all[same_var_mask],
            ground_truths_all[same_var_mask]
        )
    else:
        results['same_var'] = {'rmse': np.nan, 'mae': np.nan}
    
    return results


def evaluate_var_downstream(
    test_data: Dict,
) -> Dict[str, Dict[str, float]]:
    """Evaluate VAR-OLS on downstream effects using proper forward simulation."""
    
    X_obs = test_data['X_obs']
    X_int = test_data['X_int']
    int_targets = test_data['intervention_targets']
    int_times = test_data['intervention_times']
    int_values = test_data['intervention_values']
    query_targets = test_data['query_targets']
    query_times = test_data['query_times']
    is_downstream = test_data['is_downstream']
    
    n_test = X_obs.shape[0]
    
    baseline = VARBaseline(lag=3)
    
    predictions_all = []
    ground_truths_all = []
    is_downstream_list = []
    
    for i in range(n_test):
        X_obs_i = X_obs[i].numpy()
        X_int_i = X_int[i].numpy()
        
        # Fit VAR on observational data
        baseline.fit(X_obs_i)
        
        # Predict using downstream simulation
        pred = baseline.predict_interventional_downstream(
            X_obs_i,
            intervention_target=int_targets[i].item(),
            intervention_time=int_times[i].item(),
            intervention_value=int_values[i].item(),
            query_target=query_targets[i].item(),
            query_time=query_times[i].item(),
        )
        
        # Ground truth
        truth = X_int_i[query_times[i], query_targets[i]]
        
        predictions_all.append(pred)
        ground_truths_all.append(truth)
        is_downstream_list.append(is_downstream[i].item())
    
    predictions_all = np.array(predictions_all)
    ground_truths_all = np.array(ground_truths_all)
    is_downstream_arr = np.array(is_downstream_list)
    
    # Split by query type
    downstream_mask = is_downstream_arr
    same_var_mask = ~is_downstream_arr
    
    results = {}
    results['overall'] = compute_metrics(predictions_all, ground_truths_all)
    
    if downstream_mask.sum() > 0:
        results['downstream'] = compute_metrics(
            predictions_all[downstream_mask],
            ground_truths_all[downstream_mask]
        )
    else:
        results['downstream'] = {'rmse': np.nan, 'mae': np.nan}
    
    if same_var_mask.sum() > 0:
        results['same_var'] = compute_metrics(
            predictions_all[same_var_mask],
            ground_truths_all[same_var_mask]
        )
    else:
        results['same_var'] = {'rmse': np.nan, 'mae': np.nan}
    
    return results


def compute_metrics(predictions: np.ndarray, ground_truths: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae = np.mean(np.abs(predictions - ground_truths))
    
    return {'rmse': rmse, 'mae': mae}


def main(
    model_path: str = 'checkpoints/simple_causal_pfn_v2.pt',
    test_path: str = 'data/test_set_downstream_1k.pt',
):
    """Run downstream effects evaluation."""
    
    print("=" * 80)
    print("Downstream Effects Evaluation")
    print("=" * 80)
    
    # Load test data
    print(f"\n1. Loading test set from {test_path}...")
    
    # Check if test set exists, if not generate it
    if not os.path.exists(test_path):
        print(f"   Test set not found. Generating...")
        from causal_time_prior.generate_dataset_v2 import generate_dataset_v2
        generate_dataset_v2(
            n_scms=1000,
            T=50,
            max_nodes=10,
            seed=999,
            output_path=test_path,
            downstream_prob=0.7,
        )
    
    test_data = torch.load(test_path)
    n_test = test_data['X_obs'].shape[0]
    n_downstream = test_data['is_downstream'].sum().item()
    
    print(f"   Test set size: {n_test} samples")
    print(f"   Downstream queries: {n_downstream}/{n_test} ({100*n_downstream/n_test:.1f}%)")
    
    # Evaluate SimpleCausalPFNV2
    print("\n2. Evaluating SimpleCausalPFNV2...")
    model_results = evaluate_model_downstream(model_path, test_data, device='cpu')
    
    print(f"   Overall RMSE: {model_results['overall']['rmse']:.4f}")
    print(f"   Downstream RMSE: {model_results['downstream']['rmse']:.4f}")
    print(f"   Same-var RMSE: {model_results['same_var']['rmse']:.4f}")
    
    # Evaluate VAR-OLS
    print("\n3. Evaluating VAR-OLS...")
    var_results = evaluate_var_downstream(test_data)
    
    print(f"   Overall RMSE: {var_results['overall']['rmse']:.4f}")
    print(f"   Downstream RMSE: {var_results['downstream']['rmse']:.4f}")
    print(f"   Same-var RMSE: {var_results['same_var']['rmse']:.4f}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("Results: Downstream Effects Evaluation")
    print("=" * 80)
    print(f"{'Method':<30} {'Overall RMSE':<15} {'Downstream RMSE':<18} {'Same-Var RMSE':<15}")
    print("-" * 80)
    print(f"{'SimpleCausalPFNV2':<30} {model_results['overall']['rmse']:<15.4f} "
          f"{model_results['downstream']['rmse']:<18.4f} {model_results['same_var']['rmse']:<15.4f}")
    print(f"{'VAR-OLS':<30} {var_results['overall']['rmse']:<15.4f} "
          f"{var_results['downstream']['rmse']:<18.4f} {var_results['same_var']['rmse']:<15.4f}")
    print("=" * 80)
    
    # Key insight
    pfn_downstream = model_results['downstream']['rmse']
    var_downstream = var_results['downstream']['rmse']
    
    if pfn_downstream < var_downstream:
        improvement = (var_downstream - pfn_downstream) / var_downstream * 100
        print(f"\n✅ SimpleCausalPFNV2 outperforms VAR-OLS on downstream effects by {improvement:.1f}%")
    else:
        decline = (pfn_downstream - var_downstream) / var_downstream * 100
        print(f"\n⚠️ SimpleCausalPFNV2 underperforms VAR-OLS on downstream effects by {decline:.1f}%")
    
    print(f"\nThis measures CAUSAL UNDERSTANDING: predicting how interventions affect other variables.")
    
    # Save results
    results_path = 'results/downstream_evaluation.txt'
    with open(results_path, 'w') as f:
        f.write("Downstream Effects Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test set: {test_path}\n")
        f.write(f"Test samples: {n_test}\n")
        f.write(f"Downstream queries: {n_downstream}/{n_test} ({100*n_downstream/n_test:.1f}%)\n\n")
        f.write(f"{'Method':<30} {'Overall RMSE':<15} {'Downstream RMSE':<18} {'Same-Var RMSE':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'SimpleCausalPFNV2':<30} {model_results['overall']['rmse']:<15.4f} "
                f"{model_results['downstream']['rmse']:<18.4f} {model_results['same_var']['rmse']:<15.4f}\n")
        f.write(f"{'VAR-OLS':<30} {var_results['overall']['rmse']:<15.4f} "
                f"{var_results['downstream']['rmse']:<18.4f} {var_results['same_var']['rmse']:<15.4f}\n")
    
    print(f"\nResults saved to: {results_path}")
    
    return model_results, var_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate downstream effects")
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn_v2.pt')
    parser.add_argument('--test', type=str, default='data/test_set_downstream_1k.pt')
    
    args = parser.parse_args()
    
    main(model_path=args.model, test_path=args.test)