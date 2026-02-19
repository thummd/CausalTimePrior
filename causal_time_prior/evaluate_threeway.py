"""Three-way evaluation: intervened / downstream / non-causal.

Tests whether the model correctly predicts different magnitudes of effects
based on causal graph structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2


def evaluate_threeway(
    model_path: str,
    test_path: str,
    device: str = 'cpu',
):
    """Evaluate model on three-way query types."""
    
    print("=" * 80)
    print("Three-Way Evaluation: Intervened / Downstream / Non-Causal")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading test set from {test_path}...")
    dataset = torch.load(test_path, map_location=device)
    
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    intervention_targets = dataset['intervention_targets']
    intervention_times = dataset['intervention_times']
    intervention_values = dataset['intervention_values']
    query_targets = dataset['query_targets']
    query_times = dataset['query_times']
    query_types = dataset['query_types']  # 0=intervened, 1=downstream, 2=non-causal
    
    n_samples = X_obs.shape[0]
    print(f"   Test set size: {n_samples}")
    
    # Count query types
    n_intervened = (query_types == 0).sum().item()
    n_downstream = (query_types == 1).sum().item()
    n_noncausal = (query_types == 2).sum().item()
    
    print(f"   Query type distribution:")
    print(f"     - Intervened: {n_intervened}/{n_samples} ({100*n_intervened/n_samples:.1f}%)")
    print(f"     - Downstream: {n_downstream}/{n_samples} ({100*n_downstream/n_samples:.1f}%)")
    print(f"     - Non-causal: {n_noncausal}/{n_samples} ({100*n_noncausal/n_samples:.1f}%)")
    
    # Load model
    print(f"\n2. Loading model from {model_path}...")
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
    
    # Evaluate
    print(f"\n3. Running evaluation...")
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in range(n_samples):
            X_obs_i = X_obs[i:i+1].to(device)
            
            # Normalize
            mean = X_obs_i.mean(dim=(1, 2), keepdim=True)
            std = X_obs_i.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs_norm = (X_obs_i - mean) / std
            
            # Query
            int_target = intervention_targets[i:i+1].to(device)
            int_time = intervention_times[i:i+1].to(device)
            int_value = intervention_values[i:i+1].to(device)
            q_target = query_targets[i:i+1].to(device)
            q_time = query_times[i:i+1].to(device)
            
            # Predict
            pred, _ = model(X_obs_norm, int_target, int_time, int_value, q_target, q_time)
            
            # Denormalize
            pred_denorm = pred.item() * std.squeeze().item() + mean.squeeze().item()
            
            # Ground truth
            gt = X_int[i, q_time.item(), q_target.item()].item()
            
            predictions.append(pred_denorm)
            ground_truths.append(gt)
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Compute metrics by query type
    query_types_np = query_types.numpy()
    
    results = {}
    for qtype, qname in [(0, 'Intervened'), (1, 'Downstream'), (2, 'Non-causal')]:
        mask = query_types_np == qtype
        if mask.sum() > 0:
            preds_qt = predictions[mask]
            gts_qt = ground_truths[mask]
            
            rmse = np.sqrt(np.mean((preds_qt - gts_qt) ** 2))
            mae = np.mean(np.abs(preds_qt - gts_qt))
            
            results[qtype] = {
                'name': qname,
                'n': mask.sum(),
                'rmse': rmse,
                'mae': mae,
                'mean_pred': np.mean(preds_qt),
                'mean_gt': np.mean(gts_qt),
            }
    
    # Overall
    rmse_overall = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae_overall = np.mean(np.abs(predictions - ground_truths))
    
    # Print results
    print("\n" + "=" * 80)
    print("Results: Three-Way Evaluation")
    print("=" * 80)
    print(f"{'Query Type':<15} {'N':<8} {'RMSE':<12} {'MAE':<12} {'Mean Pred':<12} {'Mean GT':<12}")
    print("-" * 80)
    
    for qtype in [0, 1, 2]:
        if qtype in results:
            r = results[qtype]
            print(f"{r['name']:<15} {r['n']:<8} {r['rmse']:<12.2f} {r['mae']:<12.2f} {r['mean_pred']:<12.2f} {r['mean_gt']:<12.2f}")
    
    print("-" * 80)
    print(f"{'Overall':<15} {n_samples:<8} {rmse_overall:<12.2f} {mae_overall:<12.2f} {np.mean(predictions):<12.2f} {np.mean(ground_truths):<12.2f}")
    print("=" * 80)
    
    # Analysis
    print("\n4. Analysis:")
    if 0 in results and 1 in results:
        print(f"   Intervened RMSE: {results[0]['rmse']:.2f}")
        print(f"   Downstream RMSE: {results[1]['rmse']:.2f}")
        ratio = results[1]['rmse'] / results[0]['rmse']
        print(f"   Ratio (Downstream/Intervened): {ratio:.2f}x")
        
        if ratio < 1.5:
            print("   ✅ Model shows similar performance on both query types")
        else:
            print(f"   ⚠️ Downstream RMSE is {ratio:.2f}x higher than intervened")
    
    if 2 in results:
        print(f"\n   Non-causal RMSE: {results[2]['rmse']:.2f}")
        if 0 in results:
            ratio_nc = results[2]['rmse'] / results[0]['rmse']
            print(f"   Ratio (Non-causal/Intervened): {ratio_nc:.2f}x")
            
            if ratio_nc < 1.5:
                print("   ✅ Model correctly identifies non-causal queries")
            else:
                print(f"   ⚠️ Non-causal RMSE is {ratio_nc:.2f}x higher")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Three-way evaluation")
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn_v2_mixed.pt')
    parser.add_argument('--test', type=str, default='data/test_set_threeway_1k.pt')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    evaluate_threeway(args.model, args.test, args.device)