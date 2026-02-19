"""Evaluate with shuffled query type labels as a control experiment.

This tests whether the three-way pattern (intervened/downstream/non-causal) 
is a genuine property of the model's predictions or an artifact of the label assignments.

If the model truly distinguishes query types based on causal structure, 
shuffling the labels should destroy the pattern.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2


def evaluate_shuffled_queries(
    model_path: str,
    test_path: str,
    device: str = 'cpu',
    seed: int = 42,
):
    """Evaluate model on three-way query types with SHUFFLED labels."""
    
    print("=" * 80)
    print("Shuffled Query Evaluation (Control Experiment)")
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
    
    # SHUFFLE query types randomly
    print(f"\n2. Shuffling query type labels...")
    np.random.seed(seed)
    shuffled_query_types = query_types.numpy().copy()
    np.random.shuffle(shuffled_query_types)
    shuffled_query_types = torch.tensor(shuffled_query_types)
    
    print(f"   Original distribution:")
    for qtype in [0, 1, 2]:
        count = (query_types == qtype).sum().item()
        print(f"     - Type {qtype}: {count}/{n_samples} ({100*count/n_samples:.1f}%)")
    
    print(f"   Shuffled distribution:")
    for qtype in [0, 1, 2]:
        count = (shuffled_query_types == qtype).sum().item()
        print(f"     - Type {qtype}: {count}/{n_samples} ({100*count/n_samples:.1f}%)")
    
    # Load model
    print(f"\n3. Loading model from {model_path}...")
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
    print(f"\n4. Running evaluation...")
    
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
    
    # Compute metrics by SHUFFLED query type
    query_types_np = shuffled_query_types.numpy()
    
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
                'pred_gt_ratio': np.mean(preds_qt) / (np.mean(gts_qt) + 1e-8),
            }
    
    # Overall
    rmse_overall = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae_overall = np.mean(np.abs(predictions - ground_truths))
    
    # Print results
    print("\n" + "=" * 80)
    print("Results: Shuffled Query Evaluation")
    print("=" * 80)
    print(f"{'Query Type':<15} {'N':<8} {'RMSE':<12} {'Mean Pred':<12} {'Mean GT':<12} {'Pred/GT':<10}")
    print("-" * 80)
    
    for qtype in [0, 1, 2]:
        if qtype in results:
            r = results[qtype]
            print(f"{r['name']:<15} {r['n']:<8} {r['rmse']:<12.2f} {r['mean_pred']:<12.2f} {r['mean_gt']:<12.2f} {r['pred_gt_ratio']:<10.2f}")
    
    print("-" * 80)
    print(f"{'Overall':<15} {n_samples:<8} {rmse_overall:<12.2f} {np.mean(predictions):<12.2f} {np.mean(ground_truths):<12.2f}")
    print("=" * 80)
    
    # Analysis
    print("\n5. Analysis:")
    print("   If the model learned causal structure:")
    print("     - Shuffled labels should DESTROY the three-way pattern")
    print("     - Pred/GT ratios should be similar across all query types")
    print()
    
    if all(qtype in results for qtype in [0, 1, 2]):
        ratios = [results[qtype]['pred_gt_ratio'] for qtype in [0, 1, 2]]
        ratio_std = np.std(ratios)
        print(f"   Pred/GT ratio std: {ratio_std:.3f}")
        if ratio_std < 0.2:
            print("   ✅ Pattern destroyed! Model genuinely distinguishes query types.")
        else:
            print(f"   ⚠️ Pattern persists with std={ratio_std:.3f}. May indicate distributional artifact.")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shuffled query evaluation")
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn_v2_full_100k.pt')
    parser.add_argument('--test', type=str, default='data/test_set_threeway_1k.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    evaluate_shuffled_queries(args.model, args.test, args.device, args.seed)