"""Evaluate model using Normalized MSE (NMSE), comparable to Do-PFN.

NMSE = MSE(pred, true) / Var(true)
- NMSE = 1.0 means the model performs like predicting the mean
- NMSE < 1.0 means the model is better than predicting the mean
- NMSE > 1.0 means the model is worse than predicting the mean
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2


def evaluate_with_nmse(
    model_path: str,
    test_path: str,
    device: str = 'cpu',
):
    """Evaluate model with NMSE alongside standard metrics."""

    print("=" * 80)
    print("NMSE Evaluation (Do-PFN-style)")
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
    query_types = dataset['query_types']

    n_samples = X_obs.shape[0]
    print(f"   Test set size: {n_samples}")

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
    qtype_names = {0: 'Intervened', 1: 'Downstream', 2: 'Non-causal'}

    print("\n" + "=" * 90)
    print("Results: NMSE Evaluation")
    print("=" * 90)
    print(f"{'Query Type':<15} {'N':<6} {'RMSE':<10} {'NMSE':<10} {'MAE':<10} {'Pred/GT':<10} {'Var(GT)':<12}")
    print("-" * 90)

    results = {}
    for qtype in [0, 1, 2]:
        mask = query_types_np == qtype
        n = mask.sum()
        if n == 0:
            continue

        preds_qt = predictions[mask]
        gts_qt = ground_truths[mask]

        mse = np.mean((preds_qt - gts_qt) ** 2)
        var_gt = np.var(gts_qt)
        nmse = mse / var_gt if var_gt > 1e-8 else float('nan')
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_qt - gts_qt))
        pred_gt = np.mean(preds_qt) / np.mean(gts_qt) if abs(np.mean(gts_qt)) > 1e-8 else float('nan')

        results[qtype] = {
            'name': qtype_names[qtype],
            'n': n,
            'rmse': rmse,
            'nmse': nmse,
            'mae': mae,
            'pred_gt': pred_gt,
            'var_gt': var_gt,
        }

        print(f"{qtype_names[qtype]:<15} {n:<6} {rmse:<10.2f} {nmse:<10.4f} {mae:<10.2f} {pred_gt:<10.2f} {var_gt:<12.2f}")

    # Overall
    mse_all = np.mean((predictions - ground_truths) ** 2)
    var_gt_all = np.var(ground_truths)
    nmse_all = mse_all / var_gt_all if var_gt_all > 1e-8 else float('nan')
    rmse_all = np.sqrt(mse_all)
    mae_all = np.mean(np.abs(predictions - ground_truths))
    pred_gt_all = np.mean(predictions) / np.mean(ground_truths)

    print("-" * 90)
    print(f"{'Overall':<15} {n_samples:<6} {rmse_all:<10.2f} {nmse_all:<10.4f} {mae_all:<10.2f} {pred_gt_all:<10.2f} {var_gt_all:<12.2f}")
    print("=" * 90)

    # Interpretation
    print("\n4. NMSE Interpretation:")
    print(f"   NMSE = {nmse_all:.4f} (1.0 = mean baseline, <1.0 = better than mean)")
    if nmse_all < 1.0:
        print(f"   Model outperforms mean prediction by {(1 - nmse_all) * 100:.1f}%")
    else:
        print(f"   Model underperforms mean prediction by {(nmse_all - 1) * 100:.1f}%")

    return results, {
        'nmse_overall': nmse_all,
        'rmse_overall': rmse_all,
        'mae_overall': mae_all,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NMSE evaluation")
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn_v2_full_100k.pt')
    parser.add_argument('--test', type=str, default='data/test_set_threeway_1k.pt')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    evaluate_with_nmse(args.model, args.test, args.device)
