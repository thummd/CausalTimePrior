"""Shuffled-intervention control experiment.

Tests whether the model's Pred/GT gradient across query types reflects
learned causal structure or distributional artifacts. We randomly permute
intervention_targets across test samples, breaking the true correspondence
between observational data and intervention info. If the model truly
learned causal structure, the Pred/GT pattern should flatten.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2


def evaluate_shuffled(
    model_path: str,
    test_path: str,
    device: str = 'cpu',
    seed: int = 42,
):
    """Evaluate model with shuffled intervention targets as control."""

    print("=" * 80)
    print("Shuffled-Intervention Control Experiment")
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

    # Shuffle intervention targets
    print(f"\n2. Shuffling intervention targets (seed={seed})...")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    shuffled_targets = intervention_targets[perm]
    shuffled_values = intervention_values[perm]

    print(f"   Original target[0:5]: {intervention_targets[:5].tolist()}")
    print(f"   Shuffled target[0:5]: {shuffled_targets[:5].tolist()}")

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

    # Run evaluation with REAL and SHUFFLED targets
    def run_eval(int_targets_tensor, label):
        predictions = []
        ground_truths = []

        with torch.no_grad():
            for i in range(n_samples):
                X_obs_i = X_obs[i:i+1].to(device)

                mean = X_obs_i.mean(dim=(1, 2), keepdim=True)
                std = X_obs_i.std(dim=(1, 2), keepdim=True) + 1e-8
                X_obs_norm = (X_obs_i - mean) / std

                int_target = int_targets_tensor[i:i+1].to(device)
                int_time = intervention_times[i:i+1].to(device)
                int_value = intervention_values[i:i+1].to(device)
                q_target = query_targets[i:i+1].to(device)
                q_time = query_times[i:i+1].to(device)

                pred, _ = model(X_obs_norm, int_target, int_time, int_value, q_target, q_time)
                pred_denorm = pred.item() * std.squeeze().item() + mean.squeeze().item()
                gt = X_int[i, q_time.item(), q_target.item()].item()

                predictions.append(pred_denorm)
                ground_truths.append(gt)

        return np.array(predictions), np.array(ground_truths)

    print(f"\n4. Evaluating with REAL intervention targets...")
    preds_real, gts_real = run_eval(intervention_targets, "Real")

    print(f"   Evaluating with SHUFFLED intervention targets...")
    preds_shuffled, gts_shuffled = run_eval(shuffled_targets, "Shuffled")

    # Compute metrics by query type
    query_types_np = query_types.numpy()
    qtype_names = {0: 'Intervened', 1: 'Downstream', 2: 'Non-causal'}

    print("\n" + "=" * 80)
    print("Results: Real vs Shuffled Intervention Targets")
    print("=" * 80)

    print(f"\n{'Query Type':<15} {'N':<6} | {'Real Pred/GT':<14} {'Real RMSE':<12} | {'Shuf Pred/GT':<14} {'Shuf RMSE':<12}")
    print("-" * 85)

    for qtype in [0, 1, 2]:
        mask = query_types_np == qtype
        n = mask.sum()
        if n == 0:
            continue

        # Real
        mean_pred_r = np.mean(preds_real[mask])
        mean_gt_r = np.mean(gts_real[mask])
        ratio_r = mean_pred_r / mean_gt_r if abs(mean_gt_r) > 1e-8 else float('nan')
        rmse_r = np.sqrt(np.mean((preds_real[mask] - gts_real[mask]) ** 2))

        # Shuffled
        mean_pred_s = np.mean(preds_shuffled[mask])
        mean_gt_s = np.mean(gts_shuffled[mask])
        ratio_s = mean_pred_s / mean_gt_s if abs(mean_gt_s) > 1e-8 else float('nan')
        rmse_s = np.sqrt(np.mean((preds_shuffled[mask] - gts_shuffled[mask]) ** 2))

        print(f"{qtype_names[qtype]:<15} {n:<6} | {ratio_r:<14.2f} {rmse_r:<12.2f} | {ratio_s:<14.2f} {rmse_s:<12.2f}")

    # Overall
    rmse_r_all = np.sqrt(np.mean((preds_real - gts_real) ** 2))
    rmse_s_all = np.sqrt(np.mean((preds_shuffled - gts_shuffled) ** 2))
    ratio_r_all = np.mean(preds_real) / np.mean(gts_real)
    ratio_s_all = np.mean(preds_shuffled) / np.mean(gts_shuffled)

    print("-" * 85)
    print(f"{'Overall':<15} {n_samples:<6} | {ratio_r_all:<14.2f} {rmse_r_all:<12.2f} | {ratio_s_all:<14.2f} {rmse_s_all:<12.2f}")
    print("=" * 80)

    # Analysis
    print("\n5. Analysis:")

    real_ratios = []
    shuffled_ratios = []
    for qtype in [0, 1, 2]:
        mask = query_types_np == qtype
        if mask.sum() > 0:
            r_r = np.mean(preds_real[mask]) / (np.mean(gts_real[mask]) + 1e-8)
            r_s = np.mean(preds_shuffled[mask]) / (np.mean(gts_shuffled[mask]) + 1e-8)
            real_ratios.append(r_r)
            shuffled_ratios.append(r_s)

    real_spread = max(real_ratios) - min(real_ratios)
    shuffled_spread = max(shuffled_ratios) - min(shuffled_ratios)

    print(f"   Real Pred/GT spread (max-min): {real_spread:.2f}")
    print(f"   Shuffled Pred/GT spread (max-min): {shuffled_spread:.2f}")

    if shuffled_spread < real_spread * 0.5:
        print("   --> Shuffling flattens the pattern, confirming model uses causal structure")
    else:
        print("   --> Pattern persists under shuffling — may indicate distributional artifacts")

    return {
        'real_ratios': real_ratios,
        'shuffled_ratios': shuffled_ratios,
        'real_spread': real_spread,
        'shuffled_spread': shuffled_spread,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shuffled-intervention control experiment")
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn_v2_full_100k.pt')
    parser.add_argument('--test', type=str, default='data/test_set_threeway_1k.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    evaluate_shuffled(args.model, args.test, args.device, args.seed)
