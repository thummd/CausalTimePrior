"""Evaluate PCMCI+ baseline on threeway test set.

WARNING: This is SLOW! PCMCI+ runs per-sample causal discovery (~1-5 sec each).
Recommend running on a subset (e.g., 200 samples) for computational feasibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import time

from causal_time_prior.pcmci_baseline import PCMCIBaseline

def evaluate_pcmci_baseline(
    test_path: str = 'data/test_set_threeway_1k.pt',
    n_samples: int = 200,
    tau_max: int = 2,
    alpha_level: float = 0.01,
):
    """Evaluate PCMCI+ baseline on threeway test set.
    
    Parameters
    ----------
    test_path : str
        Path to test dataset.
    n_samples : int
        Number of samples to evaluate (default 200 for tractability).
    tau_max : int
        Maximum lag for PCMCI+.
    alpha_level : float
        Significance level for PCMCI+.
    """
    
    print("=" * 80)
    print(f"PCMCI+ Baseline Evaluation (n={n_samples})")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading test set from {test_path}...")
    dataset = torch.load(test_path, map_location='cpu')
    
    X_obs = dataset['X_obs'].numpy()
    X_int = dataset['X_int'].numpy()
    intervention_targets = dataset['intervention_targets'].numpy()
    intervention_times = dataset['intervention_times'].numpy()
    intervention_values = dataset['intervention_values'].numpy()
    query_targets = dataset['query_targets'].numpy()
    query_times = dataset['query_times'].numpy()
    query_types = dataset['query_types'].numpy()
    
    # Subsample
    total_samples = X_obs.shape[0]
    if n_samples > total_samples:
        n_samples = total_samples
    
    indices = np.random.RandomState(42).choice(total_samples, n_samples, replace=False)
    
    X_obs = X_obs[indices]
    X_int = X_int[indices]
    intervention_targets = intervention_targets[indices]
    intervention_times = intervention_times[indices]
    intervention_values = intervention_values[indices]
    query_targets = query_targets[indices]
    query_times = query_times[indices]
    query_types = query_types[indices]
    
    print(f"   Evaluating on {n_samples}/{total_samples} samples (subsampled)")
    
    # Count query types
    n_intervened = (query_types == 0).sum()
    n_downstream = (query_types == 1).sum()
    n_noncausal = (query_types == 2).sum()
    
    print(f"   Query type distribution:")
    print(f"     - Intervened: {n_intervened}/{n_samples} ({100*n_intervened/n_samples:.1f}%)")
    print(f"     - Downstream: {n_downstream}/{n_samples} ({100*n_downstream/n_samples:.1f}%)")
    print(f"     - Non-causal: {n_noncausal}/{n_samples} ({100*n_noncausal/n_samples:.1f}%)")
    
    # Initialize baseline
    print(f"\n2. Initializing PCMCI+ baseline (tau_max={tau_max}, alpha={alpha_level})...")
    baseline = PCMCIBaseline(tau_max=tau_max, alpha_level=alpha_level)
    
    # Evaluate
    print(f"\n3. Running PCMCI+ predictions (this will take ~{n_samples * 2 / 60:.1f} min)...")
    
    predictions = []
    ground_truths = []
    
    start_time = time.time()
    
    for i in range(n_samples):
        try:
            pred = baseline.predict_interventional(
                X_obs[i],
                target=query_targets[i],  # deprecated
                query_target=query_targets[i],
                query_time=query_times[i],
                intervention_target=intervention_targets[i],
                intervention_time=intervention_times[i],
                intervention_value=intervention_values[i],
            )
        except Exception as e:
            print(f"   Error on sample {i}: {e}")
            # Fallback to observational value
            pred = X_obs[i, query_times[i], query_targets[i]]
        
        gt = X_int[i, query_times[i], query_targets[i]]
        
        predictions.append(pred)
        ground_truths.append(gt)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"   Progress: {i + 1}/{n_samples} ({100 * (i + 1) / n_samples:.1f}%) - "
                  f"Rate: {rate:.2f} samples/sec - ETA: {eta:.1f}s")
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    elapsed_total = time.time() - start_time
    
    # Compute metrics by query type
    results = {}
    for qtype, qname in [(0, 'Intervened'), (1, 'Downstream'), (2, 'Non-causal')]:
        mask = query_types == qtype
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
    print("Results: PCMCI+ Baseline")
    print("=" * 80)
    print(f"{'Query Type':<15} {'N':<8} {'RMSE':<12} {'MAE':<12} {'Mean Pred':<12} {'Mean GT':<12}")
    print("-" * 80)
    
    for qtype in [0, 1, 2]:
        if qtype in results:
            r = results[qtype]
            print(f"{r['name']:<15} {r['n']:<8} {r['rmse']:<12.2f} {r['mae']:<12.2f} "
                  f"{r['mean_pred']:<12.2f} {r['mean_gt']:<12.2f}")
    
    print("-" * 80)
    print(f"{'Overall':<15} {n_samples:<8} {rmse_overall:<12.2f} {mae_overall:<12.2f} "
          f"{np.mean(predictions):<12.2f} {np.mean(ground_truths):<12.2f}")
    print("=" * 80)
    
    print(f"\n4. Performance summary:")
    print(f"   Total time: {elapsed_total:.1f}s ({elapsed_total / n_samples:.2f}s per sample)")
    print(f"   Overall RMSE: {rmse_overall:.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='data/test_set_threeway_1k.pt')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--tau_max', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.01)
    
    args = parser.parse_args()
    
    evaluate_pcmci_baseline(
        test_path=args.test,
        n_samples=args.n_samples,
        tau_max=args.tau_max,
        alpha_level=args.alpha,
    )