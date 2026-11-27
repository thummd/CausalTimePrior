# Copyright 2026 Dennis Thumm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare SimpleCausalPFNV2 with baselines on the three-way test set.

Every method answers the same queries, on the same samples, and is scored with one
definition of each metric. The queries are those of the three-way evaluation: each
sample has one intervention and one queried cell, which is the intervened variable, a
downstream variable, or a variable that the intervention cannot reach.

The baselines are fitted on the observational context of each sample:

- VAR-OLS fits a vector autoregression and simulates it forward from the intervention.
- PCMCI+ discovers a lagged graph and regresses the queried variable on its parents.
- The mean baseline answers with the observational mean of the queried variable.
- The constant-zero baseline shows what a predictor without any information scores.

This script replaces the earlier baseline comparison, which used ``evaluate_fair.py``.
That protocol queried the intervened cell itself, where the answer of a hard
intervention is the intervention value by definition.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ConstantInputWarning, pearsonr

from causal_time_prior.baselines import VARBaseline
from causal_time_prior.data_checks import assert_dataset_target_distribution
from causal_time_prior.pcmci_baseline import PCMCIBaseline, TIGRAMITE_AVAILABLE
from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2


# The simulator clips every value to [-1000, 1000] (utils.clip_values), so no ground
# truth lies outside this range. A simulated VAR can leave it by orders of magnitude.
DATA_RANGE = 1000.0

QUERY_TYPE_NAMES = {0: 'Intervened', 1: 'Downstream', 2: 'Non-causal'}


def effect_metrics(
    predictions: np.ndarray,
    truths: np.ndarray,
    observed: np.ndarray,
) -> Dict[str, float]:
    """Score one method on the queried cells.

    The effect of an intervention on a cell is its interventional value minus its
    observational value. A prediction is turned into a predicted effect in the same
    way. The direction accuracy asks whether the predicted effect has the sign of the
    true effect. The effect size correlation is the Pearson correlation of the two.

    Args:
        predictions: Predicted interventional value per sample, shape (n,).
        truths: True interventional value per sample, shape (n,).
        observed: Observational value of the same cell per sample, shape (n,).

    Returns:
        Dictionary with ``rmse``, ``mae``, ``effect_direction_acc``, and
        ``effect_size_corr``. The direction accuracy is computed over the samples with
        a nonzero true effect, because a zero effect has no direction. It is NaN when
        no sample has an effect. The correlation is NaN when either side is constant.

    Raises:
        Nothing.
    """
    predicted_effects = predictions - observed
    true_effects = truths - observed

    has_effect = true_effects != 0
    if has_effect.any():
        direction = float(np.mean(np.sign(predicted_effects[has_effect]) == np.sign(true_effects[has_effect])))
    else:
        direction = float('nan')

    if np.std(predicted_effects) > 0 and np.std(true_effects) > 0:
        correlation = float(pearsonr(predicted_effects, true_effects)[0])
    else:
        correlation = float('nan')

    return {
        'rmse': float(np.sqrt(np.mean((predictions - truths) ** 2))),
        'mae': float(np.mean(np.abs(predictions - truths))),
        'effect_direction_acc': direction,
        'effect_size_corr': correlation,
    }


def predict_pfn(model_path: str, dataset: Dict[str, Any], device: str = 'cpu') -> np.ndarray:
    """Predict every queried cell with a SimpleCausalPFNV2 checkpoint.

    Args:
        model_path: Checkpoint written by ``train_v2.py``.
        dataset: Three-way test set as returned by ``torch.load``, on the CPU.
        device: Torch device string for the model.

    Returns:
        Predictions in raw units, shape (n,).

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
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

    X_obs = dataset['X_obs']
    n_samples = X_obs.shape[0]
    predictions = np.zeros(n_samples)

    with torch.no_grad():
        for i in range(n_samples):
            X_obs_i = X_obs[i:i+1].to(device)
            mean = X_obs_i.mean(dim=(1, 2), keepdim=True)
            std = X_obs_i.std(dim=(1, 2), keepdim=True) + 1e-8

            pred, _ = model(
                (X_obs_i - mean) / std,
                dataset['intervention_targets'][i:i+1].to(device),
                dataset['intervention_times'][i:i+1].to(device),
                dataset['intervention_values'][i:i+1].to(device),
                dataset['query_targets'][i:i+1].to(device),
                dataset['query_times'][i:i+1].to(device),
            )
            # The model answers in the normalised units of its context. Every other
            # method and the ground truth are in raw units.
            predictions[i] = pred.item() * std.item() + mean.item()

    return predictions


def predict_var_ols(dataset: Dict[str, Any], lag: int = 3) -> np.ndarray:
    """Predict every queried cell with a VAR fitted on the observational context.

    The VAR is fitted per sample and simulated forward from the intervention to the
    queried time step.

    Args:
        dataset: Three-way test set as returned by ``torch.load``, on the CPU.
        lag: Number of lags of the VAR.

    Returns:
        Predictions in raw units, shape (n,). They are not clipped here.

    Raises:
        Nothing.
    """
    X_obs = dataset['X_obs'].numpy().astype(np.float64)
    n_samples = X_obs.shape[0]
    predictions = np.zeros(n_samples)

    for i in range(n_samples):
        baseline = VARBaseline(lag=lag)
        baseline.fit(X_obs[i])
        predictions[i] = baseline.predict_interventional_downstream(
            X_obs[i],
            int(dataset['intervention_targets'][i]),
            int(dataset['intervention_times'][i]),
            float(dataset['intervention_values'][i]),
            int(dataset['query_targets'][i]),
            int(dataset['query_times'][i]),
        )

    return predictions


def predict_pcmci(
    dataset: Dict[str, Any],
    tau_max: int = 2,
    alpha_level: float = 0.01,
) -> Tuple[np.ndarray, int]:
    """Predict every queried cell with the PCMCI+ baseline.

    Args:
        dataset: Three-way test set as returned by ``torch.load``, on the CPU.
        tau_max: Maximum lag for PCMCI+.
        alpha_level: Significance level for PCMCI+.

    Returns:
        Tuple ``(predictions, n_fallbacks)``. ``predictions`` has shape (n,) and is in
        raw units. ``n_fallbacks`` counts the samples on which the baseline raised and
        the observational value of the cell was used instead.

    Raises:
        ImportError: If the tigramite package is not installed.
    """
    X_obs = dataset['X_obs'].numpy().astype(np.float64)
    n_samples = X_obs.shape[0]
    predictions = np.zeros(n_samples)
    n_fallbacks = 0
    baseline = PCMCIBaseline(tau_max=tau_max, alpha_level=alpha_level)

    with warnings.catch_warnings():
        # Unused variables are padded with zeros, and a constant column makes every
        # partial correlation test on it warn. The warning carries no information here.
        warnings.simplefilter('ignore', category=ConstantInputWarning)
        for i in range(n_samples):
            query_target = int(dataset['query_targets'][i])
            query_time = int(dataset['query_times'][i])
            try:
                predictions[i] = baseline.predict_interventional(
                    X_obs[i],
                    target=query_target,
                    query_target=query_target,
                    query_time=query_time,
                    intervention_target=int(dataset['intervention_targets'][i]),
                    intervention_time=int(dataset['intervention_times'][i]),
                    intervention_value=float(dataset['intervention_values'][i]),
                )
            except Exception:
                # Same fallback as evaluate_pcmci.py, so that both scripts agree.
                predictions[i] = X_obs[i, query_time, query_target]
                n_fallbacks += 1

    return predictions, n_fallbacks


def evaluate_baselines_threeway(
    model_path: str,
    test_path: str,
    device: str = 'cpu',
    output_path: Optional[str] = 'results/baseline_comparison_threeway.txt',
    skip_pcmci: bool = False,
    tau_max: int = 2,
    alpha_level: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """Score the PFN and all baselines on the queries of the three-way test set.

    Args:
        model_path: Checkpoint written by ``train_v2.py``.
        test_path: Three-way test set written by ``generate_dataset_threeway.py``.
        device: Torch device string for the model. The baselines run on the CPU.
        output_path: File that receives the tables. None writes no file.
        skip_pcmci: Leave out the PCMCI+ row. Use it when tigramite is not installed.
        tau_max: Maximum lag for PCMCI+.
        alpha_level: Significance level for PCMCI+.

    Returns:
        Dictionary that maps each method to its RMSE, MAE, effect direction accuracy,
        and effect size correlation.

    Raises:
        ValueError: If the step-zero check finds zero-filled series, contexts without
            variance, or non-finite values in the test set. Nothing is evaluated in
            that case.
        ImportError: If tigramite is not installed and ``skip_pcmci`` is False.
    """
    print("=" * 80)
    print("Baseline Comparison on the Three-Way Test Set")
    print("=" * 80)

    print(f"\n1. Loading test set from {test_path}...")
    # The baselines work on numpy arrays, so the test set stays on the CPU. Only the
    # model and its inputs go to the requested device.
    dataset = torch.load(test_path, map_location='cpu')
    n_samples = dataset['X_obs'].shape[0]
    print(f"   Test set size: {n_samples}")
    assert_dataset_target_distribution(dataset, name="test set")

    if not skip_pcmci and not TIGRAMITE_AVAILABLE:
        raise ImportError(
            "tigramite is required for the PCMCI+ row. Install it with "
            "'pip install tigramite', or pass --skip_pcmci to leave the row out."
        )

    rows = torch.arange(n_samples)
    query_types = dataset['query_types'].numpy()
    truths = dataset['X_int'][rows, dataset['query_times'], dataset['query_targets']].numpy().astype(np.float64)
    observed = dataset['X_obs'][rows, dataset['query_times'], dataset['query_targets']].numpy().astype(np.float64)
    X_obs_np = dataset['X_obs'].numpy().astype(np.float64)

    print(f"\n2. Predicting with every method...")
    predictions: Dict[str, np.ndarray] = {}

    print("   - Oracle (ground truth, lower bound of every error)")
    predictions['Oracle'] = truths.copy()

    print(f"   - SimpleCausalPFN ({model_path})")
    predictions['SimpleCausalPFN'] = predict_pfn(model_path, dataset, device)

    n_fallbacks = 0
    if not skip_pcmci:
        print(f"   - PCMCI+ (tau_max={tau_max}, alpha={alpha_level}), one causal discovery per sample")
        predictions['PCMCI+'], n_fallbacks = predict_pcmci(dataset, tau_max, alpha_level)

    print("   - VAR-OLS (fitted per sample, simulated forward)")
    var_raw = predict_var_ols(dataset)
    # A VAR fitted on 50 time steps is often unstable, and its simulation can leave the
    # range of the data by orders of magnitude. Clipping to that range is what any user
    # of the baseline would do, and it keeps a few samples from deciding the row.
    predictions['VAR-OLS'] = np.clip(var_raw, -DATA_RANGE, DATA_RANGE)
    n_clipped = int((np.abs(var_raw) > DATA_RANGE).sum())

    print("   - Mean prediction (observational mean of the queried variable)")
    query_targets_np = dataset['query_targets'].numpy()
    predictions['Mean prediction'] = X_obs_np[np.arange(n_samples), :, query_targets_np].mean(axis=1)

    print("   - Constant zero")
    predictions['Constant zero'] = np.zeros(n_samples)

    results = {name: effect_metrics(values, truths, observed) for name, values in predictions.items()}
    var_raw_rmse = effect_metrics(var_raw, truths, observed)['rmse']

    # The oracle leads the table as the reference. The methods follow by RMSE.
    order = ['Oracle'] + sorted((name for name in results if name != 'Oracle'), key=lambda name: results[name]['rmse'])

    lines = []
    lines.append(f"{'Method':<18} {'RMSE':<10} {'MAE':<10} {'Effect Dir Acc':<16} {'Effect Size Corr':<16}")
    lines.append("-" * 72)
    for name in order:
        r = results[name]
        lines.append(f"{name:<18} {r['rmse']:<10.2f} {r['mae']:<10.2f} "
                     f"{r['effect_direction_acc']:<16.1%} {r['effect_size_corr']:<16.3f}")

    type_lines = []
    type_lines.append(f"{'Method':<18} " + " ".join(f"{QUERY_TYPE_NAMES[q] + ' (' + str(int((query_types == q).sum())) + ')':<18}" for q in QUERY_TYPE_NAMES))
    type_lines.append("-" * 72)
    for name in order:
        cells = []
        for q in QUERY_TYPE_NAMES:
            mask = query_types == q
            cells.append(float(np.sqrt(np.mean((predictions[name][mask] - truths[mask]) ** 2))) if mask.any() else float('nan'))
        type_lines.append(f"{name:<18} " + " ".join(f"{c:<18.2f}" for c in cells))

    notes = [
        f"Effect direction accuracy is computed over the {int((truths - observed != 0).sum())} samples with a nonzero true effect.",
        f"VAR-OLS: {n_clipped} of {n_samples} predictions left the data range and were clipped to "
        f"[-{DATA_RANGE:.0f}, {DATA_RANGE:.0f}]. Without clipping the RMSE is {var_raw_rmse:.2f}.",
    ]
    if skip_pcmci:
        notes.append("PCMCI+ was skipped.")
    else:
        notes.append(f"PCMCI+: {n_fallbacks} of {n_samples} samples fell back to the observational value.")

    print("\n" + "=" * 80)
    print("Results: all methods on the same queries")
    print("=" * 80)
    print("\n".join(lines))
    print("=" * 80)
    print("\nRMSE by query type:")
    print("\n".join(type_lines))
    print()
    for note in notes:
        print(note)

    if output_path is not None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("Baseline Comparison on the Three-Way Test Set\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test set: {test_path}\n")
            f.write(f"Test samples: {n_samples}\n")
            f.write(f"Checkpoint: {model_path}\n\n")
            f.write("\n".join(lines) + "\n\n")
            f.write("RMSE by query type:\n")
            f.write("\n".join(type_lines) + "\n\n")
            f.write("\n".join(notes) + "\n")
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline comparison on the three-way test set")
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn_v2_full_100k.pt')
    parser.add_argument('--test', type=str, default='data/test_set_threeway_1k.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='results/baseline_comparison_threeway.txt')
    parser.add_argument('--skip_pcmci', action='store_true', help='Leave out the PCMCI+ row (no tigramite needed)')
    parser.add_argument('--tau_max', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.01)

    args = parser.parse_args()

    evaluate_baselines_threeway(
        model_path=args.model,
        test_path=args.test,
        device=args.device,
        output_path=args.output,
        skip_pcmci=args.skip_pcmci,
        tau_max=args.tau_max,
        alpha_level=args.alpha,
    )
