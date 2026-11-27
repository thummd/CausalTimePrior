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

"""Find a concrete causal prediction example for the paper.

Finds a non-causal test case where:
- PFN correctly predicts small/zero causal effect
- VAR-OLS predicts a larger (incorrect) effect due to spurious correlation
"""

import sys
import os
import importlib.util

import torch
import numpy as np

# Load modules directly from file paths to avoid __init__.py chain
_dir = os.path.dirname(os.path.abspath(__file__))

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pfn_mod = _load_module('simple_causal_pfn_v2', os.path.join(_dir, 'simple_causal_pfn_v2.py'))
SimpleCausalPFNV2 = _pfn_mod.SimpleCausalPFNV2

_baselines_mod = _load_module('baselines', os.path.join(_dir, 'baselines.py'))
VARBaseline = _baselines_mod.VARBaseline

_checks_mod = _load_module('data_checks', os.path.join(_dir, 'data_checks.py'))
assert_dataset_target_distribution = _checks_mod.assert_dataset_target_distribution


def find_example(
    model_path: str = 'checkpoints/simple_causal_pfn_v2_full_100k.pt',
    test_path: str = 'data/test_set_threeway_1k.pt',
    device: str = 'cpu',
):
    """Search the non-causal test queries for examples where the PFN beats VAR-OLS.

    The paper shows one such example: the queried variable correlates with the
    intervened variable in the observational data but does not depend on it.

    Args:
        model_path: Checkpoint written by ``train_v2.py``.
        test_path: Three-way test set written by ``generate_dataset_threeway.py``.
        device: Torch device string for the model.

    Returns:
        List of dictionaries, one per example in which the PFN has the smaller error
        and the observational correlation exceeds 0.3 in magnitude. The list is
        sorted by the error margin over VAR-OLS, largest first.

    Raises:
        ValueError: If the step-zero check finds zero-filled series, contexts without
            variance, or non-finite values in the test set. No example is searched in
            that case.
    """
    # Load dataset
    dataset = torch.load(test_path, map_location=device)
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    intervention_targets = dataset['intervention_targets']
    intervention_times = dataset['intervention_times']
    intervention_values = dataset['intervention_values']
    query_targets = dataset['query_targets']
    query_times = dataset['query_times']
    query_types = dataset['query_types']
    num_vars = dataset['num_vars']

    assert_dataset_target_distribution(dataset, name="test set")

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

    # Find non-causal samples
    noncausal_mask = query_types == 2
    noncausal_indices = noncausal_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"Found {len(noncausal_indices)} non-causal samples")

    results = []

    for idx in noncausal_indices:
        n_vars = num_vars[idx].item()
        x_obs = X_obs[idx, :, :n_vars].numpy()  # (T, n_vars)
        x_int = X_int[idx, :, :n_vars].numpy()
        int_target = intervention_targets[idx].item()
        int_time = intervention_times[idx].item()
        int_value = intervention_values[idx].item()
        q_target = query_targets[idx].item()
        q_time = query_times[idx].item()

        if int_target >= n_vars or q_target >= n_vars:
            continue

        # Ground truth
        gt = x_int[q_time, q_target]
        obs_baseline = x_obs[q_time, q_target]

        # True causal effect (should be near zero for non-causal)
        true_effect = gt - obs_baseline

        # Correlation between intervention target and query target in obs data
        corr = np.corrcoef(x_obs[:, int_target], x_obs[:, q_target])[0, 1]

        # PFN prediction
        with torch.no_grad():
            X_obs_i = X_obs[idx:idx+1].to(device)
            mean = X_obs_i.mean(dim=(1, 2), keepdim=True)
            std = X_obs_i.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs_norm = (X_obs_i - mean) / std

            int_target_t = intervention_targets[idx:idx+1].to(device)
            int_time_t = intervention_times[idx:idx+1].to(device)
            int_value_t = intervention_values[idx:idx+1].to(device)
            q_target_t = query_targets[idx:idx+1].to(device)
            q_time_t = query_times[idx:idx+1].to(device)

            pred, _ = model(X_obs_norm, int_target_t, int_time_t, int_value_t, q_target_t, q_time_t)
            pred_pfn = pred.item() * std.squeeze().item() + mean.squeeze().item()

        # VAR-OLS prediction
        var = VARBaseline(lag=3)
        # Use full padded obs for VAR fitting
        x_obs_full = X_obs[idx].numpy()  # (T, max_nodes)
        var.fit(x_obs_full)
        pred_var = var.predict_interventional_downstream(
            x_obs_full, int_target, int_time, int_value, q_target, q_time
        )

        pfn_error = abs(pred_pfn - gt)
        var_error = abs(pred_var - gt)

        results.append({
            'idx': idx,
            'n_vars': n_vars,
            'int_target': int_target,
            'q_target': q_target,
            'int_time': int_time,
            'q_time': q_time,
            'int_value': int_value,
            'corr': corr,
            'obs_baseline': obs_baseline,
            'gt': gt,
            'true_effect': true_effect,
            'pred_pfn': pred_pfn,
            'pred_var': pred_var,
            'pfn_error': pfn_error,
            'var_error': var_error,
            'pfn_better': pfn_error < var_error,
        })

    # Sort by: PFN is better than VAR, high correlation, large difference
    good_examples = [r for r in results if r['pfn_better'] and abs(r['corr']) > 0.3]
    good_examples.sort(key=lambda r: r['var_error'] - r['pfn_error'], reverse=True)

    print(f"\nFound {len(good_examples)} examples where PFN beats VAR-OLS with |corr| > 0.3")
    print(f"Total non-causal evaluated: {len(results)}")
    print(f"PFN better than VAR: {sum(1 for r in results if r['pfn_better'])}/{len(results)}")

    print("\n" + "=" * 100)
    print("Top 10 examples (PFN beats VAR-OLS with highest margin + high correlation)")
    print("=" * 100)
    for i, r in enumerate(good_examples[:10]):
        print(f"\n--- Example {i+1} (sample idx={r['idx']}) ---")
        print(f"  SCM: {r['n_vars']} vars, intervene var {r['int_target']} at t={r['int_time']} (value={r['int_value']:.2f})")
        print(f"  Query: var {r['q_target']} at t={r['q_time']}")
        print(f"  Obs correlation(int_target, q_target): {r['corr']:.3f}")
        print(f"  Obs baseline (no intervention):     {r['obs_baseline']:.4f}")
        print(f"  Ground truth (under intervention):  {r['gt']:.4f}")
        print(f"  True causal effect:                 {r['true_effect']:.4f}")
        print(f"  PFN prediction:                     {r['pred_pfn']:.4f}  (error: {r['pfn_error']:.4f})")
        print(f"  VAR-OLS prediction:                 {r['pred_var']:.4f}  (error: {r['var_error']:.4f})")
        print(f"  VAR-OLS error / PFN error:          {r['var_error']/max(r['pfn_error'], 1e-8):.1f}x")

    return good_examples


if __name__ == "__main__":
    find_example()
