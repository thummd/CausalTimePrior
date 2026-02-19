"""Simple 3-variable chain benchmark: A→B→C with known ground truth.

This provides a sanity check with known interventional effects:
- Structure: A → B → C (linear chain)
- Mechanisms: B = 0.8*A + noise, C = 0.6*B + noise
- Intervention: do(A_t = 5.0)
- Known effects: B ≈ 4.0, C ≈ 2.4 (propagates through chain)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Tuple

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2
from causal_time_prior.baselines import VARBaseline


class ThreeVariableChain:
    """A→B→C linear chain with known coefficients."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the chain with fixed coefficients:
        - B = 0.8*A + noise(0, 0.1)
        - C = 0.6*B + noise(0, 0.1)
        """
        self.a_to_b = 0.8
        self.b_to_c = 0.6
        self.noise_std = 0.1
        self.rng = np.random.default_rng(seed)
    
    def sample_observational(self, T: int = 50) -> np.ndarray:
        """Sample observational data."""
        X = np.zeros((T, 3))
        
        for t in range(T):
            # A is exogenous (root node)
            X[t, 0] = self.rng.normal(0, 1)
            
            # B = 0.8*A + noise
            if t > 0:
                X[t, 1] = self.a_to_b * X[t-1, 0] + self.rng.normal(0, self.noise_std)
            else:
                X[t, 1] = self.rng.normal(0, 0.1)
            
            # C = 0.6*B + noise
            if t > 0:
                X[t, 2] = self.b_to_c * X[t-1, 1] + self.rng.normal(0, self.noise_std)
            else:
                X[t, 2] = self.rng.normal(0, 0.1)
        
        return X
    
    def sample_interventional(
        self, 
        T: int = 50, 
        intervention_time: int = 25,
        intervention_value: float = 5.0
    ) -> np.ndarray:
        """Sample interventional data with do(A_t = value)."""
        X = np.zeros((T, 3))
        
        for t in range(T):
            # Intervene on A at specified time
            if t >= intervention_time:
                X[t, 0] = intervention_value
            else:
                X[t, 0] = self.rng.normal(0, 1)
            
            # B = 0.8*A + noise (uses previous A)
            if t > 0:
                X[t, 1] = self.a_to_b * X[t-1, 0] + self.rng.normal(0, self.noise_std)
            else:
                X[t, 1] = self.rng.normal(0, 0.1)
            
            # C = 0.6*B + noise (uses previous B)
            if t > 0:
                X[t, 2] = self.b_to_c * X[t-1, 1] + self.rng.normal(0, self.noise_std)
            else:
                X[t, 2] = self.rng.normal(0, 0.1)
        
        return X
    
    def expected_effects(
        self,
        intervention_value: float = 5.0,
        steps_after: int = 1
    ) -> dict:
        """Compute expected effects of intervention after k steps.
        
        After do(A=5):
        - Step 1: B ≈ 0.8*5 = 4.0
        - Step 2: C ≈ 0.6*4 = 2.4
        - Step 3+: Effects propagate but decay with noise
        """
        if steps_after == 1:
            return {'A': intervention_value, 'B': self.a_to_b * intervention_value, 'C': 0.0}
        elif steps_after == 2:
            B_effect = self.a_to_b * intervention_value
            C_effect = self.b_to_c * B_effect
            return {'A': intervention_value, 'B': B_effect, 'C': C_effect}
        else:
            # After 2+ steps, effects continue propagating
            B_effect = self.a_to_b * intervention_value
            C_effect = self.b_to_c * B_effect
            return {'A': intervention_value, 'B': B_effect, 'C': C_effect}


def evaluate_on_chain(
    model_path: str,
    n_test: int = 100,
    device: str = 'cpu'
):
    """Evaluate SimpleCausalPFNV2 on the 3-variable chain benchmark."""
    
    print("=" * 80)
    print("3-Variable Chain Benchmark: A→B→C")
    print("=" * 80)
    
    # Create chain
    chain = ThreeVariableChain(seed=999)
    
    # Expected effects
    print("\n1. Ground Truth (Known Effects):")
    print("   Structure: A → B → C")
    print("   Mechanisms: B = 0.8*A + noise, C = 0.6*B + noise")
    print("   Intervention: do(A_25 = 5.0)")
    print("\n   Expected effects (1 step after):")
    expected_1 = chain.expected_effects(5.0, 1)
    print(f"      A: {expected_1['A']:.2f}, B: {expected_1['B']:.2f}, C: {expected_1['C']:.2f}")
    print("   Expected effects (2 steps after):")
    expected_2 = chain.expected_effects(5.0, 2)
    print(f"      A: {expected_2['A']:.2f}, B: {expected_2['B']:.2f}, C: {expected_2['C']:.2f}")
    
    # Load model
    print(f"\n2. Loading SimpleCausalPFNV2 from {model_path}...")
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
    print(f"\n3. Evaluating on {n_test} samples...")
    
    predictions_A = []
    predictions_B = []
    predictions_C = []
    ground_truths_A = []
    ground_truths_B = []
    ground_truths_C = []
    
    with torch.no_grad():
        for i in range(n_test):
            # Generate data
            X_obs = chain.sample_observational(T=50)
            X_int = chain.sample_interventional(T=50, intervention_time=25, intervention_value=5.0)
            
            # Pad to 10 variables
            X_obs_padded = np.zeros((50, 10))
            X_obs_padded[:, :3] = X_obs
            X_int_padded = np.zeros((50, 10))
            X_int_padded[:, :3] = X_int
            
            X_obs_t = torch.tensor(X_obs_padded, dtype=torch.float32).unsqueeze(0)
            
            # Normalize
            mean = X_obs_t.mean(dim=(1, 2), keepdim=True)
            std = X_obs_t.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs_norm = (X_obs_t - mean) / std
            
            # Store mean/std for denormalization
            mean_val = mean.squeeze().item()
            std_val = std.squeeze().item()
            
            # Query: Predict A, B, C at time 27 (2 steps after intervention at 25)
            int_target = torch.tensor([0], dtype=torch.long)  # Intervene on A
            int_time = torch.tensor([25], dtype=torch.long)
            int_value = torch.tensor([5.0], dtype=torch.float32)
            query_time = torch.tensor([27], dtype=torch.long)  # 2 steps after
            
            # Predict A (denormalize: pred = pred_norm * std + mean)
            query_target_A = torch.tensor([0], dtype=torch.long)
            pred_A_norm, _ = model(X_obs_norm, int_target, int_time, int_value, query_target_A, query_time)
            pred_A = pred_A_norm.item() * std_val + mean_val
            predictions_A.append(pred_A)
            ground_truths_A.append(X_int[27, 0])
            
            # Predict B (downstream, denormalize)
            query_target_B = torch.tensor([1], dtype=torch.long)
            pred_B_norm, _ = model(X_obs_norm, int_target, int_time, int_value, query_target_B, query_time)
            pred_B = pred_B_norm.item() * std_val + mean_val
            predictions_B.append(pred_B)
            ground_truths_B.append(X_int[27, 1])
            
            # Predict C (further downstream, denormalize)
            query_target_C = torch.tensor([2], dtype=torch.long)
            pred_C_norm, _ = model(X_obs_norm, int_target, int_time, int_value, query_target_C, query_time)
            pred_C = pred_C_norm.item() * std_val + mean_val
            predictions_C.append(pred_C)
            ground_truths_C.append(X_int[27, 2])
    
    predictions_A = np.array(predictions_A)
    predictions_B = np.array(predictions_B)
    predictions_C = np.array(predictions_C)
    ground_truths_A = np.array(ground_truths_A)
    ground_truths_B = np.array(ground_truths_B)
    ground_truths_C = np.array(ground_truths_C)
    
    # Compute metrics
    rmse_A = np.sqrt(np.mean((predictions_A - ground_truths_A)**2))
    rmse_B = np.sqrt(np.mean((predictions_B - ground_truths_B)**2))
    rmse_C = np.sqrt(np.mean((predictions_C - ground_truths_C)**2))
    
    mean_pred_A = np.mean(predictions_A)
    mean_pred_B = np.mean(predictions_B)
    mean_pred_C = np.mean(predictions_C)
    
    mean_true_A = np.mean(ground_truths_A)
    mean_true_B = np.mean(ground_truths_B)
    mean_true_C = np.mean(ground_truths_C)
    
    print("\n" + "=" * 80)
    print("Results: SimpleCausalPFNV2 on 3-Variable Chain")
    print("=" * 80)
    print(f"{'Variable':<15} {'Expected':<15} {'Mean Pred':<15} {'Mean True':<15} {'RMSE':<10}")
    print("-" * 80)
    print(f"{'A (intervened)':<15} {expected_2['A']:<15.2f} {mean_pred_A:<15.2f} {mean_true_A:<15.2f} {rmse_A:<10.2f}")
    print(f"{'B (1 step down)':<15} {expected_2['B']:<15.2f} {mean_pred_B:<15.2f} {mean_true_B:<15.2f} {rmse_B:<10.2f}")
    print(f"{'C (2 steps down)':<15} {expected_2['C']:<15.2f} {mean_pred_C:<15.2f} {mean_true_C:<15.2f} {rmse_C:<10.2f}")
    print("=" * 80)
    
    # Assessment
    print("\n4. Assessment:")
    if abs(mean_pred_A - expected_2['A']) < 1.0:
        print(f"   ✅ A prediction close to expected ({mean_pred_A:.2f} vs {expected_2['A']:.2f})")
    else:
        print(f"   ⚠️ A prediction deviates from expected ({mean_pred_A:.2f} vs {expected_2['A']:.2f})")
    
    if abs(mean_pred_B - expected_2['B']) < 1.5:
        print(f"   ✅ B prediction reasonably close to expected ({mean_pred_B:.2f} vs {expected_2['B']:.2f})")
    else:
        print(f"   ⚠️ B prediction deviates from expected ({mean_pred_B:.2f} vs {expected_2['B']:.2f})")
    
    if abs(mean_pred_C - expected_2['C']) < 2.0:
        print(f"   ✅ C prediction reasonably close to expected ({mean_pred_C:.2f} vs {expected_2['C']:.2f})")
    else:
        print(f"   ⚠️ C prediction deviates from expected ({mean_pred_C:.2f} vs {expected_2['C']:.2f})")
    
    return {
        'A': {'rmse': rmse_A, 'mean_pred': mean_pred_A, 'mean_true': mean_true_A},
        'B': {'rmse': rmse_B, 'mean_pred': mean_pred_B, 'mean_true': mean_true_B},
        'C': {'rmse': rmse_C, 'mean_pred': mean_pred_C, 'mean_true': mean_true_C},
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate on 3-variable chain benchmark")
    parser.add_argument('--model', type=str, default='checkpoints/simple_causal_pfn_v2.pt')
    parser.add_argument('--n_test', type=int, default=100)
    
    args = parser.parse_args()
    
    evaluate_on_chain(args.model, args.n_test)