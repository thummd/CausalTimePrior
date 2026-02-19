"""Generate dataset with three-way query labels for proper causal evaluation.

Query types:
- Intervened: query_target == intervention_target
- Downstream: query_target is reachable from intervention_target in the causal graph
- Non-causal: query_target is NOT reachable (no causal path exists)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import time
import networkx as nx
from typing import List, Tuple

from causal_time_prior.prior import CausalTimePrior
from causal_time_prior.interventions import InterventionSpec


def pad_to_max_nodes(X: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Pad time series to have max_nodes variables."""
    T, N = X.shape
    if N < max_nodes:
        padding = torch.zeros(T, max_nodes - N, dtype=X.dtype, device=X.device)
        X_padded = torch.cat([X, padding], dim=1)
    else:
        X_padded = X[:, :max_nodes]
    return X_padded


def compute_reachable_descendants(scm, intervention_target: int) -> set:
    """Compute variables reachable from intervention_target in the temporal DAG.
    
    Uses BFS on the unrolled temporal graph (instantaneous + lagged edges).
    """
    dag = scm.dag
    N = len(dag.topo_order)
    
    # Build combined graph (G_0 + all lagged edges)
    combined = nx.DiGraph()
    combined.add_nodes_from(range(N))
    
    # Add instantaneous edges
    for u, v in dag.G_0.edges():
        u_idx = dag.topo_order.index(u)
        v_idx = dag.topo_order.index(v)
        combined.add_edge(u_idx, v_idx)
    
    # Add lagged edges (G_k[i,j] = 1 means i->j at lag k)
    for k, G_k in enumerate(dag.G_lags):
        for i in range(N):
            for j in range(N):
                if G_k[i, j] > 0:
                    combined.add_edge(i, j)
    
    # Compute reachable descendants via BFS
    try:
        descendants = nx.descendants(combined, intervention_target)
    except nx.NetworkXError:
        descendants = set()
    
    return descendants


def convert_to_threeway_format(data, max_nodes, seed=42):
    """Convert a list of (X_obs, X_int, intervention, scm) tuples to threeway dataset format.

    Parameters
    ----------
    data : list of tuples
        Each tuple is (X_obs, X_int, intervention, scm) where X_obs/X_int are (T, N) tensors.
    max_nodes : int
        Pad all series to this many variables.
    seed : int
        Random seed for query type sampling.

    Returns
    -------
    dict
        Dataset dictionary with threeway query labels.
    """
    rng = np.random.RandomState(seed)

    X_obs_list = []
    X_int_list = []
    intervention_targets_list = []
    intervention_times_list = []
    intervention_values_list = []
    query_targets_list = []
    query_times_list = []
    query_types_list = []
    intervention_types_list = []
    num_vars_list = []

    for X_obs, X_int, intervention, scm in data:
        T = X_obs.shape[0]
        N = X_obs.shape[1]

        X_obs_list.append(pad_to_max_nodes(X_obs, max_nodes))
        X_int_list.append(pad_to_max_nodes(X_int, max_nodes))

        intervention_target = intervention.targets[0] if len(intervention.targets) > 0 else 0
        int_time = int(np.mean(intervention.times))

        if callable(intervention.values):
            int_value = intervention.values(int_time)
        else:
            int_value = intervention.values

        intervention_targets_list.append(intervention_target)
        intervention_times_list.append(int_time)
        intervention_values_list.append(float(int_value))

        descendants = compute_reachable_descendants(scm, intervention_target)

        query_type = rng.choice(3, p=[0.33, 0.33, 0.34])

        if query_type == 0:
            query_target = intervention_target
        elif query_type == 1 and len(descendants) > 0:
            query_target = rng.choice(list(descendants))
        elif query_type == 2:
            non_causal = [v for v in range(N) if v != intervention_target and v not in descendants]
            if len(non_causal) > 0:
                query_target = rng.choice(non_causal)
            else:
                query_target = intervention_target
                query_type = 0
        else:
            query_target = intervention_target
            query_type = 0

        query_time = min(int_time + rng.randint(1, 6), T - 1)

        query_targets_list.append(query_target)
        query_times_list.append(query_time)
        query_types_list.append(query_type)
        intervention_types_list.append(intervention.intervention_type.value)
        num_vars_list.append(N)

    return {
        'X_obs': torch.stack(X_obs_list),
        'X_int': torch.stack(X_int_list),
        'intervention_targets': torch.tensor(intervention_targets_list, dtype=torch.long),
        'intervention_times': torch.tensor(intervention_times_list, dtype=torch.long),
        'intervention_values': torch.tensor(intervention_values_list, dtype=torch.float32),
        'query_targets': torch.tensor(query_targets_list, dtype=torch.long),
        'query_times': torch.tensor(query_times_list, dtype=torch.long),
        'query_types': torch.tensor(query_types_list, dtype=torch.long),
        'intervention_types': intervention_types_list,
        'num_vars': torch.tensor(num_vars_list, dtype=torch.long),
        'metadata': {
            'n_scms': len(data),
            'max_nodes': max_nodes,
        }
    }


def generate_dataset_threeway(
    n_scms: int,
    T: int,
    max_nodes: int,
    seed: int = 42,
    output_path: str = "data/test_set_threeway_1k.pt",
):
    """Generate dataset with three-way query labels.
    
    Query types:
    - 0: Intervened (query_target == intervention_target)
    - 1: Downstream (query_target reachable from intervention_target)
    - 2: Non-causal (query_target NOT reachable)
    """
    print("=" * 80)
    print(f"Generating Three-Way Evaluation Dataset: {n_scms} SCMs, T={T}")
    print("=" * 80)
    
    # Initialize prior
    print("\n1. Initializing CausalTimePrior...")
    prior = CausalTimePrior(seed=seed, chain_prob=0.15)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Storage
    X_obs_list = []
    X_int_list = []
    intervention_targets_list = []
    intervention_times_list = []
    intervention_values_list = []
    query_targets_list = []
    query_times_list = []
    query_types_list = []  # 0=intervened, 1=downstream, 2=non-causal
    intervention_types_list = []
    num_vars_list = []
    
    # Statistics
    start_time = time.time()
    intervention_type_counts = {}
    diverged_count = 0
    query_type_counts = {0: 0, 1: 0, 2: 0}
    
    print(f"\n2. Generating {n_scms} SCM pairs with three-way queries...")
    for i in range(n_scms):
        X_obs, X_int, intervention, scm = prior.generate_pair(T=T)
        
        # Check for divergence
        if torch.isnan(X_obs).any() or torch.isnan(X_int).any():
            diverged_count += 1
            X_obs, X_int, intervention, scm = prior.generate_pair(T=T)
        
        # Pad to max_nodes
        N = X_obs.shape[1]
        X_obs_padded = pad_to_max_nodes(X_obs, max_nodes)
        X_int_padded = pad_to_max_nodes(X_int, max_nodes)
        
        # Store
        X_obs_list.append(X_obs_padded)
        X_int_list.append(X_int_padded)
        
        # Intervention info
        intervention_target = intervention.targets[0] if len(intervention.targets) > 0 else 0
        int_time = int(np.mean(intervention.times))
        
        # Intervention value
        if callable(intervention.values):
            int_value = intervention.values(int_time)
        else:
            int_value = intervention.values
        
        intervention_targets_list.append(intervention_target)
        intervention_times_list.append(int_time)
        intervention_values_list.append(float(int_value))
        
        # Compute reachable descendants
        descendants = compute_reachable_descendants(scm, intervention_target)
        
        # Sample query variable with balanced distribution
        query_type_probs = [0.33, 0.33, 0.34]  # Balanced: intervened, downstream, non-causal
        query_type = np.random.choice(3, p=query_type_probs)
        
        if query_type == 0:  # Intervened
            query_target = intervention_target
        elif query_type == 1 and len(descendants) > 0:  # Downstream
            query_target = np.random.choice(list(descendants))
        elif query_type == 2:  # Non-causal
            non_causal = [v for v in range(N) if v != intervention_target and v not in descendants]
            if len(non_causal) > 0:
                query_target = np.random.choice(non_causal)
            else:
                # Fallback: use intervened
                query_target = intervention_target
                query_type = 0
        else:
            # Fallback: use intervened
            query_target = intervention_target
            query_type = 0
        
        # Query time: 1-5 steps after intervention
        query_time = min(int_time + np.random.randint(1, 6), T - 1)
        
        query_targets_list.append(query_target)
        query_times_list.append(query_time)
        query_types_list.append(query_type)
        
        intervention_types_list.append(intervention.intervention_type.value)
        num_vars_list.append(N)
        
        # Update statistics
        int_type = intervention.intervention_type.value
        intervention_type_counts[int_type] = intervention_type_counts.get(int_type, 0) + 1
        query_type_counts[query_type] += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_scms - i - 1) / rate
            print(f"   Progress: {i + 1}/{n_scms} ({100 * (i + 1) / n_scms:.1f}%) - "
                  f"Rate: {rate:.1f} SCMs/s - ETA: {eta:.1f}s - "
                  f"Query types: Int={query_type_counts[0]}, Down={query_type_counts[1]}, NonC={query_type_counts[2]}")
    
    elapsed_total = time.time() - start_time
    
    print(f"\n3. Stacking tensors...")
    # Stack into tensors
    X_obs_tensor = torch.stack(X_obs_list)
    X_int_tensor = torch.stack(X_int_list)
    intervention_targets_tensor = torch.tensor(intervention_targets_list, dtype=torch.long)
    intervention_times_tensor = torch.tensor(intervention_times_list, dtype=torch.long)
    intervention_values_tensor = torch.tensor(intervention_values_list, dtype=torch.float32)
    query_targets_tensor = torch.tensor(query_targets_list, dtype=torch.long)
    query_times_tensor = torch.tensor(query_times_list, dtype=torch.long)
    query_types_tensor = torch.tensor(query_types_list, dtype=torch.long)
    num_vars_tensor = torch.tensor(num_vars_list, dtype=torch.long)
    
    # Create dataset dictionary
    dataset = {
        'X_obs': X_obs_tensor,
        'X_int': X_int_tensor,
        'intervention_targets': intervention_targets_tensor,
        'intervention_times': intervention_times_tensor,
        'intervention_values': intervention_values_tensor,
        'query_targets': query_targets_tensor,
        'query_times': query_times_tensor,
        'query_types': query_types_tensor,  # 0=intervened, 1=downstream, 2=non-causal
        'intervention_types': intervention_types_list,
        'num_vars': num_vars_tensor,
        'metadata': {
            'n_scms': n_scms,
            'T': T,
            'max_nodes': max_nodes,
            'seed': seed,
        }
    }
    
    # Save dataset
    print(f"\n4. Saving dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset, output_path)
    
    # Compute final statistics
    print("\n5. Dataset Statistics:")
    print(f"   Total SCMs generated: {n_scms}")
    print(f"   Time elapsed: {elapsed_total:.2f}s ({n_scms / elapsed_total:.2f} SCMs/s)")
    print(f"   Diverged (regenerated): {diverged_count}/{n_scms}")
    print(f"   Dataset shape: {X_obs_tensor.shape}")
    print(f"   Dataset size: {X_obs_tensor.element_size() * X_obs_tensor.nelement() * 2 / (1024**2):.2f} MB")
    print(f"\n   Query type distribution:")
    print(f"     - Intervened: {query_type_counts[0]}/{n_scms} ({100 * query_type_counts[0] / n_scms:.1f}%)")
    print(f"     - Downstream: {query_type_counts[1]}/{n_scms} ({100 * query_type_counts[1] / n_scms:.1f}%)")
    print(f"     - Non-causal: {query_type_counts[2]}/{n_scms} ({100 * query_type_counts[2] / n_scms:.1f}%)")
    print(f"\n   Intervention type distribution:")
    for int_type, count in sorted(intervention_type_counts.items()):
        print(f"     - {int_type}: {count}/{n_scms} ({100 * count / n_scms:.1f}%)")
    print(f"\n   Saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("Three-way evaluation dataset generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate three-way evaluation dataset")
    parser.add_argument('--n_scms', type=int, default=1000, help='Number of SCMs')
    parser.add_argument('--T', type=int, default=50, help='Time series length')
    parser.add_argument('--max_nodes', type=int, default=10, help='Max nodes')
    parser.add_argument('--seed', type=int, default=99, help='Random seed')
    parser.add_argument('--output', type=str, default='data/test_set_threeway_1k.pt')
    
    args = parser.parse_args()
    
    generate_dataset_threeway(
        n_scms=args.n_scms,
        T=args.T,
        max_nodes=args.max_nodes,
        seed=args.seed,
        output_path=args.output,
    )