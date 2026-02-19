"""Train SimpleCausalPFNV2 on downstream effects dataset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2


def train_epoch(model, dataloader, optimizer, device, normalize=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        X_obs, X_int, int_target, int_time, int_value, query_target, query_time = batch
        
        X_obs = X_obs.to(device)
        X_int = X_int.to(device)
        int_target = int_target.to(device)
        int_time = int_time.to(device)
        int_value = int_value.to(device)
        query_target = query_target.to(device)
        query_time = query_time.to(device)
        
        # Normalize inputs
        if normalize:
            mean = X_obs.mean(dim=(1, 2), keepdim=True)
            std = X_obs.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs = (X_obs - mean) / std
            X_int = (X_int - mean) / std
        
        optimizer.zero_grad()
        loss = model.loss(X_obs, X_int, int_target, int_time, int_value, query_target, query_time)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device, normalize=True):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    total_rmse = 0
    num_batches = 0
    
    for batch in dataloader:
        X_obs, X_int, int_target, int_time, int_value, query_target, query_time = batch
        
        X_obs = X_obs.to(device)
        X_int = X_int.to(device)
        int_target = int_target.to(device)
        int_time = int_time.to(device)
        int_value = int_value.to(device)
        query_target = query_target.to(device)
        query_time = query_time.to(device)
        
        # Normalize inputs
        if normalize:
            mean = X_obs.mean(dim=(1, 2), keepdim=True)
            std = X_obs.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs = (X_obs - mean) / std
            X_int = (X_int - mean) / std
        
        # Compute loss
        loss = model.loss(X_obs, X_int, int_target, int_time, int_value, query_target, query_time)
        total_loss += loss.item()
        
        # Compute RMSE
        mean_pred, std_pred = model(X_obs, int_target, int_time, int_value, query_target, query_time)
        batch_indices = torch.arange(X_int.shape[0], device=X_int.device)
        Y_true = X_int[batch_indices, query_time, query_target]
        
        rmse = torch.sqrt(torch.mean((mean_pred - Y_true) ** 2))
        total_rmse += rmse.item()
        
        num_batches += 1
    
    return total_loss / num_batches, total_rmse / num_batches


def main(
    data_path: str = "data/causal_time_prior_downstream_10k.pt",
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    num_epochs: int = 15,
    lr: float = 1e-4,
    train_split: float = 0.9,
    device: str = "cpu",
    save_path: str = "checkpoints/simple_causal_pfn_v2.pt",
):
    """Train SimpleCausalPFNV2 on downstream effects dataset."""
    
    print("=" * 80)
    print("Training SimpleCausalPFNV2 on Downstream Effects")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading dataset from {data_path}...")
    dataset = torch.load(data_path)
    
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    int_targets = dataset['intervention_targets']
    int_times = dataset['intervention_times']
    int_values = dataset['intervention_values']
    query_targets = dataset['query_targets']
    query_times = dataset['query_times']
    is_downstream = dataset.get('is_downstream', None)
    query_types = dataset.get('query_types', None)

    print(f"   Dataset shape: {X_obs.shape}")
    if is_downstream is not None:
        print(f"   Downstream queries: {is_downstream.sum()}/{len(is_downstream)} ({100*is_downstream.sum()/len(is_downstream):.1f}%)")
    elif query_types is not None:
        n_downstream = (query_types == 1).sum()
        print(f"   Downstream queries: {n_downstream}/{len(query_types)} ({100*n_downstream/len(query_types):.1f}%)")
    
    # Train/test split
    n_samples = X_obs.shape[0]
    n_train = int(train_split * n_samples)
    
    train_data = TensorDataset(
        X_obs[:n_train], X_int[:n_train], 
        int_targets[:n_train], int_times[:n_train], int_values[:n_train],
        query_targets[:n_train], query_times[:n_train]
    )
    test_data = TensorDataset(
        X_obs[n_train:], X_int[n_train:],
        int_targets[n_train:], int_times[n_train:], int_values[n_train:],
        query_targets[n_train:], query_times[n_train:]
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"\n2. Train/test split: {n_train}/{n_samples - n_train}")
    
    # Create model
    print(f"\n3. Initializing SimpleCausalPFNV2...")
    input_dim = X_obs.shape[2]
    max_nodes = X_obs.shape[2]
    
    model = SimpleCausalPFNV2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_nodes=max_nodes,
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {num_params:,}")
    print(f"   Hidden dim: {hidden_dim}, Num layers: {num_layers}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\n4. Training for {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}, Learning rate: {lr}")
    
    best_test_rmse = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device, normalize=True)
        test_loss, test_rmse = evaluate(model, test_loader, device, normalize=True)
        
        epoch_time = time.time() - epoch_start
        
        print(f"   Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s) - "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}")
        
        # Save best model
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_rmse': test_rmse,
                'config': {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'input_dim': input_dim,
                    'max_nodes': max_nodes,
                }
            }, save_path)
    
    total_time = time.time() - start_time
    
    print(f"\n5. Training complete!")
    print(f"   Total time: {total_time:.2f}s ({total_time / num_epochs:.2f}s/epoch)")
    print(f"   Best test RMSE: {best_test_rmse:.4f}")
    print(f"   Model saved to: {save_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


# Alias for backward compatibility
train_simple_pfn_v2 = main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SimpleCausalPFNV2")
    parser.add_argument('--data', type=str, default='data/causal_time_prior_downstream_10k.pt')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save', type=str, default='checkpoints/simple_causal_pfn_v2.pt')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        save_path=args.save,
    )