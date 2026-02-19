"""Train SimpleCausalPFN on CausalTimePrior dataset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from causal_time_prior.simple_causal_pfn import SimpleCausalPFN


def train_epoch(model, dataloader, optimizer, device, normalize=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X_obs, X_int, targets, intervention_times, intervention_values in dataloader:
        X_obs = X_obs.to(device)
        X_int = X_int.to(device)
        targets = targets.to(device)
        intervention_times = intervention_times.to(device)
        intervention_values = intervention_values.to(device)
        
        # Normalize inputs (per-sample z-score normalization)
        if normalize:
            mean = X_obs.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
            std = X_obs.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs = (X_obs - mean) / std
            X_int = (X_int - mean) / std
        
        optimizer.zero_grad()
        loss = model.loss(X_obs, X_int, targets, intervention_times, intervention_values)
        loss.backward()
        
        # Gradient clipping for stability
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
    
    for X_obs, X_int, targets, intervention_times, intervention_values in dataloader:
        X_obs = X_obs.to(device)
        X_int = X_int.to(device)
        targets = targets.to(device)
        intervention_times = intervention_times.to(device)
        intervention_values = intervention_values.to(device)
        
        # Normalize inputs (per-sample z-score normalization)
        if normalize:
            mean = X_obs.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
            std = X_obs.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs = (X_obs - mean) / std
            X_int = (X_int - mean) / std
        
        # Compute loss
        loss = model.loss(X_obs, X_int, targets, intervention_times, intervention_values)
        total_loss += loss.item()
        
        # Compute RMSE
        mean, std = model(X_obs, targets, intervention_times, intervention_values)
        batch_indices = torch.arange(X_int.shape[0], device=X_int.device)
        Y_true = X_int[batch_indices, intervention_times, targets]
        
        rmse = torch.sqrt(torch.mean((mean - Y_true) ** 2))
        total_rmse += rmse.item()
        
        num_batches += 1
    
    return total_loss / num_batches, total_rmse / num_batches


def main(
    data_path: str = "data/causal_time_prior_1k.pt",
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-4,
    train_split: float = 0.9,
    device: str = "cpu",
    save_path: str = "checkpoints/simple_causal_pfn.pt",
):
    """Train SimpleCausalPFN.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset file.
    hidden_dim : int
        Hidden dimension for GRU.
    num_layers : int
        Number of GRU layers.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    train_split : float
        Fraction of data to use for training.
    device : str
        Device to train on.
    save_path : str
        Path to save the trained model.
    """
    print("=" * 80)
    print("Training SimpleCausalPFN")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading dataset from {data_path}...")
    dataset = torch.load(data_path)
    
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    targets = dataset['targets']
    intervention_times = dataset['intervention_times']
    intervention_values = dataset['intervention_values']
    
    print(f"   Dataset shape: {X_obs.shape}")
    print(f"   Number of samples: {X_obs.shape[0]}")
    print(f"   Time series length: {X_obs.shape[1]}")
    print(f"   Max nodes: {X_obs.shape[2]}")
    
    # Train/test split
    n_samples = X_obs.shape[0]
    n_train = int(train_split * n_samples)
    
    X_obs_train = X_obs[:n_train]
    X_int_train = X_int[:n_train]
    targets_train = targets[:n_train]
    intervention_times_train = intervention_times[:n_train]
    intervention_values_train = intervention_values[:n_train]
    
    X_obs_test = X_obs[n_train:]
    X_int_test = X_int[n_train:]
    targets_test = targets[n_train:]
    intervention_times_test = intervention_times[n_train:]
    intervention_values_test = intervention_values[n_train:]
    
    print(f"\n2. Train/test split: {n_train}/{n_samples - n_train}")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        X_obs_train, X_int_train, targets_train, 
        intervention_times_train, intervention_values_train
    )
    test_dataset = TensorDataset(
        X_obs_test, X_int_test, targets_test,
        intervention_times_test, intervention_values_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\n3. Initializing SimpleCausalPFN...")
    input_dim = X_obs.shape[2]
    max_nodes = X_obs.shape[2]
    
    model = SimpleCausalPFN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_nodes=max_nodes,
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {num_params:,}")
    print(f"   Hidden dim: {hidden_dim}, Num layers: {num_layers}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\n4. Training for {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}, Learning rate: {lr}")
    print(f"   Device: {device}")
    
    best_test_rmse = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, normalize=True)
        
        # Evaluate
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SimpleCausalPFN")
    parser.add_argument('--data', type=str, default='data/causal_time_prior_1k.pt',
                        help='Path to dataset')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--save', type=str, default='checkpoints/simple_causal_pfn.pt',
                        help='Path to save model')
    
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