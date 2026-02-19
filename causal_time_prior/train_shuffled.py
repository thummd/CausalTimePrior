"""Train SimpleCausalPFN on SHUFFLED data (control experiment)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from causal_time_prior.simple_causal_pfn_v2 import SimpleCausalPFNV2


def train_on_shuffled_data(
    train_path: str = "data/causal_time_prior_shuffled_10k.pt",
    checkpoint_path: str = "checkpoints/simple_causal_pfn_v2_shuffled.pt",
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = 'cpu',
):
    """Train SimpleCausalPFNV2 on shuffled data."""
    
    print("=" * 80)
    print("Training SimpleCausalPFN on SHUFFLED Data (Control)")
    print("=" * 80)
    
    # Load shuffled dataset
    print(f"\n1. Loading shuffled dataset from {train_path}...")
    dataset = torch.load(train_path, map_location=device)
    
    X_obs = dataset['X_obs']
    X_int = dataset['X_int']
    intervention_targets = dataset['intervention_targets']
    intervention_times = dataset['intervention_times']
    intervention_values = dataset['intervention_values']
    query_targets = dataset['query_targets']
    query_times = dataset['query_times']
    
    n_samples = X_obs.shape[0]
    T = X_obs.shape[1]
    input_dim = X_obs.shape[2]
    
    print(f"   Dataset shape: {X_obs.shape}")
    print(f"   Total samples: {n_samples}")
    print(f"   Shuffled: {dataset['metadata'].get('shuffled', False)}")
    
    # Split train/val
    n_train = int(0.9 * n_samples)
    train_indices = np.arange(n_train)
    val_indices = np.arange(n_train, n_samples)
    
    print(f"   Train samples: {n_train}")
    print(f"   Val samples: {n_samples - n_train}")
    
    # Create model
    print(f"\n2. Creating SimpleCausalPFNV2 model...")
    model = SimpleCausalPFNV2(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        max_nodes=input_dim,
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\n3. Training for {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle training data
        np.random.shuffle(train_indices)
        
        # Training
        train_loss_accum = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_indices = train_indices[i:i+batch_size]
            
            # Get batch
            X_obs_batch = X_obs[batch_indices].to(device)
            X_int_batch = X_int[batch_indices].to(device)
            int_targets_batch = intervention_targets[batch_indices].to(device)
            int_times_batch = intervention_times[batch_indices].to(device)
            int_values_batch = intervention_values[batch_indices].to(device)
            q_targets_batch = query_targets[batch_indices].to(device)
            q_times_batch = query_times[batch_indices].to(device)
            
            # Normalize
            mean = X_obs_batch.mean(dim=(1, 2), keepdim=True)
            std = X_obs_batch.std(dim=(1, 2), keepdim=True) + 1e-8
            X_obs_norm = (X_obs_batch - mean) / std
            Y_int_norm = (X_int_batch - mean) / std
            
            # Forward pass
            loss = model.loss(
                X_obs_norm,
                Y_int_norm,
                int_targets_batch,
                int_times_batch,
                int_values_batch,
                q_targets_batch,
                q_times_batch,
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            n_batches += 1
        
        train_loss = train_loss_accum / n_batches
        
        # Validation
        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_indices), batch_size):
                batch_indices = val_indices[i:i+batch_size]
                
                X_obs_batch = X_obs[batch_indices].to(device)
                X_int_batch = X_int[batch_indices].to(device)
                int_targets_batch = intervention_targets[batch_indices].to(device)
                int_times_batch = intervention_times[batch_indices].to(device)
                int_values_batch = intervention_values[batch_indices].to(device)
                q_targets_batch = query_targets[batch_indices].to(device)
                q_times_batch = query_times[batch_indices].to(device)
                
                # Normalize
                mean = X_obs_batch.mean(dim=(1, 2), keepdim=True)
                std = X_obs_batch.std(dim=(1, 2), keepdim=True) + 1e-8
                X_obs_norm = (X_obs_batch - mean) / std
                Y_int_norm = (X_int_batch - mean) / std
                
                # Forward pass
                loss = model.loss(
                    X_obs_norm,
                    Y_int_norm,
                    int_targets_batch,
                    int_times_batch,
                    int_values_batch,
                    q_targets_batch,
                    q_times_batch,
                )
                
                val_loss_accum += loss.item()
                n_val_batches += 1
        
        val_loss = val_loss_accum / n_val_batches
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"   Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
              f"Time = {elapsed:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'max_nodes': input_dim,
                },
                'metadata': {
                    'shuffled': True,
                    'train_path': train_path,
                }
            }
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
    
    total_time = time.time() - start_time
    
    print(f"\n4. Training complete!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoint saved to: {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("Shuffled model training complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train on shuffled data")
    parser.add_argument('--train', type=str, default='data/causal_time_prior_shuffled_10k.pt')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/simple_causal_pfn_v2_shuffled.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    train_on_shuffled_data(
        train_path=args.train,
        checkpoint_path=args.checkpoint,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )