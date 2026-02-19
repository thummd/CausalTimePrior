"""Train SimpleCausalPFN on HARD-ONLY intervention data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causal_time_prior.train_v2 import train_simple_pfn_v2

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/causal_time_prior_hard_only_10k.pt')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/simple_causal_pfn_v2_hard_only.pt')
    parser.add_argument('--epochs', type=int, default=15)
    
    args = parser.parse_args()
    
    print("Training on HARD-ONLY interventions")
    
    train_simple_pfn_v2(
        data_path=args.data,
        save_path=args.checkpoint,
        num_epochs=args.epochs,
        batch_size=32,
        lr=1e-4,
    )