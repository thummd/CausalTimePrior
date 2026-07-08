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