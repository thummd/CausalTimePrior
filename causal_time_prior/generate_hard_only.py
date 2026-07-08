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

"""Generate dataset with HARD interventions only (ablation study)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Monkey-patch intervention sampler to force hard interventions
from causal_time_prior.interventions import InterventionSampler

_original_init = InterventionSampler.__init__

def hard_only_init(self, N, T, **kwargs):
    """Force hard interventions only."""
    _original_init(self, N, T, p_hard=1.0, p_soft=0.0, p_time_varying=0.0, **{k: v for k, v in kwargs.items() if k not in ['p_hard', 'p_soft', 'p_time_varying']})

InterventionSampler.__init__ = hard_only_init

# Now import and run the threeway generator
from causal_time_prior.generate_dataset_threeway import generate_dataset_threeway

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_scms', type=int, default=10000)
    parser.add_argument('--output', type=str, default='data/causal_time_prior_hard_only_10k.pt')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Generating HARD-ONLY intervention dataset")
    print("=" * 80)
    
    generate_dataset_threeway(
        n_scms=args.n_scms,
        T=50,
        max_nodes=10,
        seed=43,
        output_path=args.output,
    )
    
    print("\n✅ Hard-only dataset generated!")