# CausalTimePrior

**Interventional Time Series Priors for Causal Foundation Models**

TSALM Workshop at ICLR 2026 | [OpenReview](https://openreview.net/forum?id=JbTgx2L9Z2) <!-- | [arXiv](https://arxiv.org/abs/TODO) -->

CausalTimePrior is a framework for generating synthetic temporal structural causal models (SCMs) with paired observational and interventional time series. It addresses a critical gap in time series causal inference: existing benchmarks generate only observational data, lacking the interventional targets needed to train causal foundation models.

## Features

- **Temporal SCMs** with time-lagged causal dependencies (G_0, G_1, ..., G_K)
- **Multiple intervention types**: hard (do-operator), soft (additive shifts), time-varying
- **Diverse mechanisms**: linear, tanh, sin, cos, abs, square, ReLU, and more
- **Configurable graph priors**: Erdos-Renyi with acyclicity + lagged edges with decay
- **Regime-switching SCMs**: Markov-driven structural breaks with interventional data
- **Stability guarantees**: clipping, divergence detection, burn-in periods

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:thummd/CausalTimePrior.git
cd CausalTimePrior

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install submodule packages
pip install -e Do-PFN-prior/
pip install -e TempoPFN/

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PWD/Do-PFN-prior:$PYTHONPATH
```

## Quick Start

```python
from causal_time_prior import CausalTimePrior

# Initialize prior with default configuration
prior = CausalTimePrior(seed=42)

# Generate a single paired example
X_obs, X_int, intervention, scm = prior.generate_pair(T=100)

print(f"Observational shape: {X_obs.shape}")  # (100, N)
print(f"Interventional shape: {X_int.shape}")  # (100, N)
print(f"Intervention type: {intervention.intervention_type.value}")

# Generate a dataset
dataset = prior.generate_dataset(n_scms=1000, T=100)
```

## Reproducing the Paper

Table 2 of the paper reports a three-way evaluation of a proof-of-concept PFN. The model is a 2-layer GRU with hidden size 64 and 82,562 parameters. The published paper states a hidden size of 128. That is an error in the text. The released checkpoint is the model that was evaluated.

### Evaluating the Released Checkpoint

The checkpoint and the test set behind Table 2 are on the [Hugging Face Hub](https://huggingface.co/thummd/causal-time-prior-pfn).

```bash
hf download thummd/causal-time-prior-pfn --local-dir artifacts

python causal_time_prior/evaluate_with_nmse.py \
    --model artifacts/simple_causal_pfn_v2_full_100k.pt \
    --test artifacts/test_set_threeway_1k.pt
```

Expected output:

| Query Type | N | RMSE | NMSE | Pred/GT |
|---|---|---|---|---|
| Intervened | 573 | 226.39 | 1.41 | 0.95 |
| Downstream | 270 | 231.31 | 0.67 | 0.85 |
| Non-causal | 157 | 143.15 | 0.66 | 0.46 |
| Overall | 1000 | 216.87 | 0.99 | 0.85 |

### Regenerating the Data and Retraining

```bash
# Training set: 100,000 SCMs, about 2.5 hours on one CPU core.
python causal_time_prior/generate_dataset_v2.py --n_scms 100000 --T 50 --max_nodes 10 \
    --seed 777 --downstream_prob 0.7 --output data/causal_time_prior_full_100k.pt

# Test set. It predates the regime-switching family, hence the explicit 0.0.
python causal_time_prior/generate_dataset_threeway.py --n_scms 1000 --T 50 --max_nodes 10 \
    --seed 99 --regime_switching_prob 0.0 --output data/test_set_threeway_1k.pt

# Training uses the script defaults: hidden size 64, 2 layers, batch size 32,
# 15 epochs, learning rate 1e-4. It takes about 11 minutes on a CPU.
python causal_time_prior/train_v2.py --data data/causal_time_prior_full_100k.pt \
    --save checkpoints/simple_causal_pfn_v2_full_100k.pt

python causal_time_prior/evaluate_with_nmse.py \
    --model checkpoints/simple_causal_pfn_v2_full_100k.pt \
    --test data/test_set_threeway_1k.pt
```

The original runs used Python 3.12, PyTorch 2.10.0, NumPy 2.4.2, and NetworkX 3.6.1.

What to expect:

- **Data generation is deterministic.** With PyTorch 2.10.0 and NumPy 2.4.2, the test-set command reproduces the released test set bit for bit. The training-set command reproduces the first 5,000 samples of our stored training set bit for bit. We did not compare the remaining samples.
- **Training is not seeded.** Retraining does not return the released weights. In four reruns on the stored training set, two with hidden size 64 and two with 128, the overall NMSE on the released test set ranged from 0.77 to 1.06. The ordering of NMSE across query types was the same in every run. The Pred/GT column varies more. The non-causal ratio was 0.72 to 0.90 in the reruns, against 0.46 in the paper, so read that column as a single-run point estimate.
- **Saturated systems dominate the metrics.** Simulated values are clipped to [-1000, 1000] and the trajectory is kept. About 29% of the training SCMs and 33% of the test SCMs reach the clip, and they carry almost all of the summed squared error. The "0% divergence" in the paper means that no NaN or Inf occurs. It does not mean that the dynamics are stable.
- **Training and evaluation start with a data check.** `train_v2.py`, `evaluate_with_nmse.py`, and `evaluate_threeway.py` print the target statistics per arm and stop if they find zero-filled series or contexts without variance.

### Which Revision to Use

Use the current `main`, or the tag `tsalm2026-paper`, which marks the code that generated the paper data. Do not use commits `0208a29` to `ad9f0a4`. In that range the simulator replaced every trajectory that exceeded 500 at the end of the burn-in by zeros. About 26% of the generated contexts are then all zero, the training loss explodes, and the NMSE reaches the hundreds.

## Repository Structure

```
CausalTimePrior/
├── causal_time_prior/          # Core Python package
│   ├── prior.py                # CausalTimePrior main API
│   ├── temporal_scm.py         # Temporal SCM with forward simulation
│   ├── temporal_scm_builder.py # SCM builder with configurable priors
│   ├── temporal_graph.py       # Temporal DAG sampling
│   ├── interventions.py        # Intervention specification & sampling
│   ├── regime_switching.py     # Regime-switching SCM support
│   ├── simple_causal_pfn_v2.py # Proof-of-concept PFN model
│   ├── data_checks.py          # Step-zero data checks, rejection of failed simulations
│   ├── baselines.py            # VAR-OLS, PCMCI+ baselines
│   ├── visualization.py        # Plotting utilities
│   └── ...                     # Training, evaluation, generation scripts
├── demo_outputs/pdf/           # Paper figures
├── Do-PFN-prior/               # Git submodule: Do-PFN SCM prior
├── TempoPFN/                   # Git submodule: TempoPFN foundation model
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{
thumm2026interventional,
title={Interventional Time Series Priors for Causal Foundation Models},
author={Dennis Thumm and Ying Chen},
booktitle={1st ICLR Workshop on Time Series in the Age of Large Models},
year={2026},
url={https://openreview.net/forum?id=JbTgx2L9Z2}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

It builds on [Do-PFN-prior](https://github.com/oossen/Do-PFN-prior) and [TempoPFN](https://github.com/automl/TempoPFN), which are included as git submodules and retain their respective licenses.
