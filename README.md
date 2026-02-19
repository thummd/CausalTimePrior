# CausalTimePrior

**Interventional Time Series Priors for Causal Foundation Models**

TSALM Workshop at ICLR 2026 | [OpenReview](https://openreview.net/forum?id=TODO) <!-- | [arXiv](https://arxiv.org/abs/TODO) -->

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
@inproceedings{causaltimeprior2026,
  title={Interventional Time Series Priors for Causal Foundation Models},
  booktitle={TSALM Workshop at ICLR},
  year={2026}
}
```

## License

This project builds on [Do-PFN-prior](https://github.com/oossen/Do-PFN-prior) and [TempoPFN](https://github.com/automl/TempoPFN). Please refer to their respective licenses.
