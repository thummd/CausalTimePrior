# CausalTimePrior

**Interventional Time Series Priors for Causal Foundation Models**

CausalTimePrior is a framework for generating synthetic temporal structural causal models (SCMs) with paired observational and interventional time series data. This addresses a critical gap in time series causal inference research: existing benchmarks generate only observational data, lacking the interventional targets needed to train causal foundation models.

## Features

✅ **Temporal SCMs** with time-lagged causal dependencies (G_0, G_1, ..., G_K)  
✅ **Multiple intervention types**: Hard (do-operator), soft (additive shifts), time-varying  
✅ **Diverse mechanisms**: Linear, tanh, sin, cos, abs, square, ReLU, and more  
✅ **Configurable graph priors**: Erdős-Rényi with acyclicity + lagged edges with decay  
✅ **Stability checks**: Clipping, divergence detection, burn-in periods  
✅ **Visualization tools**: Paired time series plots, causal graphs, intervention effects  

## Installation

```bash
# Clone the repository
cd /home/dennis/repos/ctp

# Activate the virtual environment
source causal_time_env/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/home/dennis/repos/ctp:/home/dennis/repos/ctp/Do-PFN-prior:$PYTHONPATH
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

## Demo

Run the demo to see CausalTimePrior in action:

```bash
cd /home/dennis/repos/ctp
source causal_time_env/bin/activate
export PYTHONPATH=/home/dennis/repos/ctp:/home/dennis/repos/ctp/Do-PFN-prior:$PYTHONPATH
python causal_time_prior/demo.py
```

This generates:
- `demo_outputs/paired_timeseries.png` - Side-by-side observational vs interventional
- `demo_outputs/intervention_effect.png` - Causal effect over time
- `demo_outputs/temporal_dag.png` - Graph structure visualization
- `demo_outputs/all_variables.png` - All variables in grid layout

## Configuration

Default hyperparameters (matching the TSALM workshop paper):

```python
config = {
    'N_max': 10,          # Maximum number of variables
    'K_max': 3,           # Maximum number of lags
    'alpha': 2,           # Beta(2,5) for sparse graphs
    'beta': 5,
    'gamma': 0.7,         # Lag decay factor
    'sigma_w': 1.0,       # Weight std
    'sigma_b': 0.5,       # Bias std
    'T': 100,             # Time series length
    'burn_in': 50,        # Burn-in steps
}

# Use custom config
prior = CausalTimePrior(config=custom_config, seed=42)
```

## Architecture

**File Structure:**
```
causal_time_prior/
├── __init__.py           # Package exports
├── prior.py              # CausalTimePrior (main API)
├── temporal_scm.py       # TemporalSCM with time-stepped simulation
├── temporal_scm_builder.py  # Builder for sampling SCMs
├── temporal_graph.py     # Temporal DAG sampling
├── temporal_mechanism.py # Temporal structural equations
├── interventions.py      # Intervention specifications
├── utils.py              # Stability checks, noise distributions
├── visualization.py      # Plotting utilities
└── demo.py               # End-to-end demo
```

**Key Classes:**

- `CausalTimePrior`: Main orchestrator for sampling SCMs and generating data
- `TemporalSCM`: Temporal SCM with `sample_observational()` and `sample_interventional()`
- `TemporalDAG`: Dataclass holding G_0 (instantaneous) + G_lags (lagged edges)
- `InterventionSpec`: Specification for interventions (targets, times, type, values)

## Extending Do-PFN to Temporal SCMs

CausalTimePrior builds on [Do-PFN](https://github.com/oossen/Do-PFN-prior) by extending:

1. **Graph sampling**: `GraphBuilder` → `TemporalGraphBuilder` (adds lagged edges)
2. **Mechanisms**: `SimpleMechanism` → `TemporalMechanism` (handles lagged parents)
3. **SCM simulation**: Single-pass `propagate()` → Time-stepped forward simulation
4. **Interventions**: Added hard/soft/time-varying do-operator logic

## Paper Reference

This implementation is for the TSALM workshop paper:

> **Interventional Time Series Priors for Causal Foundation Models**  
> TSALM Workshop at ICLR 2026  
> Paper: `TSALM/tsalm_workshop_paper.tex`

**Key contributions:**
- Identified the interventional data gap in existing time series causal benchmarks
- Proposed CausalTimePrior: a principled framework for temporal SCMs with interventions
- Demonstrated that PFNs trained on CausalTimePrior can generalize to unseen temporal causal structures

## Citation

```bibtex
@inproceedings{causaltime2026,
  title={Interventional Time Series Priors for Causal Foundation Models},
  booktitle={TSALM Workshop at ICLR},
  year={2026}
}
```

## License

This project extends [Do-PFN-prior](https://github.com/oossen/Do-PFN-prior) and [TempoPFN](https://github.com/automl/TempoPFN). Please refer to their respective licenses.

## Contact

For questions or issues, please open an issue in the repository.