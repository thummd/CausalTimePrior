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

"""Checks that keep failed simulations and degenerate targets out of training and evaluation.

``TemporalSCM`` returns a failed simulation as an all-zero series. That series has the
shape and dtype of valid data, so no later stage can tell it from signal unless it is
checked. The trainers and evaluators normalise every sample by the standard deviation
of its observational context. A zero context therefore multiplies the target by 1e8
and silently destroys the run.

Two layers guard against this. ``generate_checked_pair`` rejects a failed simulation
while a dataset is generated. ``assert_target_distribution`` logs and asserts the
target statistics before the first training or evaluation step. Scripts that work on
a stored dataset file call it through ``assert_dataset_target_distribution``, which
knows both dataset formats.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

if TYPE_CHECKING:  # imported for type hints only, which avoids a circular import with prior.py
    from causal_time_prior.interventions import InterventionSpec
    from causal_time_prior.prior import CausalTimePrior
    from causal_time_prior.temporal_scm import TemporalSCM


# The trainers and evaluators divide by (context std + NORMALISATION_EPS). The value
# is repeated here because the statistics below must reproduce that normalisation.
NORMALISATION_EPS = 1e-8

# Smallest context standard deviation accepted. Every dataset behind the published
# results has a minimum above 0.03. A failed simulation has exactly 0.
MIN_CONTEXT_STD = 1e-6

# Largest normalised target accepted. The published training set peaks at 1.1e4. A
# zero context paired with a nonzero target gives 1e8 or more, so 1e6 separates the
# two cases with two orders of magnitude to spare on either side.
MAX_ABS_NORMALISED_TARGET = 1e6


class StepZeroCheckError(ValueError):
    """Error raised when data fails the step-zero check.

    It is a ``ValueError``, so code that catches ``ValueError`` keeps working. The
    dedicated type exists for callers that swallow ordinary errors, such as a driver
    that runs several experiments in sequence. Such a caller can let this error
    through, because corrupted data must stop the run.

    Args:
        *args: The message that names the data and every failed condition.

    Returns:
        Nothing. Exceptions are raised, not returned.

    Raises:
        Nothing by itself.
    """


def is_zero_filled(X: torch.Tensor) -> bool:
    """Tell whether a simulated series is entirely zero.

    Args:
        X: Series of shape (T, N), as returned by ``sample_observational`` or
            ``sample_interventional``.

    Returns:
        True when every entry is exactly zero. This is how ``TemporalSCM`` reports a
        failed simulation.

    Raises:
        Nothing.
    """
    return bool((X == 0).all())


def is_degenerate_pair(X_obs: torch.Tensor, X_int: torch.Tensor) -> bool:
    """Tell whether a simulated pair must not enter a dataset.

    Args:
        X_obs: Observational series of shape (T, N).
        X_int: Interventional series of shape (T, N).

    Returns:
        True when either arm is zero-filled or contains NaN or Inf.

    Raises:
        Nothing.
    """
    if is_zero_filled(X_obs) or is_zero_filled(X_int):
        return True
    return not bool(torch.isfinite(X_obs).all() and torch.isfinite(X_int).all())


def generate_checked_pair(
    prior: "CausalTimePrior",
    T: int,
    max_attempts: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, "InterventionSpec", "TemporalSCM", int]:
    """Draw one observational and interventional pair and reject failed simulations.

    The first draw is a plain ``prior.generate_pair`` call. When no simulation fails,
    the random stream is therefore consumed exactly as without this wrapper, and a
    dataset stays reproducible from its seed.

    Args:
        prior: Prior to draw from.
        T: Length of the time series.
        max_attempts: Number of draws before giving up.

    Returns:
        Tuple ``(X_obs, X_int, intervention, scm, n_rejected)``. ``n_rejected`` is the
        number of failed draws discarded before the returned pair.

    Raises:
        RuntimeError: If ``max_attempts`` consecutive draws fail. A prior that fails
            this often is misconfigured, and resampling would hide that.
    """
    for n_rejected in range(max_attempts):
        X_obs, X_int, intervention, scm = prior.generate_pair(T=T)
        if not is_degenerate_pair(X_obs, X_int):
            return X_obs, X_int, intervention, scm, n_rejected

    raise RuntimeError(
        f"{max_attempts} consecutive simulations failed (zero-filled or non-finite). "
        "Check the prior configuration instead of resampling further."
    )


def target_distribution_stats(
    X_obs: torch.Tensor,
    X_int: torch.Tensor,
    query_targets: Optional[torch.Tensor] = None,
    query_times: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute the statistics that are logged and asserted at step zero.

    Args:
        X_obs: Observational arm of shape (n, T, N).
        X_int: Interventional arm of shape (n, T, N).
        query_targets: Queried variable per sample, shape (n,). None skips the
            statistics of the query cell.
        query_times: Queried time step per sample, shape (n,). None skips the
            statistics of the query cell.

    Returns:
        Flat dictionary. Per arm it holds the fraction of series with at least one
        nonzero entry, the mean, and the variance. It also holds the minimum context
        standard deviation. With queries it adds, per arm, the nonzero fraction, mean,
        and variance of the value at the query cell, and the size of the target after
        the per-sample normalisation used by the trainers.

    Raises:
        ValueError: If the two arms differ in shape or are not three-dimensional.
    """
    if X_obs.dim() != 3 or X_obs.shape != X_int.shape:
        raise ValueError(
            f"Expected two arms of equal shape (n, T, N), got {tuple(X_obs.shape)} and {tuple(X_int.shape)}."
        )

    n = X_obs.shape[0]
    stats: Dict[str, float] = {"n_samples": float(n)}

    for arm, X in (("obs", X_obs), ("int", X_int)):
        # Series level, not entry level: padded variables are zero by design, so the
        # entry-level fraction would sit near 0.65 for perfectly valid data. `!= 0`
        # instead of a maximum, so that a NaN series is not miscounted as zero-filled.
        stats[f"{arm}_nonzero_fraction"] = (X != 0).flatten(1).any(dim=1).float().mean().item()
        stats[f"{arm}_mean"] = X.mean().item()
        stats[f"{arm}_variance"] = X.var().item()
        stats[f"{arm}_finite"] = float(torch.isfinite(X).all().item())

    context_mean = X_obs.mean(dim=(1, 2))
    context_std = X_obs.std(dim=(1, 2))
    stats["min_context_std"] = context_std.min().item()

    if query_targets is not None and query_times is not None:
        rows = torch.arange(n, device=X_obs.device)
        for arm, X in (("obs", X_obs), ("int", X_int)):
            cell = X[rows, query_times, query_targets]
            stats[f"{arm}_target_nonzero_fraction"] = (cell != 0).float().mean().item()
            stats[f"{arm}_target_mean"] = cell.mean().item()
            stats[f"{arm}_target_variance"] = cell.var().item()

        # Reproduces the normalisation in train_v2.py and the evaluators, because the
        # loss is computed on this quantity and not on the raw target.
        target = X_int[rows, query_times, query_targets]
        normalised = (target - context_mean) / (context_std + NORMALISATION_EPS)
        stats["max_abs_normalised_target"] = normalised.abs().max().item()
        stats["normalised_target_variance"] = normalised.var().item()

    return stats


def assert_target_distribution(
    X_obs: torch.Tensor,
    X_int: torch.Tensor,
    query_targets: Optional[torch.Tensor] = None,
    query_times: Optional[torch.Tensor] = None,
    name: str = "dataset",
) -> Dict[str, float]:
    """Log the target statistics and stop the run if they show corrupted data.

    Call this once, before the first training or evaluation step. A seed protocol
    guards against variance between runs. It does not guard against a systematic
    corruption of the targets, which is what this assertion is for.

    Args:
        X_obs: Observational arm of shape (n, T, N).
        X_int: Interventional arm of shape (n, T, N).
        query_targets: Queried variable per sample, shape (n,). Optional.
        query_times: Queried time step per sample, shape (n,). Optional.
        name: Label of the split in the log, for example ``"train split"``.

    Returns:
        The statistics from ``target_distribution_stats``, so a caller can store them
        next to a checkpoint or a results table.

    Raises:
        StepZeroCheckError: If an arm holds NaN or Inf, if any series is zero-filled,
            if a context has a standard deviation below ``MIN_CONTEXT_STD``, if the
            targets have no variance, or if a normalised target exceeds
            ``MAX_ABS_NORMALISED_TARGET``. It is a ``ValueError``.
    """
    stats = target_distribution_stats(X_obs, X_int, query_targets, query_times)
    has_queries = "int_target_variance" in stats

    print(f"   Target statistics at step zero [{name}, n={int(stats['n_samples'])}]:")
    print(f"     {'Arm':<16}{'Nonzero series':>15}{'Mean':>12}{'Variance':>14}")
    for arm, label in (("obs", "Observational"), ("int", "Interventional")):
        print(f"     {label:<16}{100 * stats[f'{arm}_nonzero_fraction']:>14.2f}%"
              f"{stats[f'{arm}_mean']:>12.4f}{stats[f'{arm}_variance']:>14.4f}")
    if has_queries:
        print(f"     {'Query cell':<16}{'Nonzero values':>15}{'Mean':>12}{'Variance':>14}")
        for arm, label in (("obs", "Observational"), ("int", "Interventional")):
            print(f"     {label:<16}{100 * stats[f'{arm}_target_nonzero_fraction']:>14.2f}%"
                  f"{stats[f'{arm}_target_mean']:>12.4f}{stats[f'{arm}_target_variance']:>14.4f}")
    print(f"     Minimum context std: {stats['min_context_std']:.4g}")
    if has_queries:
        print(f"     Largest |normalised target|: {stats['max_abs_normalised_target']:.4g}")

    problems: List[str] = []
    for arm, label in (("obs", "observational"), ("int", "interventional")):
        if not stats[f"{arm}_finite"]:
            problems.append(f"the {label} arm contains NaN or Inf")
        if stats[f"{arm}_nonzero_fraction"] < 1.0:
            n_zero = round((1.0 - stats[f"{arm}_nonzero_fraction"]) * stats["n_samples"])
            problems.append(f"{n_zero} {label} series are zero-filled, which marks a failed simulation")
        if not stats[f"{arm}_variance"] > 0.0:
            problems.append(f"the {label} arm has no variance")
    if stats["min_context_std"] < MIN_CONTEXT_STD:
        problems.append(
            f"the smallest context std is {stats['min_context_std']:.3g}, below {MIN_CONTEXT_STD:g}, "
            "so the per-sample normalisation would divide by about 1e-8"
        )
    if has_queries:
        if not stats["int_target_variance"] > 0.0:
            problems.append("the interventional targets have no variance")
        # `not <=` instead of `>` so that a NaN statistic is reported as a problem too.
        if not stats["max_abs_normalised_target"] <= MAX_ABS_NORMALISED_TARGET:
            problems.append(
                f"the largest normalised target is {stats['max_abs_normalised_target']:.3g}, "
                f"above {MAX_ABS_NORMALISED_TARGET:g}"
            )

    if problems:
        raise StepZeroCheckError(
            f"Step-zero check failed for the {name}: " + "; ".join(problems) + ". "
            "Regenerate the data with the current simulator. Datasets generated with "
            "commits 0208a29 to ad9f0a4 contain zero-filled series and must not be used."
        )

    print("     Step-zero check passed.")
    return stats


def assert_dataset_target_distribution(
    dataset: Dict[str, Any],
    name: str = "dataset",
    rows: Optional[Union[slice, Sequence[int], torch.Tensor]] = None,
) -> Dict[str, float]:
    """Run the step-zero check on a stored dataset of either format.

    The stored datasets come in two formats. The newer one names the queried cell
    with ``query_targets`` and ``query_times``. The older one has no separate query.
    Its models predict the intervened variable at the intervention time, so the cell
    is given by ``targets`` and ``intervention_times``. Resolving the format here
    keeps that knowledge in one place instead of in every training and evaluation
    script.

    Args:
        dataset: Dictionary as written by one of the ``generate_dataset*.py`` scripts
            and returned by ``torch.load``.
        name: Label of the data in the log, for example ``"test set"``.
        rows: Samples that the caller is going to use, as a slice, a sequence of
            indices, or an index tensor. None checks every sample. A script that
            trains or evaluates on a part of a file passes that part, so the logged
            statistics describe the data of the run.

    Returns:
        The statistics from ``target_distribution_stats`` for the selected samples.

    Raises:
        KeyError: If the dictionary lacks an arm or names no query cell in either
            format.
        StepZeroCheckError: If no sample is selected, or if the selected samples fail
            the check as described in ``assert_target_distribution``. It is a
            ``ValueError``.
    """
    if "query_targets" in dataset and "query_times" in dataset:
        targets, times = dataset["query_targets"], dataset["query_times"]
    elif "targets" in dataset and "intervention_times" in dataset:
        targets, times = dataset["targets"], dataset["intervention_times"]
    else:
        raise KeyError(
            "The dataset names no query cell. Expected 'query_targets' and 'query_times', "
            f"or 'targets' and 'intervention_times'. Found: {sorted(dataset)}."
        )

    X_obs, X_int = dataset["X_obs"], dataset["X_int"]
    if rows is not None:
        if not isinstance(rows, slice):
            # Evaluators load their test set straight onto the GPU, and a tensor on the
            # GPU can only be indexed by an index on the same device.
            rows = torch.as_tensor(rows, dtype=torch.long, device=X_obs.device)
        X_obs, X_int, targets, times = X_obs[rows], X_int[rows], targets[rows], times[rows]

    if X_obs.shape[0] == 0:
        raise StepZeroCheckError(f"Step-zero check failed for the {name}: no sample is selected.")

    return assert_target_distribution(X_obs, X_int, targets, times, name=name)
