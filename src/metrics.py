"""Exposure-consistent frequency metrics with explicit annual/period units.

The reference objective is scikit-learn's exposure-weighted rate deviance:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html
Count-deviance/exposure equivalence follows by substituting N=e*y and mu=e*r
into d(N,mu)=2*(N*log(N/mu)-N+mu). It does not justify weighting counts again.

Concentration is raw, not oracle-normalized: descending annual rate, tied score
blocks, cumulative actual claim counts versus cumulative exposure, G=2*AUC-1.
No probability-percent interpretation is attached to a deviance or Gini.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _vector(values, name: str, *, positive: bool = False,
            counts: bool = False) -> np.ndarray:
    original = np.asarray(values)
    if original.dtype.kind == "b":
        raise ValueError(f"Boolean {name} is not a numeric measurement")
    try:
        out = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    if positive and (out <= 0).any():
        raise ValueError(f"{name} must be strictly positive; no clipping")
    if counts and ((out < 0).any() or (out % 1 != 0).any()):
        raise ValueError(f"{name} must contain nonnegative integral counts")
    return out


def _same_length(*arrays: np.ndarray) -> None:
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("Metric arrays must have equal lengths; broadcasting is forbidden")


def _frequency_inputs(observed_counts, annual_rates, exposure):
    n = _vector(observed_counts, "observed_counts", counts=True)
    r = _vector(annual_rates, "annual_rates", positive=True)
    e = _vector(exposure, "exposure", positive=True)
    _same_length(n, r, e)
    with np.errstate(over="ignore", under="ignore"):
        expected = e * r
    if not np.isfinite(expected).all() or (expected <= 0).any():
        raise ValueError("Exposure times annual rate must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        totals = np.array([n.sum(), e.sum(), expected.sum()])
    if not np.isfinite(totals).all():
        raise ValueError("Portfolio count/exposure/expectation totals must be finite")
    return n, r, e, expected


def poisson_count_deviance(observed_counts, expected_counts) -> np.ndarray:
    """Per-policy count deviance d(N,mu); n=0 contributes exactly 2*mu.

    Inputs must already be aligned. There is no second exposure weighting and
    no clipping of invalid predictions. A short series near N=mu avoids loss of
    precision in z-log(1+z); log(N)-log(mu) avoids overflow in N/mu elsewhere.
    """
    n = _vector(observed_counts, "observed_counts", counts=True)
    mu = _vector(expected_counts, "expected_counts", positive=True)
    _same_length(n, mu)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            deviance = 2 * mu.copy()
            positive = n > 0
            a, b = n[positive], mu[positive]
            terms = a * (np.log(a) - np.log(b)) - a + b
            near = np.abs(b - a) <= .01 * a
            delta = (b[near] - a[near]) / a[near]
            series = np.zeros_like(delta)
            for power in range(2, 13):
                series += ((-1) ** power) * delta ** power / power
            terms[near] = a[near] * series
            deviance[positive] = 2 * terms
        except FloatingPointError as error:
            raise ValueError("Count deviance is outside finite floating-point range") from error
    if not np.isfinite(deviance).all() or (deviance < 0).any():
        raise ValueError("Count deviance is not finite and nonnegative")
    return deviance


def poisson_deviance_per_exposure(observed_counts, annual_rates, exposure) -> float:
    """sum d(N,e*r)/sum(e), equivalent to weighted mean rate deviance."""
    n, _, e, mu = _frequency_inputs(observed_counts, annual_rates, exposure)
    result = float(poisson_count_deviance(n, mu).sum() / e.sum())
    if not np.isfinite(result):
        raise ValueError("Aggregate deviance must be finite")
    return result


def mean_count_deviance(observed_counts, expected_counts) -> float:
    """sum d(N,mu)/n_policies; a separately labeled per-policy score."""
    result = float(poisson_count_deviance(observed_counts, expected_counts).mean())
    if not np.isfinite(result):
        raise ValueError("Aggregate deviance must be finite")
    return result


def frequency_summary(observed_counts, annual_rates, exposure) -> dict:
    """Portfolio count/rate totals, A/P direction, and two distinct deviances."""
    n, r, e, mu = _frequency_inputs(observed_counts, annual_rates, exposure)
    if not np.isfinite(mu.sum()) or not np.isfinite(n.sum()):
        raise ValueError("Portfolio count totals must be finite")
    return {
        "policies": len(n), "exposure_years": float(e.sum()),
        "observed_claims": int(n.sum()), "expected_claims": float(mu.sum()),
        "observed_annual_rate": float(n.sum() / e.sum()),
        "predicted_annual_rate": float(mu.sum() / e.sum()),
        "actual_over_predicted": float(n.sum() / mu.sum()),
        "poisson_deviance_per_exposure": poisson_deviance_per_exposure(n, r, e),
        "mean_count_deviance": mean_count_deviance(n, mu),
    }


def calibration_table(observed_counts, annual_rates, exposure, n_groups: int = 10) -> pd.DataFrame:
    """Approximate equal-policy groups, ascending annual rate, tied scores intact.

    Each distinct rate block is assigned by its policy-rank midpoint to one of
    n_groups nominal quantiles. Empty nominal groups disappear. A large tie can
    make groups very unequal; no outcome or exposure is used to break a tie.
    The score/rate definition and actual group supports are returned explicitly.
    """
    n, r, e, mu = _frequency_inputs(observed_counts, annual_rates, exposure)
    if isinstance(n_groups, (bool, np.bool_)) or not isinstance(n_groups, (int, np.integer)) or n_groups < 1:
        raise ValueError("n_groups must be a positive integer")
    blocks = pd.DataFrame({"annual_rate": r, "observed_claims": n,
                           "exposure_years": e, "expected_claims": mu})
    blocks = blocks.groupby("annual_rate", sort=True).agg(
        policies=("observed_claims", "size"), observed_claims=("observed_claims", "sum"),
        exposure_years=("exposure_years", "sum"), expected_claims=("expected_claims", "sum")).reset_index()
    midpoint = blocks["policies"].cumsum() - blocks["policies"] / 2
    nominal = np.floor(midpoint.to_numpy() * n_groups / len(n)).astype("int64")
    blocks["group"] = np.minimum(nominal, n_groups - 1) + 1
    result = blocks.groupby("group", sort=True).agg(
        min_annual_rate=("annual_rate", "min"), max_annual_rate=("annual_rate", "max"),
        policies=("policies", "sum"), exposure_years=("exposure_years", "sum"),
        observed_claims=("observed_claims", "sum"), expected_claims=("expected_claims", "sum")).reset_index()
    result["observed_claims"] = result["observed_claims"].astype("int64")
    result["observed_annual_rate"] = result["observed_claims"] / result["exposure_years"]
    result["predicted_annual_rate"] = result["expected_claims"] / result["exposure_years"]
    result["actual_over_predicted"] = result["observed_claims"] / result["expected_claims"]
    result.attrs["grouping"] = "ascending annual rate; policy-rank midpoint quantiles; exact ties indivisible"
    result.attrs["requested_groups"] = n_groups
    result.attrs["actual_groups"] = len(result)
    return result


def concentration_curve(observed_counts, annual_rates, exposure) -> pd.DataFrame:
    """Descending-risk frequency curve with an origin and outcome-independent ties.

    Linear interpolation within each exact tied-rate block yields G=0 for a
    constant score. The curve is undefined when the total observed count is zero.
    """
    n, r, e, _ = _frequency_inputs(observed_counts, annual_rates, exposure)
    if n.sum() == 0:
        raise ValueError("Concentration/Gini is undefined with zero total observed claims")
    blocks = pd.DataFrame({"annual_rate": r, "observed_claims": n, "exposure_years": e})
    blocks = blocks.groupby("annual_rate", sort=True).agg(
        policies=("observed_claims", "size"), observed_claims=("observed_claims", "sum"),
        exposure_years=("exposure_years", "sum")).sort_index(ascending=False).reset_index()
    x = np.r_[0., blocks["exposure_years"].cumsum().to_numpy() / e.sum()]
    y = np.r_[0., blocks["observed_claims"].cumsum().to_numpy() / n.sum()]
    x[-1], y[-1] = 1., 1.  # exact displayed endpoint after floating-point cumulative sums
    result = pd.DataFrame({"cumulative_exposure_fraction": x, "cumulative_claim_fraction": y})
    result.attrs["raw_gini"] = float(2 * np.trapezoid(y, x) - 1)
    result.attrs["ordering"] = "descending annual rate; exact tied-rate blocks"
    result.attrs["normalization"] = "raw G=2*AUC-1; not oracle-normalized"
    return result


def frequency_gini(observed_counts, annual_rates, exposure) -> float:
    """Raw frequency concentration Gini under concentration_curve's convention."""
    return concentration_curve(observed_counts, annual_rates, exposure).attrs["raw_gini"]
