"""Keyed held-out claim and all-policy loss diagnostics for S5.

Claim severity scores use one weight per positive claim row.  Policy loss scores
include zero-loss policies and use the registered exposure-weighted annual-rate
Tweedie criterion at power 1.5.  For varying exposure this criterion is not the
unweighted mean Tweedie deviance of period losses.  Prediction composition never
uses observed claim occurrence, counts, or losses.

Bootstrap draws keep complete policies together.  The intervals condition on
saved predictions and omit refitting, selection and unknown household dependence.
The caller controls release of evaluation data; no function acquires data or fits
a model.  Explicit frozen membership is required before inspecting targets.

Metric references:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data import DataContractError, exact_ids
from src.eda import add_groups, configuration
from src.validation import validate_membership

E, N, L = "Exposure", "recorded_positive_count", "recorded_loss_eur"
META = ["IDpol", "split", "fold", "model_id", "run_id"]
PP_VALUES = ["annual_rate", "severity_eur", "annual_pure_premium", "expected_period_loss"]


def _vector(values, name: str, *, positive: bool = False) -> np.ndarray:
    if np.asarray(values).dtype.kind == "b":
        raise DataContractError(f"Boolean {name} is not a measurement")
    try:
        out = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as error:
        raise DataContractError(f"{name} must be numeric") from error
    if out.ndim != 1 or len(out) == 0 or not np.isfinite(out).all():
        raise DataContractError(f"{name} must be a nonempty finite vector")
    invalid = (out <= 0).any() if positive else (out < 0).any()
    if invalid:
        raise DataContractError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return out


def gamma_deviance_rows(observed_claim, predicted_severity) -> np.ndarray:
    """Unit-weight positive-claim Gamma deviance, stable near equal amounts."""
    y = _vector(observed_claim, "observed claim", positive=True)
    mu = _vector(predicted_severity, "predicted severity", positive=True)
    if y.shape != mu.shape:
        raise DataContractError("Claim observations and predictions must have equal lengths")
    z = np.log(y) - np.log(mu)
    with np.errstate(over="ignore", invalid="ignore"):
        half = np.expm1(z) - z
    near = np.abs(z) < .01
    if near.any():
        # exp(z)-1-z starts at z**2/2; avoid cancellation at a good fit.
        term = z[near]**2/2
        series = term.copy()
        for power in range(3, 15):
            term = term*z[near]/power
            series += term
        half[near] = series
    result = 2*half
    if not np.isfinite(result).all() or (result < 0).any():
        raise DataContractError("Gamma deviance is outside finite nonnegative range")
    return result


def tweedie_deviance_rows(observed, predicted, power: float = 1.5) -> np.ndarray:
    """Registered p=1.5 deviance; zeros retained, positive predictions required.

    d_1.5(y,mu) = 4*(sqrt(y)-sqrt(mu))**2/sqrt(mu).  This algebra avoids
    cancellation in the general Tweedie formula.  A score is not a euro error.
    """
    if isinstance(power, (bool, np.bool_)) or power != 1.5:
        raise DataContractError("This comparison fixes Tweedie power at 1.5")
    y = _vector(observed, "observed loss")
    mu = _vector(predicted, "positive loss prediction", positive=True)
    if y.shape != mu.shape:
        raise DataContractError("Loss observations and predictions must have equal lengths")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        root_mu = np.sqrt(mu)
        result = 4*(np.sqrt(y)-root_mu)**2/root_mu
    if not np.isfinite(result).all() or (result < 0).any():
        raise DataContractError("Tweedie deviance is outside finite nonnegative range")
    return result


def _population(policies: pd.DataFrame, membership: pd.DataFrame, split: str,
                *, targets: bool = True) -> pd.DataFrame:
    """Verify keys before touching targets, so wrong-split rows fail early."""
    if split not in ("development", "evaluation"):
        raise DataContractError("Declare development or evaluation explicitly")
    validate_membership(membership)
    expected = membership.loc[membership["split"].eq(split)].sort_values("IDpol").reset_index(drop=True)
    d = policies.copy()
    d["IDpol"] = exact_ids(d["IDpol"])
    if d.empty or not d.IDpol.is_unique or set(d.IDpol) != set(expected.IDpol):
        raise DataContractError("Diagnostics require exactly the declared frozen split policy IDs")
    if "split" in d and not d["split"].eq(split).all():
        raise DataContractError("Policy split labels disagree with frozen membership")
    if "fold" in d:
        supplied = d[["IDpol", "fold"]].sort_values("IDpol").reset_index(drop=True)
        if not supplied.equals(expected[["IDpol", "fold"]]):
            raise DataContractError("Policy folds disagree with frozen membership")
    d = expected.merge(d.drop(columns=["split", "fold"], errors="ignore"), on="IDpol", validate="one_to_one")
    _vector(d[E], "exposure", positive=True)
    if targets:
        n, loss = _vector(d[N], "policy claim count"), _vector(d[L], "policy loss")
        if (n % 1 != 0).any() or not np.array_equal(n == 0, loss == 0):
            raise DataContractError("Nonintegral counts or inconsistent zero-count/zero-loss policies")
    return d


def _claims(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    if not {"IDpol", "claim_row_id", "ClaimAmount"}.issubset(claims):
        raise DataContractError("Claims require policy ID, unique technical row ID and amount")
    c = claims.copy()
    c["IDpol"] = exact_ids(c["IDpol"])
    if not c.claim_row_id.is_unique or c.claim_row_id.isna().any() or not set(c.IDpol).issubset(set(policies.IDpol)):
        raise DataContractError("Claim rows must be unique and remain within the declared policy split")
    for label in ("split", "fold"):
        if label in c:
            expected = c[["IDpol"]].merge(policies[["IDpol", label]], on="IDpol", validate="many_to_one")
            if not np.array_equal(c[label].to_numpy(), expected[label].to_numpy()):
                raise DataContractError("Claim split/fold labels differ from their frozen policy membership")
    c = c.loc[:, ["IDpol", "claim_row_id", "ClaimAmount"]]
    if not c.empty:
        _vector(c.ClaimAmount, "positive claim amount", positive=True)
    totals = c.groupby("IDpol", sort=False).agg(claims=("ClaimAmount", "size"), loss=("ClaimAmount", "sum"))
    check = policies.set_index("IDpol")[[N, L]].join(totals).fillna(0)
    if (not check[N].eq(check.claims).all()
            or not np.allclose(check[L], check.loss, rtol=1e-12, atol=1e-7)):
        raise DataContractError("Policy counts/losses do not reconcile to retained claim rows")
    return c.sort_values("claim_row_id").reset_index(drop=True)


def _aligned(policies: pd.DataFrame, membership: pd.DataFrame,
             predictions: pd.DataFrame, split: str, kind: str,
             *, targets: bool = True) -> list[tuple[str, pd.DataFrame]]:
    d = _population(policies, membership, split, targets=targets)
    columns = {"severity": ["severity_eur"],
               "frequency": ["annual_rate", "expected_count"], "premium": PP_VALUES}[kind]
    labels = ["frequency_model_id", "severity_model_id"] if kind == "premium" else []
    required = META + columns + labels
    if predictions.empty or not set(required).issubset(predictions) or predictions[required].isna().any().any():
        raise DataContractError("Incomplete/null prediction schema")
    p = predictions.loc[:, list(dict.fromkeys(required))].copy()
    p["IDpol"] = exact_ids(p["IDpol"])
    if not p["split"].eq(split).all() or p.model_id.astype(str).eq("").any() or p.run_id.astype(str).eq("").any():
        raise DataContractError("Predictions need declared split, nonempty model and run identities")
    result, run_reference = [], None
    base = d.drop(columns=["split", "fold"])
    expected = d[["IDpol", "split", "fold"]]
    for model, part in p.groupby("model_id", sort=True):
        if not part.IDpol.is_unique or set(part.IDpol) != set(d.IDpol) or part.run_id.nunique() != 1:
            raise DataContractError("Every model requires one prediction per policy and one run identity")
        run = str(part.run_id.iloc[0])
        if run_reference is not None and run != run_reference:
            raise DataContractError("Model predictions must share one combined run identity")
        run_reference = run
        if any(part[label].nunique() != 1 for label in labels):
            raise DataContractError("A premium model must identify one frequency/severity pair")
        aligned = base.merge(part, on="IDpol", validate="one_to_one").sort_values("IDpol").reset_index(drop=True)
        if not aligned[["IDpol", "split", "fold"]].equals(expected):
            raise DataContractError("Prediction folds/splits differ from frozen membership")
        for column in columns:
            _vector(aligned[column], column, positive=True)
        if kind == "frequency" and not np.allclose(aligned.expected_count, aligned[E]*aligned.annual_rate, rtol=1e-12, atol=0):
            raise DataContractError("Frequency annual-rate/period-count units failed")
        if kind == "premium":
            if (not np.allclose(aligned.annual_pure_premium, aligned.annual_rate*aligned.severity_eur, rtol=1e-12, atol=0)
                    or not np.allclose(aligned.expected_period_loss, aligned[E]*aligned.annual_pure_premium, rtol=1e-12, atol=0)):
                raise DataContractError("Premium must be per-policy rate times severity, then exposure times premium")
        result.append((str(model), aligned))
    return result


def compose_pure_premium(policies: pd.DataFrame, membership: pd.DataFrame,
                         frequency_predictions: pd.DataFrame, severity_predictions: pd.DataFrame,
                         pairs: Sequence[dict], run_id: str,
                         split: str = "development") -> pd.DataFrame:
    """Prospectively compose every policy; observed outcomes are not inputs.

    ``pairs`` contains model_id, frequency_model_id and severity_model_id.  Source
    frequency/severity run identities are retained beside the new combined run.
    The predictive product requires a suitable conditional-mean assumption,
    such as E[S|N,x,e]=N*m(x); count/loss reconciliation does not establish it.
    """
    if not isinstance(run_id, str) or not run_id or not pairs:
        raise DataContractError("Declare premium pairs and a combined run identity")
    frequencies = dict(_aligned(policies, membership, frequency_predictions, split, "frequency", targets=False))
    severities = dict(_aligned(policies, membership, severity_predictions, split, "severity", targets=False))
    if len({p["model_id"] for p in pairs}) != len(pairs):
        raise DataContractError("Premium pair model identities must be unique")
    rows = []
    for pair in pairs:
        frequency_id, severity_id = pair["frequency_model_id"], pair["severity_model_id"]
        if frequency_id not in frequencies or severity_id not in severities or not str(pair["model_id"]):
            raise DataContractError("A registered pair refers to an absent frequency/severity model")
        f, s = frequencies[frequency_id], severities[severity_id]
        piece = f[["IDpol", "split", "fold", E, "annual_rate"]].copy()
        piece = piece.merge(s[["IDpol", "severity_eur"]], on="IDpol", validate="one_to_one")
        piece["frequency_run_id"], piece["severity_run_id"] = f.run_id.iloc[0], s.run_id.iloc[0]
        piece["model_id"], piece["run_id"] = pair["model_id"], run_id
        piece["frequency_model_id"], piece["severity_model_id"] = frequency_id, severity_id
        with np.errstate(over="ignore", invalid="ignore"):
            piece["annual_pure_premium"] = piece.annual_rate*piece.severity_eur
            piece["expected_period_loss"] = piece[E]*piece.annual_pure_premium
        for column in PP_VALUES:
            _vector(piece[column], column, positive=True)
        rows.append(piece.drop(columns=E))
    return pd.concat(rows, ignore_index=True).sort_values(["model_id", "IDpol"]).reset_index(drop=True)


def _claim_summary(observed: np.ndarray, prediction: np.ndarray) -> dict:
    if len(observed) == 0:
        return {"claims": 0, "observed_loss_eur": 0., "predicted_total_for_observed_claims_eur": 0.,
                "observed_claim_mean_eur": None, "predicted_claim_mean_eur": None,
                "claim_actual_over_predicted": None, "mean_gamma_deviance": None,
                "claim_mae_eur": None, "claim_rmse_eur": None}
    residual = observed-prediction
    return {"claims": len(observed), "observed_loss_eur": float(observed.sum()),
            "predicted_total_for_observed_claims_eur": float(prediction.sum()),
            "observed_claim_mean_eur": float(observed.mean()),
            "predicted_claim_mean_eur": float(prediction.mean()),
            "claim_actual_over_predicted": float(observed.sum()/prediction.sum()),
            "mean_gamma_deviance": float(gamma_deviance_rows(observed, prediction).mean()),
            "claim_mae_eur": float(np.mean(np.abs(residual))),
            "claim_rmse_eur": float(np.sqrt(np.mean(residual**2)))}


def score_severity(policies: pd.DataFrame, claims: pd.DataFrame, predictions: pd.DataFrame,
                   membership: pd.DataFrame, split: str = "development",
                   support_claims: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Held-out claim scores/calibration; no exposure weights on claim amounts."""
    if isinstance(support_claims, bool) or not isinstance(support_claims, (int, np.integer)) or support_claims < 1:
        raise DataContractError("Claim support threshold must be a positive integer")
    d = _population(policies, membership, split)
    c = _claims(d, claims)
    summaries, segments, folds = [], [], []
    for model, aligned in _aligned(policies, membership, predictions, split, "severity"):
        grouped = add_groups(aligned, configuration())
        joined = c.merge(grouped, on="IDpol", validate="many_to_one")
        summary = _claim_summary(joined.ClaimAmount.to_numpy(), joined.severity_eur.to_numpy())
        summaries.append({"model_id": model, "run_id": aligned.run_id.iloc[0], "split": split,
                          "policies_predicted": len(aligned), **summary})
        for fold in sorted(aligned.fold.unique()):
            part = joined.loc[joined.fold.eq(fold)]
            folds.append({"model_id": model, "fold": int(fold), "split": split,
                          **_claim_summary(part.ClaimAmount.to_numpy(), part.severity_eur.to_numpy())})
        for dimension in ["age_band", "bm_band", "VehGas", "Region"]:
            for level in sorted(grouped[dimension].unique()):
                part = joined.loc[joined[dimension].eq(level)]
                row = _claim_summary(part.ClaimAmount.to_numpy(), part.severity_eur.to_numpy())
                segments.append({"model_id": model, "split": split, "dimension": dimension,
                                 "level": str(level), **row, "supported_claims": len(part) >= support_claims})
    return pd.DataFrame(summaries), pd.DataFrame(segments), pd.DataFrame(folds)


def _loss_summary(part: pd.DataFrame, power: float) -> dict:
    e, loss, cost = (part[c].to_numpy(dtype=float) for c in (E, L, "expected_period_loss"))
    rate = part.annual_pure_premium.to_numpy(dtype=float)
    observed_rate = loss/e
    annual_score = tweedie_deviance_rows(observed_rate, rate, power)
    period_score = tweedie_deviance_rows(loss, cost, power)
    expected_claims = float(np.dot(e, part.annual_rate))
    mean_frequency = expected_claims/e.sum()
    mean_severity = float(np.average(part.severity_eur, weights=e))
    return {"policies": len(part), "zero_loss_policies": int((loss == 0).sum()),
            "observed_claims": int(part[N].sum()), "exposure_years": float(e.sum()),
            "observed_loss_eur": float(loss.sum()), "expected_loss_eur": float(cost.sum()),
            "observed_annual_loss_cost": float(loss.sum()/e.sum()),
            "predicted_annual_loss_cost": float(cost.sum()/e.sum()),
            "actual_over_predicted": float(loss.sum()/cost.sum()),
            "tweedie_power": power,
            "tweedie_deviance_per_exposure": float(np.average(annual_score, weights=e)),
            "mean_period_tweedie_deviance": float(period_score.mean()),
            "loss_mae_period_eur": float(np.mean(np.abs(loss-cost))),
            "loss_rmse_period_eur": float(np.sqrt(np.mean((loss-cost)**2))),
            "expected_claims": expected_claims,
            "exposure_weighted_mean_annual_frequency": float(mean_frequency),
            "exposure_weighted_mean_severity_eur": mean_severity,
            "product_of_exposure_weighted_component_means": float(mean_frequency*mean_severity),
            "predicted_claim_weighted_mean_severity_eur": float(cost.sum()/expected_claims)}


def score_pure_premium(policies: pd.DataFrame, claims: pd.DataFrame, predictions: pd.DataFrame,
                       membership: pd.DataFrame, split: str = "development", power: float = 1.5,
                       support_exposure: float = 500) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All-policy loss scores and ratio-of-sums calibration, with zeros retained."""
    if not np.isfinite(support_exposure) or support_exposure <= 0:
        raise DataContractError("Loss calibration exposure support must be positive")
    d = _population(policies, membership, split)
    _claims(d, claims)
    summaries, segments, folds = [], [], []
    for model, aligned in _aligned(policies, membership, predictions, split, "premium"):
        meta = {"model_id": model, "split": split, "run_id": aligned.run_id.iloc[0],
                "frequency_model_id": aligned.frequency_model_id.iloc[0],
                "severity_model_id": aligned.severity_model_id.iloc[0]}
        summaries.append({**meta, **_loss_summary(aligned, power)})
        for fold, part in aligned.groupby("fold", sort=True):
            folds.append({**meta, "fold": int(fold), **_loss_summary(part, power)})
        grouped = add_groups(aligned, configuration())
        for dimension in ["age_band", "bm_band", "VehGas", "Region", "exposure_band", "brand12"]:
            for level, part in grouped.groupby(dimension, observed=True, sort=True):
                row = _loss_summary(part, power)
                segments.append({**meta, "dimension": dimension, "level": str(level), **row,
                                 "supported_exposure": row["exposure_years"] >= support_exposure})
    return pd.DataFrame(summaries), pd.DataFrame(segments), pd.DataFrame(folds)


def paired_policy_bootstrap(policies: pd.DataFrame, predictions: pd.DataFrame,
                            membership: pd.DataFrame, reference_model: str,
                            split: str = "development", replicates: int = 500,
                            seed: int = 20261101, power: float = 1.5) -> pd.DataFrame:
    """Fixed-prediction paired policy intervals for loss score and loss A/P.

    Each policy's aggregate loss carries all its claim rows together.  There is
    no independent resampling of claims or removal of zero-loss policies.  A/P
    intervals are also returned per model; A/P differences are candidate minus
    reference and are not differences of absolute calibration errors.
    """
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise DataContractError("Bootstrap replicates must be a positive integer")
    aligned = _aligned(policies, membership, predictions, split, "premium")
    names = [name for name, _ in aligned]
    if reference_model not in names:
        raise DataContractError("Bootstrap reference premium model is absent")
    reference = names.index(reference_model)
    e, loss = (aligned[0][1][c].to_numpy(dtype=float) for c in (E, L))
    cost = np.column_stack([frame.expected_period_loss for _, frame in aligned])
    contributions = np.column_stack([e*tweedie_deviance_rows(loss/e, frame.annual_pure_premium, power)
                                     for _, frame in aligned])
    differences = contributions-contributions[:, [reference]]
    rng = np.random.Generator(np.random.PCG64(seed))
    score_draws = np.empty((replicates, len(names)))
    calibration_draws = np.empty_like(score_draws)
    for b in range(replicates):
        indices = rng.integers(0, len(e), size=len(e))
        score_draws[b] = differences[indices].sum(axis=0)/e[indices].sum()
        calibration_draws[b] = loss[indices].sum()/cost[indices].sum(axis=0)
    calibration_differences = calibration_draws-calibration_draws[:, [reference]]
    score_bounds = np.percentile(score_draws, [2.5, 97.5], axis=0)
    ap_bounds = np.percentile(calibration_draws, [2.5, 97.5], axis=0)
    ap_difference_bounds = np.percentile(calibration_differences, [2.5, 97.5], axis=0)
    actual_over_predicted = loss.sum()/cost.sum(axis=0)
    return pd.DataFrame({"model_id": names, "reference_model": reference_model, "split": split,
                         "deviance_difference": differences.sum(axis=0)/e.sum(),
                         "lower": score_bounds[0], "upper": score_bounds[1],
                         "actual_over_predicted": actual_over_predicted,
                         "actual_over_predicted_lower": ap_bounds[0], "actual_over_predicted_upper": ap_bounds[1],
                         "actual_over_predicted_difference": actual_over_predicted-actual_over_predicted[reference],
                         "actual_over_predicted_difference_lower": ap_difference_bounds[0],
                         "actual_over_predicted_difference_upper": ap_difference_bounds[1],
                         "tweedie_power": power, "replicates": replicates, "seed": seed,
                         "estimand": "fixed policy predictions; paired whole policies including zeros; candidate minus reference"})
