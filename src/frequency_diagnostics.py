"""Development-only frequency diagnostics on explicitly aligned policy keys.

Bootstrap intervals condition on the saved predictions.  They do not include
refitting, selection, unknown household dependence, or simultaneous coverage.
Age standardization describes a fitted model on a common empirical population;
changing recorded age is not identified as a causal intervention by these data.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.data import DataContractError, exact_ids
from src.eda import add_groups, configuration
from src.metrics import (calibration_table, frequency_gini, frequency_summary,
                         poisson_count_deviance)

N, E = "recorded_positive_count", "Exposure"


def _aligned(development: pd.DataFrame, predictions: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Reject incomplete, duplicated, mixed-unit or evaluation predictions."""
    if development.empty or predictions.empty:
        raise DataContractError("Frequency diagnostics require development policies and predictions")
    d = development.copy()
    d["IDpol"] = exact_ids(d["IDpol"])
    if not d.IDpol.is_unique or ("split" in d and not d["split"].eq("development").all()):
        raise DataContractError("Diagnostics accept unique development policies only")
    needed = {"IDpol", "model_id", "annual_rate", "expected_count", "fold", "split"}
    if not needed.issubset(predictions) or predictions[list(needed)].isna().any().any():
        raise DataContractError("Invalid frequency prediction schema/nulls")
    p = predictions.copy()
    p["IDpol"] = exact_ids(p["IDpol"])
    if (not p["split"].eq("development").all()
            or not p.fold.isin([1, 2, 3, 4, 5]).all()
            or p.model_id.astype(str).eq("").any()):
        raise DataContractError("Predictions must identify development folds and models")
    result, fold_reference = [], None
    base = d.drop(columns=["split", "fold"], errors="ignore")
    for model, group in p.groupby("model_id", sort=True):
        if (not group.IDpol.is_unique or set(group.IDpol) != set(d.IDpol)
                or ("run_id" in group and (group.run_id.isna().any() or group.run_id.nunique() != 1))):
            raise DataContractError("Every model needs exactly one prediction per development policy and one run")
        aligned = base.merge(group, on="IDpol", validate="one_to_one").sort_values("IDpol").reset_index(drop=True)
        expected = aligned.annual_rate.to_numpy(dtype=float) * aligned[E].to_numpy(dtype=float)
        if (not np.isfinite(expected).all() or (expected <= 0).any()
                or not np.allclose(aligned.expected_count, expected, rtol=1e-12, atol=0)):
            raise DataContractError("Annual-rate/period-count units or exposure alignment failed")
        # The metric validates nonnegative integral counts and finite positive units.
        frequency_summary(aligned[N], aligned.annual_rate, aligned[E])
        assignment = aligned[["IDpol", "fold"]]
        if fold_reference is not None and not assignment.equals(fold_reference):
            raise DataContractError("Models must use the same policy-fold assignments")
        if "fold" in d:
            requested = d[["IDpol", "fold"]].sort_values("IDpol").reset_index(drop=True)
            if not assignment.equals(requested):
                raise DataContractError("Prediction folds differ from development membership")
        fold_reference = assignment
        result.append((str(model), aligned))
    return result


def score_predictions(development: pd.DataFrame, predictions: pd.DataFrame,
                      support: float = 500) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return model, fold, segment, and annual-rate quantile diagnostics.

    ``pearson_mean_quadratic`` is the mean OOF squared Pearson residual, with
    denominator n policies.  It is a predictive diagnostic, not an in-sample
    residual-df dispersion estimate or a test that counts truly follow Poisson.
    """
    if not np.isfinite(support) or support <= 0:
        raise DataContractError("Segment support must be positive")
    summaries, folds, segments, quantiles = [], [], [], []
    for model, aligned in _aligned(development, predictions):
        n, e, r = (aligned[c].to_numpy() for c in (N, E, "annual_rate"))
        mu = e * r
        summaries.append({"model_id": model, **frequency_summary(n, r, e),
                          "raw_frequency_gini": frequency_gini(n, r, e) if n.sum() else np.nan,
                          "pearson_mean_quadratic": float(np.mean((n-mu)**2/mu))})
        for fold, part in aligned.groupby("fold", sort=True):
            folds.append({"model_id": model, "fold": int(fold),
                          **frequency_summary(part[N], part.annual_rate, part[E])})
        grouped = add_groups(aligned, configuration())
        for column in ["age_band", "bm_band", "VehGas", "Region", "exposure_band", "brand12", "brand_new"]:
            for level, part in grouped.groupby(column, observed=True, sort=True):
                row = frequency_summary(part[N], part.annual_rate, part[E])
                segments.append({"model_id": model, "dimension": column, "level": str(level),
                                 **row, "supported": row["exposure_years"] >= support})
        quantile = calibration_table(n, r, e)
        quantile.insert(0, "model_id", model)
        quantiles.append(quantile)
    return (pd.DataFrame(summaries), pd.DataFrame(folds), pd.DataFrame(segments),
            pd.concat(quantiles, ignore_index=True))


def paired_bootstrap(development: pd.DataFrame, predictions: pd.DataFrame,
                     reference_model: str, replicates: int = 500,
                     seed: int = 20261030) -> pd.DataFrame:
    """Paired whole-policy percentile intervals for candidate-reference deviance.

    All models share exactly the same sampled indices in every draw.  The drawn
    exposure sum normalizes that draw; the full-sample denominator is not reused.
    Negative differences favor the candidate.  The reference row is exactly zero.
    """
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise DataContractError("Bootstrap replicates must be a positive integer")
    aligned = _aligned(development, predictions)
    models = [name for name, _ in aligned]
    if reference_model not in models:
        raise DataContractError("Bootstrap reference model is absent")
    e = aligned[0][1][E].to_numpy(dtype=float)
    losses = np.column_stack([poisson_count_deviance(frame[N], frame.expected_count)
                              for _, frame in aligned])
    delta = losses - losses[:, [models.index(reference_model)]]
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((replicates, len(models)))
    for b in range(replicates):
        indices = rng.integers(0, len(e), size=len(e))
        draws[b] = delta[indices].sum(axis=0) / e[indices].sum()
    bounds = np.percentile(draws, [2.5, 97.5], axis=0)
    return pd.DataFrame({"model_id": models, "reference_model": reference_model,
                         "deviance_difference": delta.sum(axis=0) / e.sum(),
                         "lower": bounds[0], "upper": bounds[1],
                         "replicates": replicates, "seed": seed,
                         "estimand": "fixed OOF predictions; paired policy resampling; candidate minus reference"})


def _support_reference(development: pd.DataFrame, support: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not np.isfinite(support) or support <= 0:
        raise DataContractError("Joint-cell support must be positive")
    d = development.copy()
    if (d.empty or not d.IDpol.is_unique
            or ("split" in d and not d["split"].eq("development").all())):
        raise DataContractError("Age profiles require unique development policies")
    d = d.loc[d.DrivAge.ge(30) & d.DrivAge.lt(60)].copy()
    if d.empty:
        raise DataContractError("No policies in the declared age30-59 reference population")
    for column in [E, N, "DrivAge", "BonusMalus", "VehAge"]:
        if not np.isfinite(d[column].to_numpy(dtype=float)).all():
            raise DataContractError("Nonfinite age-support data")
    if (d[E].le(0).any() or d[N].lt(0).any() or (d[N] % 1 != 0).any()
            or d[["VehGas", "Region"]].isna().any().any()):
        raise DataContractError("Invalid exposure, counts or support categories")
    d["age_support_band"] = pd.cut(d.DrivAge, [30, 40, 50, 60], right=False,
                                      labels=["30–39", "40–49", "50–59"]).astype(str)
    d["bm_support_band"] = pd.cut(d.BonusMalus, [50, 51, 76, 101, 351], right=False,
                                     labels=["50", "51–75", "76–100", "101+"])
    d["vehicle_support_band"] = pd.cut(d.VehAge, [0, 1, 11, 2147483648], right=False,
                                          labels=["0", "1–10", "11+"])
    strata = ["bm_support_band", "vehicle_support_band", "VehGas", "Region"]
    if d[strata].isna().any().any():
        raise DataContractError("A policy is outside the declared common-support bins")
    # Support selection uses covariates and exposure only; claims are bookkeeping.
    totals = d.groupby(strata + ["age_support_band"], observed=True).agg(
        policies=("IDpol", "size"), exposure_years=(E, "sum"), observed_claims=(N, "sum")).reset_index()
    exposure = totals.pivot(index=strata, columns="age_support_band", values="exposure_years")
    exposure = exposure.reindex(columns=["30–39", "40–49", "50–59"]).fillna(0)
    common = exposure.index[exposure.ge(support).all(axis=1)]
    keep = pd.MultiIndex.from_frame(d[strata]).isin(common)
    reference = d.loc[keep].sort_values("IDpol").reset_index(drop=True)
    cells = exposure.stack(future_stack=True).rename("exposure_years").reset_index()
    cells = cells.merge(totals.drop(columns="exposure_years"), on=strata+["age_support_band"], how="left", validate="one_to_one")
    cells[["policies", "observed_claims"]] = cells[["policies", "observed_claims"]].fillna(0).astype("int64")
    cells["common_support"] = pd.MultiIndex.from_frame(cells[strata]).isin(common)
    summary = {"reference_population": "development policies with recorded age30-59",
               "strata": strata, "minimum_exposure_each_age_band": float(support),
               "age_bands": ["30–39", "40–49", "50–59"],
               "bm_band_edges": [50, 51, 76, 101, 351],
               "vehicle_band_edges": [0, 1, 11, 2147483648],
               "common_cells": len(common), "all_cells": len(exposure),
               "eligible_policies": len(d), "eligible_exposure_years": float(d[E].sum()),
               "eligible_claims": int(d[N].sum()),
               "retained_policies": len(reference), "retained_exposure_years": float(reference[E].sum()),
               "retained_claims": int(reference[N].sum()),
               "excluded_policies": int((~keep).sum()),
               "excluded_exposure_years": float(d.loc[~keep, E].sum()),
               "excluded_claims": int(d.loc[~keep, N].sum()),
               "weights": "same retained empirical covariates at every assigned age; exposure normalized within reference",
               "support_limit": "joint coarse BM/vehicle/fuel/Region overlap; no guarantee for exact BM, density, Area, brand or power combinations",
               "uncertainty": "point predictions only; no confidence band, causal identification, refit or selection uncertainty",
               "empty_support": "no profile; rule is not expanded after checking support"}
    return reference, cells, summary


def age_standardization(development: pd.DataFrame, fitted_models: Mapping[str, object],
                        features: Sequence[str], support: float = 100,
                        ages: Sequence[int] = (35, 45, 55)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Average refitted predictions after assigning age35/45/55 on common support.

    The same covariate rows and normalized exposure weights are used at every
    age.  ``relativity_to_first_age`` is the ratio of standardized means; for a
    strictly additive log-age model it also equals exp(delta_eta).  In a model
    with interactions, the average ratio is not generally a single coefficient.
    """
    if (not features or "DrivAge" not in features or len(set(features)) != len(features)
            or not ages or len(set(ages)) != len(ages)
            or any(isinstance(a, bool) or not isinstance(a, (int, np.integer)) or a < 30 or a >= 60 for a in ages)):
        raise DataContractError("Declare unique prospective features and unique ages within30-59")
    forbidden = {"IDpol", N, E, "ClaimNb", "recorded_loss_eur", "ClaimAmount", "fold", "split"}
    if forbidden.intersection(features):
        raise DataContractError("Outcome, identifier or exposure is not an age-profile predictor")
    reference, cells, summary = _support_reference(development, support)
    summary["assigned_ages"] = list(map(int, ages))
    rows = []
    if reference.empty:
        return pd.DataFrame(columns=["model_id", "assigned_age", "standardized_annual_rate", "relativity_to_first_age"]), cells, summary
    weights = reference[E].to_numpy(dtype=float)
    for model_id, estimator in sorted(fitted_models.items()):
        first = None
        for age in ages:
            x = reference.loc[:, list(features)].copy()
            x["DrivAge"] = age
            prediction = np.asarray(estimator.predict(x), dtype=float)
            if prediction.shape != (len(reference),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise DataContractError("Invalid standardized annual predictions")
            mean = float(np.average(prediction, weights=weights))
            first = mean if first is None else first
            rows.append({"model_id": model_id, "assigned_age": int(age),
                         "standardized_annual_rate": mean,
                         "relativity_to_first_age": mean/first,
                         "reference_policies": len(reference),
                         "reference_exposure_years": float(weights.sum())})
    return pd.DataFrame(rows), cells, summary
