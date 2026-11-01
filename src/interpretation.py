"""Development-only two-source-feature tariffs and actuarial diagnostics.

Run ``python -m src.interpretation`` after the supported S1-S5 workflow.
Only development keys are analyzed after reading the shared prepared container;
no evaluation prediction file is read. Registration precedes all fits.
"""
from __future__ import annotations

import json
import argparse
import ast
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

from src.data import (DataContractError, PROJECT_ROOT as ROOT, json_bytes,
                      load_prepared, logical_hash, sha256, text_sha256)
from src.eda import development_only
from src.frequency_models import category_levels, factor_matrix, response, unknown_counts
from src.metrics import poisson_deviance_per_exposure
from src.severity_diagnostics import (_claims, _population, compose_pure_premium,
    paired_policy_bootstrap, score_pure_premium, score_severity)
from src.validation import (identity_id, open_split, prepared_location, protocol,
    read_artifacts, run_oof, write_immutable)

CONFIG = ROOT / "config/interpretation_protocol.json"
N, E, L = "recorded_positive_count", "Exposure", "recorded_loss_eur"


def configuration():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def feature_budget(frequency_features, severity_features):
    origins = set(frequency_features) | set(severity_features)
    if len(origins) != 2 or not origins <= {"DrivAge", "Area", "Density", "Region", "BonusMalus"}:
        raise DataContractError("The complete tariff requires exactly two declared source risk features")
    return sorted(origins)


class TwoFeaturePoisson(RegressorMixin, BaseEstimator):
    """Binned GLM or ridge-penalized additive cubic spline Poisson GAM.

    The GAM penalty is ridge on reduced centered basis coefficients, not S4's
    second differences. Features, knots, centering and levels use fit rows only.
    """
    def __init__(self, second="Area", family="glm"):
        self.second, self.family = second, family

    def frame(self, X):
        feature_budget(["DrivAge", self.second], [])
        if not isinstance(X, pd.DataFrame) or set(X) != {"DrivAge", self.second} or X.empty or X.isna().any().any():
            raise DataContractError("Estimator accepts exactly its two source columns")
        d = X[["DrivAge", self.second]].copy()
        numeric = ["DrivAge"] + ([self.second] if self.second in ("Density", "BonusMalus") else [])
        if not np.isfinite(d[numeric].to_numpy(float)).all():
            raise DataContractError("Nonfinite source features")
        if self.second == "Density":
            if (d.Density <= 0).any():
                raise DataContractError("Density must be positive")
            d["Density"] = np.log(d.Density)
        if self.family == "glm":
            cfg = configuration()["glm"]
            d["DrivAge"] = np.searchsorted(cfg["age_breaks"], d.DrivAge, side="right").astype(str)
            if self.second == "BonusMalus":
                d[self.second] = np.searchsorted(cfg["bm_breaks"], d[self.second], side="right").astype(str)
        return d

    def design(self, frame):
        blocks = [factor_matrix(frame, self.levels_)[0]]
        for c, (center, scale) in self.linear_.items():
            blocks.append(((frame[c].to_numpy(float)-center)/scale)[:, None])
        for c, transformer in self.splines_.items():
            blocks.append(transformer.transform(frame[[c]].to_numpy(float))[:, :-1]-self.centers_[c])
        # PoissonRegressor estimates the unpenalized intercept itself.
        return np.column_stack(blocks)

    def fit(self, X, y, sample_weight):
        if self.family not in ("glm", "gam"):
            raise DataContractError("Unregistered family")
        d = self.frame(X)
        y, w = response(y, sample_weight, len(d))
        factors = (["DrivAge"] if self.family == "glm" else [])
        if self.second in ("Area", "Region") or (self.second == "BonusMalus" and self.family == "glm"):
            factors.append(self.second)
        self.levels_ = category_levels(d, factors)
        self.linear_, self.splines_, self.centers_, self.ranges_ = {}, {}, {}, {}
        for c in [c for c in d if c not in factors]:
            a = d[c].to_numpy(float)
            self.ranges_[c] = [float(a.min()), float(a.max())]
            if self.family == "glm":
                center = float(np.average(a, weights=w))
                scale = float(np.sqrt(np.average((a-center)**2, weights=w))) or 1.
                self.linear_[c] = (center, scale)
            elif np.ptp(a) > 0:
                cfg = configuration()["gam"]
                transformer = SplineTransformer(n_knots=cfg["n_knots"], degree=cfg["degree"],
                    knots="uniform", include_bias=True, extrapolation="constant")
                b = transformer.fit_transform(a[:, None])[:, :-1]
                self.splines_[c], self.centers_[c] = transformer, np.average(b, weights=w, axis=0)
        design = self.design(d)
        if design.shape[1] == 0:
            raise DataContractError("Empty reduced design")
        self.model_ = PoissonRegressor(alpha=configuration()[self.family]["alpha"],
            fit_intercept=True, solver="newton-cholesky", max_iter=100, tol=1e-9)
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.model_.fit(design, y, sample_weight=w)
        return self

    def predict(self, X):
        d = self.frame(X)
        self.last_support_ = {"unknown_categories": unknown_counts(d, self.levels_),
            "outside_training_range": {c: int(((d[c] < a) | (d[c] > b)).sum()) for c, (a, b) in self.ranges_.items()}}
        with threadpool_limits(limits=2):
            return self.model_.predict(self.design(d))


def fold_severity(d, c, membership, run_id):
    """Only claims belonging to training policies contribute to each mean."""
    d = _population(d, membership, "development")
    c = _claims(d, c)
    joined = c.merge(d[["IDpol", "fold"]], on="IDpol", validate="many_to_one")
    pieces, audit = [], []
    for f in range(1, 6):
        training = joined.loc[joined.fold.ne(f)]
        if training.empty:
            raise DataContractError("A fold has no training claims")
        mean = float(training.ClaimAmount.mean())
        p = d.loc[d.fold.eq(f), ["IDpol", "split", "fold"]].copy()
        p["model_id"], p["run_id"], p["severity_eur"] = "severity_mean", run_id, mean
        pieces.append(p)
        audit.append({"fold": f, "training_claims": len(training), "training_loss_eur": float(training.ClaimAmount.sum()),
            "severity_eur": mean, "training_claim_keys_sha256": logical_hash(training[["claim_row_id"]].sort_values("claim_row_id")),
            "validation_policies": len(p), "source_features": []})
    return pd.concat(pieces, ignore_index=True), audit


def groups(d):
    cfg = configuration()["groups"]
    out = d.copy()
    for name, original, edges in [("age", "DrivAge", cfg["age_edges"]),
        ("density", "Density", cfg["density_edges"]), ("bm", "BonusMalus", cfg["bm_edges"]),
        ("vehicle", "VehAge", cfg["vehicle_edges"]), ("duration", E, cfg["duration_edges"])]:
        # Duration endpoints intentionally include .25/.75 in the lower group.
        out[name] = np.searchsorted(edges, out[original], side="left" if name == "duration" else "right").astype(str)
    out["geo"] = out.Area.map({a: g for g, areas in cfg["geo"].items() for a in areas})
    if out.geo.isna().any():
        raise DataContractError("Unregistered Area level")
    return out


def describe_groups(d, c, dimensions):
    _claims(d, c)
    rows = []
    for level, p in d.groupby(dimensions, observed=True, sort=True):
        level = level if isinstance(level, tuple) else (level,)
        amounts = c.loc[c.IDpol.isin(p.IDpol), "ClaimAmount"].to_numpy(float)
        exposure, loss, n = float(p[E].sum()), float(p[L].sum()), int(p[N].sum())
        row = dict(zip(dimensions, map(str, level)))
        row.update(policies=len(p), exposure_years=exposure, claims=n, zero_loss_policies=int(p[N].eq(0).sum()),
            loss_eur=loss, frequency=n/exposure, severity_eur=loss/n if n else None,
            annual_loss_eur=loss/exposure, mean_duration=float(p[E].mean()),
            mean_age=float(np.average(p.DrivAge, weights=p[E])), mean_bm=float(np.average(p.BonusMalus, weights=p[E])),
            mean_vehicle_age=float(np.average(p.VehAge, weights=p[E])),
            diesel_exposure_share=float(p.loc[p.VehGas.eq("Diesel"), E].sum()/exposure),
            brand12_exposure_share=float(p.loc[p.VehBrand.eq("B12"), E].sum()/exposure),
            short_duration_policy_share=float(p[E].le(.25).mean()),
            supported=bool(exposure >= 500 and n >= 100),
            severity_median_eur=float(np.median(amounts)) if n else None,
            severity_p90_eur=float(np.quantile(amounts, .9)) if n else None,
            severity_p99_eur=float(np.quantile(amounts, .99)) if n else None,
            maximum_claim_eur=float(amounts.max()) if n else None)
        for threshold in configuration()["groups"]["tail_thresholds_eur"]:
            tail = amounts >= threshold
            row[f"claims_ge_{threshold}"] = int(tail.sum())
            row[f"loss_share_ge_{threshold}"] = float(amounts[tail].sum()/loss) if loss else None
            row[f"annual_loss_capped_{threshold}"] = float(np.minimum(amounts, threshold).sum()/exposure)
        row["density_min"], row["density_max"] = float(p.Density.min()), float(p.Density.max())
        rows.append(row)
    return pd.DataFrame(rows)


def support_cells(d, duration=False):
    """Fixed covariate/count support; never generate synthetic Area-Density pairs."""
    strata = configuration()["common_support"]["strata"] + (["duration"] if duration else [])
    t = d.groupby([*strata, "geo"], observed=True).agg(exposure=(E, "sum"), claims=(N, "sum"), policies=(N, "size")).reset_index()
    spec = configuration()["common_support"]
    good = t.loc[t.exposure.ge(spec["minimum_exposure_each_geo"]) & t.claims.ge(spec["minimum_claims_each_geo"])]
    keys = good.groupby(strata, observed=True).geo.nunique()
    keys = keys.loc[keys.eq(3)].reset_index()[strata]
    if keys.empty:
        return d.iloc[:0].copy(), t.assign(retained=False), strata
    t = t.merge(keys.assign(retained=True), on=strata, how="left")
    t["retained"] = t.retained.eq(True).astype(bool)
    retained = d.merge(keys, on=strata, validate="many_to_one").sort_values("IDpol").reset_index(drop=True)
    return retained, t, strata


def standardized(d, strata, columns):
    """Ratio-of-sums in cells, then one shared pooled exposure distribution."""
    if d.empty:
        return pd.DataFrame()
    ref = d.groupby(strata, observed=True)[E].sum()
    ref /= ref.sum()
    rows = []
    for g, part in d.groupby("geo", observed=True):
        exposure = part.groupby(strata, observed=True)[E].sum()
        rates = {col: part.groupby(strata, observed=True)[col].sum()/exposure for col in columns}
        rows.append({"geo": g, "policies": len(part), "exposure_years": float(part[E].sum()),
                     **{col: float(value.reindex(ref.index).dot(ref)) for col, value in rates.items()}})
    return pd.DataFrame(rows)


def descriptive_intervals(d, replicates, seed):
    """Shared complete-policy draws; each claim aggregate follows its policy."""
    codes, levels = pd.factorize(d.geo, sort=True)
    e, n, loss = (d[c].to_numpy(float) for c in (E, N, L))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(levels), 3))
    for b in range(replicates):
        i = rng.integers(0, len(d), len(d))
        denominator = np.bincount(codes[i], weights=e[i], minlength=len(levels))
        count = np.bincount(codes[i], weights=n[i], minlength=len(levels))
        total = np.bincount(codes[i], weights=loss[i], minlength=len(levels))
        draws[b] = np.column_stack([count/denominator, total/count, total/denominator])
    rows = []
    for j, g in enumerate(levels):
        for k, metric in enumerate(["frequency", "severity_eur", "annual_loss_eur"]):
            low, high = np.percentile(draws[:, j, k], [2.5, 97.5])
            rows.append({"geo": g, "metric": metric, "lower": low, "upper": high})
    # Direct paired contrast, avoiding the invalid inference from interval overlap.
    ab, ef = list(levels).index("AB"), list(levels).index("EF")
    for k, metric in enumerate(["frequency", "severity_eur", "annual_loss_eur"]):
        low, high = np.percentile(draws[:, ef, k]-draws[:, ab, k], [2.5, 97.5])
        rows.append({"geo": "EF-minus-AB", "metric": metric, "lower": low, "upper": high})
    return pd.DataFrame(rows)


def standardized_intervals(d, strata, columns, replicates, seed):
    """Fixed-support, fixed-reference-mix whole-policy intervals, no refits."""
    if d.empty:
        return pd.DataFrame()
    keys = pd.MultiIndex.from_frame(d[strata])
    cell, levels = pd.factorize(keys, sort=True)
    geo, geos = pd.factorize(d.geo, sort=True)
    k = len(levels)
    combined = geo*k+cell
    exposure = d[E].to_numpy(float)
    values = d[columns].to_numpy(float)
    reference = np.bincount(cell, weights=exposure, minlength=k)
    reference /= reference.sum()
    draws = np.empty((replicates, len(geos), len(columns)+1))
    rng = np.random.default_rng(seed)
    for b in range(replicates):
        i = rng.integers(0, len(d), len(d))
        den = np.bincount(combined[i], weights=exposure[i], minlength=k*len(geos)).reshape(-1, k)
        if (den <= 0).any():
            raise DataContractError("Bootstrap emptied a supported cell")
        for j in range(len(columns)):
            numerator = np.bincount(combined[i], weights=values[i, j], minlength=k*len(geos)).reshape(-1, k)
            draws[b, :, j] = (numerator/den)@reference
        draws[b, :, -1] = draws[b, :, columns.index(L)]/draws[b, :, columns.index(N)]
    rows = []
    labels = [*columns, "standardized_severity_eur"]
    for j, g in enumerate(geos):
        for v, metric in enumerate(labels):
            low, high = np.percentile(draws[:, j, v], [2.5, 97.5])
            rows.append({"geo": g, "metric": metric, "lower": low, "upper": high})
    ab, ef = list(geos).index("AB"), list(geos).index("EF")
    for v, metric in enumerate(labels):
        low, high = np.percentile(draws[:, ef, v]-draws[:, ab, v], [2.5, 97.5])
        rows.append({"geo": "EF-minus-AB", "metric": metric, "lower": low, "upper": high})
    return pd.DataFrame(rows)


def cost_scenarios(q):
    a = np.asarray(q, float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise DataContractError("Annual loss cost must be finite and nonnegative")
    return a+50, a/.85


def load_inputs():
    policies, claims, prepared = load_prepared(prepared_location(None))
    frames, frozen = open_split(ROOT/"data/validation", prepared, protocol())
    d, c = development_only(policies, claims, frames)
    snapshot = json.loads((ROOT/"reports/s5/input_snapshot.json").read_text(encoding="utf-8"))
    if (prepared["source_id"] != snapshot["source_id"] or frozen["split_id"] != snapshot["split_id"]
        or prepared["preparation_id"] != snapshot["preparation_id"]):
        raise DataContractError("S6a inputs differ from the verified S5 source/split bridge")
    return d, c, frames["policy_membership"], prepared, frozen, snapshot


def verified_fit_reuse(previous, current_identity):
    """Explicit reuse after an analysis-only repair, with executed-source proof.

    Compare the exact AST of every local function used in fitting plus all
    dependency, input, software and configuration hashes. No unchecked cache
    globbing or inference from a model name is permitted.
    """
    if previous is None:
        return None
    if Path(previous).name != previous:
        raise DataContractError("Declare a single previous run identity")
    directory = ROOT/"data/interpretation"/previous
    old = json.loads((directory/"registration.json").read_text(encoding="utf-8"))
    source = directory/"executed_source.py"
    if sha256(source) != old["identity"]["source_sha256"]["src/interpretation.py"]:
        raise DataContractError("Previous executed source snapshot hash differs")
    before, after = old["identity"], current_identity
    for key in before:
        if key != "source_sha256" and before[key] != after[key]:
            raise DataContractError(f"Fit reuse identity changed: {key}")
    for path, digest in before["source_sha256"].items():
        if path != "src/interpretation.py" and digest != after["source_sha256"][path]:
            raise DataContractError(f"Fit dependency changed: {path}")
    def fit_ast(text):
        names = {"feature_budget", "TwoFeaturePoisson", "configuration", "fold_severity"}
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names}
    if fit_ast(source.read_text(encoding="utf-8")) != fit_ast(Path(__file__).read_text(encoding="utf-8")):
        raise DataContractError("Estimator/training-claim implementation changed")
    return directory


def register_run(path, registration):
    """UTF-8 roundtrip, with the original cache provenance retained on repeats."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if registration["verified_frequency_cache_origin"] is None:
            registration["verified_frequency_cache_origin"] = previous.get("verified_frequency_cache_origin")
        if previous != registration:
            raise DataContractError("Registration changed")
    else:
        path.write_bytes(json_bytes(registration))
    return registration


def persist_source_snapshot(directory, identity, source_path):
    """Create the registered UTF-8/LF source snapshot once, before any fit."""
    raw = source_path.read_bytes()
    expected = identity["source_sha256"]["src/interpretation.py"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise DataContractError("Execution source changed after registration")
    snapshot = directory/"executed_source.py"
    try:
        with snapshot.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if snapshot.read_bytes() != raw:
            raise DataContractError("Immutable executed source snapshot differs")
    return snapshot


def run(reuse_frequency_from=None):
    started = time.perf_counter()
    cfg = configuration()
    d, c, membership, prepared, frozen, snapshot = load_inputs()
    inputs = {p: sha256(ROOT/p) for p in ["reports/s5/input_snapshot.json", "reports/s5/development_report.json",
        "reports/s5/development_decision.json", "reports/s6/reproduction.json", "reports/s6/reproducibility.json"]}
    identity = {"source_id": prepared["source_id"], "preparation_id": prepared["preparation_id"], "split_id": frozen["split_id"],
        "config_sha256": text_sha256(CONFIG), "source_sha256": {p: text_sha256(ROOT/p) for p in
        ["src/interpretation.py", "src/data.py", "src/validation.py", "src/eda.py", "src/metrics.py", "src/frequency_models.py", "src/severity_diagnostics.py"]},
        "inputs_sha256": inputs, "software": {"python": platform.python_version(), **{p: version(p) for p in
        ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "joblib"]}}}
    run_id = identity_id("s6a", identity)
    directory = ROOT/"data/interpretation"/run_id
    directory.mkdir(parents=True, exist_ok=True)
    registration = {"run_id": run_id, "identity": identity, "specification": cfg,
                    "verified_frequency_cache_origin": reuse_frequency_from}
    rp = directory/"registration.json"
    registration = register_run(rp, registration)
    persist_source_snapshot(directory, identity, Path(__file__))
    reuse_frequency_from = registration["verified_frequency_cache_origin"]
    reuse = verified_fit_reuse(reuse_frequency_from, identity)
    print(f"Registered {run_id}: 40 OOF frequency fits, 5 training claim means", flush=True)
    severity, severity_audit = fold_severity(d, c, membership, run_id)
    all_frequency, fit_audit = [], []
    for second in cfg["pairs"]:
        for family in cfg["families"]:
            model_id = f"two_{second}_{family}"
            path = directory/model_id
            if path.exists():
                frames, audit = read_artifacts(path)
                f = frames["oof_predictions"]
                diagnostics = audit["folds"]
            elif reuse is not None:
                frames, audit = read_artifacts(reuse/model_id)
                f = frames["oof_predictions"].assign(run_id=run_id)
                diagnostics = audit["folds"]
                write_immutable(path, {"oof_predictions": f}, {"run_id": run_id, "folds": diagnostics,
                    "source_features": feature_budget(["DrivAge", second], []),
                    "reused_from": reuse_frequency_from, "original_manifest_sha256": sha256(reuse/model_id/"manifest.json")})
            else:
                f, diagnostics = run_oof(d, membership, lambda: TwoFeaturePoisson(second, family),
                    model_id, run_id, features=("DrivAge", second))
                write_immutable(path, {"oof_predictions": f}, {"run_id": run_id, "folds": diagnostics,
                    "source_features": feature_budget(["DrivAge", second], [])})
            for a in diagnostics:
                train_keys = membership.loc[membership.split.eq("development") & membership.fold.ne(a["fold"]), "IDpol"]
                train, held = d.loc[d.IDpol.isin(train_keys)], d.loc[d.IDpol.isin(f.loc[f.fold.eq(a["fold"]), "IDpol"])]
                a["numeric_support"] = {v: {"training_min": float(train[v].min()), "training_max": float(train[v].max()),
                    "outside_training_range": int((held[v].lt(train[v].min()) | held[v].gt(train[v].max())).sum())}
                    for v in ["DrivAge", *([second] if second in ("Density", "BonusMalus") else [])]}
            all_frequency.append(f)
            fit_audit.extend({"model_id": model_id, **a} for a in diagnostics)
            print(f"Verified {model_id}", flush=True)
    frequency = pd.concat(all_frequency, ignore_index=True)
    pairs = [{"model_id": m, "frequency_model_id": m, "severity_model_id": "severity_mean"} for m in frequency.model_id.unique()]
    pp = compose_pure_premium(d, membership, frequency, severity, pairs, run_id)
    # Verify original immutable development cache, then use only the locked pair.
    report = json.loads((ROOT/"reports/s5/development_report.json").read_text(encoding="utf-8"))
    cache, cache_manifest = read_artifacts(ROOT/"data/severity"/report["run_id"]/"development")
    full = cache["pp_predictions"].loc[lambda x: x.model_id.eq("pp_hgb_mean")].copy()
    if set(full.IDpol) != set(d.IDpol):
        raise DataContractError("Full OOF reference keys differ")
    full["run_id"] = run_id
    pp = pd.concat([pp, full], ignore_index=True)
    print("Scoring keyed all-policy predictions", flush=True)
    scores, segments, folds = score_pure_premium(d, c, pp, membership)
    for m, part in frequency.groupby("model_id"):
        aligned = d[["IDpol", N, E]].merge(part, on="IDpol", validate="one_to_one")
        scores.loc[scores.model_id.eq(m), "frequency_deviance_per_exposure"] = poisson_deviance_per_exposure(aligned[N], aligned.annual_rate, aligned[E])
    ss, _, sf = score_severity(d, c, severity, membership)
    print("Paired fixed-prediction policy intervals", flush=True)
    boot = paired_policy_bootstrap(d, pp, membership, "pp_hgb_mean", replicates=cfg["bootstrap"]["replicates"], seed=cfg["bootstrap"]["seed"])
    best = str(scores.loc[scores.model_id.ne("pp_hgb_mean")].sort_values("tweedie_deviance_per_exposure").iloc[0].model_id)
    boot_best = paired_policy_bootstrap(d, pp, membership, best, replicates=cfg["bootstrap"]["replicates"], seed=cfg["bootstrap"]["seed"])
    gd = groups(d)
    print("Geography, tail and mix diagnostics", flush=True)
    tables = {"scores": scores, "segments": segments, "fold_scores": folds, "bootstrap_full": boot,
        "bootstrap_best": boot_best, "severity_scores": ss, "severity_folds": sf,
        "Area": describe_groups(gd, c, ["Area"]), "geo": describe_groups(gd, c, ["geo"]),
        "Region": describe_groups(gd, c, ["Region"]), "density": describe_groups(gd, c, ["density"]),
        "age": describe_groups(gd, c, ["age"]), "Area_age": describe_groups(gd, c, ["Area", "age"]),
        "geo_duration": describe_groups(gd, c, ["geo", "duration"]),
        "geo_intervals": descriptive_intervals(gd, cfg["bootstrap"]["replicates"], cfg["bootstrap"]["seed"])}
    # Use verified S4 refits for supported associations without new core fits.
    for m in cfg["common_support"]["profile_models"]:
        path = ROOT/"data/frequency"/snapshot["frequency_run_id"]/f"{m}.joblib"
        freport = json.loads((ROOT/"reports/s4/frequency_report.json").read_text(encoding="utf-8"))
        if sha256(path) != freport["development_refits"][m]["sha256"]:
            raise DataContractError("Full profile model checksum mismatch")
        from src.frequency_models import FEATURES
        gd[m] = joblib.load(path).predict(gd[list(FEATURES)])*gd[E]
    for duration in [False, True]:
        print(f"Common support and intervals: duration={duration}", flush=True)
        retained, cells, strata = support_cells(gd, duration)
        tag = "with_duration" if duration else "without_duration"
        tables[f"support_{tag}"] = cells
        tables[f"standardized_{tag}"] = standardized(retained, strata, [N, L, *cfg["common_support"]["profile_models"]])
        tables[f"standardized_intervals_{tag}"] = standardized_intervals(retained, strata,
            [N, L, *cfg["common_support"]["profile_models"]], cfg["bootstrap"]["replicates"], cfg["bootstrap"]["seed"])
    # OOF risk groups use only age and the registered second feature.
    selected = pp.loc[pp.model_id.eq(best), ["IDpol", "expected_period_loss"]]
    costs = gd.merge(selected, on="IDpol", validate="one_to_one")
    second = best.split("_")[1]
    cost_group = {"Area": "Area", "Density": "density", "Region": "Region", "BonusMalus": "bm"}[second]
    t = costs.groupby(["age", cost_group], observed=True).agg(policies=("IDpol", "size"), exposure_years=(E, "sum"), predicted_loss=("expected_period_loss", "sum")).reset_index()
    t["q_eur_per_year"] = t.predicted_loss/t.exposure_years
    t["P_fix"], t["P_var"] = cost_scenarios(t.q_eur_per_year)
    t["supported"] = t.exposure_years.ge(500)
    tables["costs"] = t
    summary = {"run_id": run_id, "identity": identity, "registration_sha256": sha256(rp),
        "verified_frequency_cache_origin": reuse_frequency_from,
        "frequency_fold_audit": fit_audit, "severity_fold_audit": severity_audit,
        "reference_original_run_id": report["run_id"], "reference_cache_manifest_sha256": sha256(ROOT/"data/severity"/report["run_id"]/"development/manifest.json"),
        "best_registered_candidate": best, "scope": cfg["scope"], "elapsed_seconds": time.perf_counter()-started,
        "population": {"policies": len(d), "exposure_years": float(d[E].sum()), "claims": len(c), "loss_eur": float(c.ClaimAmount.sum())}}
    result_dir = directory/"results"
    frames = {**tables, "pp_predictions": pp, "frequency_predictions": frequency, "severity_predictions": severity}
    if result_dir.exists():
        existing, _ = read_artifacts(result_dir)
        for name, frame in frames.items():
            pd.testing.assert_frame_equal(existing[name], frame)
    else:
        print("Writing and roundtrip-checking immutable results", flush=True)
        write_immutable(result_dir, frames, {"run_id": run_id})
    reports = ROOT/"reports/s6a"
    reports.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        (reports/f"{name}.json").write_text(frame.to_json(orient="records", indent=2, force_ascii=False, double_precision=15)+"\n", encoding="utf-8")
    summary["results_manifest_sha256"] = sha256(result_dir/"manifest.json")
    (reports/"report.json").write_bytes(json_bytes(summary))
    render(tables, reports/"figures")
    print(json.dumps({"run_id": run_id, "best": best, "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return summary


def render(t, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    s = t["scores"].sort_values("tweedie_deviance_per_exposure")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.barh(s.model_id, s.tweedie_deviance_per_exposure, color=["#bc6c25" if m == "pp_hgb_mean" else "#317589" for m in s.model_id])
    ax.invert_yaxis(); ax.set_xlabel("Jahresraten-Tweedie-Deviance je Vertragsjahr (p = 1,5)")
    ax.set_title("Entwicklungs-OOF: vier Paare, zwei Spezifikationen")
    fig.savefig(destination/"zwei_merkmale.png", dpi=150); plt.close(fig)
    g = t["geo"].set_index("geo")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, metric, title in zip(axes, ["frequency", "severity_eur", "annual_loss_eur"],
        ["Positive Schadenzeilen / Vertragsjahr", "Schadenmittel (historische EUR)", "Schadenbedarf (EUR / Jahr)"]):
        b = t["geo_intervals"].query("metric == @metric").set_index("geo").reindex(g.index)
        ax.bar(g.index, g[metric], color="#317589")
        ax.errorbar(g.index, g[metric], yerr=np.vstack([g[metric]-b.lower, b.upper-g[metric]]), fmt="none", color="black", capsize=4)
        ax.set_title(title, fontsize=10); ax.set_xlabel("Area-Gruppen am Wohnort")
    fig.suptitle("Beobachtete Muster und bedingte 95%-Vertragsbootstrap-Intervalle")
    fig.savefig(destination/"geografie.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for tag, label in [("without_duration", "Mix ohne Dauer"), ("with_duration", "Mix mit Dauer")]:
        s = t[f"standardized_{tag}"]
        if not s.empty:
            axes[0].plot(s.geo, s[N], marker="o", label=label)
            axes[1].plot(s.geo, s[L], marker="o", label=label)
    axes[0].set_ylabel("Standardisierte Frequenz / Jahr"); axes[1].set_ylabel("Standardisierter Schadenbedarf (EUR / Jahr)")
    for ax in axes:
        ax.set_xlabel("Area-Gruppen; gemeinsame Zellen"); ax.legend()
    fig.suptitle("Deskriptive Standardisierung: unterschiedliche unterstützte Populationen")
    fig.savefig(destination/"mix_dauer.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-frequency-from", help="Explicit source-verified reuse after analysis-only repair")
    run(parser.parse_args().reuse_frequency_from)
