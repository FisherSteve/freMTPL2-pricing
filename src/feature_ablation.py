"""Exploratory S6a feature ablation on frozen development folds only.

python -m src.feature_ablation; original S6a artifacts remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

from src.data import DataContractError, json_bytes, text_sha256, sha256
from src.frequency_models import category_levels, response
from src.interpretation import (ROOT, E, N, TwoFeaturePoisson, configuration,
                                load_inputs, fold_severity)
from src.metrics import poisson_deviance_per_exposure
from src.severity_diagnostics import (compose_pure_premium, paired_policy_bootstrap,
                                     score_pure_premium)
from src.validation import identity_id, read_artifacts, run_oof, write_immutable

CONFIG = ROOT / "config/feature_ablation_protocol.json"


class FeaturePoisson(TwoFeaturePoisson):
    """Same reduced-model construction for one or two explicit source features."""
    def __init__(self, features=("BonusMalus", "Area"), family="gam"):
        self.features, self.family = features, family

    def frame(self, X):
        if (not 1 <= len(self.features) <= 2 or len(set(self.features)) != len(self.features)
                or not set(self.features) <= {"DrivAge", "BonusMalus", "Area"}
                or not isinstance(X, pd.DataFrame) or set(X) != set(self.features)
                or X.empty or X.isna().any().any()):
            raise DataContractError("Exactly the declared one or two source features required")
        d = X[list(self.features)].copy()
        for col in set(self.features) - {"Area"}:
            if not np.isfinite(d[col].to_numpy(float)).all():
                raise DataContractError("Nonfinite source feature")
            if self.family == "glm":
                key = "age_breaks" if col == "DrivAge" else "bm_breaks"
                d[col] = np.searchsorted(configuration()["glm"][key], d[col], side="right").astype(str)
        return d

    def fit(self, X, y, sample_weight):
        if self.family not in ("glm", "gam"):
            raise DataContractError("Unregistered family")
        d = self.frame(X)
        y, w = response(y, sample_weight, len(d))
        factors = list(d) if self.family == "glm" else [c for c in d if c == "Area"]
        self.levels_ = category_levels(d, factors)
        self.linear_, self.splines_, self.centers_, self.ranges_ = {}, {}, {}, {}
        for col in [c for c in d if c not in factors]:
            a = d[col].to_numpy(float)
            self.ranges_[col] = [float(a.min()), float(a.max())]
            if np.ptp(a) > 0:
                cfg = configuration()["gam"]
                transformer = SplineTransformer(n_knots=cfg["n_knots"], degree=cfg["degree"],
                    knots="uniform", include_bias=True, extrapolation="constant")
                b = transformer.fit_transform(a[:, None])[:, :-1]
                self.splines_[col], self.centers_[col] = transformer, np.average(b, weights=w, axis=0)
        design = self.design(d)
        # Degenerate single-feature training samples still need an intercept.
        self.constant_design_ = design.shape[1] == 0
        if self.constant_design_:
            design = np.zeros((len(d), 1))
        self.model_ = PoissonRegressor(alpha=configuration()[self.family]["alpha"],
            fit_intercept=True, solver="newton-cholesky", max_iter=100, tol=1e-9)
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.model_.fit(design, y, sample_weight=w)
        return self

    def predict(self, X):
        if self.constant_design_:
            return self.model_.predict(np.zeros((len(self.frame(X)), 1)))
        return super().predict(X)


def weighted_corr(a, b, w):
    a, b, w = map(lambda x: np.asarray(x, dtype=float), (a, b, w))
    a, b = a-np.average(a, weights=w), b-np.average(b, weights=w)
    denominator = np.sqrt(np.dot(w, a*a)*np.dot(w, b*b))
    return float(np.dot(w, a*b)/denominator) if denominator else None


def dependence(d):
    a, b, w = d.DrivAge, d.BonusMalus, d[E]
    out = {"policies": len(d), "exposure_years": float(w.sum()),
        "pearson": weighted_corr(a, b, np.ones(len(d))),
        "exposure_weighted_pearson": weighted_corr(a, b, w),
        "spearman": weighted_corr(a.rank(), b.rank(), np.ones(len(d))),
        "exposure_weighted_midrank_correlation": weighted_corr(a.rank(), b.rank(), w)}
    # Midranks are unweighted ranks; weighting is applied to their covariance.
    out["rank_definition"] = "Average unweighted ranks for ties; exposure weights applied to rank covariance"
    for col in ("DrivAge", "BonusMalus"):
        mean = np.average(d[col], weights=w)
        between = sum(part[E].sum()*(np.average(part[col], weights=part[E])-mean)**2
                      for _, part in d.groupby("Area", observed=True))
        out[f"Area_{col}_weighted_eta_squared"] = float(between/np.dot(w, (d[col]-mean)**2))
    bands = pd.cut(a, [0, 25, 35, 50, 65, np.inf], right=False).astype(str)
    out["age_bands"] = [{"age_band": name, "policies": len(part),
        "exposure_years": float(part[E].sum()),
        "weighted_bm_mean": float(np.average(part.BonusMalus, weights=part[E])),
        "weighted_bm50_share": float(np.average(part.BonusMalus.eq(50), weights=part[E]))}
        for name, part in d.groupby(bands, observed=True)]
    return out


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    d, c, membership, prepared, frozen, _ = load_inputs()
    old = ROOT/"data/interpretation"/cfg["old_reference_run"]
    original, original_manifest = read_artifacts(old/"results")
    identity = {"config_sha256": sha256(CONFIG), "preparation_id": prepared["preparation_id"],
        "split_id": frozen["split_id"], "source_id": prepared["source_id"],
        "source_sha256": {p: text_sha256(ROOT/p) for p in ["src/feature_ablation.py", "src/interpretation.py",
            "src/validation.py", "src/frequency_models.py", "src/severity_diagnostics.py", "src/metrics.py", "src/data.py", "src/eda.py"]},
        "original_manifest_sha256": sha256(old/"results/manifest.json"),
        "original_registration_sha256": sha256(old/"registration.json"),
        "software": json.loads((old/"registration.json").read_text(encoding="utf-8"))["identity"]["software"]}
    # Actual runtime versions must match the original before comparing caches.
    import platform
    from importlib.metadata import version
    actual = {"python": platform.python_version(), **{p: version(p) for p in identity["software"] if p != "python"}}
    if actual != identity["software"]:
        raise DataContractError("Software differs from original S6a")
    run_id = identity_id("s6aablation", identity)
    directory = ROOT/"data/interpretation"/run_id
    directory.mkdir(parents=True, exist_ok=True)
    registration = {"run_id": run_id, "identity": identity, "specification": cfg}
    reg = directory/"registration.json"
    raw = json_bytes(registration)
    if reg.exists():
        if reg.read_bytes() != raw:
            raise DataContractError("Registration changed")
    else:
        reg.write_bytes(raw)
    source = directory/"executed_source.py"
    if source.exists() and source.read_bytes() != Path(__file__).read_bytes():
        raise DataContractError("Source snapshot changed")
    if not source.exists():
        source.write_bytes(Path(__file__).read_bytes())
    print(f"Registered {run_id}: 40 new frequency fits", flush=True)
    severity, severity_audit = fold_severity(d, c, membership, run_id)
    frequencies, audits = [], []
    for features in cfg["new_feature_sets"]:
        for family in cfg["families"]:
            model = "abl_"+"_".join(features)+"_"+family
            path = directory/model
            if path.exists():
                frames, audit = read_artifacts(path)
                f, diagnostics = frames["oof_predictions"], audit["folds"]
            else:
                f, diagnostics = run_oof(d, membership, lambda: FeaturePoisson(tuple(features), family),
                    model, run_id, features=tuple(features))
                write_immutable(path, {"oof_predictions": f}, {"run_id": run_id,
                    "folds": diagnostics, "source_features": features})
            frequencies.append(f)
            audits.extend({"model_id": model, **a} for a in diagnostics)
            print(f"Verified {model}", flush=True)
    f = pd.concat(frequencies, ignore_index=True)
    pairs = [{"model_id": m, "frequency_model_id": m, "severity_model_id": "severity_mean"} for m in f.model_id.unique()]
    pp = compose_pure_premium(d, membership, f, severity, pairs, run_id)
    # Use verified original products, with scientific values unchanged.
    pp = pd.concat([pp, original["pp_predictions"].assign(run_id=run_id)], ignore_index=True)
    scores, segments, folds = score_pure_premium(d, c, pp, membership)
    for name, part in pp.groupby("model_id"):
        aligned = d[["IDpol", N, E]].merge(part, on="IDpol", validate="one_to_one")
        scores.loc[scores.model_id.eq(name), "frequency_deviance_per_exposure"] = poisson_deviance_per_exposure(aligned[N], aligned.annual_rate, aligned[E])
    reduced = scores.loc[scores.model_id.ne("pp_hgb_mean")]
    best = str(reduced.sort_values("tweedie_deviance_per_exposure").iloc[0].model_id)
    tables = {"scores": scores, "fold_scores": folds, "segments": segments}
    for reference in ["two_BonusMalus_gam", best]:
        print(f"Paired bootstrap versus {reference}", flush=True)
        tables["bootstrap_"+reference] = paired_policy_bootstrap(d, pp, membership, reference,
            replicates=cfg["bootstrap"]["replicates"], seed=cfg["bootstrap"]["seed"])
    manifest = write_immutable(directory/"results", {"pp_predictions": pp, **tables}, {"run_id": run_id})
    target = ROOT/"reports/s6a/ablation"
    target.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (target/f"{name}.json").write_text(table.to_json(orient="records", indent=2, force_ascii=False, double_precision=15)+"\n", encoding="utf-8", newline="\n")
    (target/"dependence.json").write_bytes(json_bytes(dependence(d)))
    report = {**registration, "best_expanded_candidate": best, "new_frequency_fits": 40,
        "severity_audit": severity_audit, "frequency_audit": audits,
        "results_manifest_sha256": sha256(directory/"results/manifest.json"),
        "files_sha256": {p.name: sha256(p) for p in target.glob("*.json") if p.name != "report.json"}}
    (target/"report.json").write_bytes(json_bytes(report))
    print(scores[["model_id", "tweedie_deviance_per_exposure", "frequency_deviance_per_exposure"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
