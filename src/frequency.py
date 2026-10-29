"""Registered, bounded frequency comparison; all fitting/scoring is development-only."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data import (PROJECT_ROOT, DataContractError, json_bytes, load_prepared,
                      logical_hash, root_path, sha256, text_sha256)
from src.eda import development_only
from src.metrics import poisson_deviance_per_exposure, concentration_curve
from src.validation import (FEATURES, MEMBERSHIP_COLUMNS, ExposureIntercept,
                            align_predictions, identity_id, open_split,
                            prepared_location, protocol, read_artifacts,
                            write_immutable)

CONFIG = PROJECT_ROOT / "config/frequency_protocol.json"
CODE = ["src/frequency.py", "src/frequency_models.py", "src/frequency_diagnostics.py",
        "src/metrics.py", "src/validation.py", "src/eda.py", "config/eda_protocol.json"]


def configuration() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    configs = cfg["configurations"]
    ids = [c["model_id"] for c in configs]
    budget = protocol()["candidate_budget"]
    if len(set(ids)) != len(ids):
        raise DataContractError("Duplicate registered configuration IDs")
    for family in ("glm", "gam", "histogram_poisson"):
        if not 1 <= sum(c["family"] == family for c in configs) <= budget[family]["maximum_configurations"]:
            raise DataContractError("Registered family missing or over frozen search budget")
    if sum(c["family"] == "intercept" for c in configs) != 1:
        raise DataContractError("Exactly one intercept is required")
    if len({c["encoding"] for c in configs if c["family"] == "glm"}) > 2:
        raise DataContractError("At most two registered GLM encodings")
    if set(c["family"] for c in configs) != {"intercept", "glm", "gam", "histogram_poisson"}:
        raise DataContractError("Unknown configuration family")
    for c in configs:
        if c["family"] == "glm" and c["alpha"] not in budget["glm"]["regularization_levels"]:
            raise DataContractError("Unregistered GLM penalty")
        if c["family"] == "gam" and (c["n_splines"] not in budget["gam"]["spline_basis_sizes"]
                                         or c["lam"] not in budget["gam"]["smoothing_penalties"]):
            raise DataContractError("Unregistered GAM basis/penalty")
        if c["family"] == "histogram_poisson" and (c["max_leaf_nodes"] not in budget["histogram_poisson"]["max_leaf_nodes"]
                    or c["l2_regularization"] not in budget["histogram_poisson"]["l2_regularization"]):
            raise DataContractError("Unregistered boosting configuration")
    if cfg["histogram_poisson"]["early_stopping"] is not False:
        raise DataContractError("Automatic early stopping violates the frozen boundary")
    for key in ("learning_rate", "max_iter", "early_stopping", "random_state"):
        if cfg["histogram_poisson"][key] != budget["histogram_poisson"][key]:
            raise DataContractError("Fixed boosting budget differs from frozen validation protocol")
    if cfg["histogram_poisson"]["min_samples_leaf"] != 20:
        raise DataContractError("This implementation fixes boosting minimum leaf size at 20")
    return cfg


def estimator(config: dict):
    from src.frequency_models import make_estimator
    if config["family"] == "intercept":
        return ExposureIntercept()
    params = {k: v for k, v in config.items() if k not in ("family", "model_id")}
    if config["family"] == "histogram_poisson":
        fixed = configuration()["histogram_poisson"]
        params.update({key: fixed[key] for key in ("learning_rate", "max_iter", "random_state")})
    return make_estimator({"family": config["family"], "parameters": params})


def fit_oof(development: pd.DataFrame, membership: pd.DataFrame,
            config: dict, run_id: str) -> tuple[pd.DataFrame, list, list]:
    expected = membership.loc[membership.split.eq("development")]
    if (not development.IDpol.is_unique or set(development.IDpol) != set(expected.IDpol)
            or "fold" in development or "split" in development):
        raise DataContractError("Frequency fitting requires exactly development IDs, no input fold predictors")
    d = expected.merge(development, on="IDpol", validate="one_to_one").sort_values("IDpol")
    pieces, evidence, timing = [], [], []
    for fold in range(1, 6):
        train, valid = d.loc[d.fold.ne(fold)], d.loc[d.fold.eq(fold)]
        model = estimator(config)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train[list(FEATURES)], train.recorded_positive_count / train.Exposure,
                      sample_weight=train.Exposure.to_numpy())
            rate = np.asarray(model.predict(valid[list(FEATURES)]), dtype=float)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise DataContractError(f"Unresolved convergence: {config['model_id']} fold {fold}")
        part = valid[list(MEMBERSHIP_COLUMNS)].copy()
        part["model_id"], part["run_id"] = config["model_id"], run_id
        part["annual_rate"], part["expected_count"] = rate, rate * valid.Exposure.to_numpy()
        pieces.append(part)
        evidence.append({"fold": fold, "training_policies": len(train), "validation_policies": len(valid),
            "training_id_sha256": logical_hash(train[["IDpol"]]),
            "validation_id_sha256": logical_hash(valid[["IDpol"]]),
            "training_exposure_years": float(train.Exposure.sum()), "training_claims": int(train.recorded_positive_count.sum()),
            "validation_exposure_years": float(valid.Exposure.sum()), "validation_claims": int(valid.recorded_positive_count.sum()),
            "deviance_per_exposure": poisson_deviance_per_exposure(valid.recorded_positive_count, rate, valid.Exposure),
            "unknown_validation_categories": {c: int((~valid[c].isin(train[c])).sum()) for c in ("VehPower", "VehBrand", "VehGas", "Area", "Region")},
            "fit": getattr(model, "diagnostics_", {"rate": float(model.rate_)} if isinstance(model, ExposureIntercept) else {}),
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]})
        timing.append({"fold": fold, "seconds": time.perf_counter()-started})
        print(f"{config['model_id']} fold {fold}: {evidence[-1]['deviance_per_exposure']:.8f}, {timing[-1]['seconds']:.1f}s", flush=True)
    aligned = align_predictions(pd.concat(pieces, ignore_index=True), membership,
                                development[["IDpol", "Exposure"]], config["model_id"], run_id)
    return aligned, evidence, timing


def choose_models(scores: pd.DataFrame, configs: list) -> tuple[dict, str]:
    """Pick empirical family minima; practical override must be a recorded decision."""
    mapping = {c["model_id"]: c["family"] for c in configs}
    if set(scores.model_id) != set(mapping) or not scores.model_id.is_unique:
        raise DataContractError("Scores do not cover the registered candidate set")
    score = scores.set_index("model_id")["poisson_deviance_per_exposure"]
    if not np.isfinite(score).all():
        raise DataContractError("Nonfinite model-selection score")
    winners = {family: min((m for m in score.index if mapping[m] == family), key=lambda m: (score[m], m))
               for family in ("glm", "gam", "histogram_poisson")}
    return winners, str(score.idxmin())


def dependence(d: pd.DataFrame) -> dict:
    weights = d.Exposure.to_numpy()
    a = np.column_stack([d.DrivAge, d.BonusMalus, d.VehAge, np.log(d.Density)])
    centered = a - np.average(a, axis=0, weights=weights)
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    scale = np.sqrt(np.diag(covariance))
    correlations = covariance / np.outer(scale, scale)
    result = {"variables": ["DrivAge", "BonusMalus", "VehAge", "logDensity"],
              "exposure_weighted_pearson_correlation": correlations.tolist(),
              "scope": "Deskriptive Kovariatenabhängigkeit, keine Kausalität und kein vollständiger Konkurvitätstest"}
    temp = d.assign(z=np.log(d.Density), wz=d.Exposure*np.log(d.Density))
    total_mean = float(temp.wz.sum()/temp.Exposure.sum())
    denominator = float(np.sum(weights*(temp.z.to_numpy()-total_mean)**2))
    for feature in ("Area", "Region"):
        groups = temp.groupby(feature, observed=True).agg(w=("Exposure", "sum"), wz=("wz", "sum"))
        result[f"logDensity_between_{feature}_variance_fraction"] = float(np.sum(groups.w*(groups.wz/groups.w-total_mean)**2)/denominator)
    return result


def effect_profiles(d: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Fixed-reference relativities, each variable and support explicitly named."""
    anchor = {}
    for c in FEATURES:
        if c in ("VehPower", "VehBrand", "VehGas", "Area", "Region"):
            anchor[c] = d.groupby(c, observed=True).Exposure.sum().idxmax()
        else:
            anchor[c] = float(d[c].median())
    rows = []
    for feature, grid in {"DrivAge": list(range(18, 91)), "VehAge": list(range(0, 41)),
                           "BonusMalus": list(range(50, 151, 5)),
                           "VehBrand": sorted(d.VehBrand.unique()), "VehGas": sorted(d.VehGas.unique())}.items():
        frame = pd.DataFrame([dict(anchor, **{feature: value}) for value in grid])[list(FEATURES)]
        reference = pd.DataFrame([anchor])[list(FEATURES)]
        for model_id, model in models.items():
            base = float(model.predict(reference)[0])
            rates = model.predict(frame)
            for value, rate in zip(grid, rates):
                part = d.loc[d[feature].eq(value)]
                rows.append({"model_id": model_id, "feature": feature, "value": str(value),
                             "annual_rate": float(rate), "reference_rate": base,
                             "relativity": float(rate/base), "policies_at_exact_value": len(part),
                             "exposure_at_exact_value": float(part.Exposure.sum()),
                             "anchor": json.dumps(anchor, ensure_ascii=False, default=str, sort_keys=True),
                             "interpretation": "Illustrative fixed-reference profile; marginal support is not joint support. Use supported standardization for adjusted age comparison."})
    return pd.DataFrame(rows)


def concentration_points(d: pd.DataFrame, predictions: pd.DataFrame, models: list) -> pd.DataFrame:
    rows = []
    for model_id in models:
        p = predictions.loc[predictions.model_id.eq(model_id)]
        aligned = d[["IDpol", "Exposure", "recorded_positive_count"]].merge(p, on="IDpol", validate="one_to_one")
        curve = concentration_curve(aligned.recorded_positive_count, aligned.annual_rate, aligned.Exposure)
        x = np.linspace(0, 1, 101)
        y = np.interp(x, curve.cumulative_exposure_fraction, curve.cumulative_claim_fraction)
        rows.extend({"model_id": model_id, "cumulative_exposure_fraction": float(a),
                     "cumulative_claim_fraction": float(b), "exact_raw_gini": curve.attrs["raw_gini"]} for a, b in zip(x, y))
    return pd.DataFrame(rows)


def table_json(frame: pd.DataFrame) -> bytes:
    return json_bytes(json.loads(frame.to_json(orient="records", double_precision=15)))


def figures(tables: dict, destination: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    def save(fig, name):
        path = destination / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, metadata={"Software": "freMTPL2 frequency comparison"})
        plt.close(fig)
        paths[name] = {"filename": path.name, "sha256": sha256(path)}
    score = tables["scores"].sort_values("poisson_deviance_per_exposure")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(score.model_id, score.poisson_deviance_per_exposure)
    ax.set(xlabel="OOF-Poisson-Deviance je Vertragsjahr (kleiner ist besser)", title="Frequenzvergleich im Entwicklungsbestand")
    save(fig, "modellvergleich")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, feature, label in zip(axes, ["DrivAge", "VehAge", "BonusMalus"], ["Fahreralter", "Fahrzeugalter", "Bonus-Malus"]):
        part = tables["effects"].loc[tables["effects"].feature.eq(feature)]
        for model_id, group in part.groupby("model_id", sort=False):
            ax.plot(group.value.astype(float), group.relativity, label=model_id)
        ax.axhline(1, color="grey", lw=.7)
        ax.set(xlabel=label, ylabel="Relative Jahresfrequenz", title="Feste Referenzmerkmale; keine Kausalwirkung")
    axes[0].legend(fontsize=7)
    save(fig, "relativitaeten")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model_id, group in tables["calibration"].groupby("model_id", sort=False):
        axes[0].plot(group.predicted_annual_rate, group.observed_annual_rate, marker="o", ms=3, label=model_id)
    axes[0].plot([0, .3], [0, .3], color="grey", lw=.7)
    axes[0].set(xlabel="Vorhergesagte Jahresfrequenz", ylabel="Beobachtete Jahresfrequenz", title="OOF-Gruppenkalibrierung; gebundene Raten ungeteilt")
    axes[0].legend(fontsize=6)
    if not tables["standardized_age"].empty:
        for model_id, group in tables["standardized_age"].groupby("model_id", sort=False):
            axes[1].plot(group.assigned_age, group.standardized_annual_rate, marker="o", label=model_id)
        axes[1].legend(fontsize=7)
    axes[1].set(xlabel="Gesetztes Fahreralter", ylabel="Standardisierte Jahresfrequenz", title="Gleicher unterstützter Merkmalsmix; Entwicklungsrefits")
    save(fig, "kalibrierung_standardisierung")
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_id, group in tables["concentration"].groupby("model_id", sort=False):
        ax.plot(group.cumulative_exposure_fraction, group.cumulative_claim_fraction, label=model_id)
    ax.plot([0, 1], [0, 1], color="grey", lw=.7)
    ax.set(xlabel="Kumulierte Exponierung", ylabel="Kumulierter Anteil beobachteter Schäden", title="OOF-Konzentration nach absteigender Jahresrate")
    ax.legend(fontsize=8)
    save(fig, "konzentration")
    return paths


def run(output: Path, reports: Path) -> dict:
    from src.frequency_diagnostics import score_predictions, paired_bootstrap, age_standardization
    cfg = configuration()
    policies, claims, prepared = load_prepared(prepared_location(None))
    frames, frozen = open_split(PROJECT_ROOT / "data/validation", prepared, protocol())
    d, _ = development_only(policies, claims, frames)
    del policies, claims
    membership = frames["policy_membership"]
    software = {"python": platform.python_version(), **{name: version(name) for name in
                 ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib"]}}
    identity = {"source_id": prepared["source_id"], "preparation_id": prepared["preparation_id"],
                "split_id": frozen["split_id"], "configuration_sha256": text_sha256(CONFIG),
                "validation_protocol_sha256": text_sha256(PROJECT_ROOT / "config/validation_protocol.json"),
                "code_sha256": {p: text_sha256(PROJECT_ROOT / p) for p in CODE}, "software": software}
    run_id = identity_id("s4freq-", identity)
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=True)
    registration = {"run_id": run_id, "identity": identity, "registered_configurations": cfg["configurations"],
                    "registered_before_first_fit": True, "evaluation_predictions": 0, "evaluation_scores": 0}
    reg_path = directory / "registration.json"
    if reg_path.exists() and reg_path.read_bytes() != json_bytes(registration):
        raise DataContractError("Pre-fit registration differs")
    reg_path.write_bytes(json_bytes(registration))
    predictions, model_reports, runtimes = [], {}, {}
    with threadpool_limits(limits=cfg["compute"]["threads"]):
        for config in cfg["configurations"]:
            model_id = config["model_id"]
            cache = directory / model_id
            if cache.exists():
                saved, model_report = read_artifacts(cache)
                if model_report["registration_sha256"] != sha256(reg_path) or model_report["configuration"] != config:
                    raise DataContractError("Model cache registration mismatch")
                prediction = align_predictions(saved["oof_predictions"], membership, d[["IDpol", "Exposure"]], model_id, run_id)
                print(f"Verified cached {model_id}", flush=True)
            else:
                try:
                    prediction, diagnostic, timing = fit_oof(d, membership, config, run_id)
                except Exception as error:
                    (directory / f"{model_id}.failure.json").write_bytes(json_bytes({"model_id": model_id,
                        "error_type": type(error).__name__, "error": str(error), "registration_sha256": sha256(reg_path)}))
                    raise
                model_report = write_immutable(cache, {"oof_predictions": prediction}, {
                    "run_id": run_id, "model_id": model_id, "configuration": config,
                    "registration_sha256": sha256(reg_path), "fold_diagnostics": diagnostic})
                runtimes[model_id] = timing
                (directory / f"{model_id}.timing.json").write_bytes(json_bytes(timing))
            predictions.append(prediction)
            model_reports[model_id] = model_report
        prediction = pd.concat(predictions, ignore_index=True)
        scores, folds, segments, calibration = score_predictions(d, prediction)
        winners, selected = choose_models(scores, cfg["configurations"])
        bootstrap = paired_bootstrap(d, prediction, selected, **{k: cfg["bootstrap"][k] for k in ("replicates", "seed")})
        fitted, refits = {}, {}
        for model_id in dict.fromkeys([*winners.values(), selected]):
            config = next(c for c in cfg["configurations"] if c["model_id"] == model_id)
            path = directory / f"{model_id}.joblib"
            info_path = directory / f"{model_id}.refit.json"
            if path.exists() and info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                if info["sha256"] != sha256(path) or info["run_id"] != run_id:
                    raise DataContractError("Development refit cache checksum/identity mismatch")
                model = joblib.load(path)
            else:
                model = estimator(config)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(d[list(FEATURES)], d.recorded_positive_count/d.Exposure, sample_weight=d.Exposure.to_numpy())
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise DataContractError("Unresolved convergence in development refit")
                joblib.dump(model, path)
                info = {"run_id": run_id, "model_id": model_id, "sha256": sha256(path),
                        "training_policies": len(d), "training_id_sha256": logical_hash(d[["IDpol"]]),
                        "fit": getattr(model, "diagnostics_", {"rate": float(model.rate_)} if isinstance(model, ExposureIntercept) else {}),
                        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
                info_path.write_bytes(json_bytes(info))
                (directory / f"{model_id}.refit.timing.json").write_bytes(json_bytes({"seconds": time.perf_counter()-started}))
            fitted[model_id], refits[model_id] = model, info
        rule = cfg["age_standardization"]
        age, cells, support = age_standardization(d, fitted, FEATURES,
            support=rule["minimum_exposure_years_each_age_band"], ages=tuple(rule["set_ages"]))
        effects = effect_profiles(d, fitted)
        concentration = concentration_points(d, prediction, ["intercept", *winners.values()])
    baseline_score = float(scores.loc[scores.model_id.eq("intercept"), "poisson_deviance_per_exposure"].iloc[0])
    scores["relative_deviance_reduction_vs_intercept"] = 1-scores.poisson_deviance_per_exposure/baseline_score
    scores["family"] = scores.model_id.map({c["model_id"]: c["family"] for c in cfg["configurations"]})
    tables = {"scores": scores, "folds": folds, "segments": segments, "calibration": calibration,
              "paired_bootstrap": bootstrap, "standardized_age": age, "standardization_cells": cells,
              "effects": effects, "concentration": concentration}
    reports.mkdir(parents=True, exist_ok=True)
    images = figures(tables, reports / "figures")
    for name, table in tables.items():
        (reports / f"{name}.json").write_bytes(table_json(table))
    decision = {"family_winners": winners, "selected_frequency_model": selected,
                "rule": cfg["selection_rule"], "empirical_minimum_used": True,
                "rationale": "Gemeinsame OOF-Deviance bestimmt empirisches Minimum. Falten, Segmentkalibrierung, bedingte Bootstrap-Differenzen und Rechenaufwand sind zusammen zu lesen; verbleibende Exponierungsabweichungen begrenzen jede Tarifaussage.",
                "evaluation_released": False, "next_gate": "S5: Schadenhöhe und reine Risikoprämie; gemeinsame Evaluation erst nach gesperrter Entwicklungsentscheidung."}
    report = {"run_id": run_id, "identity": identity, "registration_sha256": sha256(reg_path),
              "source_id": prepared["source_id"], "preparation_id": prepared["preparation_id"], "split_id": frozen["split_id"],
              "scope": cfg["scope"], "units": {"annual_rate": "Schäden je Vertragsjahr", "expected_count": "Schäden je Vertragsperiode"},
              "models": model_reports, "development_refits": refits, "decision": decision,
              "dependence": dependence(d), "age_reference_support": support, "figures": images,
              "public_tables": {name: {"filename": f"{name}.json", "sha256": sha256(reports / f"{name}.json")} for name in tables},
              "evaluation_predictions": 0, "evaluation_scores": 0,
              "limits": ["OOF-Auswahlwerte sind Selektionsschätzungen, keine Nested-CV oder externe Evaluation.",
                         "Bootstrap bei festen OOF-Vorhersagen enthält keine Refitting-/Auswahlunsicherheit.",
                         "Poisson-Mittelwertmodell begründet weder äquidisperse Anzahlen noch kausale Effekte.",
                         "Refitprofile sind deskriptiv; unbekannte Haushalte, Nutzung und Beendigung bleiben unbeobachtet."]}
    saved = write_immutable(directory / "results", tables, report)
    (reports / "frequency_report.json").write_bytes(json_bytes(saved))
    (reports / "development_decision.json").write_bytes(json_bytes({"run_id": run_id, **decision,
        "selected_configuration": next(c for c in cfg["configurations"] if c["model_id"] == selected),
        "selected_refit_sha256": refits[selected]["sha256"]}))
    print(f"{run_id}: selected {selected}; {len(d)} development policies per model; evaluation closed", flush=True)
    return saved


def load_report(reports: Path = PROJECT_ROOT / "reports/s4") -> tuple[dict, dict]:
    report = json.loads((reports / "frequency_report.json").read_text(encoding="utf-8"))
    identity = report["identity"]
    if report["run_id"] != identity_id("s4freq-", identity) or identity["configuration_sha256"] != text_sha256(CONFIG):
        raise DataContractError("Frequency identity/configuration mismatch")
    if identity["validation_protocol_sha256"] != text_sha256(PROJECT_ROOT / "config/validation_protocol.json"):
        raise DataContractError("Frequency validation protocol has changed")
    for name, digest in identity["code_sha256"].items():
        if text_sha256(PROJECT_ROOT / name) != digest:
            raise DataContractError(f"Frequency code has changed: {name}")
    tables, local = read_artifacts(PROJECT_ROOT / "data/frequency" / report["run_id"] / "results")
    if local != report:
        raise DataContractError("Public/local frequency reports differ")
    for name, spec in report["public_tables"].items():
        path = reports / spec["filename"]
        if sha256(path) != spec["sha256"] or path.read_bytes() != table_json(tables[name]):
            raise DataContractError(f"Public aggregate mismatch: {name}")
    for spec in report["figures"].values():
        if sha256(reports / "figures" / spec["filename"]) != spec["sha256"]:
            raise DataContractError("Frequency figure mismatch")
    return report, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/frequency"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/s4"))
    args = parser.parse_args()
    run(root_path(args.output), root_path(args.report_dir))


if __name__ == "__main__":
    main()
