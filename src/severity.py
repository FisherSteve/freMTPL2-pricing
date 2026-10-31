"""Claim-level severity, all-policy loss cost, and locked joint evaluation.

Run `develop` before `evaluate`. The evaluation registry rejects a new selection
identity after release. All learned severity quantities use claim rows belonging
to training policies; every held-out policy receives the same prediction rule.
"""
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

from src.data import (PROJECT_ROOT as ROOT, DataContractError, json_bytes,
                      load_prepared, logical_hash, root_path, sha256, text_sha256)
from src.eda import add_groups, configuration as eda_configuration, development_only
from src.metrics import frequency_summary, frequency_gini, calibration_table
from src.severity_models import FEATURES, make_severity_estimator
from src.severity_diagnostics import (score_severity, compose_pure_premium,
    score_pure_premium, paired_policy_bootstrap)
from src.validation import (ExposureIntercept, identity_id, open_split,
    prepared_location, protocol, read_artifacts, write_immutable)

CONFIG = ROOT / "config/severity_protocol.json"
CODE = ["src/severity.py", "src/severity_models.py", "src/severity_diagnostics.py",
        "src/frequency_models.py", "src/validation.py", "src/data.py", "src/metrics.py",
        "src/eda.py", "config/eda_protocol.json", "config/validation_protocol.json",
        "scripts/verify_severity_inputs.py"]


def configuration() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = [
        {"model_id": "severity_mean", "family": "mean", "parameters": {}},
        {"model_id": "severity_gamma", "family": "gamma", "parameters": {"alpha": 0.0, "max_iter": 200, "tol": 1e-8}},
        {"model_id": "severity_log_smear", "family": "lognormal", "parameters": {}}]
    if (cfg["configurations"] != expected or cfg["features"] != list(FEATURES)
            or cfg["loss_metrics"]["tweedie_power"] != 1.5
            or cfg["frequency"]["selected_model_id"] != "hgb_l15"):
        raise DataContractError("Severity specification differs from the registered bounded comparison")
    from src.severity_models import AGE_BREAKS, VEHICLE_BREAKS, BM_BREAKS
    for key, values in [("age_edges", AGE_BREAKS), ("vehicle_edges", VEHICLE_BREAKS), ("bm_edges", BM_BREAKS)]:
        if cfg["design"][key] != list(values):
            raise DataContractError("Severity protocol bins differ from implementation")
    return cfg


def inputs() -> tuple:
    """Verify the current replay bridge before loading analytical populations."""
    snapshot_path = ROOT / "reports/s5/input_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    policies, claims, prepared = load_prepared(prepared_location(None))
    frames, frozen = open_split(ROOT / "data/validation", prepared, protocol())
    from src.frequency import load_report as load_frequency
    frequency_report, _ = load_frequency()
    decision = json.loads((ROOT / "reports/s4/development_decision.json").read_text(encoding="utf-8"))
    if (snapshot["source_id"] != prepared["source_id"] or snapshot["preparation_id"] != prepared["preparation_id"]
            or snapshot["split_id"] != frozen["split_id"] or snapshot["frequency_run_id"] != frequency_report["run_id"]
            or snapshot["frequency_decision"] != decision
            or decision["run_id"] != frequency_report["run_id"] or decision["selected_frequency_model"] != "hgb_l15"
            or decision["selected_refit_sha256"] != frequency_report["development_refits"]["hgb_l15"]["sha256"]):
        raise DataContractError("Current source/split/frequency decision differs from the verified S5 input bridge")
    for model_id, info in frequency_report["development_refits"].items():
        if sha256(ROOT / "data/frequency" / frequency_report["run_id"] / f"{model_id}.joblib") != info["sha256"]:
            raise DataContractError("Frequency development refit checksum mismatch")
    identity = {"source_id": prepared["source_id"], "preparation_id": prepared["preparation_id"],
        "split_id": frozen["split_id"], "frequency_run_id": frequency_report["run_id"],
        "input_snapshot_sha256": sha256(snapshot_path), "frequency_decision_sha256": sha256(ROOT / "reports/s4/development_decision.json"),
        "configuration_sha256": text_sha256(CONFIG), "code_sha256": {p: text_sha256(ROOT / p) for p in CODE},
        "software": {"python": platform.python_version(), **{name: version(name) for name in
            ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "joblib", "matplotlib", "threadpoolctl"]}}}
    return policies, claims, prepared, frames, frequency_report, identity


def join_training_claims(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    if (not policies.IDpol.is_unique or not claims.claim_row_id.is_unique
            or not set(claims.IDpol).issubset(set(policies.IDpol)) or claims.empty):
        raise DataContractError("Training claim/policy keys invalid or crossing a split")
    return claims.merge(policies[["IDpol", *FEATURES]], on="IDpol", validate="many_to_one").sort_values("claim_row_id")


def fit_checked(config: dict, training: pd.DataFrame):
    model = make_severity_estimator(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(training[list(FEATURES)], training.ClaimAmount.to_numpy())
    if (any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or not model.diagnostics_["converged"]
            or model.diagnostics_.get("normalized_score_max", 0) > 1e-6):
        raise DataContractError("Unresolved severity convergence failure")
    return model, [{"category": w.category.__name__, "message": str(w.message)} for w in caught]


def predict_severity(model, policies: pd.DataFrame, membership: pd.DataFrame,
                     model_id: str, run_id: str, split: str) -> pd.DataFrame:
    expected = membership.loc[membership["split"].eq(split)].sort_values("IDpol")
    if not policies.IDpol.is_unique or set(policies.IDpol) != set(expected.IDpol):
        raise DataContractError("Severity prediction requires exactly the requested policy split")
    ordered = expected.merge(policies[["IDpol", *FEATURES]], on="IDpol", validate="one_to_one")
    rates = np.asarray(model.predict(ordered[list(FEATURES)]), dtype=float)
    if rates.shape != (len(ordered),) or not np.isfinite(rates).all() or (rates <= 0).any():
        raise DataContractError("Invalid prospective severity predictions")
    result = ordered[["IDpol", "split", "fold"]].copy()
    result["model_id"], result["run_id"], result["severity_eur"] = model_id, run_id, rates
    return result


def fit_oof(d: pd.DataFrame, c: pd.DataFrame, membership: pd.DataFrame,
            config: dict, run_id: str) -> tuple:
    expected = membership.loc[membership["split"].eq("development")]
    if not d.IDpol.is_unique or set(d.IDpol) != set(expected.IDpol):
        raise DataContractError("Severity OOF accepts exactly development policies")
    pieces, evidence, timing = [], [], []
    for fold in range(1, 6):
        train_ids = expected.loc[expected.fold.ne(fold), "IDpol"]
        valid_ids = expected.loc[expected.fold.eq(fold), "IDpol"]
        train = join_training_claims(d.loc[d.IDpol.isin(train_ids)], c.loc[c.IDpol.isin(train_ids)])
        valid = d.loc[d.IDpol.isin(valid_ids)].sort_values("IDpol")
        started = time.perf_counter()
        model, caught = fit_checked(config, train)
        predicted = np.asarray(model.predict(valid[list(FEATURES)]), dtype=float)
        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise DataContractError("Invalid held-fold severity")
        part = valid[["IDpol"]].merge(expected, on="IDpol", validate="one_to_one")
        part["model_id"], part["run_id"], part["severity_eur"] = config["model_id"], run_id, predicted
        pieces.append(part)
        evidence.append({"fold": fold, "training_policy_ids_sha256": logical_hash(pd.DataFrame({"IDpol": train_ids.sort_values().to_numpy()})),
            "validation_policy_ids_sha256": logical_hash(valid[["IDpol"]]),
            "training_claim_keys_sha256": logical_hash(train[["claim_row_id", "IDpol"]]),
            "training_policies_with_claims": int(train.IDpol.nunique()), "training_claim_rows": len(train),
            "training_loss_eur": float(train.ClaimAmount.sum()), "validation_policies": len(valid),
            "unknown_validation_categories": getattr(model, "last_unknown_categories_", {}),
            "fit": model.diagnostics_, "warnings": caught})
        timing.append({"fold": fold, "seconds": time.perf_counter()-started})
        print(f"{config['model_id']} fold {fold}: {len(train)} training claims, {len(valid)} policy predictions", flush=True)
    prediction = pd.concat(pieces, ignore_index=True).sort_values("IDpol").reset_index(drop=True)
    if (not prediction.IDpol.is_unique or set(prediction.IDpol) != set(expected.IDpol)
            or not np.array_equal(prediction.fold, expected.sort_values("IDpol").fold)):
        raise DataContractError("Severity OOF exactly-once policy/fold contract failed")
    return prediction, evidence, timing


def choose_severity(scores: pd.DataFrame) -> tuple[str, str]:
    choices = scores.loc[scores.model_id.isin(["pp_hgb_mean", "pp_hgb_gamma", "pp_hgb_log_smear"])]
    if set(choices.model_id) != {"pp_hgb_mean", "pp_hgb_gamma", "pp_hgb_log_smear"} or not choices.model_id.is_unique:
        raise DataContractError("Severity selection lacks the registered common-frequency PP candidates")
    score_column = "tweedie_deviance_per_exposure"
    values = choices.set_index("model_id")[score_column]
    if not np.isfinite(values).all():
        raise DataContractError("Nonfinite severity-selection scores")
    selected = str(values.idxmin())
    if values["pp_hgb_mean"] <= values.min()*1.005:
        selected = "pp_hgb_mean"
    return selected, {"pp_hgb_mean": "severity_mean", "pp_hgb_gamma": "severity_gamma", "pp_hgb_log_smear": "severity_log_smear"}[selected]


def frequency_oof(report: dict, run_id: str) -> pd.DataFrame:
    pieces = []
    for model_id in ("intercept", "hgb_l15"):
        frames, manifest = read_artifacts(ROOT / "data/frequency" / report["run_id"] / model_id)
        if manifest != report["models"][model_id]:
            raise DataContractError("Frequency OOF manifest differs from verified report")
        p = frames["oof_predictions"].copy()
        pieces.append(p)
    return pd.concat(pieces, ignore_index=True)


def refit_models(d: pd.DataFrame, c: pd.DataFrame, cfg: dict, directory: Path, run_id: str) -> tuple:
    train = join_training_claims(d, c)
    models, infos = {}, {}
    for config in cfg["configurations"]:
        model_id = config["model_id"]
        model_path, info_path = directory / f"{model_id}.joblib", directory / f"{model_id}.refit.json"
        if model_path.exists() and info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info["run_id"] != run_id or sha256(model_path) != info["sha256"]:
                raise DataContractError("Severity development refit identity/checksum mismatch")
            model = joblib.load(model_path)
        else:
            model, caught = fit_checked(config, train)
            joblib.dump(model, model_path)
            info = {"run_id": run_id, "model_id": model_id, "sha256": sha256(model_path),
                "training_policy_ids_sha256": logical_hash(d[["IDpol"]]),
                "training_claim_keys_sha256": logical_hash(train[["claim_row_id", "IDpol"]]),
                "fit": model.diagnostics_, "warnings": caught}
            info_path.write_bytes(json_bytes(info))
        models[model_id], infos[model_id] = model, info
    return models, infos


def tail_diagnostics(d: pd.DataFrame, c: pd.DataFrame, models: dict, cfg: dict, frequency_report: dict) -> dict:
    amounts = c.ClaimAmount.to_numpy()
    ordered = np.sort(amounts)[::-1]
    totals = c.groupby("IDpol").ClaimAmount.sum()
    removed_id = totals.idxmax()
    removed = c.loc[c.IDpol.eq(removed_id)]
    train = join_training_claims(d.loc[d.IDpol.ne(removed_id)], c.loc[c.IDpol.ne(removed_id)])
    # No evaluation rows: influence is measured on the identical development
    # prediction population, with the unchanged development frequency refit.
    f = joblib.load(ROOT / "data/frequency" / frequency_report["run_id"] / "hgb_l15.joblib")
    expected_claim_weights = d.Exposure.to_numpy()*f.predict(d[list(FEATURES)])
    rows = []
    for config in cfg["configurations"]:
        full = models[config["model_id"]]
        reduced, caught = fit_checked(config, train)
        before, after = full.predict(d[list(FEATURES)]), reduced.predict(d[list(FEATURES)])
        sum_before, sum_after = float(np.dot(expected_claim_weights, before)), float(np.dot(expected_claim_weights, after))
        rows.append({"model_id": config["model_id"], "expected_development_loss_before_eur": sum_before,
            "expected_development_loss_after_eur": sum_after, "relative_change": sum_after/sum_before-1,
            "annual_portfolio_loss_cost_before_eur": sum_before/float(d.Exposure.sum()),
            "annual_portfolio_loss_cost_after_eur": sum_after/float(d.Exposure.sum()),
            "severity_ratio_after_before_quantiles": {str(q): float(np.quantile(after/before, q)) for q in [.01, .5, .99]},
            "removed_claim_rows": len(removed), "removed_loss_eur": float(removed.ClaimAmount.sum()),
            "remaining_training_claim_rows": len(train), "remaining_training_loss_eur": float(train.ClaimAmount.sum()),
            "warnings": caught, "scope": "Training influence only, same development policies; no evaluation, no alternative primary target"})
    leverage = []
    train_all = join_training_claims(d, c)
    largest_indices = np.argsort(train_all.ClaimAmount.to_numpy())[::-1][:10]
    for model_id, model in models.items():
        for rank, index in enumerate(largest_indices, 1):
            leverage.append({"model_id": model_id, "claim_amount_rank": rank,
                "claim_amount_eur": float(train_all.ClaimAmount.iloc[index]),
                "design_leverage": float(model.training_leverage_[index])})
    cap = cfg["tail_diagnostics"]["descriptive_cap_eur"]
    return {"population": "development positive claim rows, no primary exclusions", "claim_rows": len(c),
        "loss_eur": float(amounts.sum()), "quantiles_eur": {str(q): float(np.quantile(amounts, q)) for q in cfg["tail_diagnostics"]["quantiles"]},
        "largest_claim_share": float(ordered[0]/ordered.sum()),
        "largest_ceil_one_percent_share": float(ordered[:int(np.ceil(.01*len(c)))].sum()/ordered.sum()),
        "descriptive_cap_eur": cap, "capped_loss_eur": float(np.minimum(amounts, cap).sum()),
        "removed_eur_if_capped": float(np.maximum(amounts-cap, 0).sum()), "claim_rows_above_cap": int((amounts > cap).sum()),
        "influence_removes_one_whole_policy": True, "influence_removed_policy_exposure_years": float(d.set_index("IDpol").loc[removed_id, "Exposure"]),
        "influence_refits": rows, "largest_claim_leverage": leverage,
        "limits": "Observed-tail sensitivity is not coverage of unseen losses. Leverage depends on covariates; deletion refits measure actual fitted influence. No cap/deletion is used in the primary study."}


def loss_components(d: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    groups = add_groups(d, eda_configuration())[["IDpol", "Exposure", "recorded_loss_eur", "age_band"]]
    a = groups.merge(pp, on="IDpol", validate="one_to_many")
    rows = []
    for (model_id, age), part in a.groupby(["model_id", "age_band"], observed=True, sort=True):
        e = part.Exposure.to_numpy()
        r, m = part.annual_rate.to_numpy(), part.severity_eur.to_numpy()
        rows.append({"model_id": model_id, "age_band": str(age), "policies": len(part), "exposure_years": float(e.sum()),
            "observed_loss_eur": float(part.recorded_loss_eur.sum()),
            "observed_annual_loss_cost": float(part.recorded_loss_eur.sum()/e.sum()),
            "mean_annual_frequency_exposure_weighted": float(np.average(r, weights=e)),
            "mean_predicted_severity_policy_exposure_weighted": float(np.average(m, weights=e)),
            "mean_product_annual_pp": float(np.average(r*m, weights=e)),
            "product_of_separate_exposure_means": float(np.average(r, weights=e)*np.average(m, weights=e))})
    return pd.DataFrame(rows)


def score_tables(d, c, membership, severity, frequency, cfg, run_id, split):
    sev, sev_segments, sev_folds = score_severity(d, c, severity, membership, split=split)
    pp = compose_pure_premium(d, membership, frequency, severity, cfg["pairs"], run_id, split=split)
    loss, loss_segments, loss_folds = score_pure_premium(d, c, pp, membership, split=split)
    return {"severity_predictions": severity, "pp_predictions": pp, "severity_scores": sev,
            "severity_segments": sev_segments, "severity_folds": sev_folds, "loss_scores": loss,
            "loss_segments": loss_segments, "loss_folds": loss_folds, "loss_components": loss_components(d, pp)}


def publish(tables: dict, manifest: dict, reports: Path, prefix: str) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name.endswith("predictions"):
            continue
        payload = json.loads(table.to_json(orient="records", double_precision=15))
        (reports / f"{prefix}_{name}.json").write_bytes(json_bytes(payload))
    (reports / f"{prefix}_report.json").write_bytes(json_bytes(manifest))


def development_boundary(output: Path, run_id: str) -> bool:
    path = output / "evaluation_release.json"
    if not path.exists():
        return False
    released = json.loads(path.read_text(encoding="utf-8"))
    if released["run_id"] != run_id:
        raise DataContractError("Evaluation is already open; no new development identity/selection is allowed")
    if not (output / run_id / "development/manifest.json").exists():
        raise DataContractError("Evaluation is open; missing development evidence must not trigger new fitting")
    return True


def develop(output: Path, reports: Path) -> dict:
    cfg = configuration()
    policies, claims, prepared, frames, frequency_report, identity = inputs()
    run_id = identity_id("s5sev-", identity)
    already_released = development_boundary(output, run_id)
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=True)
    registration = {"run_id": run_id, "identity": identity, "configurations": cfg["configurations"],
                    "registered_before_first_fit": True, "evaluation_released": False}
    reg_path = directory / "registration.json"
    if reg_path.exists() and reg_path.read_bytes() != json_bytes(registration):
        raise DataContractError("Severity registration changed")
    reg_path.write_bytes(json_bytes(registration))
    d, c = development_only(policies, claims, frames)
    del policies, claims
    membership = frames["policy_membership"]
    if (directory / "development").exists():
        tables, manifest = read_artifacts(directory / "development")
        if (manifest["run_id"] != run_id or manifest["identity"] != identity
                or manifest["registration_sha256"] != sha256(reg_path)):
            raise DataContractError("Development cache identity/registration mismatch")
    else:
        pieces, model_reports = [], {}
        with threadpool_limits(limits=cfg["compute"]["threads"]):
            for config in cfg["configurations"]:
                model_id = config["model_id"]
                cache = directory / model_id
                if cache.exists():
                    saved, info = read_artifacts(cache)
                    if (info["registration_sha256"] != sha256(reg_path) or info["configuration"] != config
                            or info["run_id"] != run_id or info["model_id"] != model_id):
                        raise DataContractError("Severity OOF cache registration mismatch")
                    prediction = saved["predictions"]
                else:
                    prediction, diagnostic, timing = fit_oof(d, c, membership, config, run_id)
                    info = write_immutable(cache, {"predictions": prediction}, {"run_id": run_id,
                        "model_id": model_id, "configuration": config, "registration_sha256": sha256(reg_path), "fold_diagnostics": diagnostic})
                    (directory / f"{model_id}.timing.json").write_bytes(json_bytes(timing))
                pieces.append(prediction)
                model_reports[model_id] = info
            frequency = frequency_oof(frequency_report, run_id)
            tables = score_tables(d, c, membership, pd.concat(pieces, ignore_index=True), frequency, cfg, run_id, "development")
            selected_pp, selected_sev = choose_severity(tables["loss_scores"])
            tables["bootstrap"] = paired_policy_bootstrap(d, tables["pp_predictions"], membership, selected_pp,
                replicates=cfg["bootstrap"]["replicates"], seed=cfg["bootstrap"]["seed"])
            models, refits = refit_models(d, c, cfg, directory, run_id)
            tails = tail_diagnostics(d, c, models, cfg, frequency_report)
        manifest = write_immutable(directory / "development", tables, {"run_id": run_id, "identity": identity,
            "registration_sha256": sha256(reg_path), "models": model_reports, "development_refits": refits,
            "selected_pp_model_id": selected_pp, "selected_severity_model_id": selected_sev,
            "selection_rule": cfg["selection"], "tail_diagnostics": tails,
            "reconciliation": {"policies": len(d), "claim_rows": len(c), "exposure_years": float(d.Exposure.sum()),
                "loss_eur": float(c.ClaimAmount.sum()), "excluded_policies": 0, "excluded_claim_rows": 0, "excluded_loss_eur": 0},
            "evaluation_predictions": 0, "evaluation_scores": 0})
    decision = {"run_id": run_id, "identity": identity, "selected_pp_model_id": manifest["selected_pp_model_id"],
        "selected_severity_model_id": manifest["selected_severity_model_id"], "selection_rule": cfg["selection"],
        "development_manifest_sha256": sha256(directory / "development/manifest.json"),
        "severity_refits": {m: info["sha256"] for m, info in manifest["development_refits"].items()},
        "frequency_refits": {m: info["sha256"] for m, info in frequency_report["development_refits"].items()},
        "frequency_intercept_annual_rate": float(d.recorded_positive_count.sum()/d.Exposure.sum()),
        "frequency_selected_model_id": "hgb_l15", "evaluation_power": 1.5, "evaluation_used_for_selection": False}
    decision["decision_id"] = identity_id("s5decision-", decision)
    path = directory / "development_decision.json"
    if path.exists() and path.read_bytes() != json_bytes(decision):
        raise DataContractError("Locked severity development decision differs")
    path.write_bytes(json_bytes(decision))
    publish(tables, manifest, reports, "development")
    (reports / "development_decision.json").write_bytes(json_bytes(decision))
    state = "evaluation already released; verified existing selection" if already_released else "evaluation closed"
    print(f"{run_id}: development complete; selected {decision['selected_pp_model_id']}; {state}", flush=True)
    return manifest


def evaluate(output: Path, reports: Path) -> dict:
    cfg = configuration()
    policies, claims, prepared, frames, frequency_report, identity = inputs()
    run_id = identity_id("s5sev-", identity)
    directory = output / run_id
    decision_path = directory / "development_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    verify = {k: v for k, v in decision.items() if k != "decision_id"}
    if (decision["decision_id"] != identity_id("s5decision-", verify) or decision["identity"] != identity
            or decision["development_manifest_sha256"] != sha256(directory / "development/manifest.json")
            or (reports / "development_decision.json").read_bytes() != decision_path.read_bytes()):
        raise DataContractError("Final evaluation requires an unchanged locked development decision")
    for model_id, digest in decision["severity_refits"].items():
        if sha256(directory / f"{model_id}.joblib") != digest:
            raise DataContractError("Locked severity refit changed before evaluation")
    if decision["frequency_refits"] != {m: info["sha256"] for m, info in frequency_report["development_refits"].items()}:
        raise DataContractError("Locked frequency refits changed before evaluation")
    release = {"run_id": run_id, "decision_id": decision["decision_id"], "decision_sha256": sha256(decision_path),
        "tweedie_power": 1.5, "historically_unseen": False, "evaluation_released": True,
        "boundary": "No post-evaluation changes to selection, features, retransformation or scoring power"}
    release_path = output / "evaluation_release.json"
    if release_path.exists() and release_path.read_bytes() != json_bytes(release):
        raise DataContractError("Evaluation already released for a different locked decision")
    release_path.write_bytes(json_bytes(release))
    membership = frames["policy_membership"]
    keys = membership.loc[membership["split"].eq("evaluation"), "IDpol"]
    d = policies.loc[policies.IDpol.isin(keys)].sort_values("IDpol").reset_index(drop=True)
    c = claims.loc[claims.IDpol.isin(keys)].sort_values("claim_row_id").reset_index(drop=True)
    del policies, claims
    if (directory / "evaluation").exists():
        tables, manifest = read_artifacts(directory / "evaluation")
        if (manifest["run_id"] != run_id or manifest["identity"] != identity
                or manifest["decision_id"] != decision["decision_id"]
                or manifest["decision_sha256"] != sha256(decision_path)
                or manifest["release_sha256"] != sha256(release_path)
                or manifest["selected_pp_model_id"] != decision["selected_pp_model_id"]):
            raise DataContractError("Evaluation cache identity/decision/release mismatch")
    else:
        with threadpool_limits(limits=cfg["compute"]["threads"]):
            severity = pd.concat([predict_severity(joblib.load(directory / f"{model_id}.joblib"), d, membership,
                model_id, run_id, "evaluation") for model_id in decision["severity_refits"]], ignore_index=True)
            fp, fs, fc = [], [], []
            for model_id in ["intercept", *frequency_report["development_refits"]]:
                if model_id == "intercept":
                    model = ExposureIntercept()
                    model.rate_ = decision["frequency_intercept_annual_rate"]
                else:
                    model = joblib.load(ROOT / "data/frequency" / frequency_report["run_id"] / f"{model_id}.joblib")
                rates = model.predict(d[list(FEATURES)])
                prediction = membership.loc[membership["split"].eq("evaluation")].sort_values("IDpol").copy()
                if not np.array_equal(prediction.IDpol, d.IDpol):
                    raise DataContractError("Frequency evaluation key alignment failed")
                prediction["model_id"], prediction["run_id"] = model_id, run_id
                prediction["annual_rate"], prediction["expected_count"] = rates, d.Exposure.to_numpy()*rates
                fp.append(prediction)
                fs.append({"model_id": model_id, **frequency_summary(d.recorded_positive_count, rates, d.Exposure),
                           "raw_frequency_gini": frequency_gini(d.recorded_positive_count, rates, d.Exposure)})
                cal = calibration_table(d.recorded_positive_count, rates, d.Exposure)
                cal.insert(0, "model_id", model_id)
                fc.append(cal)
            frequency = pd.concat(fp, ignore_index=True)
            tables = score_tables(d, c, membership, severity, frequency, cfg, run_id, "evaluation")
            tables["frequency_predictions"] = frequency
            tables["frequency_scores"] = pd.DataFrame(fs)
            tables["frequency_calibration"] = pd.concat(fc, ignore_index=True)
            tables["bootstrap"] = paired_policy_bootstrap(d, tables["pp_predictions"], membership,
                decision["selected_pp_model_id"], split="evaluation", replicates=cfg["bootstrap"]["replicates"], seed=cfg["bootstrap"]["seed"])
        manifest = write_immutable(directory / "evaluation", tables, {"run_id": run_id, "identity": identity,
            "decision_id": decision["decision_id"], "decision_sha256": sha256(decision_path),
            "release_sha256": sha256(release_path), "selected_pp_model_id": decision["selected_pp_model_id"],
            "selection_changed_after_evaluation": False,
            "reconciliation": {"policies": len(d), "claim_rows": len(c), "exposure_years": float(d.Exposure.sum()),
                "loss_eur": float(c.ClaimAmount.sum()), "excluded_policies": 0, "excluded_claim_rows": 0, "excluded_loss_eur": 0},
            "units": {"severity_eur": "EUR per positive recorded claim", "annual_pure_premium": "EUR per policy-year",
                      "expected_period_loss": "EUR per observed policy period"}})
    publish(tables, manifest, reports, "evaluation")
    (reports / "evaluation_release.json").write_bytes(json_bytes(release))
    print(f"{run_id}: joint evaluation complete; decision unchanged", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["develop", "evaluate"])
    parser.add_argument("--output", type=Path, default=Path("data/severity"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/s5"))
    args = parser.parse_args()
    (develop if args.action == "develop" else evaluate)(root_path(args.output), root_path(args.report_dir))


if __name__ == "__main__":
    main()
