"""Frozen policy splits and actual development-only OOF prediction contracts.

The real S2 run fits only exposure intercepts. Future estimators must implement
the weighted-rate API and are cloned with all learned preprocessing per fold.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.data import (PROJECT_ROOT, DataContractError, exact_ids, json_bytes,
                      load_prepared, logical_hash, root_path, sha256, text_sha256)

PROTOCOL_PATH = PROJECT_ROOT / "config/validation_protocol.json"
FEATURES = ("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")
MEMBERSHIP_COLUMNS = ("IDpol", "split", "fold")
PREDICTION_COLUMNS = (*MEMBERSHIP_COLUMNS, "model_id", "run_id", "annual_rate", "expected_count")


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def identity_id(prefix: str, value: dict) -> str:
    return prefix + hashlib.sha256(json_bytes(value)).hexdigest()[:16]


def make_membership(ids: pd.Series, recipe: dict) -> pd.DataFrame:
    ids = exact_ids(ids).sort_values().reset_index(drop=True)
    folds = recipe["development_folds"]
    if not ids.is_unique:
        raise DataContractError("Split universe must have unique policy IDs")
    if (recipe["algorithm"] != "sorted-ID-PCG64-permutation-evaluation-first-ceil-round-robin-development-v1"
            or recipe["evaluation_fold_code"] != 0 or folds != 5
            or not 0 < recipe["evaluation_fraction"] < 1):
        raise DataContractError("Unsupported split recipe")
    evaluation_n = math.ceil(len(ids) * recipe["evaluation_fraction"])
    if len(ids) - evaluation_n < folds or evaluation_n < 1:
        raise DataContractError("Too few policies for evaluation plus five development folds")
    rng = np.random.Generator(np.random.PCG64(recipe["seed"]))
    order = rng.permutation(len(ids))
    split = np.full(len(ids), "development", dtype=object)
    fold = np.zeros(len(ids), dtype="int64")
    split[order[:evaluation_n]] = "evaluation"
    fold[order[evaluation_n:]] = np.arange(len(ids) - evaluation_n) % folds + 1
    return pd.DataFrame({"IDpol": ids, "split": pd.Series(split, dtype="str"), "fold": fold})


def propagate_claims(claim_keys: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    if not claim_keys["claim_row_id"].is_unique or claim_keys["claim_row_id"].isna().any():
        raise DataContractError("Claim row lineage must be unique and non-null")
    keys = claim_keys.loc[:, ["claim_row_id", "IDpol"]].copy()
    keys["IDpol"] = exact_ids(keys["IDpol"])
    linked = keys.merge(membership, on="IDpol", how="left", validate="many_to_one", sort=False)
    if linked["split"].isna().any():
        raise DataContractError("Every claim must inherit an existing policy membership")
    return linked.sort_values("claim_row_id").reset_index(drop=True)


def write_immutable(directory: Path, files: dict[str, pd.DataFrame], report: dict) -> dict:
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".validation-") as temporary:
        stage = Path(temporary) / directory.name
        stage.mkdir()
        artifacts = {}
        for name, frame in files.items():
            path = stage / f"{name}.parquet"
            frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_dtype=True)
            artifacts[name] = {"filename": path.name, "rows": len(frame), "sha256": sha256(path),
                               "logical_csv_sha256": logical_hash(frame),
                               "columns": {c: str(frame[c].dtype) for c in frame}}
        result = {**report, "artifacts": artifacts}
        (stage / "manifest.json").write_bytes(json_bytes(result))
        if directory.exists():
            for path in stage.iterdir():
                old = directory / path.name
                if not old.exists() or old.read_bytes() != path.read_bytes():
                    raise DataContractError(f"Frozen artifact differs; preserve/investigate: {old}")
        else:
            stage.rename(directory)
    return result


def read_artifacts(directory: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    report = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for name, spec in report["artifacts"].items():
        if spec["filename"] != f"{name}.parquet":
            raise DataContractError("Unexpected artifact filename")
        path = directory / spec["filename"]
        if sha256(path) != spec["sha256"]:
            raise DataContractError(f"Frozen artifact checksum mismatch: {path}")
        frame = pd.read_parquet(path)
        if ({c: str(frame[c].dtype) for c in frame} != spec["columns"]
                or len(frame) != spec["rows"] or logical_hash(frame) != spec["logical_csv_sha256"]):
            raise DataContractError(f"Frozen artifact schema/content mismatch: {path}")
        frames[name] = frame
    return frames, report


def freeze_split(policy_ids: pd.Series, claim_keys: pd.DataFrame, prepared: dict,
                 output: Path, specification: dict) -> tuple[dict, Path]:
    """Assign using keys alone; lock a registry to prevent accidental seed fishing."""
    membership = make_membership(policy_ids, specification["split"])
    claim_membership = propagate_claims(claim_keys, membership)
    identity = {"source_id": prepared["source_id"],
                "policy_universe_sha256": logical_hash(membership[["IDpol"]]),
                "recipe": specification["split"]}
    split_id = identity_id("s2split-", identity)
    output.mkdir(parents=True, exist_ok=True)
    registry = output / "frozen_split.json"
    if registry.exists():
        previous = json.loads(registry.read_text(encoding="utf-8"))
        if previous["split_id"] != split_id or previous["identity"] != identity:
            raise DataContractError("A different split is already frozen; do not reshuffle the evaluation set")
    destination = output / split_id
    if destination.exists():
        saved_frames, saved = read_artifacts(destination)
        if saved["identity"] != identity or saved["split_id"] != split_id:
            raise DataContractError("Saved split identity mismatch")
        pd.testing.assert_frame_equal(saved_frames["policy_membership"], membership)
        pd.testing.assert_frame_equal(saved_frames["claim_membership"], claim_membership)
        if not registry.exists():
            registry.write_bytes(json_bytes({"split_id": split_id, "identity": identity}))
        return saved, destination
    report = {"split_id": split_id, "identity": identity,
              "preparation_id_at_freeze": prepared["preparation_id"],
              "prepared_policy_sha256": prepared["analysis_tables"]["policies"]["sha256"],
              "prepared_claim_sha256": prepared["analysis_tables"]["claims"]["sha256"],
              "protocol_sha256_at_freeze": text_sha256(PROTOCOL_PATH),
              "split_code_sha256_at_freeze": text_sha256(Path(__file__)),
              "software_at_freeze": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "membership_sizes": {str(name): len(group) for name, group in membership.groupby("split")},
              "development_fold_sizes": {str(fold): len(group) for fold, group in membership.loc[membership["split"] == "development"].groupby("fold")},
              "evaluation_outcomes_used_for_split_assignment": False, "historically_unseen": False,
              "history": specification["analysis_history"], "grouping_limits": specification["grouping_limits"],
              "units": {"fold": "1..5 development; 0 evaluation", "split": "policy-level membership inherited by all linked claim rows"}}
    result = write_immutable(destination, {"policy_membership": membership,
                                         "claim_membership": claim_membership}, report)
    registry.write_bytes(json_bytes({"split_id": split_id, "identity": identity}))
    return result, destination


def validate_membership(membership: pd.DataFrame) -> None:
    if list(membership) != list(MEMBERSHIP_COLUMNS) or membership.isna().any().any():
        raise DataContractError("Invalid membership schema/nulls")
    ids = exact_ids(membership["IDpol"])
    if not ids.is_unique or set(membership["split"]) != {"development", "evaluation"}:
        raise DataContractError("Invalid membership IDs/splits")
    dev = membership["split"] == "development"
    if (set(membership.loc[dev, "fold"]) != {1, 2, 3, 4, 5}
            or not membership.loc[~dev, "fold"].eq(0).all()):
        raise DataContractError("Invalid development folds/evaluation fold code")


def open_split(output: Path, prepared: dict, specification: dict) -> tuple[dict, dict]:
    """Check registry, recipe, lineage and contents before development fitting."""
    registry = json.loads((output / "frozen_split.json").read_text(encoding="utf-8"))
    frames, frozen = read_artifacts(output / registry["split_id"])
    if (set(frames) != {"policy_membership", "claim_membership"}
            or frozen["split_id"] != identity_id("s2split-", frozen["identity"])
            or registry != {"split_id": frozen["split_id"], "identity": frozen["identity"]}
            or frozen["identity"]["source_id"] != prepared["source_id"]
            or frozen["prepared_policy_sha256"] != prepared["analysis_tables"]["policies"]["sha256"]
            or frozen["prepared_claim_sha256"] != prepared["analysis_tables"]["claims"]["sha256"]
            or frozen["identity"]["recipe"] != specification["split"]):
        raise DataContractError("Prepared data/registry/protocol no longer match the frozen split")
    membership = frames["policy_membership"]
    validate_membership(membership)
    expected = make_membership(membership["IDpol"], specification["split"])
    if (logical_hash(expected[["IDpol"]]) != frozen["identity"]["policy_universe_sha256"]
            or not membership.equals(expected)):
        raise DataContractError("Frozen policy membership differs from its declared key-only recipe")
    claim_membership = frames["claim_membership"]
    inherited = propagate_claims(claim_membership[["claim_row_id", "IDpol"]], membership)
    if not inherited.equals(claim_membership):
        raise DataContractError("Claim membership differs from policy membership")
    return frames, frozen


def align_predictions(predictions: pd.DataFrame, membership: pd.DataFrame,
                      policy_exposure: pd.DataFrame, model_id: str, run_id: str) -> pd.DataFrame:
    """Require exactly the requested development policies; align by key, never row order."""
    validate_membership(membership)
    expected = membership.loc[membership["split"] == "development"].copy()
    if list(predictions) != list(PREDICTION_COLUMNS) or predictions.isna().any().any():
        raise DataContractError("Invalid prediction schema/nulls")
    prediction = predictions.copy()
    prediction["IDpol"] = exact_ids(prediction["IDpol"])
    if (not prediction["IDpol"].is_unique or set(prediction["IDpol"]) != set(expected["IDpol"])
            or not prediction["model_id"].eq(model_id).all() or not prediction["run_id"].eq(run_id).all()):
        raise DataContractError("Prediction ID/model/run contract failed")
    ordered = expected.merge(prediction, on="IDpol", validate="one_to_one", suffixes=("_expected", ""))
    if not (ordered["split"] == ordered["split_expected"]).all() or not (ordered["fold"] == ordered["fold_expected"]).all():
        raise DataContractError("Prediction membership contract failed")
    ex = policy_exposure.loc[:, ["IDpol", "Exposure"]].copy()
    ex["IDpol"] = exact_ids(ex["IDpol"])
    if not ex["IDpol"].is_unique or set(ex["IDpol"]) != set(expected["IDpol"]):
        raise DataContractError("Exposure key alignment contract failed")
    ordered = ordered.merge(ex, on="IDpol", validate="one_to_one")
    values = ordered[["annual_rate", "expected_count", "Exposure"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values <= 0).any()
            or not np.allclose(values[:, 1], values[:, 0] * values[:, 2], rtol=1e-12, atol=0)):
        raise DataContractError("Positive annual-rate/period-count/exposure units contract failed")
    return ordered.loc[:, list(PREDICTION_COLUMNS)]


class ExposureIntercept(RegressorMixin, BaseEstimator):
    """Weighted-rate MLE; zero training claims cannot give strictly positive scores."""

    def fit(self, X, y, sample_weight):
        self.rate_ = float(np.dot(np.asarray(y, dtype=float), sample_weight) / np.sum(sample_weight))
        if not np.isfinite(self.rate_) or self.rate_ <= 0:
            raise DataContractError("No positive training claim rate; cannot score a strictly positive Poisson mean")
        return self

    def predict(self, X):
        return np.full(len(X), self.rate_, dtype=float)


def run_oof(development: pd.DataFrame, membership: pd.DataFrame, factory,
            model_id: str, run_id: str, features: tuple[str, ...] = FEATURES) -> tuple[pd.DataFrame, list[dict]]:
    """Five independent fits; only development rows and prospective features accepted."""
    from src.metrics import poisson_deviance_per_exposure
    validate_membership(membership)
    if not features or len(set(features)) != len(features) or not set(features).issubset(FEATURES):
        raise DataContractError("Only explicitly prospective unique features are permitted")
    expected = membership.loc[membership["split"] == "development"]
    dev = development.copy()
    dev["IDpol"] = exact_ids(dev["IDpol"])
    if not dev["IDpol"].is_unique or set(dev["IDpol"]) != set(expected["IDpol"]):
        raise DataContractError("OOF accepts exactly development policies, never evaluation policies")
    dev = expected.merge(dev, on="IDpol", how="left", validate="one_to_one", suffixes=("", "_input"))
    # Validate the response/units before fitting; no full-table preprocessing is fit.
    n = dev["recorded_positive_count"].to_numpy(dtype=float)
    e = dev["Exposure"].to_numpy(dtype=float)
    if (not np.isfinite(n).all() or (n < 0).any() or (n % 1 != 0).any()
            or not np.isfinite(e).all() or (e <= 0).any()):
        raise DataContractError("Invalid development response/exposure")
    pieces, diagnostics = [], []
    for fold in range(1, 6):
        training, validation = dev.loc[dev["fold"] != fold], dev.loc[dev["fold"] == fold]
        estimator = clone(factory())
        weights = training["Exposure"].to_numpy(dtype=float)
        y = training["recorded_positive_count"].to_numpy(dtype=float) / weights
        fit_key = f"{estimator.steps[-1][0]}__sample_weight" if isinstance(estimator, Pipeline) else "sample_weight"
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(training.loc[:, list(features)], y, **{fit_key: weights})
            rates = np.asarray(estimator.predict(validation.loc[:, list(features)]), dtype=float)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise DataContractError(f"Unresolved convergence warning in fold {fold}")
        if rates.shape != (len(validation),) or not np.isfinite(rates).all() or (rates <= 0).any():
            raise DataContractError(f"Invalid annual predictions in fold {fold}")
        prediction = validation.loc[:, list(MEMBERSHIP_COLUMNS)].copy()
        prediction["model_id"], prediction["run_id"] = model_id, run_id
        prediction["annual_rate"] = rates
        prediction["expected_count"] = rates * validation["Exposure"].to_numpy()
        pieces.append(prediction)
        diagnostics.append({"fold": fold, "training_policies": len(training), "validation_policies": len(validation),
            "training_id_sha256": logical_hash(training[["IDpol"]].sort_values("IDpol")),
            "validation_id_sha256": logical_hash(validation[["IDpol"]].sort_values("IDpol")),
            "training_exposure_years": float(weights.sum()), "training_claims": int(training["recorded_positive_count"].sum()),
            "validation_exposure_years": float(validation["Exposure"].sum()),
            "validation_claims": int(validation["recorded_positive_count"].sum()),
            "unknown_validation_categories": {c: int((~validation[c].isin(training[c])).sum())
                for c in ("VehBrand", "VehGas", "Area", "Region") if c in features},
            "deviance_per_exposure": poisson_deviance_per_exposure(validation["recorded_positive_count"].to_numpy(), rates, validation["Exposure"].to_numpy()),
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            "elapsed_seconds": time.perf_counter() - started})
    predictions = pd.concat(pieces, ignore_index=True)
    return align_predictions(predictions, membership, development[["IDpol", "Exposure"]], model_id, run_id), diagnostics


def prepared_location(argument: Path | None) -> Path:
    if argument is not None:
        return root_path(argument)
    published = json.loads((PROJECT_ROOT / "reports/s1/preparation_report.json").read_text(encoding="utf-8"))
    return PROJECT_ROOT / "data/prepared" / published["preparation_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "oof-baseline"])
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/validation"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/s2"))
    args = parser.parse_args()
    policies, claims, prepared = load_prepared(prepared_location(args.prepared))
    output, reports = root_path(args.output), root_path(args.report_dir)
    specification = protocol()
    reports.mkdir(parents=True, exist_ok=True)
    if args.action == "freeze":
        frozen, directory = freeze_split(policies["IDpol"], claims[["claim_row_id", "IDpol"]], prepared, output, specification)
        (reports / "split_manifest.json").write_bytes(json_bytes(frozen))
        print(f"{frozen['split_id']}: {directory}")
        return
    frames, frozen = open_split(output, prepared, specification)
    membership = frames["policy_membership"]
    # Explicitly remove evaluation rows before passing any data to fitting/scoring.
    dev_ids = membership.loc[membership["split"] == "development", "IDpol"]
    development = policies.loc[policies["IDpol"].isin(dev_ids)].copy()
    identity = {"split_id": frozen["split_id"], "preparation_id": prepared["preparation_id"],
                "protocol_sha256": text_sha256(PROTOCOL_PATH), "runner_sha256": text_sha256(Path(__file__)),
                "metrics_sha256": text_sha256(PROJECT_ROOT / "src/metrics.py"),
                "model_id": "exposure_intercept_v1", "model_parameters": {},
                "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": version("scikit-learn")}}
    run_id = identity_id("s2oof-", identity)
    prediction, diagnostics = run_oof(development, membership, ExposureIntercept, identity["model_id"], run_id)
    from src.metrics import frequency_summary, calibration_table, concentration_curve
    aligned = development[["IDpol", "Exposure", "recorded_positive_count"]].merge(prediction, on="IDpol", validate="one_to_one")
    n, e, rates = (aligned[c].to_numpy() for c in ("recorded_positive_count", "Exposure", "annual_rate"))
    summary = frequency_summary(n, rates, e)
    calibration = calibration_table(n, rates, e)
    concentration = concentration_curve(n, rates, e)
    # Runtime is separate from deterministic artifacts, so repeated outputs can match.
    elapsed = [{"fold": d["fold"], "elapsed_seconds": d.pop("elapsed_seconds")} for d in diagnostics]
    report = {"run_id": run_id, "identity": identity, "source_id": prepared["source_id"],
              "split_id": frozen["split_id"], "model_id": identity["model_id"],
              "scope": "Echte OOF-Vorhersagen des Exponierungs-Basismodells auf allen Entwicklungsverträgen; kein Modellvergleich und keine Evaluation.",
              "units": {"annual_rate": "claims per policy-year", "expected_count": "claims per observed policy period"},
              "summary": summary, "fold_diagnostics": diagnostics,
              "baseline_concentration_caveat": specification["baseline_oof_note"],
              "protocol_sha256_at_split_freeze": frozen["protocol_sha256_at_freeze"],
              "evaluation_predictions": 0, "evaluation_scores": 0}
    saved = write_immutable(output / run_id, {"oof_predictions": prediction, "calibration": calibration,
                                            "concentration": concentration}, report)
    (reports / "baseline_oof_report.json").write_bytes(json_bytes(saved))
    (output / f"{run_id}.timing.json").write_bytes(json_bytes({"run_id": run_id, "folds": elapsed}))
    print(f"{run_id}: {len(prediction)} development OOF predictions; zero evaluation predictions")


if __name__ == "__main__":
    main()
