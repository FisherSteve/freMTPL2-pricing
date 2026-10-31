"""Identify and verify the existing frequency inputs before severity fitting.

This command fits no model and aggregates no evaluation outcomes. Historical
reproducibility records retain their original run identities. The snapshot
bridges those records to the current, explicitly verified input artifacts.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DataContractError, json_bytes, load_prepared, sha256, text_sha256
from src.frequency import load_report as load_frequency_report
from src.validation import (FEATURES, align_predictions, identity_id, open_split,
                            protocol)

PUBLISHED_BASELINE = "d8a4773c3ef89d4ddde52b4febd22e024d476d4a"


def notebook_source_hash(path: Path) -> str:
    """Hash substantive source cells, excluding outputs, IDs and empty cells."""
    import hashlib
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [{"cell_type": cell["cell_type"], "source": cell["source"]}
             for cell in notebook["cells"] if "".join(cell["source"]).strip()]
    return hashlib.sha256(json_bytes(cells)).hexdigest()


def verify_inputs() -> dict:
    """Return a deterministic current-input snapshot; no writes or fitting."""
    prep_path = ROOT / "reports/s1/preparation_report.json"
    public_prepared = json.loads(prep_path.read_text(encoding="utf-8"))
    preparation_path = ROOT / "data/prepared" / public_prepared["preparation_id"]
    policies, claims, prepared = load_prepared(preparation_path)
    if prepared != public_prepared:
        raise DataContractError("Current public/prepared manifests differ")
    for key, path in [("code_sha256", ROOT / "src/data.py"),
                      ("contract_sha256", ROOT / "config/data_contract.json"),
                      ("source_manifest_sha256", ROOT / "reports/s0/source_manifest.json")]:
        if prepared["identity"][key] != text_sha256(path):
            raise DataContractError(f"Current preparation source identity differs: {path}")
    specification = protocol()
    if tuple(specification["prospective_features"]) != FEATURES or len(FEATURES) != 9:
        raise DataContractError("The frozen nine prospective features have changed")
    if set(FEATURES).intersection(specification["forbidden_predictors"]):
        raise DataContractError("A forbidden predictor entered the prospective feature contract")
    frames, split = open_split(ROOT / "data/validation", prepared, specification)
    membership = frames["policy_membership"]
    development_keys = membership.loc[membership.split.eq("development"), "IDpol"]
    development_exposure = policies.loc[policies.IDpol.isin(development_keys), ["IDpol", "Exposure"]]
    del policies, claims  # No analytical function receives evaluation outcomes.
    frequency, _ = load_frequency_report()
    identity = frequency["identity"]
    if (identity["source_id"] != prepared["source_id"]
            or identity["preparation_id"] != prepared["preparation_id"]
            or identity["split_id"] != split["split_id"]):
        raise DataContractError("Frequency/preparation/split lineage differs")
    run_path = ROOT / "data/frequency" / frequency["run_id"]
    registration_path = run_path / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if (registration["identity"] != identity
            or registration["run_id"] != identity_id("s4freq-", identity)
            or frequency["registration_sha256"] != sha256(registration_path)):
        raise DataContractError("Frequency pre-fit registration mismatch")
    decision_path = ROOT / "reports/s4/development_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selected = decision["selected_frequency_model"]
    if (decision["run_id"] != frequency["run_id"]
            or any(decision[key] != value for key, value in frequency["decision"].items())
            or decision["selected_configuration"] != frequency["models"][selected]["configuration"]
            or decision["selected_refit_sha256"] != frequency["development_refits"][selected]["sha256"]
            or decision["evaluation_released"] is not False
            or frequency["evaluation_predictions"] != 0 or frequency["evaluation_scores"] != 0):
        raise DataContractError("Locked frequency decision/current report mismatch or evaluation released")
    if (selected != "hgb_l15" or decision["selected_configuration"] != {
            "model_id": "hgb_l15", "family": "histogram_poisson",
            "max_leaf_nodes": 15, "l2_regularization": 1.0}):
        raise DataContractError("S5 requires the documented S4 histogram-Poisson decision")
    model_inputs = {}
    for name, model in frequency["models"].items():
        if Path(name).name != name:
            raise DataContractError("Unexpected frequency model identifier")
        manifest_path = run_path / name / "manifest.json"
        if json.loads(manifest_path.read_text(encoding="utf-8")) != model:
            raise DataContractError("Public/local frequency model manifests differ")
        artifact = model["artifacts"]["oof_predictions"]
        if artifact["filename"] != "oof_predictions.parquet":
            raise DataContractError("Unexpected OOF filename")
        path = run_path / name / artifact["filename"]
        if sha256(path) != artifact["sha256"]:
            raise DataContractError(f"Frequency OOF checksum mismatch: {name}")
        prediction = pd.read_parquet(path)
        align_predictions(prediction, membership, development_exposure, name, frequency["run_id"])
        if (len(prediction) != artifact["rows"]
                or {column: str(prediction[column].dtype) for column in prediction} != artifact["columns"]):
            raise DataContractError("Frequency prediction row/schema contract mismatch")
        model_inputs[name] = {"configuration": model["configuration"], "oof_prediction_sha256": artifact["sha256"],
                              "oof_predictions": len(prediction), "manifest_sha256": sha256(manifest_path)}
    for name, refit in frequency["development_refits"].items():
        path = run_path / f"{name}.joblib"
        if refit["run_id"] != frequency["run_id"] or sha256(path) != refit["sha256"]:
            raise DataContractError(f"Frequency refit checksum/identity mismatch: {name}")
    historical_path = ROOT / "reports/s4/reproducibility.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    sources = ["src/data.py", "src/validation.py", "src/metrics.py", "src/eda.py",
               "src/frequency.py", "src/frequency_models.py", "src/frequency_diagnostics.py",
               "config/data_contract.json", "config/validation_protocol.json", "config/eda_protocol.json",
               "config/frequency_protocol.json", "reports/s0/source_manifest.json", "requirements.txt", "requirements.lock.txt"]
    notebooks = [ROOT / "notebooks" / name for name in ["01_data_preparation.ipynb", "02_validation_contracts.ipynb",
                  "03_development_eda.ipynb", "04_frequency_models.ipynb"]]
    return {
        "snapshot_version": 1, "producer_sha256": text_sha256(Path(__file__)),
        "scope": "Geprüfte aktuelle Eingaben vor Schadenhöhenanpassung; keine Modellanpassung oder Aggregation von Evaluationszielgrößen.",
        "source_id": prepared["source_id"], "preparation_id": prepared["preparation_id"],
        "split_id": split["split_id"], "frequency_run_id": frequency["run_id"],
        "frequency_decision": decision, "frequency_model_refits": frequency["development_refits"],
        "frequency_model_inputs": model_inputs, "prospective_features": list(FEATURES),
        "prepared_table_sha256": {name: spec["sha256"] for name, spec in prepared["analysis_tables"].items()},
        "split_table_sha256": {name: spec["sha256"] for name, spec in split["artifacts"].items()},
        "source_sha256": {name: text_sha256(ROOT / name) for name in sources},
        "current_report_sha256": {"preparation": sha256(prep_path),
                                  "frequency": sha256(ROOT / "reports/s4/frequency_report.json"),
                                  "frequency_decision": sha256(decision_path)},
        "notebook_nonempty_source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): notebook_source_hash(path) for path in notebooks},
        "software": {"python": platform.python_version(), **{name: version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pyarrow", "joblib", "threadpoolctl"]}},
        "previous_published_frequency_identity": {
            "commit": PUBLISHED_BASELINE, "run_id": historical["run_id"], "source_id": historical["source_id"],
            "preparation_id": historical["preparation_id"], "split_id": historical["split_id"],
            "python": historical["software"]["python"], "historical_metadata_sha256": sha256(historical_path)},
        "independently_verified_historical_transition": {
            "finding": "Vor S5 wurde der Wechsel von CPython 3.12.14 auf 3.12.10 unabhängig geprüft: alle zehn OOF-Tabellen waren nach Entfernen ausschließlich der Laufkennung exakt gleich; ebenso Faltendiagnostik, numerische EDA-/Modelltabellen, Abbildungen und drei Refit-Dateien.",
            "execution_limit": "Die Ausführung des vorhandenen Wiederholungslaufs wurde dabei nicht beobachtet. Diese historische Prüfung belegt Inhaltsgleichheit; die aktuellen Hash- und Schlüsselprüfungen identifizieren getrennt den gegenwärtigen Stand.",
            "policy_predictions_per_model": 542392, "models_compared": 10,
        },
        "evaluation_outcomes_aggregated": 0,
    }


def compare_historical_cache(snapshot: dict) -> dict:
    """Optional local content comparison; absence never blocks a fresh clone."""
    historical = ROOT / "data/frequency" / snapshot["previous_published_frequency_identity"]["run_id"]
    current = ROOT / "data/frequency" / snapshot["frequency_run_id"]
    if not historical.is_dir():
        return {"available": False, "comparison_performed": False}
    for name in snapshot["frequency_model_inputs"]:
        before = pd.read_parquet(historical / name / "oof_predictions.parquet").drop(columns="run_id")
        after = pd.read_parquet(current / name / "oof_predictions.parquet").drop(columns="run_id")
        pd.testing.assert_frame_equal(before, after, check_exact=True)
    return {"available": True, "comparison_performed": True,
            "models_exactly_equal_excluding_only_run_id": len(snapshot["frequency_model_inputs"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-historical-cache", action="store_true")
    args = parser.parse_args()
    snapshot = verify_inputs()
    if args.compare_historical_cache:
        comparison = compare_historical_cache(snapshot)
        destination = ROOT / "data/s5/historical_cache_comparison.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(json_bytes(comparison))
    destination = ROOT / "reports/s5/input_snapshot.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(json_bytes(snapshot))
    print(f"{snapshot['frequency_run_id']}: verified frequency inputs; wrote {destination}; evaluation outcomes aggregated: 0")


if __name__ == "__main__":
    main()
