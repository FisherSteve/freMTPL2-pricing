"""Reproduce independent fixed-OOF age x Bonus-Malus GLM contrasts.

This scientific check uses the registered S4 development predictions and
scipy.special.xlogy, independently of the project's metric implementation.
It fits no model and never aggregates evaluation outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DataContractError, json_bytes, sha256
from src.validation import identity_id, open_split, protocol, read_artifacts


def check(run_id: str | None = None) -> dict:
    if run_id is None:
        report_path = ROOT / "reports/s4/frequency_report.json"
        run_id = json.loads(report_path.read_text(encoding="utf-8"))["run_id"]
    if Path(run_id).name != run_id or not run_id.startswith("s4freq-"):
        raise DataContractError("Expected one registered S4 run identifier")
    run = ROOT / "data/frequency" / run_id
    registration_path = run / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    identity = registration["identity"]
    if registration["run_id"] != run_id or identity_id("s4freq-", identity) != run_id:
        raise DataContractError("S4 registration identity mismatch")
    preparation = ROOT / "data/prepared" / identity["preparation_id"]
    prepared = json.loads((preparation / "manifest.json").read_text(encoding="utf-8"))
    policy_path = preparation / "policies.parquet"
    if (prepared["preparation_id"] != identity["preparation_id"]
            or prepared["source_id"] != identity["source_id"]
            or sha256(policy_path) != prepared["analysis_tables"]["policies"]["sha256"]):
        raise DataContractError("Prepared policy/source lineage mismatch")
    frames, frozen = open_split(ROOT / "data/validation", prepared, protocol())
    if frozen["split_id"] != identity["split_id"]:
        raise DataContractError("S4 and frozen policy memberships differ")
    membership = frames["policy_membership"]
    keys = membership.loc[membership.split.eq("development"), ["IDpol", "fold"]]
    policies = pd.read_parquet(policy_path, columns=["IDpol", "recorded_positive_count", "Exposure"])
    d = keys.merge(policies, on="IDpol", validate="one_to_one").sort_values("IDpol").reset_index(drop=True)
    del policies
    exposure = d.Exposure.to_numpy()
    observed = d.recorded_positive_count.to_numpy()
    configurations = registration["registered_configurations"]

    def find(encoding: str, alpha: float) -> str:
        matches = [c["model_id"] for c in configurations
                   if c["family"] == "glm" and c["encoding"] == encoding and c["alpha"] == alpha]
        if len(matches) != 1:
            raise DataContractError("Expected exactly one registered GLM for each declared contrast")
        return matches[0]

    references = [find("additive", 0.0), find("additive", .0001)]
    candidate = find("age_bm_interaction", .0001)
    losses, summaries, hashes = {}, {}, {}
    for model in [*references, candidate]:
        tables, model_report = read_artifacts(run / model)
        if (model_report["run_id"] != run_id or model_report["model_id"] != model
                or model_report["registration_sha256"] != sha256(registration_path)
                or model_report["configuration"] != next(c for c in configurations if c["model_id"] == model)):
            raise DataContractError("OOF model registration mismatch")
        prediction = tables["oof_predictions"].sort_values("IDpol").reset_index(drop=True)
        if (not np.array_equal(prediction.IDpol, d.IDpol) or not prediction.IDpol.is_unique
                or not prediction["split"].eq("development").all()
                or not prediction.model_id.eq(model).all() or not prediction.run_id.eq(run_id).all()
                or not np.array_equal(prediction.fold, d.fold)):
            raise DataContractError("OOF policy/fold/model alignment mismatch")
        expected = prediction.expected_count.to_numpy()
        if (not np.isfinite(expected).all() or (expected <= 0).any()
                or not np.allclose(prediction.annual_rate.to_numpy() * exposure, expected, rtol=1e-12, atol=0)):
            raise DataContractError("Annual-rate/period-count units mismatch")
        loss = 2 * (xlogy(observed, observed) - xlogy(observed, expected) - observed + expected)
        losses[model] = loss
        summaries[model] = {
            "poisson_deviance_per_exposure": float(loss.sum() / exposure.sum()),
            "retained_parameters_by_fold": {
                str(row["fold"]): row["fit"]["retained_design"]["design_columns"]
                for row in model_report["fold_diagnostics"]
            },
        }
        hashes[model] = sha256(run / model / "oof_predictions.parquet")
    contrasts = []
    for reference in references:
        delta = losses[candidate] - losses[reference]
        rng = np.random.Generator(np.random.PCG64(20261030))
        draws = np.empty(500)
        for b in range(500):
            indices = rng.integers(0, len(d), size=len(d))
            draws[b] = delta[indices].sum() / exposure[indices].sum()
        bounds = np.percentile(draws, [2.5, 97.5])
        contrasts.append({
            "candidate": candidate, "reference": reference,
            "deviance_difference": float(delta.sum() / exposure.sum()),
            "relative_deviance_reduction": float(-delta.sum() / losses[reference].sum()),
            "conditional_percentile_95_lower": float(bounds[0]),
            "conditional_percentile_95_upper": float(bounds[1]),
            "fold_deviance_differences": {
                str(fold): float(delta[d.fold.eq(fold)].sum() / exposure[d.fold.eq(fold)].sum())
                for fold in range(1, 6)
            },
        })
    first = contrasts[0]
    first_better_folds = sum(value < 0 for value in first["fold_deviance_differences"].values())
    second_better_folds = sum(value < 0 for value in contrasts[1]["fold_deviance_differences"].values())
    relative_percent = f"{first['relative_deviance_reduction'] * 100:.3f}".replace(".", ",")
    direction = ("Die bedingten Intervalle liefern gerichtete Evidenz für eine Devianceverbesserung. "
                 if all(c["conditional_percentile_95_upper"] < 0 for c in contrasts)
                 else "Die bedingten Intervalle belegen keine durchgehend gerichtete Devianceverbesserung. ")
    practical = ("Der praktische Nutzen ist klein; " if abs(first["relative_deviance_reduction"]) < .005
                 else "Die praktische Relevanz ist anhand der übrigen Diagnostik zu bewerten; ")
    conclusion = (direction + f"Gegenüber dem unbestraften additiven GLM beträgt die relative Verbesserung {relative_percent} Prozent; "
                  f"{first_better_folds} von fünf Falten verbessern sich. Gegenüber dem gleich bestraften additiven GLM verbessern sich "
                  f"{second_better_folds} von fünf Falten. " + practical
                  + "eine Notwendigkeit der Interaktion oder eine kausale Alterswirkung ist nicht belegt.")
    return {
        "run_id": run_id, "source_id": identity["source_id"],
        "preparation_id": identity["preparation_id"], "split_id": identity["split_id"],
        "producer_sha256": sha256(Path(__file__)),
        "scope": "Ausschließlich Entwicklungsverträge und gespeicherte OOF-Vorhersagen; keine zusätzliche Modellanpassung oder Evaluation.",
        "development_policies": len(d), "exposure_years": float(exposure.sum()),
        "observed_claims": int(observed.sum()),
        "input_sha256": {
            "policy_membership": frozen["artifacts"]["policy_membership"]["sha256"],
            "policies": sha256(policy_path), "oof_predictions": hashes,
        },
        "models": summaries, "contrasts": contrasts,
        "calculation": "d(N,mu)=2*(scipy.special.xlogy(N,N)-scipy.special.xlogy(N,mu)-N+mu). Differenz Kandidat minus Referenz, summiert und durch dieselbe Exponierung geteilt. Negative Werte begünstigen die Interaktion.",
        "bootstrap": {
            "replicates": 500, "seed": 20261030, "rng": "NumPy Generator(PCG64)",
            "sampling": "Ganze, nach IDpol sortierte Vertragszeilen mit Zurücklegen; pro Ziehung dieselben Indizes für beide Modelle und den gezogenen Exponierungsnenner. Je Referenz wird derselbe RNG-Seed neu initialisiert.",
            "percentiles": [2.5, 97.5],
            "limits": "Stichprobenstabilität bei festen OOF-Vorhersagen. Keine Refitting-, Auswahl- oder unbekannte Haushaltsunsicherheit; kein Hypothesentest nach Auswahl.",
        },
        "conclusion": conclusion,
        "evaluation_outcomes_aggregated": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Registered run; defaults to the current public frequency report")
    args = parser.parse_args()
    result = check(args.run_id)
    destination = ROOT / "reports/s4/interaction_check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(json_bytes(result))
    print(f"{result['run_id']}: two independent GLM contrasts; evaluation closed; wrote {destination}")


if __name__ == "__main__":
    main()
