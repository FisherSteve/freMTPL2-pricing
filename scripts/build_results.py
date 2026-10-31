"""Build a small German results table from the published numerical evidence.

This is a reporting step: no raw data, prediction, fitting or selection occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows: list[dict], key: str = "model_id") -> dict:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} in result evidence")
    return result


def collect(root: Path) -> dict:
    names = ["reports/s0/source_manifest.json", "reports/s4/scores.json",
             "reports/s5/development_decision.json", "reports/s5/evaluation_release.json",
             "reports/s5/development_loss_scores.json", "reports/s5/evaluation_loss_scores.json",
             "reports/s5/evaluation_frequency_scores.json", "reports/s5/evaluation_report.json"]
    documents = {name: json.loads((root/name).read_text(encoding="utf-8")) for name in names}
    decision = documents[names[2]]
    release = documents[names[3]]
    evaluation = documents[names[-1]]
    if (release["decision_sha256"] != digest(root/names[2])
            or evaluation["decision_sha256"] != digest(root/names[2])
            or evaluation["release_sha256"] != digest(root/names[3])
            or evaluation["run_id"] != decision["run_id"]
            or evaluation["decision_id"] != decision["decision_id"]
            or evaluation["selected_pp_model_id"] != decision["selected_pp_model_id"]
            or release["decision_id"] != decision["decision_id"]
            or not release["evaluation_released"]
            or evaluation["selection_changed_after_evaluation"]
            or decision["evaluation_used_for_selection"]):
        raise ValueError("Results require the unchanged locked evaluation decision")
    identity = decision["identity"]
    for phase in ["development", "evaluation"]:
        report = json.loads((root/f"reports/s5/{phase}_report.json").read_text(encoding="utf-8"))
        names.append(f"reports/s5/{phase}_report.json")
        if report["run_id"] != decision["run_id"] or report["identity"] != decision["identity"]:
            raise ValueError("Phase report identity differs from locked decision")
        if phase == "development" and digest(root/names[-1]) != decision["development_manifest_sha256"]:
            raise ValueError("Development report differs from locked decision")
        path = root/f"reports/s5/{phase}_loss_scores.json"
        local = root/"data/severity"/report["run_id"]/phase/"loss_scores.parquet"
        if local.exists():
            import pandas as pd
            spec = report["artifacts"]["loss_scores"]
            if digest(local) != spec["sha256"]:
                raise ValueError("Local loss-score table differs from its manifest")
            records = json.loads(pd.read_parquet(local).to_json(orient="records", double_precision=15))
            if records != json.loads(path.read_text(encoding="utf-8")):
                raise ValueError("Public loss-score table differs from verified local results")
        else:
            # A reader of the published repository need not download individual
            # predictions. The historical execution record verifies its tables.
            record = root/"reports/s5/reproducibility.json"
            expected = json.loads(record.read_text(encoding="utf-8"))["files_sha256"][f"reports/s5/{phase}_loss_scores.json"]
            if digest(path) != expected:
                raise ValueError("Public loss-score table differs from published execution evidence")
        if any(report["reconciliation"][key] != 0 for key in
               ["excluded_policies", "excluded_claim_rows", "excluded_loss_eur"]):
            raise ValueError("Result summary assumes the registered zero-exclusion population")
    names.append("reports/s5/reproducibility.json")
    frequency_report_path = root/"reports/s4/frequency_report.json"
    frequency_report = json.loads(frequency_report_path.read_text(encoding="utf-8"))
    if (digest(root/"reports/s4/scores.json") != frequency_report["public_tables"]["scores"]["sha256"]
            or frequency_report["source_id"] != identity["source_id"]
            or frequency_report["split_id"] != identity["split_id"]):
        raise ValueError("Frequency table or population differs from its evidence")
    names.append("reports/s4/frequency_report.json")
    frequency = unique(documents["reports/s4/scores.json"])
    development = unique(documents["reports/s5/development_loss_scores.json"])
    held = unique(documents["reports/s5/evaluation_loss_scores.json"])
    selected = decision["selected_pp_model_id"]
    if selected not in held or set(development) != set(held):
        raise ValueError("Candidate identities differ between development and evaluation")
    for row in [*development.values(), *held.values()]:
        if row["run_id"] != decision["run_id"] or row["tweedie_power"] != 1.5:
            raise ValueError("Score run or power differs from the fixed contract")
    loss = [{"model_id": name,
             "development_deviance": development[name]["tweedie_deviance_per_exposure"],
             "evaluation_deviance": held[name]["tweedie_deviance_per_exposure"],
             "evaluation_actual_over_predicted": held[name]["actual_over_predicted"]}
            for name in ["pp_intercept_mean", "pp_hgb_mean", "pp_hgb_gamma", "pp_hgb_log_smear"]]
    chosen = held[selected]
    return {
        "source_id": identity["source_id"], "source_version": documents[names[0]]["cas_version"],
        "source_commit": documents[names[0]]["cas_commit"],
        "preparation_id": identity["preparation_id"], "split_id": identity["split_id"],
        "frequency_input_run_id": identity["frequency_run_id"],
        "severity_run_id": decision["run_id"], "decision_id": decision["decision_id"],
        "selected_model_id": selected,
        "units": {"frequency_deviance": "Poisson-Deviance je Vertragsjahr",
                  "loss_deviance": "Exponierungsgewichtete Jahresraten-Tweedie-Deviance, Power 1,5",
                  "annual_loss_cost": "Historische EUR je Vertragsjahr", "period_loss": "Historische EUR je Vertragsperiode"},
        "development_frequency": [frequency[name] for name in
            ["intercept", "glm_age_bm_a0001", "gam_b20_l10", "hgb_l15"]],
        "loss_comparison": loss,
        "selected_evaluation": chosen,
        "relative_loss_deviance_reduction": 1-chosen["tweedie_deviance_per_exposure"]/held["pp_intercept_mean"]["tweedie_deviance_per_exposure"],
        "evidence_sha256": {name: digest(root/name) for name in sorted(set(names))},
        "limits": ["Keine neue Auswahl nach Evaluation.", "Historische Datenexploration; keine externe oder zeitliche Validierung.",
                   "Feste-Vorhersage-Intervalle enthalten keine Refitting- oder Auswahlunsicherheit.",
                   "Dauer- und Alterskalibrierung sowie Extremschäden begrenzen die Tarifanwendung."]}


def german(value: float, places: int) -> str:
    return f"{value:.{places}f}".replace(".", ",")


def markdown(result: dict) -> str:
    labels = {"pp_intercept_mean": "Intercept × Schadenmittel", "pp_hgb_mean": "HGB × Schadenmittel",
              "pp_hgb_gamma": "HGB × Gamma", "pp_hgb_log_smear": "HGB × Log-Smearing"}
    labels[result["selected_model_id"]] += " (gewählt)"
    lines = ["# Berechnete Kernergebnisse", "",
             "Automatisch aus den [numerischen Nachweisen](../reports/s6/results.json) erzeugt; keine neue Modellwahl.", "",
             f"Quelle `{result['source_id']}`, Split `{result['split_id']}`, Schadenhöhenlauf `{result['severity_run_id']}`.", "",
             "| Produkt | OOF-Verlustdeviance | Evaluations-Verlustdeviance | Verlust-A/P |", "|---|---:|---:|---:|"]
    for row in result["loss_comparison"]:
        lines.append(f"| {labels[row['model_id']]} | {german(row['development_deviance'],6)} | {german(row['evaluation_deviance'],6)} | {german(row['evaluation_actual_over_predicted'],4)} |")
    selected = result["selected_evaluation"]
    lines.extend(["", "Score: exponierungsgewichtete Jahresraten-Tweedie-Deviance mit fester Power 1,5.", "",
        f"Das gewählte Produkt senkt sie gegenüber der Interceptbasis um {german(100*result['relative_loss_deviance_reduction'],2)} %. "
        f"Die vorhergesagten Jahreskosten betragen {german(selected['predicted_annual_loss_cost'],2)} EUR, beobachtet werden {german(selected['observed_annual_loss_cost'],2)} EUR.", "",
        "Die Auswahl bleibt vor der Evaluation festgelegt. Abweichungen in Dauer- und Altersgruppen sowie seltene große Schäden begrenzen die Anwendung. "
        "Ein guter Gesamtscore belegt keinen unmittelbar verwendbaren Tarif. Einzelheiten: [Methodik und Grenzen](schadenhoehe_risikopraemie.md).", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated public results are stale")
    args = parser.parse_args()
    result = collect(ROOT)
    files = {ROOT/"reports/s6/results.json": json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n",
             ROOT/"docs/ergebnisse.md": markdown(result)}
    for path, content in files.items():
        if args.check:
            if not path.exists() or path.read_bytes() != content.encode("utf-8"):
                sys.exit(f"Generated result is stale: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    print("Computed results and German table agree with locked evidence.")


if __name__ == "__main__":
    main()
