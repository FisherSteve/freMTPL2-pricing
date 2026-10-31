"""Render four German S5 figures from verified, released aggregate evidence.

No fitting, source acquisition, target capping, or model selection occurs here.
The figure manifest identifies every consumed report/table and the exact PNGs.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT as ROOT, DataContractError, json_bytes, root_path, sha256, text_sha256
from src.validation import identity_id


SEVERITY_IDS = ["severity_mean", "severity_gamma", "severity_log_smear"]
PREMIUM_IDS = ["pp_intercept_mean", "pp_hgb_mean", "pp_hgb_gamma", "pp_hgb_log_smear"]
NAMES = {"severity_mean": "Schadenmittel", "severity_gamma": "Gamma-GLM",
    "severity_log_smear": "Log-OLS + Smearing", "pp_intercept_mean": "Basis × Schadenmittel",
    "pp_hgb_mean": "HGB × Schadenmittel", "pp_hgb_gamma": "HGB × Gamma",
    "pp_hgb_log_smear": "HGB × Log-Smearing"}
AGES = ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–64", "65–74", "75+"]


def load_inputs(reports: Path) -> tuple[dict, dict, dict]:
    """Verify release, local/public manifests and every consumed aggregate table."""
    paths = {name: reports / f"{name}.json" for name in
             ["development_report", "evaluation_report", "development_decision", "evaluation_release"]}
    documents = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    development, evaluation = documents["development_report"], documents["evaluation_report"]
    decision, release = documents["development_decision"], documents["evaluation_release"]
    run_id = development["run_id"]
    local = ROOT / "data/severity" / run_id
    if (evaluation["run_id"] != run_id or evaluation["identity"] != development["identity"]
            or identity_id("s5sev-", development["identity"]) != run_id
            or decision["decision_id"] != identity_id("s5decision-", {k: v for k, v in decision.items() if k != "decision_id"})
            or decision["run_id"] != run_id or decision["identity"] != development["identity"]
            or release["run_id"] != run_id or release["evaluation_released"] is not True
            or release["decision_id"] != decision["decision_id"]
            or evaluation["decision_id"] != decision["decision_id"]
            or release["decision_sha256"] != sha256(paths["development_decision"])
            or evaluation["decision_sha256"] != sha256(paths["development_decision"])
            or evaluation["release_sha256"] != sha256(paths["evaluation_release"])
            or evaluation["selection_changed_after_evaluation"] is not False
            or decision["evaluation_power"] != 1.5 or release["tweedie_power"] != 1.5):
        raise DataContractError("Figures require one unchanged, locked and released S5 decision")
    if sha256(local / "development/manifest.json") != decision["development_manifest_sha256"]:
        raise DataContractError("Locked development manifest differs")
    for phase, report in [("development", development), ("evaluation", evaluation)]:
        if json.loads((local / phase / "manifest.json").read_text(encoding="utf-8")) != report:
            raise DataContractError("Figure input public/local manifests differ")
    for name, digest in development["identity"]["code_sha256"].items():
        if text_sha256(ROOT / name) != digest:
            raise DataContractError(f"Recorded model source has changed: {name}")
    inputs = {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in paths.values()}
    wanted = {"development": ["severity_scores", "loss_scores", "bootstrap"],
              "evaluation": ["severity_scores", "loss_scores", "bootstrap", "loss_segments", "loss_components"]}
    tables = {}
    for phase, names in wanted.items():
        report = documents[f"{phase}_report"]
        for name in names:
            spec = report["artifacts"][name]
            if spec["filename"] != f"{name}.parquet":
                raise DataContractError("Unexpected aggregate artifact filename")
            path = local / phase / spec["filename"]
            if sha256(path) != spec["sha256"]:
                raise DataContractError("Figure input table checksum differs")
            frame = pd.read_parquet(path)
            public = reports / f"{phase}_{name}.json"
            expected = json_bytes(json.loads(frame.to_json(orient="records", double_precision=15)))
            if public.read_bytes() != expected:
                raise DataContractError("Figure input public aggregate differs from immutable table")
            inputs[str(public.relative_to(ROOT).as_posix())] = sha256(public)
            tables[f"{phase}_{name}"] = frame
    return documents, tables, inputs


def render(reports: Path, output: Path) -> dict:
    documents, tables, inputs = load_inputs(reports)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    outputs = {}

    def save(fig, name):
        path = output / f"{name}.png"
        fig.savefig(path, dpi=150, metadata={"Software": "freMTPL2 severity evidence"})
        plt.close(fig)
        outputs[name] = {"filename": path.name, "sha256": sha256(path)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(SEVERITY_IDS))
    for phase, shift, color, label in [("development", -.18, "#426b9b", "Entwicklungs-OOF"),
                                       ("evaluation", .18, "#dc8a35", "Evaluation")]:
        values = tables[f"{phase}_severity_scores"].set_index("model_id").loc[SEVERITY_IDS]
        axes[0].bar(x+shift, values.mean_gamma_deviance, width=.36, color=color, label=label)
    held = tables["evaluation_severity_scores"].set_index("model_id").loc[SEVERITY_IDS]
    axes[1].bar(x, held.predicted_claim_mean_eur, color="#426b9b", label="Vorhergesagtes Schadenmittel")
    axes[1].axhline(float(held.observed_claim_mean_eur.iloc[0]), color="#b84535", linestyle="--", label="Beobachtetes Schadenmittel")
    for ax in axes:
        ax.set_xticks(x, [NAMES[m] for m in SEVERITY_IDS], rotation=12)
        ax.legend(fontsize=8)
    axes[0].set(ylabel="Mittlere Gamma-Deviance je positiver Schadenzeile", title="Schadenhöhe: Gewicht eins je Schadenzeile")
    axes[1].set(ylabel="EUR je beobachteter positiver Schadenzeile", title="Schadenhöhenkalibrierung in der Evaluation")
    save(fig, "schadenhoehe_vergleich")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    y = np.arange(len(PREMIUM_IDS))
    for phase, shift, color, label in [("development", -.12, "#426b9b", "Entwicklungs-OOF"),
                                       ("evaluation", .12, "#dc8a35", "Evaluation")]:
        boot = tables[f"{phase}_bootstrap"].set_index("model_id").loc[PREMIUM_IDS]
        for j, model_id in enumerate(PREMIUM_IDS):
            row = boot.loc[model_id]
            # A percentile interval need not surround the original estimate.
            axes[0].plot([row.actual_over_predicted_lower, row.actual_over_predicted_upper], [j+shift]*2, color=color, lw=2)
            axes[0].plot(row.actual_over_predicted, j+shift, "o", color=color, label=label if j == 0 else None)
        values = tables[f"{phase}_loss_scores"].set_index("model_id").loc[PREMIUM_IDS]
        axes[1].bar(y+shift*1.5, values.tweedie_deviance_per_exposure, width=.36, color=color, label=label)
    axes[0].axvline(1, color="grey", linestyle="--", lw=1)
    axes[0].set_yticks(y, [NAMES[m] for m in PREMIUM_IDS])
    axes[0].set(xlabel="Verlust-A/P: beobachtet / vorhergesagt", title="95%-Perzentile bei festen Vorhersagen\nGanze Verträge einschließlich Nullverlusten")
    axes[1].set_xticks(y, [NAMES[m] for m in PREMIUM_IDS], rotation=18, ha="right")
    axes[1].set(ylabel="Exponierungsgewichtete Jahresraten-Tweedie-Deviance", title="Gemeinsamer Verlustscore, Power = 1,5\nKeine EUR-Fehler und keine Periodenscore-Identität")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "risikopraemie_kalibrierung")

    selected = documents["development_decision"]["selected_pp_model_id"]
    components = tables["evaluation_loss_components"]
    segments = tables["evaluation_loss_segments"]
    reference = components.loc[components.model_id.eq(selected)].set_index("age_band").reindex(AGES)
    support = segments.loc[segments.model_id.eq(selected) & segments.dimension.eq("age_band")].set_index("level").reindex(AGES)
    present = reference.policies.notna()
    ages = np.asarray(AGES)[present.to_numpy()]
    reference, support = reference.loc[present], support.loc[present]
    x = np.arange(len(ages))
    sparse = support.exposure_years.lt(500).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    axes[0, 0].plot(x, support.observed_claims/support.exposure_years, "o-", color="#333333", label="Beobachtete Frequenz")
    axes[0, 0].plot(x, reference.mean_annual_frequency_exposure_weighted, "o-", label="Vorhergesagte Frequenz des gewählten Produkts")
    axes[0, 0].set(ylabel="Schäden je Vertragsjahr", title="Frequenz im tatsächlichen Altersgruppenmix")
    axes[0, 1].plot(x, reference.mean_predicted_severity_policy_exposure_weighted, "o-", label=NAMES[selected])
    axes[0, 1].set(ylabel="Exponierungsgewichtetes Modellmittel EUR/Schaden", title="Schadenhöhenvorhersagen auf allen Gruppenverträgen")
    axes[1, 0].plot(x, reference.observed_annual_loss_cost, "o-", color="#333333", label="Beobachteter Verlust / Vertragsjahre")
    for model_id in dict.fromkeys([selected, "pp_hgb_mean", "pp_intercept_mean"]):
        part = components.loc[components.model_id.eq(model_id)].set_index("age_band").reindex(ages)
        axes[1, 0].plot(x, part.mean_product_annual_pp, "o-", label=NAMES[model_id])
    if sparse.any():
        axes[1, 0].scatter(x[sparse], reference.observed_annual_loss_cost.to_numpy()[sparse], s=100, facecolors="none", edgecolors="#b84535", label="Weniger als 500 Vertragsjahre")
    axes[1, 0].set(ylabel="EUR je Vertragsjahr", title="Reine Risikoprämie gegen tatsächliche Verlustkosten")
    for model_id in ["pp_hgb_mean", "pp_hgb_gamma", "pp_hgb_log_smear"]:
        part = components.loc[components.model_id.eq(model_id)].set_index("age_band").reindex(ages)
        axes[1, 1].plot(x, part.mean_product_annual_pp-part.product_of_separate_exposure_means, "o-", label=NAMES[model_id])
    axes[1, 1].axhline(0, color="grey", lw=.7)
    axes[1, 1].set(ylabel="Differenz EUR je Vertragsjahr", title="Mittel(r × m) minus Mittel(r) × Mittel(m)\nIdentische Exponierungsgewichte; keine Kausalwirkung")
    support_labels = [f"{age}\n{years:,.0f} J.".replace(",", ".") for age, years in zip(ages, support.exposure_years)]
    for ax in axes.flat:
        ax.set_xticks(x, support_labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlabel("Fahreraltersgruppe und beobachtete Vertragsjahre")
    fig.suptitle("Evaluation im tatsächlichen Merkmalsmix: Gruppenprofile, keine adjustierten oder kausalen Alterskurven", fontsize=12)
    save(fig, "alter_zerlegung")

    tail = documents["development_report"]["tail_diagnostics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    quantiles = sorted((float(q), float(value)) for q, value in tail["quantiles_eur"].items())
    axes[0].bar(np.arange(len(quantiles)), [value for _, value in quantiles], color="#426b9b")
    axes[0].set_xticks(np.arange(len(quantiles)), [f"{100*q:g} %".replace(".", ",") for q, _ in quantiles])
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Empirisches Quantil positiver Entwicklungs-Schadenzeilen", ylabel="Tatsächlicher Schadenbetrag EUR, logarithmische Achse", title="Ungekappte beobachtete Schadenhöhen\nKeine Verteilung vorhergesagter Prämien")
    influence = pd.DataFrame(tail["influence_refits"]).set_index("model_id").loc[SEVERITY_IDS]
    axes[1].bar(np.arange(len(SEVERITY_IDS)), influence.relative_change*100, color="#dc8a35")
    axes[1].axhline(0, color="grey", lw=.7)
    axes[1].set_xticks(np.arange(len(SEVERITY_IDS)), [NAMES[m] for m in SEVERITY_IDS], rotation=12)
    axes[1].set(ylabel="Änderung der vorhergesagten Portfoliosumme in %", title="Refit ohne größten Verlustvertrag\nGleiche Entwicklungs-Prognoseverträge und Frequenz")
    share_label = f"{tail['largest_claim_share']:.2%}".replace(".", ",")
    fig.suptitle(f"Entwicklungsdiagnostik: größter Schaden {share_label} der Summe; kein Ausschluss im Primärbestand", fontsize=11)
    save(fig, "tail_einfluss")

    manifest = {"run_id": documents["development_report"]["run_id"],
        "decision_id": documents["development_decision"]["decision_id"],
        "source_sha256": {"src/severity_figures.py": text_sha256(Path(__file__))},
        "input_sha256": inputs, "figures": outputs,
        "software": {"matplotlib": version("matplotlib"), "numpy": np.__version__, "pandas": pd.__version__},
        "scope": "Verified released aggregate evidence; fixed-prediction bootstrap intervals; no fitting, reselection, capping or causal claims"}
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    print(f"{manifest['run_id']}: four figures and provenance manifest in {output}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/s5"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = root_path(args.report_dir)
    output = root_path(args.output) if args.output else reports / "figures"
    render(reports, output)


if __name__ == "__main__":
    main()
