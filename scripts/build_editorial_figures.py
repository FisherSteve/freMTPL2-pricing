"""Render publication figures from the existing aggregate tables."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "reports/editorial_figures"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    inputs = {}
    figures = {}

    def table(path):
        data = (ROOT / path).read_bytes()
        inputs[path] = sha256(data).hexdigest()
        rows = json.loads(data)
        return pd.DataFrame(rows["rows"] if isinstance(rows, dict) else rows)

    def save(fig, name):
        path = DESTINATION / name
        fig.savefig(path, dpi=140, metadata={"Software": "freMTPL2"})
        plt.close(fig)
        figures[name] = sha256(path.read_bytes()).hexdigest()

    duration = table("reports/s3/exposure_band.json")
    brands = table("reports/s3/b12_new.json")
    tail = table("reports/s3/tail_concentration.json")
    order = ["<1 Tag", "1–<7 Tage", "7 Tage–<0,1 J.", "0,1–<0,5 J.", "0,5–<1 J.", "1+ J."]
    duration = duration.set_index("exposure_band").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    axes[0].bar(duration.index, duration.frequency)
    axes[0].set(xlabel="Vertragsdauer", ylabel="Schäden je Vertragsjahr")
    axes[0].tick_params(axis="x", rotation=40)
    for brand in ("B12", "andere Marken"):
        part = brands.loc[brands.brand12.eq(brand)].set_index("brand_new").loc[["VehAge=0", "VehAge>0"]]
        axes[1].plot(["0 Jahre", "über 0 Jahre"], part.frequency, "o-", label=brand)
    axes[1].set(xlabel="Fahrzeugalter", ylabel="Schäden je Vertragsjahr")
    axes[1].legend()
    axes[2].plot(tail.claim_row_share * 100, tail.loss_share * 100, "o-")
    axes[2].set(xscale="log", xlabel="Anteil der größten Schadenzeilen (%)",
                ylabel="Anteil am Gesamtbetrag (%)")
    fig.suptitle("Schadenfrequenz und Konzentration der Schadenbeträge im Entwicklungsbestand")
    save(fig, "frequenz_und_grossschaeden.png")

    effects = table("reports/s4/effects.json")
    models = {"glm_age_bm_a0001": "GLM", "gam_b20_l10": "GAM", "hgb_l15": "Boosting"}
    features = {"DrivAge": "Fahreralter (Jahre)", "VehAge": "Fahrzeugalter (Jahre)", "BonusMalus": "Bonus-Malus"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, (feature, label) in zip(axes, features.items()):
        for model, group in effects.loc[effects.feature.eq(feature)].groupby("model_id", sort=False):
            ax.plot(group.value.astype(float), group.relativity, label=models[model])
        ax.axhline(1, color="grey", lw=.7)
        ax.set(xlabel=label, ylabel="Relative Jahresfrequenz")
    axes[0].legend()
    fig.suptitle("Frequenzrelativitäten der auf dem Entwicklungsbestand angepassten Modelle")
    save(fig, "frequenzrelativitaeten.png")

    scores = table("reports/s6a/scores.json").sort_values("tweedie_deviance_per_exposure")
    labels = {"pp_hgb_mean": "Boosting × Schadenmittel",
              "two_BonusMalus": "Alter + Bonus-Malus", "two_Density": "Alter + log(Density)",
              "two_Area": "Alter + Area", "two_Region": "Alter + Region"}
    names = []
    for model in scores.model_id:
        if model == "pp_hgb_mean":
            names.append(labels[model])
        else:
            pair, family = model.rsplit("_", 1)
            names.append(f"{labels[pair]} ({family.upper()})")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    ax.barh(names, scores.tweedie_deviance_per_exposure,
            color=["#bc6c25" if model == "pp_hgb_mean" else "#317589" for model in scores.model_id])
    ax.invert_yaxis()
    ax.set(xlabel="Exponierungsgewichtete Tweedie-Deviance der Jahresraten (p = 1,5)",
           title="Verlustdeviance im Entwicklungs-OOF-Vergleich")
    save(fig, "verlustdeviance_merkmalspaare.png")

    manifest = {"inputs_sha256": inputs, "figures_sha256": figures,
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
