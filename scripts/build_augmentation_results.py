"""Render the existing O2 report into German prose/figure; no training or scoring."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from src.data import json_bytes, sha256
    r = json.loads((ROOT / "reports/o2/report.json").read_text(encoding="utf-8"))
    scores = r["scores"]
    real = next(s for s in scores if s["method"] == "real")
    ct = [s for s in scores if s["method"] == "ctgan"]
    boots = {(s["ratio"], s["seed"]): s for s in scores if s["method"] == "bootstrap"}
    figures = ROOT / "reports/o2/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ratio in zip(axes, (.25, 1.)):
        rows = sorted([s for s in ct if s["ratio"] == ratio], key=lambda s: s["seed"])
        x = np.arange(len(rows))
        ax.axhline(0, color="black", linewidth=1)
        ax.bar(x-.18, [boots[(ratio, s["seed"])]["vs_real"]["difference"] for s in rows],
               width=.35, label="Zusätzliche Bootstrap-Zeilen", color="#7290aa")
        ax.bar(x+.18, [s["vs_real"]["difference"] for s in rows], width=.35,
               label="Zusätzliche CTGAN-Zeilen", color="#c47840")
        for offset, values in ((-.18, [boots[(ratio,s["seed"])] for s in rows]), (.18, rows)):
            means = np.array([s["vs_real"]["difference"] for s in values])
            intervals = np.array([s["vs_real"]["conditional_95_interval"] for s in values])
            if len(values):
                ax.errorbar(x+offset, means, yerr=[means-intervals[:,0],intervals[:,1]-means],
                            fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.set(xticks=x, xticklabels=[str(s["seed"]) for s in rows], xlabel="Generator-/Kontrollseed",
               title="Zusätzliche Zeilen je realer Trainingszeile: "+f"{ratio:g}".replace(".",","))
    axes[0].set_ylabel("Deviance-Differenz je Vertragsjahr")
    fig.suptitle("Poisson-Deviance gegenüber rein realem Training",y=.98,fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.89),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.045,"Reale Entwicklungs-OOF-Verträge · 95-%-Intervalle bei festen Vorhersagen · unterschiedliche y-Skalen",ha="center",fontsize=9)
    fig.subplots_adjust(left=.10,right=.985,bottom=.20,top=.68,wspace=.29)
    fig.savefig(figures / "vergleich.png", dpi=150)
    plt.close(fig)
    fmt = lambda x, d=6: f"{x:.{d}f}".replace(".", ",")
    grouped = lambda n: format(n,",").replace(",",".")
    lines = ["| Zusätzliche Zeilen je realer Zeile | Seed | Bootstrap-Deviance | CTGAN-Deviance | CTGAN − reale Basis | CTGAN − Bootstrap | CTGAN A/P |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ct:
        b = boots[(s["ratio"], s["seed"])]
        lines.append(f"| {fmt(s['ratio'],2)} | {s['seed']} | {fmt(b['poisson_deviance_per_exposure'])} | "
            f"{fmt(s['poisson_deviance_per_exposure'])} | {fmt(s['vs_real']['difference'])} | "
            f"{fmt(s['vs_bootstrap']['difference'])} | {fmt(s['actual_over_predicted'],4)} |")
    audits = r["generator_audits"]
    support = json.loads((ROOT / "reports/o2/joint_support.json").read_text(encoding="utf-8"))
    support_rows = support["rows"]
    area_fraction = [s["outside_training_Area_density_fraction"] for s in support_rows]
    area_range = f"{fmt(100*min(area_fraction),2)}–{fmt(100*max(area_fraction),2)} %" if area_fraction else "nicht verfügbar"
    near_copies = sum(s["near_training_copies_at_tolerance_0_001"] for s in support_rows)
    attempted = sum(a["validity"]["attempted"] for a in audits)
    rejected = sum(a["validity"]["rejected"] for a in audits)
    matches = sum(a["fidelity"]["exact_training_row_matches"] for a in audits)
    total_fit = sum(a["worker"]["fit_seconds"] for a in audits)
    def bounds(path):
        vals = [a["fidelity"]["synthetic"][path] for a in audits]
        return f"{fmt(min(vals),4)}–{fmt(max(vals),4)}" if vals else "nicht verfügbar"
    decision = ("Können zusätzliche synthetische Verträge die Frequenzprognose bei kleinem Trainingsbestand verbessern? In diesem Pilot erreicht CTGAN den geforderten konsistenten Vorteil gegenüber beiden Kontrollen nicht."
                if r["decision"] == "discard_for_this_pilot" else
                "Der Pilot erfüllt die vorab festgelegte Bedingung für eine weitere unabhängige Untersuchung. Er belegt noch keinen bestätigten Vorteil im vollständigen Bestand.")
    if len(ct)==6 and all(s["vs_real"]["difference"]>0 and s["vs_bootstrap"]["difference"]>0 for s in ct):
        deterioration = [100*s["vs_real"]["difference"]/real["poisson_deviance_per_exposure"] for s in ct]
        decision += f" Alle sechs CTGAN-Varianten verschlechtern die reale OOF-Deviance gegenüber beiden Kontrollen; gegenüber der rein realen Basis beträgt die Verschlechterung {fmt(min(deterioration),2)}–{fmt(max(deterioration),2)} %."
    document = f"""# Synthetische Trainingsverträge: begrenzter CTGAN-Pilot

{decision} Der Vergleich betrifft ein Poisson-GLM mit jeweils 20.000 realen Trainingsverträgen pro Fold.

## Poisson-Deviance je Vertragsjahr auf realen Entwicklungs-OOF-Verträgen

Die fünf Entwicklungs-OOF-Folds liefern {grouped(real['policies'])} reale OOF-Verträge mit {fmt(real['exposure_years'],2)} Vertragsjahren und {grouped(real['observed_claims'])} erfassten positiven Schäden. Pro Fold wird einmal ein festes, ergebnisunabhängig zufällig gezogenes Trainingssubset von 20.000 Verträgen aus den übrigen vier Folds gewählt. Alle Methoden und Generatorseeds verwenden dasselbe Subset. Die Basisdeviance beträgt **{fmt(real['poisson_deviance_per_exposure'])} je Vertragsjahr**, A/P **{fmt(real['actual_over_predicted'],4)}**. Der vollständige Frequenzvergleich verwendet größere Trainingsmengen.

{chr(10).join(lines)}

![Differenzen der Poisson-Deviance je Vertragsjahr auf realen Entwicklungs-OOF-Verträgen](../reports/o2/figures/vergleich.png)

Kleinere Deviance ist besser. Negative Differenzen sprechen für die jeweils zuerst genannte Methode. Die Fehlerbalken zeigen bedingte 95-%-Intervalle. A/P ist die beobachtete geteilt durch die prognostizierte Schadenanzahl. Die Tabelle zeigt die sechs CTGAN-Varianten mit ihren Kontrollen. Im [Notebook](../notebooks/07_synthetic_augmentation.ipynb) stehen alle {len(scores)} OOF-Varianten einschließlich der rein realen Basis. Die GLM-Spezifikation mit Alters-BM-Interaktion und alpha=0,0001 gilt für alle Fits.

## Verfahren und Abgrenzung zur Originalstudie

[Xu et al. (2019), Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf) modellieren gemischte Tabellen: Eine variationale Gaußmischung repräsentiert numerische Werte durch Komponente und normierten Abstand innerhalb der Komponente. Kategoriale Spalten erhalten One-Hot-Darstellungen. Der Generator verwendet Zufallsrauschen und eine Kategorienbedingung. Beim Training werden seltene Kategorien über logarithmische Häufigkeiten häufiger berücksichtigt. Eine zusätzliche Kreuzentropie bestraft verletzte Bedingungen. Der Critic unterscheidet reale und generierte Zeilengruppen; WGAN-Verlust und Gradientenstrafe stabilisieren dieses Gegenspiel.

Für einen numerischen Wert c und die gezogene Mischungskomponente k lautet der skalierte Abstand (c − mu_k) / (4 sigma_k); zusätzlich wird k kodiert. Die Zuordnung erfolgt anhand der trainierten Mischungswahrscheinlichkeiten. Damit wird kein global normalverteilter Versicherungsbestand vorausgesetzt. Erforderlich bleibt, dass die trainierte gemeinsame Verteilung für den gewünschten realen Vorhersagebestand informativ ist.

Im verwendeten [Release 0.11.1](https://github.com/sdv-dev/CTGAN/tree/v0.11.1) minimiert der Critic schematisch E[C(fake)] − E[C(real)] + Gradientenstrafe, der Generator −E[C(fake)] + bedingte Kreuzentropie; hinzu kommen die registrierten Optimiererparameter. Die Funktion sample_original_condvec verwendet die logarithmischen Kategorienwahrscheinlichkeiten aus dem Training. Für das Sampling stellt restore_natural_sampling die empirischen Häufigkeiten anhand der Trainingszeilen wieder her. Dieser Schritt betrifft die Auswahl der Kategorien beim Ziehen. Die gemeinsame Verteilung wird anschließend an den akzeptierten Zeilen untersucht. Alle diskreten Merkmale einschließlich Anzahl werden als Kategorien gelernt; nur Density und Exposure sind kontinuierlich. Eine kleinere Architektur mit 128 statt 256 Einheiten und 30 statt 300 Epochen begrenzt den Pilot. Das begrenzte Training lässt die Konvergenz des Generators offen. In der Originalstudie ersetzen die erzeugten Zeilen das Training. Dieser Pilot fügt sie zu realen Trainingsdaten hinzu.

[TabDDPM (Kotelnikov et al., 2023)](https://proceedings.mlr.press/v202/kotelnikov23a.html) ist eine veröffentlichte Diffusionsalternative für gemischte Tabellen. Deren allgemeine Benchmarkbefunde entscheiden diese exponierungsgewichtete Versicherungsfrage nicht; eine zweite Generatorfamilie wurde nicht implementiert.

## Trainingsgrenzen, Einheiten und Strafe

CTGAN sieht ausschließlich neun prospektive Merkmale, positive Exposure und recorded_positive_count aus dem jeweiligen Trainingssubset. Reale Vertragskennungen, Fold-/Splitkennungen, Schadenbeträge und Modellvorhersagen fehlen im Generatoreingang. Synthetische Zeilen besitzen eigene Kennungen und Herkunft zum Lauf, Seed, ausgeschlossenen Fold und Hash der realen Trainingskennungen. Sie gelangen ausschließlich ins Training. Alle Normalisierer, Kategorien und GLM-Designparameter werden dort gelernt. Reale Validierungsmerkmale fließen erst in predict ein.

Die Antwort ist Anzahl / Exposure; Gewichte sind Exposure. Kombinierte Gewichte werden global auf die reale Trainings-Exponierung normiert. Der GLM normiert seine Likelihood bereits durch die Gewichtssumme: Die globale Skalierung verändert sein alpha deshalb nicht. Eine vollständige Verdoppelung identischer Zeilen liefert im Regressionstest dieselben Vorhersagen. Bei gewöhnlichem Bootstrap ändern sich relative Gewichte zufällig; genau dieser Effekt dient als Kontrolle. Eine Ratio bezeichnet zusätzliche Zeilen, keinen gleichen Anteil an Vertragsjahren. Die tatsächlichen zusätzlichen Exponierungen sind im lokalen Fit-Audit gespeichert.

## Qualität der synthetischen Zeilen

Innerhalb der festen Budgets wurden {len(audits)} Generatoren ausgeführt. Von {grouped(attempted)} Kandidaten wurden {grouped(rejected)} verworfen ({fmt(100*rejected/attempted if attempted else 0,2)} %); Gründe können sich überschneiden. Es gibt kein nachträgliches Runden oder Clipping. Diskrete Werte müssen im Trainingssupport liegen; Density und Exposure müssen endlich, positiv und innerhalb der jeweiligen Trainingsbereiche sein. Dichte wird vor der Transformation explizit als float übergeben. Die Auswahl nimmt die ersten gültigen Zeilen aus höchstens doppelt so vielen Kandidaten wie benötigt. Die Verteilung der akzeptierten Zeilen hängt damit auch von den Annahmekriterien ab.

Die Nullanteile der akzeptierten vollständigen synthetischen Samples reichen von {bounds('zero_fraction')}; Anteile mit mindestens zwei Schäden von {bounds('count_ge2_fraction')}; jährliche Schadenraten von {bounds('annual_rate')}. Exponierungsquantile, Anzahl/Exponierungs-Zusammenhang, Ratenextreme, Alters-/BM-Support, sämtliche Kategorienhäufigkeiten und unbeobachtete Area/Region-, Alter/BM- und Marke/Kraftstoff-Paarungen sind je Generator im Ergebnis-JSON ausgewiesen. Unbeobachtete Paarungen sind kein Nachweis physischer Unmöglichkeit und werden nicht pauschal verworfen.

Der Hashvergleich findet {grouped(matches)} exakte vollständige Trainingszeilen unter den akzeptierten vollständigen synthetischen Samples. Auch null exakte Treffer belegen weder fehlende ungefähre Memorierung noch Datenschutz. Der Pilot besitzt keine Differential-Privacy-Garantie. Neue Zeilen sind keine neue unabhängige Informationsquelle.

Eine nach der ersten Generatorausgabe ergänzte [Diagnostik des gemeinsamen Supports](../reports/o2/joint_support.json) zählt zusätzlich Kombinationen von Area und Density außerhalb des für dieselbe Area beobachteten Trainingsbereichs. Ihr Anteil liegt über die Generatoren bei **{area_range}**. Das zeigt eine Schwäche der gemeinsamen Verteilung; die empirischen Subset-Grenzen sind kein Beweis physischer Unmöglichkeit. Die Diagnose beschreibt die bereits gezogenen Zeilen. Ein zusätzlicher Nähecheck findet {grouped(near_copies)} Zeilen mit identischer vollständiger diskreter Signatur und höchstens 0,001 Abstand in beiden durch Trainingsbereiche skalierten Koordinaten logDensity und Exposure zu einer Trainingszeile. Auch diese operationalisierte Ähnlichkeit ist keine Datenschutzprüfung.

## Unsicherheit und Entscheidung

Der [Bericht](../reports/o2/report.json) enthält gepaarte Verlustdifferenzen samt konditionalen 95-%-Intervallen und alle fünf Fold-Ergebnisse; [Segmenttabellen](../reports/o2/segments.json) zeigen Area, Alter, Bonus-Malus und Vertragsdauer. Für feste Vorhersagen ist Delta = sum(d_neu − d_basis) / sum(Exposure). Das asymptotische Intervall verwendet die über reale Policen streuenden Beiträge (d_neu − d_basis − Delta × Exposure). Refit-, Generator- und Modellauswahlunsicherheit sind darin nicht enthalten. Seed- und Foldstreuung werden gesondert gelesen; überlappende Trainingsfolds sind keine unabhängigen Wiederholungen. Unbekannte Haushaltsabhängigkeit bleibt offen.

Die Gütewerte stammen aus den realen Entwicklungs-OOF-Folds. Der Pilot ist explorativ. Eine Bestätigung auf unabhängigen Daten steht aus. Das Ergebnis gilt für dieses Trainingsbudget, diese diskrete Kodierung, diese Architektur, 30 Epochen und den festgelegten GLM. Es widerlegt oder bestätigt CTGAN nicht allgemein und erlaubt keine Aussagen über synthetische Schadenhöhen oder Risikoprämien.

## Reproduktion und Aufwand

```powershell
python -m venv data/o2/venv
data/o2/venv/Scripts/python.exe -m pip install -r requirements-o2.lock.txt
data/o2/venv/Scripts/python.exe -m src.augmentation
data/o2/venv/Scripts/python.exe scripts/augmentation_support_diagnostics.py
data/o2/venv/Scripts/python.exe scripts/build_augmentation_results.py
data/o2/venv/Scripts/python.exe scripts/verify_augmentation.py
python scripts/execute_notebook.py notebooks/07_synthetic_augmentation.ipynb
```

Die vollständige Wiederholung setzt die vorbereiteten Vertragsdaten und die feste Aufteilung voraus. Die optionale Umgebung ist von der Kernumgebung getrennt. Ein identischer Laufpfad wird nicht überschrieben; für eine vollständige Wiederholung muss ein separates Checkout mit vorbereitetem Bestand verwendet werden. Pro Generator gelten 300 Sekunden für Prozessstart, Fit und Sampling; nach 7.200 Sekunden startet kein weiterer Generator. Fehler werden gespeichert, ohne Ersatzseeds oder zusätzliche Suche.

Gemessen: {fmt(r['seconds'],1)} Sekunden für den Analyselauf, davon {fmt(total_fit,1)} Sekunden Generatorfits; Installation, nachgelagerte Verifikation und Darstellung sind darin nicht enthalten. Verwendet wurden CPU mit zwei Torch-/BLAS-Threads, CPython {r['identity']['python']}, torch {r['identity']['dependencies']['torch']} und CTGAN {r['identity']['dependencies']['ctgan']}; kein GPU-Fit. Lauf **{r['run_id']}**. Der Bericht enthält Quell-/Konfigurations-/Daten-/Splithashes und Abhängigkeiten; vollständige lokale Gewichte, Kandidaten, synthetische Zeilen und keyed OOF-Vorhersagen liegen ausgeschlossen unter data/o2/. Öffentlich werden aggregierte Ergebnisse bereitgestellt.
"""
    (ROOT / "docs/synthetische_trainingsdaten.md").write_text(document, encoding="utf-8", newline="\n")
    (ROOT / "reports/o2/presentation.json").write_bytes(json_bytes({"run_id": r["run_id"],
        "report_sha256": sha256(ROOT / "reports/o2/report.json"),
        "figure_sha256": sha256(figures / "vergleich.png"), "decision": r["decision"]}))
    print("Rendered O2 results without fits")


if __name__ == "__main__":
    build()
