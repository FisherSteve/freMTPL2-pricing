# Kfz-Haftpflichtanalyse in sieben Notebooks

Die verlinkten Notebooks enthalten ihre Tabellen und Abbildungen. Alle lesen gespeicherte aggregierte Ergebnisse. Im [Begleitpaper](../paper/begleitpaper.pdf) stehen Modellherleitungen und Literatur.

| Notebook | Fragestellung |
|---|---|
| [01 · Datenbasis](01_data_preparation.ipynb) | Wie werden Vertragsdauer, erfasste Schadenanzahl und Schadenbetrag zusammengeführt? |
| [02 · Validierung](02_validation_contracts.ipynb) | Wie funktionieren Vertragsaufteilung, OOF-Vorhersagen und Poisson-Deviance je Vertragsjahr? |
| [03 · Bestandsanalyse](03_development_eda.ipynb) | Wie unterscheiden sich Frequenz und Verlustkosten nach Alter, Bestandsmix und Vertragsdauer? |
| [04 · Frequenzmodelle](04_frequency_models.ipynb) | Wie viel gewinnen GLM, GAM und Boosting auf denselben Entwicklungsverträgen? |
| [05 · Schadenhöhe und Risikoprämie](05_severity_pure_premium.ipynb) | Verbessern Schadenhöhenmodelle die Verlustprognose und wie fällt die gemeinsame Evaluation aus? |
| [06 · Aktuarische Einordnung](06_actuarial_interpretation.ipynb) | Welche zwei Merkmale tragen die Prognose und wie lassen sich Geografie, Alter und Kosten einordnen? |
| [07 · Synthetische Trainingsverträge](07_synthetic_augmentation.ipynb) | Verbessert CTGAN die Frequenzprognose mit begrenztem realen Training? |

## Gespeicherte Ergebnisse erneut darstellen

Im Projektstamm mit der Umgebung aus [requirements.lock.txt](../requirements.lock.txt):

```powershell
python scripts/execute_notebook.py notebooks/04_frequency_models.ipynb
```

Der Befehl verwendet einen frischen Kernel des aufrufenden Interpreters und schreibt die Ausgaben bei Erfolg an den regulären Notebookpfad. Optional schreibt `--output data/notebooks/04_frequency_models.executed.ipynb` eine lokale Kopie. Die [Reproduktion](../docs/reproduktion.md) beschreibt getrennt, wie die statistischen Ergebnisse berechnet wurden.

[legacy/](legacy/) enthält die erste Projektfassung mit ihren ursprünglichen Ausgaben und Datenpfaden.
