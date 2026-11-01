# Ausführbare Analyse und historische Notebooks

[07_synthetic_augmentation.ipynb](07_synthetic_augmentation.ipynb) zeigt den optionalen CTGAN-Pilot mit gespeicherten Ergebnissen und einer Abbildung. Es liest ausschließlich vorhandene aggregierte Entwicklungsresultate; ein Notebookaufruf trainiert keine Generatoren oder Modelle. Die [Auswertung](../docs/synthetische_trainingsdaten.md) erklärt die begrenzte Trainingsmenge, Kontrollen, Samplingkorrektur und Unsicherheit. Die optionale Umgebung und die vollständigen Befehle stehen dort gesondert, da der Pilot nicht zum ursprünglichen Kernvergleich gehört.

[01_data_preparation.ipynb](01_data_preparation.ipynb) ist der aktuelle Einstieg. Nach dem dokumentierten Download ruft es ausschließlich `src.data.prepare()` und `load_prepared()` auf. Es zeigt Quelle, Einheiten, Verknüpfung, erhaltene Daten und hypothetische Kappungsauswirkungen mit einem frischen Kernel. Relative Datenpfade beziehen sich auf den Projektstamm.

```powershell
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/01_data_preparation.ipynb --output data/notebooks/01_data_preparation.executed.ipynb
```

Das Ausführungswerkzeug verwendet genau den aufrufenden Python-Interpreter und schreibt die ausgeführte Kopie lokal unter `data/notebooks/`. Das versionierte Notebook bleibt ohne gespeicherte Ausgaben. Bei interaktiver Verwendung muss der Kernel dieselbe Projektumgebung verwenden; JupyterLab ist dafür eine optionale Oberfläche.

[02_validation_contracts.ipynb](02_validation_contracts.ipynb) prüft anschließend die gespeicherte Vertragsaufteilung und die tatsächlichen OOF-Vorhersagen des Exponierungs-Basismodells. Es zeigt ausschließlich Entwicklungskennzahlen und die Größen der Aufteilung. Die Ergebnisgrößen der Evaluation werden nicht geladen. Die statistischen Verträge und die Analysegeschichte werden in [docs/validierung.md](../docs/validierung.md) erläutert.

```powershell
.\.venv\Scripts\python.exe -m src.validation freeze
.\.venv\Scripts\python.exe -m src.validation oof-baseline
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/02_validation_contracts.ipynb --output data/notebooks/02_validation_contracts.executed.ipynb
```

`freeze` schreibt eine feste Zuordnung anhand von Vertragsschlüsseln; ein anderer Seed ersetzt eine bereits festgelegte Aufteilung nicht stillschweigend. `oof-baseline` passt fünf Exponierungs-Intercepts ohne Evaluationsbeobachtungen an. Das Notebook selbst passt kein Modell an und erzeugt keine neue Zuordnung, sondern verifiziert die angegebenen lokalen Artefakte und ihre öffentlichen Nachweise.

[03_development_eda.ipynb](03_development_eda.ipynb) liest anschließend die mit `python -m src.eda` berechneten Entwicklungsergebnisse. Es prüft die unveränderlichen lokalen Tabellen, die öffentlichen aggregierten JSON-Spiegel, alle Bild-Hashes und aktuelle Code-/Konfigurationsidentitäten. Die sechs Grafiken und unterstützten Gruppen unterscheiden marginale Muster, deskriptive BM-Standardisierung und nicht identifizierbare Mechanismen. Die [fünf Befunde](../docs/eda.md) enthalten Unsicherheit, Sensitivitäten und konkrete spätere Modellfragen.

```powershell
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/03_development_eda.ipynb --output data/notebooks/03_development_eda.executed.ipynb
```

Keine Evaluationszeile wird exploriert; es entstehen keine Prämienvorhersagen oder GLM-/GAM-/Boosting-Fits. Die 500 Vertragsbootstrap-Ziehungen halten Anzahl, Exponierung und Betrag zusammen. Marginale Poisson-Referenzintervalle sind ausdrücklich annahmegebunden, keine Bänder einer geglätteten Kurve.

[04_frequency_models.ipynb](04_frequency_models.ipynb) prüft den registrierten GLM-/GAM-/Histogramm-Poisson-Vergleich im Entwicklungsbestand. Es liest gespeicherte echte OOF-Vorhersagen, Aggregat-/Segmentkalibrierung, Faltenstabilität, gepaarte Vertragsziehungen und unterstützte standardisierte Altersprofile. Die Tabellen unterscheiden Jahresraten und erwartete Periodenanzahlen. Modellrefits dienen der Erklärung; ihre Profile sind keine OOF-Gütemaße oder kausalen Wirkungen. Dieses Notebook öffnet die Evaluation nicht; deren spätere gemeinsame Bewertung steht im fünften Notebook.

```powershell
.\.venv\Scripts\python.exe -m src.frequency
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/04_frequency_models.ipynb --output data/notebooks/04_frequency_models.executed.ipynb
```

Der erste Modellaufruf registriert die Spezifikation vor allen Portfoliofits und berechnet fünf Fits je Einstellung. Wiederholungen prüfen die unveränderlichen Vorhersageartefakte und berechnen Diagnostik erneut; sie passen vorhandene Modelle nicht nochmals an. Code- oder Konfigurationsänderungen erzeugen andere Laufkennungen. Die [Methodik und Grenzen](../docs/frequenzmodelle.md) erläutern Strafen, Identifizierung, Unterstützung und Unsicherheit.

[05_severity_pure_premium.ipynb](05_severity_pure_premium.ipynb) prüft anschließend den Schadenhöhenvergleich und die gemeinsame Evaluation. Es zeigt echte Entwicklungs-OOF-Ergebnisse, die davor gesperrte Entscheidung, gehaltene Schaden- und Verlustscores, A/P, ganze Vertragsbootstrap-Ziehungen, Segmentkalibrierung und Tail-/Smearing-Diagnostik. Alle Evaluationsverträge einschließlich der Nullverlustverträge erhalten dieselbe prospektive Vorhersageregel. Das Notebook überprüft lokale Modell- und Vorhersagehashes sowie Entscheidung, Freigabe und vier Abbildungen; es passt kein Modell an und wählt nach der Evaluation nichts neu aus.

```powershell
.\.venv\Scripts\python.exe scripts/verify_severity_inputs.py
.\.venv\Scripts\python.exe -m src.severity develop
.\.venv\Scripts\python.exe -m src.severity evaluate
.\.venv\Scripts\python.exe -m src.severity_figures
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/05_severity_pure_premium.ipynb --output data/notebooks/05_severity_pure_premium.executed.ipynb
```

Vor dem zweiten Modellbefehl müssen Entwicklungsentscheidung und Refits feststehen. Die feste Tweedie-Power 1,5 und der exponierungsgewichtete Jahresratenscore werden nicht aus der Evaluation ausgewählt. Der getrennte Periodenscore hat eine andere Exponierungsskalierung. [Schadenhöhe, Produktannahme, Ergebnisse und Grenzen](../docs/schadenhoehe_risikopraemie.md) erläutern, warum die unverändert gewählte konstante Schadenhöhe trotz kleinerer einzelner Evaluationsscores anderer Modelle beibehalten wird und weshalb die Verlustkalibrierung keinen unmittelbar verwendbaren Tarif begründet.

Die vier ursprünglichen Notebooks liegen bytegetreu in [legacy/](legacy/). Ihre gespeicherten Ergebnisse, alten Datenpfade und Abhängigkeiten dokumentieren die frühere Analyse. Sie gehören nicht zum aktuellen ausführbaren Ablauf und belegen keine korrigierte Modellgüte. Der neue Datenstand ist von ihren Ergebnissen getrennt. Das Ausführungswerkzeug lehnt diese historischen Dateien ausdrücklich ab.

[06_actuarial_interpretation.ipynb](06_actuarial_interpretation.ipynb) ergänzt den begrenzten Zwei-Merkmale-Vergleich und die aktuarische Interpretation mit gespeicherten Ergebnissen. Vier Merkmalskombinationen werden als GLM und additives Spline-GAM auf denselben Entwicklungs-OOF-Folds verglichen; die Schadenhöhe bleibt ein Trainingsclaim-Mittel. Das Notebook zeigt Geografie-/Altersmuster, Tail-Einfluss, gemeinsame Unterstützung mit und ohne Dauergruppen, bedingte Intervalle, Segmentkalibrierung und Kostenbeispiele. Die [Einordnung](../docs/aktuarische_einordnung.md) trennt Befunde, externe Mechanismusevidenz und offene Kausalfragen. Nach der Kernanalyse folgen `python -m src.interpretation`, `python scripts/s6a_diagnostics.py` und `python scripts/verify_interpretation.py`; danach kann das Notebook erneut ausgeführt werden. Der dort dokumentierte explizite Cache-Befehl wurde für die Wiederaufnahme nach der Diagnosereparatur verwendet. Das Notebook selbst passt keine Modelle an.

Die Datenvorbereitung enthält keine Modellanpassung und keine Auswahl von Merkmalen, Glättung, Kappungen oder Ausschlüssen. Das zweite Notebook belegt die Validierungsinfrastruktur anhand des Exponierungs-Basismodells; der substantielle Vergleich folgt im vierten Notebook. Inhaltliche Exploration und Auswahl erfolgen im Entwicklungsbestand gemäß dem vorab festgelegten Protokoll. Die vollständige frühere Datenexploration bleibt Teil der dokumentierten Analysegeschichte.
