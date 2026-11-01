# Mathematisch-statistisches Begleitpaper

[PDF lesen](begleitpaper.pdf) · [LaTeX-Quelle](begleitpaper.tex) · [Bibliografie](literatur.bib)

**Von der Exposition zum Schadenbedarf** erklärt die tatsächlich verwendeten Frequenz- und Schadenhöhenmodelle, ihre Annahmen und die Ergebnisse des historischen französischen Kfz-Portfolios. Zum Haupttext kommen Literatur und drei kurze technische Anhänge. Entwicklungs-OOF, Evaluation und die späteren explorativen Zwei-Merkmale- und CTGAN-Vergleiche bleiben getrennt.

Die CTGAN-Ergänzung untersucht zusätzliche synthetische Trainingsverträge bei jeweils 20.000 realen Trainingsverträgen je Fold. Drei Seeds, zwei Ratios und gewöhnliche Bootstrap-Kontrollen werden ausschließlich auf realen OOF-Verträgen verglichen. Alle sechs CTGAN-Varianten verschlechtern die Deviance gegenüber beiden Kontrollen; die Kernmodellwahl bleibt unverändert. Der Haupttext enthält beide Kontrollen und die Diagnose des gemeinsamen Supports. Ein zusätzlicher technischer Anhang leitet das normalisierte Trainingsziel und die tatsächlich verwendeten gepaarten Delta-Methoden-Intervalle her. [Auswertung und vollständige Nachweise](../docs/synthetische_trainingsdaten.md) erläutern auch die explizite Samplingkorrektur im gepinnten Release und die begrenzte Aussagekraft dieses Pilots.

Die Herleitungen behandeln Offset und exponierungsgewichtete Rate, Verlustnormierung und Strafen, Mittelwertkonsistenz mit Momentbedingungen, Trainingsbalance, Spline-Identifizierung und effektive Freiheitsgrade, Schadenzeilengewichte, Log-Smearing, das Frequenz-Schadenhöhen-Produkt sowie Kappung und Kosten. Die ONISR-/Cerema-Evidenz wird inhaltlich mit den beobachteten geografischen und Altersbefunden verglichen.

## Kompilieren

Aus dem Repositorywurzelverzeichnis:

```powershell
python scripts/build_paper.py --compile
python scripts/build_paper.py --check
```

Benötigt werden Python 3.12 (Standardbibliothek), Git mit Repositoryhistorie, pdfLaTeX und BibTeX. Die TeX-Pakete sind `babel` mit `ngerman`, `fontenc`, `inputenc`, `lmodern`, `microtype`, `geometry`, `amsmath`, `amssymb`, `amsthm`, `booktabs`, `array`, `tabularx`, `xcolor`, `hyperref`, `url` und `fancyhdr`; Bibliografiestil `plain`. Geprüft mit Windows x64, Python 3.12.10 und MiKTeX 26.2 / pdfTeX 1.40.29. Andere TeX-Installationen sind nicht ausgeführt.

Das Skript liest ausschließlich vorhandene kleine Berichte und Spezifikationen, erzeugt [Zahlenmakros und Tabellen](generated.tex) und führt pdfLaTeX, BibTeX und zwei weitere pdfLaTeX-Durchläufe aus. Es trainiert keine Modelle und berechnet keine Evaluation neu. Fehlende Belege, geänderte generierte Eingaben, TeX-Fehler, unaufgelöste Verweise oder Überläufe führen zum Abbruch. Temporäre Dateien liegen unter dem ausgeschlossenen `paper/build/`.

Die S3-Berichte `tail_concentration.json` und `eda_report.json` werden aus dem festgelegten Git-Commit `a44c7e91c481fab9312af66bdc08cb96ffd1ab85` gelesen. So bleibt die Zahlenherkunft stabil, wenn lokale Analysewiederholungen deren Laufmetadaten ändern. Die übrigen Eingaben stammen aus dem Checkout; [evidence.json](evidence.json) kennzeichnet die beiden versionierten Eingaben ausdrücklich. Für einen flachen Clone muss die Referenzhistorie zunächst verfügbar gemacht werden.

## Ergebnisse und Quellen nachvollziehen

- [Zahlenherkunft und Eingabehashes](evidence.json): gelesene Artefakte, Makrowerte, Rundung und Tabellenursprünge. Einige Textzahlen sind direkt mit den unmittelbar angegebenen Ergebnisdateien belegt.
- [Geprüfte externe Quellen](sources.json): Primärquellen, Abrufdatum, Gegenstand und Grenzen; die [Bibliografie](literatur.bib) enthält die zitierfähigen Angaben.
- [Vollständige berechnete Ergebnisse](../reports/s6/results.json) und [aktuarische Einordnung](../docs/aktuarische_einordnung.md).
- [Aktueller ergänzender Ergebnisstand](../reports/s6a/release.json), Lauf `s6af633eebaf65aa470`.

Die Bootstrapintervalle der Kernstudie und die Delta-Methoden-Intervalle des CTGAN-Pilots gelten bedingt auf feste Vorhersagen und gegebenenfalls ausgewählte Unterstützung. Das Paper weist keine kausalen Effekte, Profitabilität oder Übertragbarkeit auf einen deutschen Markttarif nach.
