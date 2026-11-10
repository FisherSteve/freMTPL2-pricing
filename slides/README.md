# Fachvortrag: Von der Exponierung zum Schadenaufwand

[Gesamte Präsentation](fachvortrag.pdf) · [Nur Hauptvortrag](hauptvortrag.pdf) · [Technische Backups](backup.pdf) · [Sprechernotizen und Zeitbudget](sprechernotizen.md)

Zehn Hauptfolien erläutern die wichtigsten Ergebnisse des historischen französischen Haftpflichtbestands: Quellenabgleich, Alter und Bestandsmix, Geografie und externe Verkehrssicherheitsevidenz, Frequenzmodelle, Schadenhöhe, Merkmalswahl und der begrenzte CTGAN-Pilot. Die 13 getrennten Backups beantworten mathematische und aktuarielle Nachfragen. Das Inhaltsbudget beträgt 11:40 Minuten, mit Übergängen rund zwölf Minuten; es beruht auf Textumfang und Folienbudget, nicht auf einer tatsächlich gehaltenen Probe.

Die Abwägung zwischen zusätzlichem Prognosewert, Erklärbarkeit und bislang ungeprüfter Effektstabilität steht im Hauptvortrag. ONISR und Cerema werden nach Population, Zeitraum, Geografie und Messgröße mit den beobachteten Portfolioergebnissen verglichen. Die explorativen Merkmals- und CTGAN-Ergänzungen ändern die bereits evaluierte vollständige Modellregel nicht.

## Aus vorhandenen Ergebnissen bauen

Im Repositorywurzelverzeichnis:

```powershell
.\.venv\Scripts\python.exe scripts/build_slides.py --compile
.\.venv\Scripts\python.exe scripts/build_slides.py --check
```

Erforderlich sind Python 3.12 (nur Standardbibliothek), Git mit Referenzhistorie und pdfLaTeX. Verwendete TeX-Pakete: Beamer, babel/ngerman, fontenc, inputenc, lmodern, amsmath, amssymb, booktabs, array, tabularx und pgfplots 1.18 einschließlich TikZ. Geprüft mit MiKTeX 26.2 / pdfTeX 1.40.29 unter Windows x64. Andere Plattformen wurden nicht ausgeführt. Der Build ruft weder Modelle noch Generatoren oder Resampling auf.

[Eine Beamer-Quelle](fachvortrag.tex) erzeugt mit je zwei pdfLaTeX-Durchläufen alle drei PDFs. Für Einzelfassungen setzt der Build `MainOnly` beziehungsweise `BackupOnly`. Die Backups beginnen jeweils bei B1. Offene Verweise, fehlende Zeichen und Überläufe brechen den Build ab. TeX-Zwischendateien bleiben unter dem ausgeschlossenen `build/`.

## Zahlen und Quellen

- [evidence.json](evidence.json) enthält Eingabehashes mit Herkunft, Zahlenmakros, Tabellen- und Diagrammwerte sowie Populationen, Einheiten und Unsicherheit je Hauptfolie. Die Kernkennzahlen werden mit den ursprünglichen Scoredateien für Frequenz und Verlust abgeglichen.
- [sources.json](sources.json) beschreibt die für diesen Vortrag gelesenen Originalseiten von CASdatasets, ONISR und Cerema und die Grenzen ihrer Übertragung. Methodische Literatur mit vollständigen Angaben steht in [paper/literatur.bib](../paper/literatur.bib) und im [Begleitpaper](../paper/begleitpaper.pdf).
- Die Aggregate der deskriptiven Bestandsanalyse werden ausdrücklich aus Git-Commit `a44c7e91c481fab9312af66bdc08cb96ffd1ab85` gelesen. Dadurch ändern lokale Replays mit anderen Metadaten die Herkunft nicht. Die übrigen Eingänge stammen aus dem Checkout. Ein flacher Clone benötigt die Referenzhistorie.
- Diagramme sind native PGFPlots-Darstellungen aus [Altersdaten](age.dat), [BM-Standardisierung](bm.dat) und [CTGAN-Differenzen](augmentation.dat). Werte, Achsen und asymmetrische Intervalle stammen aus den aggregierten Berichten, ohne erneute Schätzung.

Die Intervalle gelten bei festen Vorhersagen beziehungsweise festgehaltener Unterstützung. Refit-, Auswahl-, Generator- und unbekannte Haushaltsunsicherheit sowie bisher unbeobachtete Extremverluste sind nicht vollständig erfasst. Es fehlt eine externe, zeitliche und deutsche Zielbestandsvalidierung. Der Vortrag begründet keinen unmittelbar verwendbaren Markttarif.
