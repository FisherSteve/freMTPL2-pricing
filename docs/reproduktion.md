# Die Studie vollständig nachrechnen

## Notebooks und Darstellungen

Die [sieben Notebooks](../notebooks/README.md) enthalten ihre Ergebnisse am regulären Pfad. Die Darstellung liest geprüfte Aggregate. Neue statistische Läufe sind dafür entbehrlich. In der Umgebung aus [requirements.lock.txt](../requirements.lock.txt) führt der folgende Befehl ein Notebook mit einem frischen Kernel aus und speichert die Ergebnisse nach erfolgreicher Ausführung.

```powershell
python scripts/execute_notebook.py notebooks/04_frequency_models.ipynb
```

Paper und Beamer lesen ihre Zahlen aus den Ergebnisdateien. Beide Builds benötigen pdfLaTeX, das Paper zusätzlich BibTeX. Die Befehle und Quellen stehen bei [Paper](../paper/README.md) und [Vortrag](../slides/README.md). Der [CTGAN-Bericht](synthetische_trainingsdaten.md) und das zugehörige Diagramm lassen sich aus dem gespeicherten Pilotbericht mit `python scripts/build_augmentation_results.py` erzeugen. Danach aktualisiert der Notebook-Befehl die Darstellung im siebten Notebook.

## Statistischen Kernlauf reproduzieren

Der gemessene Kernlauf basierte auf `53018083970c9b1e6def77f7783518b9703ca6f0`. Commit `b29b38d95cca5d46b8cdca2336dc1b8af96d2682` enthält die ausgeführten Skripte und Notebook-Quellen mit denselben im Laufnachweis verzeichneten Prüfsummen. In einem separaten Clone unter Windows x64 mit CPython 3.12:

```powershell
git switch --detach b29b38d95cca5d46b8cdca2336dc1b8af96d2682
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts/reproduce.py --fresh
.\.venv\Scripts\python.exe scripts/build_results.py --check
```

Der Runner prüft die Umgebung, lädt die CAS-Dateien mit Prüfsummenabgleich und berechnet Vorbereitung, Vertragsaufteilung, OOF-Basis, Bestandsanalyse, Frequenz, Schadenhöhe und gemeinsame Evaluation. Er erzeugt Abbildungen und führt die fünf damaligen Kernnotebooks in eigenen Kerneln aus. `--fresh` setzt leere Analysecaches voraus. Logs und Laufzeitmessungen stehen unter `data/reproduction/`.

Rohdaten, Vertragstabellen, Vorhersagen und Refits bleiben lokal unter `data/`. Öffentliche JSON-Dateien enthalten aggregierte Ergebnisse. Die dokumentierte Laufzeit und numerische Übereinstimmung beziehen sich auf Windows x64 und CPython 3.12.10.

## Ergebnisse und Quellen zuordnen

Die Quelle ist CASdatasets 1.2-1, Commit `227fb56b8734bdb7c0327a41180e01d2ddaeaf26`, Kennung `cas-744e13a80283a07a`. Die Splitkennung `s2split-fe11d414168572dd` benennt dieselben Vertrags- und Schadenmitgliedschaften in allen Kandidaten. Ihre Identität hängt von Schlüsseln und Aufteilungsregel ab; Modellkennungen enthalten zusätzlich Code, Konfiguration und Software.

Die ursprünglichen Entwicklungsnachweise verwenden CPython 3.12.14: Vorbereitung `s1-cb931cb73e410280` und Frequenzlauf `s4freq-4b7f4f42e310c304`. Vor der Schadenhöhenanalyse wurde die numerisch identische Wiederholung unter CPython 3.12.10 geprüft: Vorbereitung `s1-6b71086d8adc0e3c` und Frequenzeingang `s4freq-737bcb82931ca813`. Der [Eingangsnachweis](../reports/s5/input_snapshot.json) dokumentiert diese Zuordnung. Die Schadenhöhen- und Evaluationsresultate gehören zu `s5sev-2c4bc0d8ad1fd1c1`, Entscheidung `s5decision-df9a82872ec16be6`.

Das [kleine Ergebnis-JSON](../reports/s6/results.json) und die daraus [erzeugte Tabelle](ergebnisse.md) enthalten Einheiten und die Hashes ihrer verwendeten Nachweise. `scripts/build_results.py` prüft die unveränderte Entscheidungskette. Mit lokalen Ergebnissen prüft es auch den JSON-Spiegel gegen die verifizierte aggregierte Parquet-Tabelle; ohne lokale Analysecaches verwendet es die veröffentlichte Ausführungsevidenz. `--check` meldet veraltete Ausgaben. So können weitere Darstellungen dieselben berechneten Werte verwenden.

Die laufgebundenen Grafiken liegen bei den jeweiligen Berichten. `python scripts/build_editorial_figures.py` rendert drei neu beschriftete Darstellungen aus den publizierten Aggregaten. Das [Bildmanifest](../reports/editorial_figures/manifest.json) hält die Eingabe- und Ausgabehashes fest. Die Verzeichnisse `figures/` und `reports/figs/` enthalten ausdrücklich historische Bilder. Alte Notebookausgaben und Kennzahlen dienen nicht als Nachweis korrigierter Modellgüte.

Der [Herkunftsnachweis der Darstellung](../reports/editorial.json) verbindet die ausgeführten Notebooks, die redigierten Abbildungen und die PDF-Fassungen mit dem numerischen Referenzstand. Die ursprünglichen Lauf- und Veröffentlichungsnachweise beschreiben jeweils ihren damaligen Dateisatz. Die Basismodell-Diagnostik für Notebook 02 enthält ausschließlich aggregierte Kalibrierungs- und Konzentrationswerte aus den im Basismodellbericht bezeichneten Parquet-Tabellen.

## Entscheidungen, die die Interpretation bestimmen

- **Einheitliches Verlustziel:** Die Anzahl erfasster positiver Schadenzeilen stimmt in CAS mit der mitgelieferten Anzahl überein. Gemeldete Schäden insgesamt und wirtschaftliche Vollkosten sind damit nicht beobachtet. Primär werden weder Anzahlen noch Exponierungen oder Beträge gekappt.
- **Exponierung:** Das Frequenzmodell skaliert erwartete Anzahlen proportional zur Dauer. Die starke Fehlkalibrierung nach Dauer ist eine Grenze dieser Annahme und des beobachteten Risikomixes. Ein Dauerterm würde unbekannte Nutzung oder Beendigung nicht automatisch erklären.
- **Eigenständiger Modellvergleich:** GLM und penalisiertes Spline-GAM werden tatsächlich abgestimmt und geprüft. Ein begrenzter CPU-Boosting-Benchmark misst den zusätzlichen Vorhersagegewinn. Das Boosting erreicht die kleinste Entwicklungs-OOF-Deviance.
- **Schadenhöhe und Aggregation:** Jede positive Trainingszeile hat Gewicht eins. Die Produktannahme betrifft den bedingten Erwartungswert. Frequenz und Schadenhöhe werden für denselben Vertrag multipliziert, bevor Periodenbeträge oder Jahresportfoliowerte aggregiert werden.
- **Auswahl und Evaluation:** OOF dient der Entwicklung. Die vorab gewählte konstante Schadenhöhe bleibt nach der gemeinsamen Evaluation gewählt. Kleine bessere Evaluationswerte anderer Kandidaten sind sichtbar und führen im festgelegten Verfahren zu keiner neuen Auswahl.
- **Grenzen der Unsicherheit:** Ganze Vertragsziehungen erhalten Nullschäden und Mehrfachschäden zusammen. Die Intervalle gelten bei festen Vorhersagen; erneute Anpassung, Auswahl, unbekannte Haushalte und noch nicht beobachtete Extremschäden fehlen. Eine nahezu passende Gesamtsumme genügt bei starken Segmentabweichungen nicht für einen Tarif.

## Gemessener vollständiger Lauf

Der Lauf vom 2026-10-31 verwendet Windows x64 und CPython 3.12.10 mit allen 63 festgelegten Distributionen. Er startet ohne Rohdaten- oder Analysecache in einem neuen GitHub-Clone mit separat installierter Umgebung. Die neuen Ausführungs- und Berichtsskripte sind durch ihre Quellprüfsummen festgehalten; der Code der statistischen Modelle bleibt unverändert.

Alle 19 Befehle endeten erfolgreich. Einschließlich Download, Kaltstart, Tests, Datenprüfungen, Modellanpassungen, Auswertungen und Notebook-Kerneln dauerte die Folge **28,09 Minuten**. Die höchste im Abstand von 0,2 Sekunden beobachtete Summe des RSS im jeweiligen Prozessbaum betrug **2,45 GiB**. Dies ist eine abgetastete Prozessspeichergröße, kein exakter Spitzenwert oder eindeutig belegter physischer Speicher: gemeinsam genutzte Seiten können mehrfach zählen. Installation, separate Importdiagnostik und nachgelagerter Referenzvergleich sind nicht in diesen Laufzeiten enthalten.

Die 54 aggregierten Vergleichstabellen stimmen nach Entfernung ausschließlich der Laufkennungen exakt mit den veröffentlichten Referenzwerten überein. Alle 14 aktuellen PNGs sind bytegleich. 15 vollständige Vorhersagetabellen stimmen einschließlich Schlüsseln, Modellkennung, Fold, Reihenfolge, Datentypen und Vorhersagefloats exakt überein; allein die softwareabhängige Laufkennung darf abweichen. Alle fünf Notebooks liefen in neuen Kerneln mit insgesamt 32 ausgeführten Codezellen und ohne Zellenfehler. Die Mitgliedschaftsdateien des Splits sind identisch; die Metadaten des damaligen und heutigen Einfrierzeitpunkts bleiben getrennt.

Der [maschinenlesbare Ausführungs- und Vergleichsnachweis](../reports/s6/reproduction.json) enthält sämtliche Einzelbefehle, Log- und Quellprüfsummen, Laufzeiten, Speicherstichproben und Referenzzuordnungen. Der [zusammenfassende Reproduzierbarkeitsnachweis](../reports/s6/reproducibility.json) verbindet dies mit Paketprüfung, statistischen Prüfungen und unabhängigen Nachrechnungen. Gleichheit auf demselben Datenstand ist Reproduktion und keine neue externe Validierung; die Modellauswahl bleibt unverändert.

## Ergänzender Vergleich der Merkmalsmengen

Nach dem Vergleich der vier Paare mit Alter ergänzt `python -m src.feature_ablation` Alter, Bonus-Malus und Area jeweils allein sowie Bonus-Malus + Area. Die [Spezifikation](../config/feature_ablation_protocol.json) begrenzt die Erweiterung auf 40 zusätzliche Frequenzfits auf den eingefrorenen Entwicklungs-OOF-Folds. Die ursprünglichen Vorhersagen werden über ihren geprüften Cache übernommen, neue Fits separat registriert und gespeichert. Die neuen Regeln nutzen dieselben fünf Trainingsclaim-Mittel; die Evaluation wird nicht ausgewertet.

`python scripts/verify_feature_ablation.py` prüft die neuen Quell-/Artefakthashes, die unveränderten ursprünglichen Vorhersagen, vollständige Entwicklungsschlüssel und Foldzuordnungen, Trainingsclaim-Mittel sowie Deviance und Einheiten mit einer unabhängigen Rechenformel. Der gemessene 19-Befehle-Kernlauf endet vor dieser Erweiterung. Ihre [Ergebnisse](../reports/s6a/ablation/report.json) sind separat dokumentiert.
