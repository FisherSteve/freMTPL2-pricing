# Von der Exponierung zum erwarteten Schadenaufwand

Eine aktuarielle Modellierungsstudie zum französischen Kfz-Haftpflichtportfolio freMTPL2.

**Forschungsfrage:** Wie viel der Vorhersagestruktur erfassen ein interpretierbares Poisson-GLM und ein Spline-GAM, und welchen zusätzlichen Nutzen liefert ein Histogramm-Boosting-Modell als Vergleich? Eine gesonderte Analyse von Schadenhöhe und reiner Risikoprämie untersucht die Auswirkungen auf den erwarteten Schadenaufwand.

**Aktueller Stand: Datenquelle und Zielgröße geprüft; der korrigierte Modellvergleich steht noch aus.** Frühere Leistungskennzahlen und Schlussfolgerungen sind historische, nicht verifizierte Ergebnisse und werden hier nicht als bestätigte Befunde dargestellt.

## Was bisher verifiziert ist

Der Download und Abgleich zweier festgelegter öffentlicher Datenquellen zeigt: Die Wahl der Datensatzversion verändert auch die Zielgröße der Schadenanzahl.

| Größe | CASdatasets 1.2-1, festgelegter Commit | OpenML 41214/41215, Version 1 |
|---|---:|---:|
| Vertragszeilen | 677,991 | 678,013 |
| Exponierung in Vertragsjahren | 358,482.835463 | 358,499.445463 |
| Mitgelieferte Schadenanzahl | 26,444 | 36,102 |
| Mit Verträgen verknüpfte Schadenzeilen mit positivem Betrag | 26,444 | 26,444 |
| Verknüpfte erfasste Schadenbeträge, EUR | 59,909,216.50 | 59,909,216.50 |
| Nicht zuordenbare Schadenzeilen | 0 | 195 |

Die 195 nicht zuordenbaren OpenML-Zeilen summieren sich auf **EUR 788,714.18**. Unter den in beiden Quellen enthaltenen Verträgen unterscheidet sich die mitgelieferte Schadenanzahl bei 9,095 Verträgen. Das betrifft die Zielgröße der Schadenfrequenz und geht über einen Dateiformatunterschied hinaus. Der Quellenvergleich klärt nicht, warum der Anbieter diese Anzahlen geändert hat.

Die überarbeitete Studie verwendet den **CASdatasets-Datenstand** und modelliert die Anzahl **erfasster Schadenzeilen mit positivem Betrag je Vertragsperiode**. Die mitgelieferte und die rekonstruierte Schadenanzahl werden getrennt aufbewahrt und auf Gleichheit geprüft. Diese Zielgröße ermöglicht eine konsistente Zerlegung in Schadenfrequenz und Schadenhöhe; sie darf nicht als Gesamtheit aller gemeldeten Schäden oder aller wirtschaftlichen Verluste bezeichnet werden.

Die primäre Analyse verwendet ungekappte Anzahlen, Exponierungen und Schadenbeträge und behält Verträge ohne Schaden bei. Mehrere Schadenzeilen desselben Vertrags bleiben erhalten; ohne eindeutige Schaden-ID belegen auch gleiche Beträge keine Dublette. Für die spätere Produktzerlegung aus Jahresfrequenz und mittlerer Schadenhöhe ist zusätzlich eine Annahme an den bedingten Erwartungswert nötig: beispielsweise `E[S | N,x,e] = N*m(x)` bei geeigneten endlichen Momenten, mit Gesamtbetrag `S`, Schadenanzahl `N`, Merkmalen `x`, Exponierung `e` und mittlerer Schadenhöhe `m(x)`. Der Abgleich der Anzahlen allein belegt diese Annahme nicht. Die jährliche reine Risikoprämie und der erwartete Schadenaufwand der tatsächlichen Vertragsperiode werden getrennt ausgewiesen.

Dies sind Ergebnisse des Quellenabgleichs, keine Schätzungen der Modellgüte. Der [berechnete Vergleich](reports/s0/source_comparison.json) und das [Quellenmanifest mit Prüfsummen](reports/s0/source_manifest.json) dokumentieren die Ergebnisse und die verwendeten Dateien, Lauf `s0-2a902c26d6f96e7a`.

## Quellenvergleich reproduzieren

Benötigt wird **64-Bit-Python 3.12**. Die folgende kleine Umgebung ist ausschließlich für den Quellenvergleich vorgesehen. Die alte `requirements.txt` im Projektstamm und die bisherigen Notebooks bilden noch keinen unterstützten, durchgängig ausführbaren Ablauf.

```powershell
# Im Projektstamm (Windows PowerShell)
py -3.12 -m venv .venv-s0
.\.venv-s0\Scripts\python.exe -m pip install -r scripts/requirements-s0.txt
.\.venv-s0\Scripts\python.exe scripts/s0_compare_sources.py --download
.\.venv-s0\Scripts\python.exe -m unittest discover -s tests -v
```

Unter Linux/macOS wird die Umgebung mit `python3.12 -m venv .venv-s0` erstellt. Für dieselben Befehle dient anschließend `.venv-s0/bin/python` als Interpreter. Die Pakete wurden unter Windows x64 mit CPython 3.12.14 geprüft; die [aufgelösten Paketversionen](reports/s0/environment-windows-py312.txt) sind dokumentiert.

`--download` lädt ausschließlich fehlende Dateien und prüft die festgelegten SHA-256-Prüfsummen sowie die vom OpenML-Anbieter angegebenen MD5-Prüfsummen. Rohdateien bleiben im von Git ignorierten Verzeichnis `data/s0/raw/`. Ein erneuter Aufruf ohne `--download` verwendet ausschließlich diese zwischengespeicherten Dateien. `--output data/s0/recheck.json` schreibt einen gesonderten Vergleich zur Reproduzierbarkeitsprüfung. Das Prüfskript schätzt kein Modell, kappt keine Beobachtungen und exportiert keine Analysetabellen.

## Die Modellierungsstudie

Geplant sind ein ausschließlich auf der Exponierung beruhendes Basismodell, ein substanzielles aktuarielles Poisson-GLM, ein penalisiertes Poisson-Spline-GAM und ein Histogramm-Boosting-Modell als Vergleich. GPU-Experimente sind optional.

Vor der Auswahl von Modelltermen untersucht die Analyse Alter, Bonus-Malus und Portfoliozusammensetzung. Insbesondere ist ein vermeintlicher Anstieg im mittleren Alter zunächst eine zu prüfende Frage: Marginale Raten, adjustierte Zusammenhänge und plausible Mechanismen werden unterschieden. Die vorhandenen Variablen können eine Erklärung durch Haushaltszusammensetzung oder weitere Fahrer nicht belegen.

Die Validierung verwendet Aufteilungen auf Vertragsebene, die für verknüpfte Schäden übernommen werden, tatsächliche Out-of-Fold-Vorhersagen im Entwicklungsdatensatz, ausschließlich auf Trainingsdaten geschätzte Vorverarbeitung und ein gesondert festgelegtes Evaluationsprotokoll. Jahresraten, erwartete Schadenanzahlen je Periode, Schadenhöhe, jährliche reine Risikoprämie und Kosten je Periode werden ausdrücklich unterschieden. Die frühere Exploration des gesamten Datensatzes wird offengelegt; eine neue Zufallsaufteilung macht diese Vorgeschichte nicht rückgängig.

Die Erweiterung um Schadenhöhe und reine Risikoprämie wird vollständig evaluiert. Ein erfolgloses Schadenhöhenmodell, ein dem GLM unterlegenes GAM oder ein geringer Zusatznutzen des Boostings sind zulässige Ergebnisse. Ein marktfähiger Tarif oder eine Übertragbarkeit auf Deutschland werden nicht behauptet.

## Daten und Reproduzierbarkeit

- [Berechneter Quellenvergleich](reports/s0/source_comparison.json): Schemata, Kategorien, Vertragsabgleich sowie Anzahl-, Exponierungs- und Schadenssummen.
- [Quellenmanifest](reports/s0/source_manifest.json): Datenstand, Bezugsadressen, Abrufzeitpunkte, Prüfsummen, Quellenangaben und deklarierte Nutzungsbedingungen.
- [Paketversionen der geprüften Umgebung](reports/s0/environment-windows-py312.txt): tatsächlich aufgelöste Abhängigkeiten des Quellenvergleichs.

`notebooks/`, `src/`, `figures/` und `reports/figs/` enthalten die historische Analyse. Ihre gespeicherten Kennzahlen und Abbildungen müssen durch den korrigierten Ablauf ersetzt werden. Die abschließende Darstellung der Studie soll eine kleine mathematische Ausarbeitung und eine 10–15-minütige Präsentation für Vorstellungsgespräche umfassen.

## Quelle, Kontext und Nutzungsbedingungen

Verbindliche Datenquelle: CASdatasets **1.2-1**, Commit [`227fb56b8734bdb7c0327a41180e01d2ddaeaf26`](https://github.com/dutangc/CASdatasets/commit/227fb56b8734bdb7c0327a41180e01d2ddaeaf26). Quellenangabe: C. Dutang und A. Charpentier (2026), *CASdatasets: Insurance datasets*, [DOI 10.57745/P0KHAG](https://doi.org/10.57745/P0KHAG).

Die [Dokumentation der Datenbetreuer](https://github.com/dutangc/CASdatasets/blob/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/man/freMTPL.Rd) beschreibt einen unbekannten privaten Versicherer, ungefähr den Zeitraum 2011–2013, Exponierung in Jahren und teilweise feste Schadenbeträge aus dem IRSA-IDA-Abkommen. Die erfassten Beträge sind historische französische Schadenregulierungsdaten und keine heutigen deutschen Preise.

Der Projektcode steht unter der [MIT-Lizenz](LICENSE). CASdatasets [deklariert GPL ≥2](https://github.com/dutangc/CASdatasets/blob/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/DESCRIPTION); die OpenML-[Frequenzmetadaten](https://www.openml.org/api/v1/json/data/41214) und -[Schadenhöhenmetadaten](https://www.openml.org/api/v1/json/data/41215) deklarieren jeweils CC0. Diese Angaben werden getrennt dokumentiert; die MIT-Lizenz des Projekts lizenziert die Quelldaten nicht um. Rohdatensätze und lokale Referenz-PDFs sind nicht Bestandteil der Veröffentlichung.
