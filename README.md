# Von der Exponierung zum erwarteten Schadenaufwand

Eine aktuarielle Modellierungsstudie zum französischen Kfz-Haftpflichtportfolio freMTPL2.

**Forschungsfrage:** Wie viel der Vorhersagestruktur erfassen ein interpretierbares Poisson-GLM und ein Spline-GAM, und welchen zusätzlichen Nutzen liefert ein Histogramm-Boosting-Modell als Vergleich? Eine gesonderte Analyse von Schadenhöhe und reiner Risikoprämie untersucht die Auswirkungen auf den erwarteten Schadenaufwand.

**Aktueller Stand: Datenaufbereitung und Messgrößen geprüft, Vertragsaufteilung festgelegt und echte OOF-Vorhersagen des Exponierungs-Basismodells erzeugt. Der GLM-/GAM-/Boosting-Vergleich steht noch aus.** Frühere Leistungskennzahlen und Schlussfolgerungen sind historische, nicht verifizierte Ergebnisse und werden hier nicht als bestätigte Befunde dargestellt.

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

## Datenaufbereitung reproduzieren

Benötigt wird **64-Bit-Python 3.12**. Die [direkten Abhängigkeiten](requirements.txt) und die [vollständig aufgelösten Paketversionen](requirements.lock.txt) ersetzen die frühere, widersprüchliche Umgebung. Die Lockdatei beschreibt die geprüfte Windows-x64-Umgebung; sie ist keine behauptete plattformübergreifende Binärgarantie.

```powershell
# Im Projektstamm (Windows PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/01_data_preparation.ipynb --output data/notebooks/01_data_preparation.executed.ipynb
```

Unter Linux/macOS wird die Umgebung mit `python3.12 -m venv .venv` angelegt und mit `.venv/bin/python -m pip install -r requirements.txt` installiert. Für alle weiteren Befehle dient `.venv/bin/python` als Interpreter. Die tatsächliche Paketauflösung ist dort zusätzlich aufzuzeichnen und zu prüfen; ein Linux-/macOS-Lauf wird bislang nicht behauptet. Interaktives JupyterLab und GPU-Pakete sind optional. Der Notebook-Aufruf benötigt kein JupyterLab und startet einen frischen Kernel mit genau dem aufrufenden Python-Interpreter.

Die Kompatibilitätsprüfung passt kleine künstliche GLM-, GAM- und Boosting-Beispiele an, keine Portfoliomodelle. Sie dokumentiert eine beobachtete DeprecationWarning in der UBRE-Zusammenfassung von pyGAM 0.12.0 unter Python 3.12. Diese Zusammenfassung ist hier nicht als Auswahlkriterium validiert; der spätere Modellvergleich benötigt das eigene geprüfte Scoring.

**Ein Befehl für Download und Vorbereitung:** `python -m src.data --download` lädt ausschließlich fehlende CAS-Rohdateien vom festgelegten Commit. Jede Datei wird vor dem Lesen gegen Größe, SHA-256 und den archivierten MD5 geprüft. Der lokale Cache liegt unter `data/raw/cas-<Quellcommit>/`. Vorhandene abweichende Dateien werden weder ersetzt noch stillschweigend korrigiert; der Lauf bricht mit einer Fehlermeldung ab. Rohdaten und vollständige Analysetabellen bleiben von Git ausgeschlossen.

Die Vorbereitung prüft Schema, Kategorien, exakte ganzzahlige Schlüssel, eindeutige Vertragszeilen, Nullwerte, Endlichkeit, positive Exponierung/Dichte/Schadenbeträge und ganzzahlige Anzahlen. Wissenschaftliche ID-Schreibweisen werden exakt interpretiert und nie gerundet. Region- und Markenbezeichnungen bleiben erhalten. Verwaiste Schadenzeilen oder Unterschiede zwischen mitgelieferter und rekonstruierter Anzahl stoppen den Lauf mit einer Bilanz zur Untersuchung.

Die beiden Tabellen `policies.parquet` und `claims.parquet` entstehen unter `data/prepared/<Vorbereitungskennung>/`, zusammen mit `manifest.json`. Die Vertragsdaten enthalten ursprüngliche Merkmale, `ClaimNb`, `recorded_positive_count` und `recorded_loss_eur`; die Schadenzeilen enthalten `IDpol`, `ClaimAmount` und die technische Zeilenherkunft `claim_row_id`. Diese Kennung ist keine echte Schaden-ID und kein Modellmerkmal. Ein späterer Schadenhöhenlauf verknüpft die Tabellen anhand `IDpol` innerhalb der festgelegten Vertragsaufteilung.

Der [berechnete Vorbereitungsnachweis](reports/s1/preparation_report.json) dokumentiert Quellstand, Rohdatei- und Code-Hashes, Paketversionen, Schema, Einheiten, beide Anti-Joins, Anzahlabgleich, erhaltene und ausgeschlossene Summen sowie Prüfgruppen. Die Vorbereitungskennung umfasst Quelle, Manifest, Konfiguration, Code und Software; zusätzliche Hashes kennzeichnen die exakten Parquet-Dateien und ihren Inhalt in deklarierter Spalten-/Zeilenreihenfolge. Keine dieser Kennungen ist eine Modell- oder Validierungskennung. Unterschiedliche Vorbereitungsversionen erhalten eigene Verzeichnisse.

Ein Wiederholungslauf arbeitet ohne Netzwerk und prüft dieselben Tabellen erneut:

```powershell
.\.venv\Scripts\python.exe -m src.data --report data/s1/preparation_recheck.json
```

Alle relativen CLI-Datenpfade beziehen sich auf den Projektstamm. Das [aktive Notebook](notebooks/01_data_preparation.ipynb) ruft dieselben wiederverwendbaren Funktionen auf und prüft die Tabellen vor der Anzeige. Die vier ursprünglichen Notebooks sind unverändert unter [notebooks/legacy/](notebooks/legacy/) erhalten und gehören nicht zum unterstützten Ablauf.

### Datenprüfung statt automatischer Kappung

Die primäre Datenbasis schließt keine Vertrags- oder Schadenzeile aus und verändert weder Exponierung, Anzahl noch Betrag. Der Nachweis quantifiziert Exponierungen über einem Jahr, kurze Exponierungen, extreme Alter, hohe Anzahlen, B12 und große Schäden sowie die Auswirkungen hypothetischer Kappungen. Überlappende Prüfgruppen dürfen nicht zu einer Ausschlusssumme addiert werden.

Eine Exponierungskappung verändert den Nenner einer Jahresrate; eine Anzahlkappung verändert die Zielgröße und unterbricht die Übereinstimmung mit den unveränderten Schadenzeilen. Eine Betragkappung verändert den erwarteten Schadenaufwand. Für entfernte Anzahlen wird daher kein willkürlicher Eurobetrag zugeordnet. Die Ursachen ungewöhnlicher Exponierungen und Alterswerte sind nicht geklärt. Ihre Erhaltung ist eine explizite Modellierungsentscheidung; sie bestätigt nicht die inhaltliche Richtigkeit jedes Quellwerts.

Die Vorbereitung schätzt keine Encoder, Modelle oder Glättungen. Sie erstellt keine Evaluationsaufteilung und keine Vorhersagen. Ihre vollständige Datenbuchhaltung ist keine Erlaubnis, spätere Evaluationsbeobachtungen zur Modell- oder Merkmalswahl zu verwenden.

### Historischen Quellenvergleich wiederholen

Der gesonderte Vergleich von CAS und OpenML bleibt mit `python scripts/s0_compare_sources.py --download` ausführbar. Er schreibt Rohdateien nach `data/s0/raw/` und den Vergleich nach `reports/s0/source_comparison.json`; `--output data/s0/recheck.json` erstellt eine getrennte Wiederholung. Er verändert keine Analysetabellen und ist für den täglichen CAS-Aufbau nicht erforderlich. Die kleine frühere Vergleichsumgebung bleibt in `scripts/requirements-s0.txt` dokumentiert.

Code, Konfiguration und Nachweise werden mit UTF-8/LF versioniert. Provenienz-Hashes beziehen sich auf die exakten Dateibytes; abweichende Zeilenenden in den identifizierten Quelldateien führen vor der Speicherung zu einem Fehler. Rohdatei-Prüfsummen werden bytegetreu ohne Textnormalisierung berechnet.

## Messgrößen und Validierung reproduzieren

Nach der Datenvorbereitung werden die Vertragszuordnung festgelegt und anschließend echte Out-of-Fold-Vorhersagen im Entwicklungsbestand erzeugt:

```powershell
.\.venv\Scripts\python.exe -m src.validation freeze
.\.venv\Scripts\python.exe -m src.validation oof-baseline
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/02_validation_contracts.ipynb --output data/notebooks/02_validation_contracts.executed.ipynb
```

Die Aufteilung verwendet ausschließlich die Vertragsschlüssel: 80 % Entwicklung mit fünf Falten, 20 % Evaluation, feste Zufallsregel mit Seed `20261029`. Jede Schadenzeile übernimmt die Zuordnung ihres Vertrags. Die Vorhersagen tragen eindeutige Vertragsschlüssel, Falte, Modell- und Laufkennung sowie getrennte Jahresraten und erwartete Periodenanzahlen. Der [Splitnachweis](reports/s2/split_manifest.json) enthält die Identität und Prüfsummen der lokalen Zuordnungstabellen. Eine bereits festgelegte lokale Aufteilung wird nicht stillschweigend durch einen anderen Seed ersetzt.

Der aktuelle [berechnete OOF-Nachweis](reports/s2/baseline_oof_report.json) betrifft ausschließlich das Exponierungs-Basismodell: In jeder Falte wird die Rate aus Anzahl- und Exponierungssumme der vier Trainingsfalten geschätzt. Alle Entwicklungsverträge erhalten genau eine Vorhersage aus einem Fit ohne ihre eigene Falte. Es wurden keine Evaluationsvorhersagen oder Evaluationsscores erzeugt. Das [Validierungsnotebook](notebooks/02_validation_contracts.ipynb) liest und prüft die gespeicherten Artefakte; es passt kein Modell an und teilt keine Verträge neu auf.

Der primäre Frequenzscore ist `sum d(N, Exposure * annual_rate) / sum Exposure`. Er stimmt mit der exponierungsgewichteten mittleren Poisson-Deviance von `N/Exposure` und `annual_rate` überein. Erwartete Anzahlen werden nicht zusätzlich mit Exponierung gewichtet; die Deviance ist kein Prozentwert. Kalibrierung verwendet A/P in der Richtung tatsächlich durch vorhergesagt und zeigt Summen und Unterstützung. Konzentrationskurven ordnen Jahresraten, behandeln Gleichstände gemeinsam und verwenden je Modell dessen eigene kumulierte Exponierung. Die [statistische Erläuterung](docs/validierung.md) beschreibt Formeln, Grenzen, Prüfungen und das vorab begrenzte Auswahlprotokoll.

Die ursprüngliche Exploration des gesamten Datenstands und die retrospektive Quellenwahl bleiben offengelegt. Die neue Evaluation ist keine externe oder buchstäblich nie gesehene Validierung. Ab ihrer Festlegung bleibt sie aus neuer Exploration, Vorverarbeitung und Auswahl ausgeschlossen. Vorhersagen und Scores zur Modellwahl entstehen ausschließlich in der Entwicklung. OOF-Werte, die zur Auswahl dienen, sind Selektionsschätzungen; sie bewerten nicht unverzerrt das gesamte abgestimmte Verfahren.

## Die Modellierungsstudie

Geplant sind ein ausschließlich auf der Exponierung beruhendes Basismodell, ein substanzielles aktuarielles Poisson-GLM, ein penalisiertes Poisson-Spline-GAM und ein Histogramm-Boosting-Modell als Vergleich. GPU-Experimente sind optional.

Vor der Auswahl von Modelltermen untersucht die Analyse Alter, Bonus-Malus und Portfoliozusammensetzung. Insbesondere ist ein vermeintlicher Anstieg im mittleren Alter zunächst eine zu prüfende Frage: Marginale Raten, adjustierte Zusammenhänge und plausible Mechanismen werden unterschieden. Die vorhandenen Variablen können eine Erklärung durch Haushaltszusammensetzung oder weitere Fahrer nicht belegen.

Das [Validierungsprotokoll](config/validation_protocol.json) begrenzt den späteren Vergleich auf höchstens sechs GLM-, sechs GAM- und vier Histogramm-Boosting-Konfigurationen auf gemeinsamen Entwicklungsfalten. Konkrete Kodierungen und Terme werden aus der Entwicklungs-EDA begründet und vor Anpassung registriert. Gelernte Vorverarbeitung wird je Trainingsfalte angepasst; unbekannte Validierungskategorien müssen ausdrücklich behandelt werden. Für Boosting sind feste Iterationszahl und deaktivierter Frühstopp vorgesehen. Jahresraten, erwartete Schadenanzahlen je Periode, Schadenhöhe, jährliche reine Risikoprämie und Kosten je Periode werden getrennt behandelt.

Geplant sind Faltenstabilität und 500 gepaarte Bootstrap-Ziehungen auf Vertragszeilen bei festen Vorhersagen. Solche Intervalle enthalten weder erneutes Fitten noch Modellwahlunsicherheit. Die vorab festgelegte Toleranz von 0,5 % relativer Deviance für praktisch ähnliche Kandidaten ist kein Signifikanztest. Diese späteren Vergleiche wurden noch nicht ausgeführt; es ist kein Modellgewinner festgelegt.

Die Erweiterung um Schadenhöhe und reine Risikoprämie wird vollständig evaluiert. Ein erfolgloses Schadenhöhenmodell, ein dem GLM unterlegenes GAM oder ein geringer Zusatznutzen des Boostings sind zulässige Ergebnisse. Ein marktfähiger Tarif oder eine Übertragbarkeit auf Deutschland werden nicht behauptet.

## Daten und Reproduzierbarkeit

- [Berechneter Quellenvergleich](reports/s0/source_comparison.json): Schemata, Kategorien, Vertragsabgleich sowie Anzahl-, Exponierungs- und Schadenssummen.
- [Quellenmanifest](reports/s0/source_manifest.json): Datenstand, Bezugsadressen, Abrufzeitpunkte, Prüfsummen, Quellenangaben und deklarierte Nutzungsbedingungen.
- [Vorbereitungsnachweis](reports/s1/preparation_report.json) und [Datenvertrag](config/data_contract.json): Zielgröße, Einheiten, Verknüpfung, Prüfgruppen und Tabellenidentität.
- [Paketversionen der geprüften Projektumgebung](reports/s1/environment-windows-py312.json) und [Lockdatei](requirements.lock.txt): tatsächlich aufgelöste Abhängigkeiten.
- [Reproduzierbarkeitsnachweis](reports/s1/reproducibility.json): ausgeführte Prüfungen, Wiederholung aus frischem Download und Cache sowie Notebook- und Tabellenkennungen.
- [Validierungsprotokoll](config/validation_protocol.json) und [statistische Erläuterung](docs/validierung.md): Einheiten, Metriken, Lern- und Auswahlgrenzen.
- [Splitnachweis](reports/s2/split_manifest.json) und [OOF-Basismodellnachweis](reports/s2/baseline_oof_report.json): tatsächliche Entwicklungszuordnung und berechnete Vorhersagebewertung; keine Evaluation.
- [Validierungsprüfungen](reports/s2/reproducibility.json): ausgeführte Tests, unabhängiger Referenzabgleich, bytegleiche Wiederholung und Notebook-Nachweise.
- [Paketversionen des historischen Quellenvergleichs](reports/s0/environment-windows-py312.txt): frühere gesonderte Vergleichsumgebung.

`notebooks/legacy/`, `src/legacy/metrics.py`, `src/features_glm.py`, `figures/` und `reports/figs/` enthalten die historische Analyse. Ihre gespeicherten Kennzahlen und Abbildungen müssen durch den korrigierten Ablauf ersetzt werden. `src/data.py` bildet die neue Datenaufbereitung; `src/metrics.py` und `src/validation.py` enthalten die geprüften Mess- und Validierungsfunktionen. Die aktiven Notebooks dokumentieren diesen ausführbaren Ablauf. Die abschließende Darstellung der Studie soll eine kleine mathematische Ausarbeitung und eine 10–15-minütige Präsentation für Vorstellungsgespräche umfassen.

## Quelle, Kontext und Nutzungsbedingungen

Verbindliche Datenquelle: CASdatasets **1.2-1**, Commit [`227fb56b8734bdb7c0327a41180e01d2ddaeaf26`](https://github.com/dutangc/CASdatasets/commit/227fb56b8734bdb7c0327a41180e01d2ddaeaf26). Quellenangabe: C. Dutang und A. Charpentier (2026), *CASdatasets: Insurance datasets*, [DOI 10.57745/P0KHAG](https://doi.org/10.57745/P0KHAG).

Die [Dokumentation der Datenbetreuer](https://github.com/dutangc/CASdatasets/blob/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/man/freMTPL.Rd) beschreibt einen unbekannten privaten Versicherer, ungefähr den Zeitraum 2011–2013, Exponierung in Jahren und teilweise feste Schadenbeträge aus dem IRSA-IDA-Abkommen. Die erfassten Beträge sind historische französische Schadenregulierungsdaten und keine heutigen deutschen Preise.

Der Projektcode steht unter der [MIT-Lizenz](LICENSE). CASdatasets [deklariert GPL ≥2](https://github.com/dutangc/CASdatasets/blob/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/DESCRIPTION); die OpenML-[Frequenzmetadaten](https://www.openml.org/api/v1/json/data/41214) und -[Schadenhöhenmetadaten](https://www.openml.org/api/v1/json/data/41215) deklarieren jeweils CC0. Diese Angaben werden getrennt dokumentiert; die MIT-Lizenz des Projekts lizenziert die Quelldaten nicht um. Rohdatensätze und lokale Referenz-PDFs sind nicht Bestandteil der Veröffentlichung.
