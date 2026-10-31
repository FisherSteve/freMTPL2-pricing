# Von der Exponierung zum erwarteten Schadenaufwand

Eine aktuarielle Modellierungsstudie zum französischen Kfz-Haftpflichtportfolio freMTPL2.

**Forschungsfrage:** Wie viel der Vorhersagestruktur erfassen ein interpretierbares Poisson-GLM und ein Spline-GAM, und welchen zusätzlichen Nutzen liefert ein Histogramm-Boosting-Modell als Vergleich? Eine gesonderte Analyse von Schadenhöhe und reiner Risikoprämie untersucht die Auswirkungen auf den erwarteten Schadenaufwand.

**Aktueller Stand: Frequenz-, Schadenhöhen- und Risikoprämienvergleich sind auf gemeinsamen festgelegten Entwicklungs- und Evaluationsverträgen berechnet. Vor der Evaluation gewählt wurden HGB-Frequenz und konstante mittlere Schadenhöhe. Die Verlustdeviance sinkt gegenüber der vollständigen Interceptbasis um 9,80 %, während erhebliche Segmentabweichungen und Tail-Unsicherheit bleiben.** Frühere Leistungskennzahlen und Schlussfolgerungen sind historische, nicht verifizierte Ergebnisse und werden hier nicht als bestätigte Befunde dargestellt.

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

## Was die Entwicklungs-EDA zeigt

Die [fünf dokumentierten Befunde](docs/eda.md) beziehen sich ausschließlich auf die 542.392 Entwicklungsverträge des festgelegten Splits. Die Evaluation bleibt aus neuer Exploration und Auswahl ausgeschlossen.

- Die beobachtete Frequenz ist bei jungen Fahrern hoch und steigt nach einem Rückgang im mittleren Alter erneut an. Schadenhöhe und jährliche Verlustkosten verlaufen anders. Das spricht für flexible Altersverläufe und eine getrennte Schadenhöhenanalyse.
- Eine gemeinsame unterstützte Bonus-Malus-Mischung verstärkt den deskriptiven Altersunterschied. Das ist keine vollständige Adjustierung oder kausale Alterswirkung; Fahrzeug-, Kraftstoff- und geografischer Mix bleiben gemeinsam zu prüfen.
- B12 weist auch bei getrennt betrachteten Fahrzeugen mit Alter null bzw. höherem Alter niedrigere marginale Frequenzen auf. Die unbekannte Markenklasse bleibt erhalten und wird gemeinsam mit dem Fahrzeugmix modelliert.
- Kurze Exponierungen tragen überproportional viele Schäden je Vertragsjahr. Das begründet eine Prüfung der Exponierungsannahme und des Risikomixes; Beendigungsprozesse sind in den Daten nicht beobachtet.
- Die größte Schadenzeile trägt 8,58 % des Entwicklungsverlusts; rund 1 % der Zeilen tragen 38,04 %. Eine Kappung bei 25.000 EUR je Zeile entfernt 27,86 % des Betrags und ändert das Ziel. Der Primärbestand bleibt ungekappt.

[Gruppen, Unsicherheit, alternative Erklärungen und Modellierungsfolgen](docs/eda.md), [berechneter Nachweis mit Laufkennung und Prüfsummen](reports/s3/eda_report.json), [ausführbares Notebook](notebooks/03_development_eda.ipynb). Alle 27 Tabellen sind als aggregierte JSON-Dateien öffentlich lesbar; vollständige Vertragstabellen bleiben lokal. Die Altersintervalle verwenden 500 ganze Vertragsziehungen, während die übrigen Merkmalsgrafiken ausdrücklich Poisson-Referenzintervalle zeigen. Beides ersetzt keine Prüfung unbekannter Abhängigkeit oder unbeobachteter Extremverluste.

```powershell
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/03_development_eda.ipynb --output data/notebooks/03_development_eda.executed.ipynb
```

Das Notebook verifiziert lokale Tabellen, öffentliche Spiegel und Bild-Hashes; der EDA-Befehl berechnet sie aus der geprüften Quelle und der bestehenden Vertragszuordnung. Es werden keine Portfoliomodelle angepasst oder Evaluationsscores erzeugt.

## Die Modellierungsstudie

Der registrierte Vergleich umfasst zehn Konfigurationen einschließlich Basismodell und **50 echte Faltenfits** auf jeweils denselben 542.392 Entwicklungsverträgen. Alle neun Merkmale sind prospektiv; gelernte Kodierungen, Skalierung und Splines werden ausschließlich innerhalb der Trainingsfalte angepasst. Der Quellstand und die ungekappte positive-Schadenzeilen-Zielgröße bleiben unverändert.

| Kandidat | OOF-Deviance je Vertragsjahr | Reduktion zum Intercept | A/P |
|---|---:|---:|---:|
| Exponierungs-Intercept | 0,478094 | 0,00 % | 1,0000 |
| GLM, numerisches Familienminimum | 0,457123 | 4,39 % | 1,0000 |
| GAM, numerisches Familienminimum | 0,454637 | 4,91 % | 1,0000 |
| Histogramm-Poisson, 15 Blätter | 0,446749 | 6,56 % | 0,9996 |

Ausgewählt wird `hgb_l15`: gegenüber dem besten GAM senkt es die Deviance in allen fünf Falten. Die gepaarte 500-Vertragsbootstrap-Auswertung bei festen OOF-Vorhersagen stützt den Unterschied; sie enthält keine Refitting- oder Auswahlunsicherheit. Die Einstellung mit 31 Blättern verbessert den Score nicht. GLM und GAM bleiben substantielle interpretierbare Kandidaten. Die Alters-BM-Interaktion liefert gegenüber dem einfachen additiven GLM lediglich etwa 0,071 % Gewinn bei 24 zusätzlichen Parametern; ihre praktische Notwendigkeit ist damit nicht belegt.

**Gute Aggregatkalibrierung genügt nicht:** Beim ausgewählten Modell beträgt A/P für 7 Tage bis unter 0,1 Vertragsjahre etwa 1,79, für mindestens ein Jahr etwa 0,83. Diese OOF-Abweichungen bleiben nach der Modellierung des beobachteten Merkmalsmixes bestehen. Die Diagnose kann Fehlanpassung, unbeobachtete Nutzung und Beendigungsprozesse nicht trennen. Dies begrenzt eine Tarifanwendung und wird nicht durch den besseren Gesamtscore aufgehoben.

Unter gemeinsam unterstütztem Merkmalsmix bleibt eine höhere modellierte Jahresfrequenz bei gesetztem Alter 45 gegenüber 35 sichtbar; bis 55 sinkt sie wieder. Die Referenz umfasst 291.234 Verträge und 81,73 % der Exponierung der 30- bis 59-Jährigen. Das ist ein deskriptiver adjustierter Modellvergleich mit ausgewiesener grober Unterstützung, keine kausale Alterswirkung.

[Vollständige Methodik, Auswahl und Grenzen](docs/frequenzmodelle.md), [berechnete Kennzahlen aller Einstellungen](reports/s4/scores.json), [gesperrte Entwicklungsentscheidung](reports/s4/development_decision.json), [Laufnachweis mit Prüfsummen](reports/s4/frequency_report.json) und [ausführbares Notebook](notebooks/04_frequency_models.ipynb), Lauf `s4freq-4b7f4f42e310c304`.

```powershell
.\.venv\Scripts\python.exe -m src.frequency
.\.venv\Scripts\python.exe scripts/check_frequency_interaction.py
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/04_frequency_models.ipynb --output data/notebooks/04_frequency_models.executed.ipynb
```

Die Frequenzentscheidung wurde vor der gemeinsamen Evaluation festgelegt. Die anschließende Schadenhöhen- und Verlustbewertung wird im folgenden Abschnitt dokumentiert; die Auswahl bleibt unverändert. Ein marktfähiger Tarif und eine Übertragbarkeit auf Deutschland werden nicht behauptet. GPU-Experimente bleiben optional.

## Schadenhöhe und reine Risikoprämie

Der begrenzte Vergleich verwendet einzelne positive Schadenzeilen mit Gewicht eins: arithmetisches Trainingsschadenmittel, Gamma-GLM mit Log-Link und Log-OLS mit ausschließlich im Training geschätztem Smearing. Alle späteren Verträge erhalten dieselbe prospektive Vorhersageregel, auch die Verträge ohne Schaden. Die Jahresrisikoprämie ist Frequenz × Schadenhöhe; Periodenkosten sind Exponierung × Jahresrisikoprämie.

Die Entwicklungs-OOF-Auswahl bevorzugt die konstante Schadenhöhe. Nach gesperrter Entscheidung wurden alle Varianten gemeinsam auf **135.599 Evaluationsverträgen mit 5.378 Schadenzeilen** bewertet, einschließlich 130.499 Verträgen ohne Verlust. Das ist nach offengelegter historischer Exploration keine externe oder buchstäblich nie zuvor gesehene Validierung.

| Produkt | Entwicklung: Verlustdeviance | Evaluation: Verlustdeviance | Evaluation: Verlust-A/P |
|---|---:|---:|---:|
| Interceptfrequenz × Schadenmittel | 87,233849 | 88,917790 | 1,0488 |
| HGB × Schadenmittel (vorab gewählt) | 79,194844 | 80,206534 | 1,0481 |
| HGB × Gamma | 80,386955 | 80,063797 | 1,0807 |
| HGB × Log-Smearing | 79,370343 | 80,026128 | 1,0308 |

Der gemeinsame Verlustscore ist die exponierungsgewichtete Jahresraten-Tweedie-Deviance mit vorab festgelegter Power 1,5. Gamma und Log-Smearing schneiden auf dem Evaluations-Verlustscore leicht besser ab als die vorab gewählte Konstante; die bedingten gepaarten Intervalle der Unterschiede enthalten null. Die Auswahl wird deshalb nicht nach Einsicht geändert. Für die Schaden-Gamma-Deviance ist das Gamma-Modell in der Evaluation besser als die Konstante; auch dieses Ergebnis bleibt sichtbar.

Das gewählte Produkt prognostiziert **165,60 EUR je Vertragsjahr**, beobachtet werden **173,58 EUR**, A/P **1,048**. Gute Gesamtwerte reichen nicht: Verlust-A/P liegt beispielsweise bei 3,057 für sieben Tage bis unter 0,1 Jahre und 0,607 ab einem Jahr. Schwere Schäden beeinflussen erwartete Summen stark; der zusätzliche Entwicklungsrefit ohne den größten Verlustvertrag verändert die Portfoliovorhersage um etwa 6–9 %. Primär werden keine Beträge gekappt oder Zeilen entfernt.

[Methodik, vollständige Ergebnisse und Grenzen](docs/schadenhoehe_risikopraemie.md), [Entwicklungsentscheidung](reports/s5/development_decision.json), [Evaluationsergebnisse](reports/s5/evaluation_report.json), [Notebook](notebooks/05_severity_pure_premium.ipynb) und [Reproduzierbarkeitsnachweis](reports/s5/reproducibility.json). Lauf `s5sev-2c4bc0d8ad1fd1c1`.

```powershell
.\.venv\Scripts\python.exe scripts/verify_severity_inputs.py
.\.venv\Scripts\python.exe -m src.severity develop
.\.venv\Scripts\python.exe -m src.severity evaluate
.\.venv\Scripts\python.exe -m src.severity_figures
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/05_severity_pure_premium.ipynb --output data/notebooks/05_severity_pure_premium.executed.ipynb
```

Die neue Eingangsprüfung dokumentiert den überprüften Wiederholungslauf mit CPython 3.12.10 und unveränderten numerischen Frequenzvorhersagen. Historische Herkunftsnachweise bleiben ihrem damaligen Lauf zugeordnet. Bootstrap-Intervalle verwenden ganze Verträge bei festen Vorhersagen; sie enthalten keine neue Anpassung, Modellwahl- oder unbekannte Haushaltsunsicherheit. Die integrierte Reproduktion des gesamten Ablaufs folgt als nächster Arbeitsschritt.

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
