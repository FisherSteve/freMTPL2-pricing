# Die Studie vollständig nachrechnen

Der unterstützte Ablauf beginnt mit den Rohdateien vom festgelegten CASdatasets-Commit und endet mit aggregierten Ergebnissen, Abbildungen und fünf ausgeführten Notebooks. Die einzelnen Rechenschritte sind dieselben Funktionen, die auch die Notebooks verwenden. Es gibt keine versteckten Notebookvariablen zwischen den Schritten.

## Umgebung und vollständiger Ablauf

In einem separaten Clone unter Windows x64 mit CPython 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts/reproduce.py --fresh
.\.venv\Scripts\python.exe scripts/build_results.py --check
```

Die [Lockdatei](../requirements.lock.txt) legt alle aufgelösten Paketversionen fest. Der Runner führt `pip check`, API-Beispiele und die Tests aus. Er lädt die fehlenden CAS-Dateien von den festgelegten URLs, prüft Größe, SHA-256 und archivierten MD5 und erstellt die ungekappte Datenbasis. Danach folgen schlüsselbasierte Vertragsaufteilung, OOF-Intercept, Entwicklungs-EDA, Frequenzvergleich, separate Interaktionsprüfung, Schadenhöhenentwicklung, gesperrte gemeinsame Evaluation, Abbildungen und berechnete Kernergebnisse. Abschließend laufen alle fünf aktiven Notebooks jeweils in einem neuen Kernel desselben Interpreters.

`--fresh` prüft, dass keine Analyse- oder Notebookcaches vorliegen. Es löscht nichts. Vorhandene Rohdaten, Modelle, Ergebnisse und die Sperre einer geöffneten Evaluation bleiben erhalten. Nach einem fehlgeschlagenen vollständigen Lauf stehen Logs und Teilnachweise unter `data/reproduction/`; die gemeldete Stufe lässt sich mit ihrem einzelnen dokumentierten Befehl untersuchen. Für eine neue vollständige Laufzeitmessung ist ein weiterer Clone erforderlich. Die spezifischen Einzelbefehle stehen im [Notebookverzeichnis](../notebooks/README.md).

Die wissenschaftlichen Befehle veröffentlichen ihre aktuellen aggregierten JSON-Dateien unter `reports/s1` bis `reports/s5`. Ein neuer Python-Patchstand kann andere Laufkennungen und damit andere Datei-Hashes erzeugen, auch bei gleichen Vorhersagen. Die ursprünglichen Reproduzierbarkeitsnachweise behalten ihre damalige Bedeutung; sie sind keine Behauptung einer erneuten Ausführung. Ein separater Clone schützt sie und andere lokale Arbeiten. Der [integrierte Nachweis](../reports/s6/reproduction.json) ordnet die erneut berechneten Ergebnisse den veröffentlichten Referenzdateien zu.

Vollständige Rohdaten, Vertragstabellen, Vorhersagen, Refits und ausgeführte Notebooks bleiben unter `data/` lokal und sind von Git ausgeschlossen. Öffentliche JSON-Tabellen enthalten aggregierte Ergebnisse. Linux/macOS wurden nicht ausgeführt; dort müssen Paketauflösung und numerische Übereinstimmung gesondert geprüft werden. Die Windows-Lockdatei ist keine plattformübergreifende Binärgarantie.

## Ergebnisse und Quellen zuordnen

Die Quelle ist CASdatasets 1.2-1, Commit `227fb56b8734bdb7c0327a41180e01d2ddaeaf26`, Kennung `cas-744e13a80283a07a`. Die Splitkennung `s2split-fe11d414168572dd` benennt dieselben Vertrags- und Schadenmitgliedschaften in allen Kandidaten. Ihre Identität hängt von Schlüsseln und Aufteilungsregel ab; Modellkennungen enthalten zusätzlich Code, Konfiguration und Software.

Die ursprünglichen Entwicklungsnachweise verwenden CPython 3.12.14: Vorbereitung `s1-cb931cb73e410280` und Frequenzlauf `s4freq-4b7f4f42e310c304`. Vor der Schadenhöhenanalyse wurde die numerisch identische Wiederholung unter CPython 3.12.10 geprüft: Vorbereitung `s1-6b71086d8adc0e3c` und Frequenzeingang `s4freq-737bcb82931ca813`. Der [Eingangsnachweis](../reports/s5/input_snapshot.json) dokumentiert diese Zuordnung. Die Schadenhöhen- und Evaluationsresultate gehören zu `s5sev-2c4bc0d8ad1fd1c1`, Entscheidung `s5decision-df9a82872ec16be6`.

Das [kleine Ergebnis-JSON](../reports/s6/results.json) und die daraus [erzeugte Tabelle](ergebnisse.md) enthalten Einheiten und die Hashes ihrer verwendeten Nachweise. `scripts/build_results.py` prüft die unveränderte Entscheidungskette. Mit lokalen Ergebnissen prüft es auch den JSON-Spiegel gegen die verifizierte aggregierte Parquet-Tabelle; ohne lokale Analysecaches verwendet es die veröffentlichte Ausführungsevidenz. `--check` meldet veraltete Ausgaben. So können weitere Darstellungen dieselben berechneten Werte verwenden.

Aktuelle Abbildungen stehen unter `reports/s3/figures`, `reports/s4/figures` und `reports/s5/figures`. Die Verzeichnisse `figures/` und `reports/figs/` enthalten ausdrücklich historische Bilder. Alte Notebookausgaben und Kennzahlen dienen nicht als Nachweis korrigierter Modellgüte.

## Entscheidungen, die die Interpretation bestimmen

- **Einheitliches Verlustziel:** Die Anzahl erfasster positiver Schadenzeilen stimmt in CAS mit der mitgelieferten Anzahl überein. Gemeldete Schäden insgesamt und wirtschaftliche Vollkosten sind damit nicht beobachtet. Primär werden weder Anzahlen noch Exponierungen oder Beträge gekappt.
- **Exponierung:** Das Frequenzmodell skaliert erwartete Anzahlen proportional zur Dauer. Die starke Fehlkalibrierung nach Dauer ist eine Grenze dieser Annahme und des beobachteten Risikomixes. Ein Dauerterm würde unbekannte Nutzung oder Beendigung nicht automatisch erklären.
- **Eigenständiger Modellvergleich:** GLM und penalisiertes Spline-GAM werden tatsächlich abgestimmt und geprüft. Ein begrenzter CPU-Boosting-Benchmark misst den zusätzlichen Vorhersagegewinn. Fehlende GAM-Überlegenheit wäre kein Mangel der Untersuchung.
- **Schadenhöhe und Aggregation:** Jede positive Trainingszeile hat Gewicht eins. Die Produktannahme betrifft den bedingten Erwartungswert. Frequenz und Schadenhöhe werden für denselben Vertrag multipliziert, bevor Periodenbeträge oder Jahresportfoliowerte aggregiert werden.
- **Auswahl und Evaluation:** OOF dient der Entwicklung. Die vorab gewählte konstante Schadenhöhe bleibt nach der gemeinsamen Evaluation gewählt. Kleine bessere Evaluationswerte anderer Kandidaten sind sichtbar und keine Erlaubnis zur Neuwahl.
- **Grenzen der Unsicherheit:** Ganze Vertragsziehungen erhalten Nullschäden und Mehrfachschäden zusammen. Die Intervalle gelten bei festen Vorhersagen; erneute Anpassung, Auswahl, unbekannte Haushalte und noch nicht beobachtete Extremschäden fehlen. Eine nahezu passende Gesamtsumme genügt bei starken Segmentabweichungen nicht für einen Tarif.

Diese Entscheidungen sind Modellierungs- und Bewertungsentscheidungen. Die historischen Einheiten- und Leakagefehler sind im aktiven Ablauf durch geprüfte Verträge ersetzt. Fehlende Nutzungs-, Kalender- und Haushaltsdaten bleiben fehlende Evidenz. GPU, direkte Tweedie-Modelle oder neue Dauerterme sind mögliche spätere Untersuchungen und werden nicht als ausgeführte Verbesserungen dargestellt.

## Gemessener vollständiger Lauf

Der Lauf vom 2026-10-31 verwendet Windows x64 und CPython 3.12.10 mit allen 63 festgelegten Distributionen. Er startet ohne Rohdaten- oder Analysecache in einem neuen GitHub-Clone mit separat installierter Umgebung. Die neuen Ausführungs- und Berichtsskripte sind durch ihre Quellprüfsummen festgehalten; der Code der statistischen Modelle bleibt unverändert.

Alle 19 Befehle endeten erfolgreich. Einschließlich Download, Kaltstart, Tests, Datenprüfungen, Modellanpassungen, Auswertungen und Notebook-Kerneln dauerte die Folge **28,09 Minuten**. Die höchste im Abstand von 0,2 Sekunden beobachtete Summe des RSS im jeweiligen Prozessbaum betrug **2,45 GiB**. Dies ist eine abgetastete Prozessspeichergröße, kein exakter Spitzenwert oder eindeutig belegter physischer Speicher: gemeinsam genutzte Seiten können mehrfach zählen. Installation, separate Importdiagnostik und nachgelagerter Referenzvergleich sind nicht in diesen Laufzeiten enthalten.

Die 54 aggregierten Vergleichstabellen stimmen nach Entfernung ausschließlich der Laufkennungen exakt mit den veröffentlichten Referenzwerten überein. Alle 14 aktuellen PNGs sind bytegleich. 15 vollständige Vorhersagetabellen stimmen einschließlich Schlüsseln, Modellkennung, Fold, Reihenfolge, Datentypen und Vorhersagefloats exakt überein; allein die softwareabhängige Laufkennung darf abweichen. Alle fünf Notebooks liefen in neuen Kerneln mit insgesamt 32 ausgeführten Codezellen und ohne Zellenfehler. Die Mitgliedschaftsdateien des Splits sind identisch; die Metadaten des damaligen und heutigen Einfrierzeitpunkts bleiben getrennt.

Der [maschinenlesbare Ausführungs- und Vergleichsnachweis](../reports/s6/reproduction.json) enthält sämtliche Einzelbefehle, Log- und Quellprüfsummen, Laufzeiten, Speicherstichproben und Referenzzuordnungen. Der [zusammenfassende Reproduzierbarkeitsnachweis](../reports/s6/reproducibility.json) verbindet dies mit Paketprüfung, statistischen Prüfungen und unabhängigen Nachrechnungen. Gleichheit auf demselben Datenstand ist Reproduktion und keine neue externe Validierung; die Modellauswahl bleibt unverändert.
