# Sprechernotizen zum Fachvortrag

Für die zehn Hauptfolien sind **11:40 Minuten Sprech- und Abbildungszeit** vorgesehen. Mit 20–40 Sekunden für Übergänge ergibt sich ein Zeitbudget von rund **12 Minuten**. Die Backups sind für Rückfragen vorgesehen. Das Budget ist anhand des Sprechtexts und der Abbildungen geschätzt.

| Folie | Dauer | Kumuliert | Schwerpunkt |
|---|---:|---:|---|
| 1 | 0:50 | 0:50 | Ziel und zwei Ergebnisse |
| 2 | 1:00 | 1:50 | Population und Quellenabgleich |
| 3 | 1:25 | 3:15 | Altersverlauf und BM-Mischung |
| 4 | 1:30 | 4:45 | Geografie, Mechanismen, ONISR/Cerema |
| 5 | 1:00 | 5:45 | Gründe für die Modellklassen |
| 6 | 1:15 | 7:00 | OOF-Gewinn und Dauerlücke |
| 7 | 1:15 | 8:15 | Schadenhöhe und Verlustziel |
| 8 | 1:30 | 9:45 | Merkmalswahl und Stabilität |
| 9 | 1:05 | 10:50 | Kontrollierter CTGAN-Pilot |
| 10 | 0:50 | 11:40 | Konsequenzen und benötigte Daten |

## 1 · Fragestellung

<!-- speech:start -->
Wie hoch ist der erwartete Schadenaufwand eines Vertrags? Die Prognose verbindet zwei Größen: die Schadenfrequenz je Vertragsjahr und den erwarteten Betrag je Schaden. Die beiden Komponenten haben unterschiedliche Einheiten und werden erst je Vertrag miteinander multipliziert. Die Studie prüft, welche zusätzliche Komplexität sich in der Vorhersage auszahlt.

Zwei Ergebnisse geben die Richtung vor: Histogramm-Boosting mit konstantem Schadenmittel reduziert die Verlustdeviance in der bestehenden Evaluation um 9,80 Prozent gegenüber einer vollständigen Interceptbasis. Eine variable Schadenhöhe hilft in der Entwicklung dagegen nicht. Auch der spätere Pilot mit synthetischen Trainingszeilen liefert keinen Nutzen. Daraus folgt eine begründete Modellregel für diesen Bestand, aber noch kein verwendungsfertiger Tarif.
<!-- speech:end -->

Beleg: `reports/s6/results.json`, `reports/o2/report.json`. Beim Prozentwert ausdrücklich **relative Scoreverbesserung**, keine Einsparung von Schadenaufwand.

## 2 · Daten und Ziel

<!-- speech:start -->
Zunächst muss klar sein, was überhaupt gezählt wird. Verwendet wird der festgelegte CASdatasets-Stand eines unbekannten französischen Haftpflichtversicherers, ungefähr aus den Jahren 2011 bis 2013. Eine Vertragszeile trägt Merkmale und Exponierung, eine Schadenzeile einen positiven Zahlbetrag.

Die Tabelle zeigt, weshalb die Quellenwahl wesentlich ist: CAS und OpenML enthalten dieselbe Zahl verknüpfter positiver Schadenzeilen, aber deutlich unterschiedliche mitgelieferte Anzahlen. Im CAS-Stand stimmen mitgelieferte Anzahl und rekonstruierte Zeilenanzahl überein. Das belegt keine vollständige Erfassung aller wirtschaftlichen Schäden. Die Unterschiede zwischen den Quellen sind dokumentiert, ihre Ursache ist nicht geklärt.

Nullschäden und große Beträge bleiben erhalten. Exponierung wird in Vertragsjahren gemessen. Die Beträge sind historische EUR; teilweise spielen feste Konventionsabrechnungen eine Rolle. Wohnortmerkmale sind keine Unfallorte.
<!-- speech:end -->

Beleg: `reports/s0/source_comparison.json`, Q1. B8 vergleicht den historischen Versicherungsbestand mit den externen Verkehrsdaten.

## 3 · Altersverlauf und Bestandsmix

<!-- speech:start -->
Links steht die beobachtete Frequenz: Anzahl durch Vertragsjahre, mit punktweisen Vertragsbootstrap-Intervallen. Junge eingetragene Fahrer haben eine hohe Rate. Danach fällt sie und steigt im mittleren Alter wieder an. Die mittleren Gruppen sind breit unterstützt; einzelne benachbarte Unterschiede sind trotzdem unsicher. Ich würde hier keinen exakten Altersgipfel benennen.

Rechts wird nur die gemeinsam unterstützte Population betrachtet. Die graue Linie zeigt ihre marginalen Raten, die türkise dieselben Altersgruppen mit gemeinsamer Bonus-Malus-Mischung. Der Wiederanstieg verschwindet nicht; die günstigere BM-Mischung älterer Gruppen verdeckt sogar einen Teil des Unterschieds. Die beiden Diagramme haben unterschiedliche Populationen und lassen sich nicht direkt voneinander abziehen.

Das begründet eine flexible gemeinsame Modellierung von Alter und BM. Es erklärt noch kein Fahrverhalten. Die Intervalle sind weder simultane Bänder noch eine vollständige Unsicherheit aller Modellentscheidungen.
<!-- speech:end -->

Beim Zeigen etwa 10–15 Sekunden für die beiden Linien vorsehen. Belege: gepinnte `reports/s3/age_bootstrap.json`, `bm_standardized.json`, `bm_common_cells.json`, `age_contrasts.json`. Die BM-Referenzanteile sind etwa 65,75/25,52/8,73 Prozent. Modellstandardisierung und Unterstützung in B6.

## 4 · Geografie und externe Evidenz

<!-- speech:start -->
Geografisch nimmt die Frequenz von den ländlicheren zu den urbaneren Wohnortgruppen zu. Dieser Kontrast bleibt auch auf gemeinsamen Mix- und Dauerzellen positiv. Eine Erklärung allein durch die gemessene grobe Zusammensetzung reicht daher nicht. Fahrleistung, Nutzung und Erfassung bleiben aber mögliche Erklärungen.

Beim Schadenmittel ist die Lage schwächer: Das ländlichere Mittel liegt im Vollbestand höher, aber die Kontraste sind unsicher und reagieren auf große Schäden und die ausgewählte Unterstützung. Eine Geschichte von vielen kleinen Stadtunfällen und schweren Landunfällen wäre damit nicht belegt.

Auf Außerortsstraßen ohne Autobahnen entfallen laut ONISR 60 Prozent der Toten und geschätzt 47 Prozent der Schwerverletzten. Cerema findet nach der 80-km/h-Regel geringere Geschwindigkeiten und Mortalität im Straßenvergleich. Das macht Geschwindigkeit als Mechanismus körperlicher Schwere plausibel. Es überträgt sich aber nicht direkt auf diese Wohnort-Zahlbeträge: Andere Verkehrsteilnehmer, andere Zeiten, Unfallort statt Wohnort und Verletzung statt Versicherungszahlung. Auch die adjustierten Altersprofile identifizieren weder Erfahrung noch den tatsächlichen Unfallfahrer.
<!-- speech:end -->

Die Schlussfolgerung betrifft die untersuchten groben Mixzellen, nicht sämtliche unbeobachtete Komposition. B7 zeigt die gemeinsam besetzten Gruppen. B8 erläutert die Methoden der externen Quellen. ONISR 2012 ist zeitnäher, seine Übersichtsseite enthält widersprüchliche Gesamtzahlen; keine davon wird übernommen. Für kausale Trennung fehlen Unfallort/Straßentyp, Kilometer, Geschwindigkeit, Fahrer, Personenschaden und Regulierungsverlauf.

## 5 · Modellklassen

<!-- speech:start -->
Alle Modelle schätzen dieselbe positive Jahresrate. Die Exponierung skaliert daraus die erwartete Anzahl der tatsächlichen Vertragsperiode. Im Anzahlmodell ist ihr Logarithmus ein Offset mit Koeffizient eins; implementiert wird die äquivalente Ratenzielfunktion mit Exponierungsgewichten.

Das GLM bietet einfache gruppierte Relativitäten und prüft auch eine Alter-BM-Interaktion. Das GAM beschreibt nichtlineare Verläufe durch bestrafte Splines, behält aber additive Struktur. Histogramm-Boosting dient als einzelner moderner Benchmark dafür, was diese Struktur möglicherweise verpasst. Es gibt zwei begrenzte Einstellungen derselben Familie.

Kategorien, Skalierung und Knoten entstehen jeweils nur im Training. Für den Vergleich zählen anschließend gehaltene Vorhersagen auf denselben Verträgen. Eine schöne Effektkurve erklärt das Modell; sie validiert seine Vorhersage nicht.
<!-- speech:end -->

Beleg: `config/frequency_protocol.json`, `src/frequency_models.py`. B1 erläutert die Offsetidentität, B3 Strafen und EDF. Keine numerische Gleichsetzung von GLM-alpha und GAM-lambda.

## 6 · Frequenzgewinn und Kalibrierung

<!-- speech:start -->
Die Entwicklung umfasst achtzig Prozent der Verträge mit fünf festen OOF-Folds. Jede Entwicklungsprognose stammt aus einem Modell ohne diesen Vertrag. Erst nach der Modellentscheidung wird die gemeinsame Evaluation geöffnet. Der historische Bestand war früher insgesamt exploriert worden; daraus wird keine externe oder zeitliche Prüfung.

Beide Tabellen zeigen Entwicklungs-OOF. Links sinkt die Frequenzdeviance vom GLM über das GAM zum Boosting. Der ausgewählte HGB liegt auch im gepaarten bedingten Intervall vor dem GAM. Das ist Evidenz innerhalb dieses Suchbudgets, keine universelle Rangfolge von Modellklassen.

Rechts wird die Grenze sichtbar: Trotz fast ausgeglichener Gesamtanzahl unterscheiden sich tatsächliche und erwartete Anzahlen deutlich nach Vertragsdauer. Kurze unterstützte Perioden werden unterschätzt, lange überschätzt. Ein Dauerterm könnte diese Abhängigkeit beschreiben, löst ohne Beginn-, Ende- und Nutzungsinformationen aber nicht automatisch die Frage nach einem prospektiven Jahresrisiko.
<!-- speech:end -->

Belege: Scores des Frequenzvergleichs, `paired_bootstrap.json`, `segments.json`. Die gezeigten Dauergruppen haben mindestens 500 Vertragsjahre. Die unter sieben Tagen liegenden, viel schwächer unterstützten Gruppen bleiben in den Originaltabellen. Intervalle halten die Vorhersagen fest und enthalten keine neue Auswahl oder Refitstreuung.

## 7 · Schadenhöhe und Verlustziel

<!-- speech:start -->
Für die Schadenhöhe werden einzelne positive Trainingsschadenzeilen verwendet, jeweils mit Gewicht eins. Verglichen werden arithmetisches Mittel, Gamma-GLM und ein Logmodell mit Trainings-Smearing. Dieselbe Regel prognostiziert anschließend jeden Vertrag, auch wenn er später schadenfrei bleibt.

In der Entwicklung erzielt das konstante Mittel den besten gemeinsamen Verlustscore. Gamma und Smearing sind in der Evaluation geringfügig besser, aber die gepaarten Differenzintervalle enthalten null. Die Wahl stammt aus der Entwicklung. Die Intervalle zeigen, wie unsicher die kleinen Unterschiede in der Evaluation sind. Gamma verbessert dort den separaten Schaden-Score; das ist kein allgemeines Argument gegen Merkmalsmodelle der Schadenhöhe.

Das gewählte Produkt unterschätzt den aggregierten Jahresbedarf etwas. Zudem trägt ein einziger Entwicklungsschaden 8,58 Prozent der Verlustsumme. Das erklärt die Vorsicht bei Schadenmittelwerten und Unsicherheit. Aggregiert werden erwartete Vertragsbeträge, also Exponierung mal Frequenz mal Schadenhöhe, erst danach wird durch die Gesamtexponierung geteilt.
<!-- speech:end -->

Belege: Verlustscores aus Entwicklung und Evaluation, `evaluation_bootstrap.json`, gepinnte Tail-Tabelle. Der Score hat feste Power 1,5 und ist kein Eurofehler. B4 zeigt log(Y)-Smearing, B5 Produktannahme/Tail. Die Verlust-A/P-Dauerwerte 3,057 beziehungsweise 0,607 stehen in der Originaltabelle und in der Präsentations-Evidenz als `LossDuration`.

## 8 · Merkmalswahl

<!-- speech:start -->
Der spätere reduzierte Vergleich ist explorativ. Ursprünglich stand Alter in jedem Paar fest; die Ergänzung prüft auch Einzelmerkmale und Bonus-Malus plus Area. Alle reduzierten Regeln verwenden dieselben Entwicklungs-OOF-Folds und pro Fold dasselbe arithmetische Mittel der positiven Trainingsbeträge.

Alter plus Bonus-Malus erreicht den kleinsten Score unter den reduzierten Modellen. Seine Deviance liegt gut dreieinhalb Prozent über der des vollständigen Modells. Der Abstand zu BM plus Area ist im bedingten Intervall nicht eindeutig getrennt. Ebenso unsicher bleibt die Trennung von reduziertem GLM und GAM.

Alter und BM sind deutlich abhängig. Das macht Alter nicht vollständig redundant: Neben BM verbessert es den Verlustscore messbar im bedingten Vergleich. Daraus folgt aber keine stabile Aufteilung des Risikos in zwei separat interpretierbare Effekte. Genau dafür fehlen Refit- und Concurvity-Diagnostik. Ein möglicher Vorteil leichter erklärbarer oder weniger überlappender Merkmale muss gegen Prognoseleistung und Effektstabilität abgewogen werden. Auch für Area ist ein Stabilitätsvorteil bislang nicht nachgewiesen.
<!-- speech:end -->

Belege: `reports/s6a/scores.json`, `ablation/scores.json`, `dependence.json`, `bootstrap_two_BonusMalus_gam.json`, `bootstrap_best.json`. B9 enthält alle acht Merkmalsmengen. Eine kleinere Frequenzdeviance von BM + Area gegenüber BM allein bedeutet hier keinen besseren primären Verlustscore. Keine Aussage über sämtliche denkbaren Tarife.

## 9 · Synthetische Zeilen

<!-- speech:start -->
Der CTGAN-Pilot fragt, ob zusätzliche synthetische Zeilen bei begrenzter realer Trainingsmenge helfen. Pro bestehendem Fold werden dieselben zwanzigtausend realen Trainingsverträge verwendet. Verglichen werden real-only, zusätzliche gewöhnliche Bootstrap-Zeilen und CTGAN, mit drei Seeds und zwei Zusatzratios. Bewertet wird ausschließlich auf realen OOF-Verträgen.

Alle gezeigten Differenzen sind positiv: Jede der sechs CTGAN-Varianten liegt hinter beiden Kontrollen. Gegenüber real-only steigen die Deviances um 0,37 bis 3,06 Prozent. Unter diesen Bedingungen hat CTGAN die Prognose verschlechtert.

Das begrenzte 30-Epochen-Experiment widerlegt CTGAN nicht allgemein und untersucht keine Augmentation des vollständigen Bestands. Die Intervalle halten die Vorhersagen fest; Seed- und Generatorstreuung sind darin nicht enthalten. Zusätzliche Zeilen schaffen weder unabhängige Schadenerfahrung noch eine Datenschutzgarantie.
<!-- speech:end -->

Beleg: `reports/o2/report.json`. Legende klar lesen: Türkis mit Intervall gegen real-only, orange Punktwerte gegen jeweilige Bootstrap-Kontrolle. B11 erläutert Samplingkorrektur, Ablehnung und schwachen Area/Density-Support. Der ursprüngliche Algorithmus ist M4, die tatsächliche Implementierung und Lesereichweite stehen im O2-Quellenmanifest.

## 10 · Schluss

<!-- speech:start -->
Verteidigen lässt sich ein nachvollziehbarer Gewinn durch flexible Frequenzstruktur sowie eine begrenzte Empfehlung für einfache Merkmalsmengen. Nicht nachgewiesen sind stabile kausale Einzeleffekte oder die unmittelbare Tarifverwendbarkeit.

Als Nächstes braucht es einen aktuellen, zeitlich gehaltenen Zielbestand und bessere Informationen zu Fahrleistung, Nutzung und Vertragsende. Große Schäden erfordern außerdem passende Schaden- und Regulierungsverläufe. Für Deutschland müssen Deckung, Haftung, Inflation und Merkmalsbedeutung neu abgeglichen werden; französischer Bonus-Malus ist keine deutsche SF-Klasse.

Der Schadenbedarf ist schließlich nur eine Grundlage des Preises. Kosten, Kapital, Vertrieb und Nachfrage brauchen eigene Daten. Ein niedriger prognostizierter Betrag belegt noch keinen profitablen Tarif. Die technischen Einzelheiten und Kostenszenarien liegen als Backups bereit.
<!-- speech:end -->

## Backups: kurze Antworten und Belege

| Backup | Typische Frage | Antwort und Beleg |
|---|---|---|
| B1 | Weshalb Exponierung als Gewicht? | Dieselben parameterabhängigen Poisson-Terme wie beim log(e)-Offset. Jahresrate und Periodenerwartung unterscheiden. Unbestrafte Interceptbalance gilt im Training. `src/frequency_models.py`, Paper Abschnitte 3–4. |
| B2 | Welcher Score? | Poisson je Vertragsjahr; beim Verlust exponierungsgewichtete Jahresraten-Tweedie-Deviance mit p=1,5. Periodenscore ist anders gewichtet. `src/metrics.py`, `src/severity.py`. |
| B3 | Sind die GAMs gleich? | Nein: GAM des vollständigen Frequenzvergleichs mit D2-Strafe auf Poisson-Summe, GAM mit zwei Merkmalen mit Ridge auf normiertem Verlust. Zentrierung und EDF bedingt auf Glättung erläutern. Protokolle und Paper Abschnitt 5. |
| B4 | Warum nicht exp(log-Prognose)? | Der arithmetische Mittelwert benötigt Rücktransformation. Tatsächlich log(Y), globaler Trainings-Smearingfaktor, kein minus eins. Claim-Zeilen mit Gewicht eins. `src/severity_models.py`. |
| B5 | Warum darf man multiplizieren? | Hinreichend ist E[S\|N,x,e]=N m(x) bei endlichen Mitteln. Erst Vertragsprodukte, dann Summen; eine Kappung ändert das erwartete Ziel. Paper Abschnitt 8. |
| B6 | Ist der mittlere Altersanstieg adjustiert? | Auf gemeinsamer grober Unterstützung bleiben die Modellprofile nicht monoton; unterschiedliche Referenzpopulationen nicht direkt subtrahieren. Keine Refitbänder und kein lokalisierter Gipfel. Modellierte Altersprofile und Zellen der Bonus-Malus-Standardisierung. |
| B7 | Erklärt Dauer den geografischen Unterschied? | Nicht isoliert bestimmbar: andere unterstützte Populationen, positiv bleibende Frequenzkontraste, tailabhängige Schadenhöhe. Geografische Standardisierungen und ihre Intervalle. |
| B8 | Was genau stützt ONISR/Cerema? | Körperliche Schwere und Geschwindigkeit auf Straßen; keine verknüpften Wohnort-Vertragszahlungen. Originalseiten Q1–Q3, Abruf 03.11.2026. |
| B9 | Sind Dummies weitere Merkmale? | Es zählen Ursprungsmerkmale. Encoder verändern die Spaltenzahl, nicht die deklarierte Risikomerkmalszahl. Reduzierte Prognose, Überlappung und fehlende Effektstabilität getrennt beurteilen. |
| B10 | Wie belastbar sind die Intervalle? | Ganze Verträge gepaart, Vorhersagen fest; bei O2 Delta-Methode statt Bootstrap. Kein Refit, keine Auswahlkorrektur, keine unbekannten Haushalte oder neuen Extremverluste. |
| B11 | Warum schlechte synthetische Zeilen? | Die gemeinsame Verteilung zeigt Schwächen, aber die Diagnose identifiziert keine alleinige Ursache des Scoreverlusts. Samplingadapter und Ablehnung sind dokumentiert; keine Datenschutzprüfung. |
| B12 | Wie werden Kosten aufgeschlagen? | q+50 versus q/0,85; Schnittpunkt 850/3 EUR/Jahr. Prämienanteil ist kein gleich hoher Aufschlag. Kanal-/Profitabilitätsaussagen bleiben an tatsächliche Vollkosten und Selektion gebunden. |
| B13 | Wo sind Literatur und Zahlen? | Originalberichte, `evidence.json`, `sources.json` und die vollständige Paper-Bibliografie. Modellspezifikationen zusätzlich gegen die lokale Implementierung geprüft. |
