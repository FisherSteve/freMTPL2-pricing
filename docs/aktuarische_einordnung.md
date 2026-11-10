# Zwei Tarifmerkmale und die Erklärung des Schadenbedarfs

Wie viel Prognoseleistung bleibt mit zwei Tarifmerkmalen erhalten? Im untersuchten Kreis erreicht das GAM mit Fahreralter und Bonus-Malus den besten Punktwert. Der Vergleich zeigt außerdem, wie die Merkmale überlappen und welche Fragen zur Stabilität offenbleiben. Die geografischen und altersbezogenen Befunde ergänzen diese Prognosefrage um mögliche Erklärungen.

## Daten, Ziel und Einheiten

Die [gepinnte CAS-Dokumentation](https://github.com/dutangc/CASdatasets/blob/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/man/freMTPL.Rd) beschreibt einen unbekannten privaten französischen Versicherer, ungefähr 2011–2013. `Area` bezeichnet ein Wohngebiet von A bis F, `Density` die Einwohnerdichte am Wohnort und `Region` eine historische Vertragsregion. Keines ist der Unfallort. `DrivAge` identifiziert nicht zwingend den Unfallfahrer. Teilweise beruhen Beträge auf der IRSA-IDA-Konvention; sie messen nicht unmittelbar physische Unfallschwere.

Die Vergleichsmodelle verwenden dieselben Entwicklungs-OOF-Verträge wie die vollständige Modellregel. Vertragsjahre und positive Beträge bleiben ungekappt. Die Merkmalsvergleiche entstanden nach der gemeinsamen Evaluation. Sie verwenden ausschließlich Entwicklungsdaten und sind explorativ. [Validierung](validierung.md) · [Kernergebnisse](ergebnisse.md)

Je Vertrag gilt `q_i=r_i*m_i` in EUR/Jahr und `C_i=e_i*q_i` in EUR je Periode. Erst Vertragsprodukte werden summiert. `sum(C_i)/sum(e_i)` ist der jährliche Schadenbedarf, `sum(C_i)/sum(e_i*r_i)` die vorhergesagte claimgewichtete Schadenhöhe. Ein Produkt separat exponierungsgewichteter Komponentenmittel ist im Allgemeinen etwas anderes. Die Zerlegung benötigt etwa `E[S | N,x,e]=N*m(x)` und `E[N | x,e]=e*r(x)` sowie geeignete endliche Momente. Der Abgleich von Schadenzeilen und Anzahlen beweist diese bedingte Mittelwertannahme nicht.

## Der begrenzte Modellvergleich

Die [vor den Fits registrierte Spezifikation](../config/interpretation_protocol.json) legt vier Paare fest: `DrivAge + Area`, `DrivAge + log(Density)`, `DrivAge + Region`, `DrivAge + BonusMalus`. Je Paar werden ein Poisson-GLM und ein additives Poisson-Spline-GAM auf denselben fünf OOF-Folds berechnet. Dummy- und Spline-Spalten zählen zu ihrem Ursprungsmerkmal. Exponierung skaliert die Periodenerwartung und dient als Gewicht.

Das GLM verwendet feste Alters-/BM-Grenzen, Referenzkodierung und einen linearen, trainingsweise standardisierten logarithmischen Dichteterm. Das reduzierte GAM verwendet kubische B-Splines mit acht uniformen Trainingsbereichsknoten, exponierungsgewichteter Zentrierung und Ridge-Strafe `alpha=0.0001` auf nichtkonstante Koeffizienten. Diese Strafe unterscheidet sich von der Differenzenstrafe des GAM im vollständigen Frequenzvergleich. Die beiden Ansätze glätten mit unterschiedlichen Strafen. [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) und [SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) dokumentieren die APIs.

Kategorien, Skalierung und Splines werden nur im Trainingsfold gelernt. Unbekannte Faktoren verwenden die Trainingsreferenz; numerische Werte außerhalb des Trainingsbereichs werden protokolliert. Spline-Ränder werden konstant fortgesetzt. Die Schadenhöhe ist bei allen acht Regeln das arithmetische Mittel sämtlicher positiver **Trainingsschadenzeilen** je Fold. Auch Nullschadenverträge erhalten diese Regel. Das Budget umfasst 40 Frequenzfits und fünf Schadenmittel, ohne Hyperparametersuche oder neue Vollbestandsrefits.

Der [vollständige Schadenhöhenvergleich](schadenhoehe_risikopraemie.md) untersucht daneben ein Gamma-GLM und ein Logmodell mit trainingsweise geschätztem Smearing. Ein Log-Fit modelliert nicht unmittelbar den arithmetischen Schadenmittelwert; die Rücktransformation benötigt passende Annahmen über das bedingte Fehlerverhalten. Für den Vergleich der Merkmalsmengen bleibt das Schadenmittel je OOF-Fold bei allen reduzierten Regeln gleich. Der [Frequenzvergleich](frequenzmodelle.md) begründet GLM, GAM und den einen nichtadditiven Benchmark anhand ihrer tatsächlich gemessenen Entwicklungsleistung.

Hauptkriterium ist `sum(e_i*d_1.5(S_i/e_i,q_i))/sum(e_i)` mit fester Tweedie-Power 1,5. Vollständige Referenz sind die unveränderten Entwicklungs-OOF-Produkte `pp_hgb_mean` aus der vollständigen Modellierung. Frequenzdeviance, Gamma-Deviance je Schadenzeile, A/P, Segmente und Fold-Scores ergänzen den Vergleich. Die 500 gepaarten ganzen Vertragsziehungen verwenden gemeinsame Indizes und den gezogenen Exponierungsnenner. Ihre 95%-Perzentilintervalle enthalten keine erneute Anpassung, Auswahlunsicherheit oder unbekannte Haushaltsabhängigkeit.

Der empirische Bootstrap kann zudem noch nicht beobachtete Extremverluste nicht darstellen. Seine Intervalle sind keine umfassenden Prämien- oder Reserveunsicherheitsintervalle.

### Ergebnis des ursprünglichen Vergleichs

Lauf `s6af633eebaf65aa470` verwendet 542.392 Entwicklungsverträge, 286.880,82 Vertragsjahre und 21.066 positive Schadenzeilen mit 47.480.754,65 historischen EUR. Alle Kandidaten prognostizieren dieselben Verträge, darunter 522.548 Nullschadenverträge. Quelle `cas-744e13a80283a07a`, Split `s2split-fe11d414168572dd`. [Scores](../reports/s6a/scores.json), [Foldmengen und Trainingsclaims](../reports/s6a/report.json).

| Zwei Ursprungsmerkmale | GLM: Jahresraten-Deviance | GAM: Jahresraten-Deviance |
| --- | --- | --- |
| Fahreralter + Area | 84,620291 | 84,181046 |
| Fahreralter + log(Density) | 84,503809 | 84,057302 |
| Fahreralter + Region | 84,667702 | 84,216007 |
| Fahreralter + Bonus-Malus | 82,620512 | 82,027856 |

**Im ursprünglichen Vier-Paare-Kreis lag Fahreralter + Bonus-Malus mit dem reduzierten Spline-GAM vorn.** Sein Score beträgt 82,027856 gegenüber 79,194844 beim vollständigen OOF-Modell: 3,58 % höher. Die gepaarte Scoreverschlechterung beträgt 2,833012, bedingtes 95%-Intervall [1,462608; 4,807079]. Unter den acht reduzierten Regeln gewinnt das BM-GAM zwei Fold-Punktvergleiche, das BM-GLM die übrigen drei. Das vollständige Modell liegt in allen fünf Folds vor sämtlichen reduzierten Regeln. Nach der vorab festgelegten 0,5%-Nähe-Regel liegt nur `two_BonusMalus_gam` im engsten Kandidatenkreis. Das ist keine globale Paaroptimalität. [Gepaarter Vergleich zur vollständigen Regel](../reports/s6a/bootstrap_full.json), [Fold-Scores](../reports/s6a/fold_scores.json).

Der Punktabstand des BM-GLM zum BM-GAM beträgt 0,592656, sein bedingtes Intervall [-0,188276; 1,959059]. Das Intervall enthält null: Der Vorteil des GAM gegenüber dem BM-GLM ist damit unter dieser bedingten Unsicherheitsbetrachtung nicht eindeutig getrennt. Die Paarwahl ist belastbarer als eine eindeutige Rangbehauptung zwischen diesen beiden Spezifikationen. Diese Intervallinformation ergänzt die praktische Nähe-Regel; sie korrigiert nicht für nachträgliche Auswahl. Das Portfolio-A/P des empfohlenen Modells ist 1,000146. Fast exakte Gesamtbalance ersetzt weder Segmentkalibrierung noch eine unabhängige Validierung. Das konstante Trainingsclaim-Mittel begrenzt alle reduzierten Regeln besonders dort, wo Schadenhöhen stark variieren. [Claim-Score](../reports/s6a/severity_scores.json), [Segmentkalibrierung](../reports/s6a/geography_calibration.json).

![Vier Paare im Entwicklungs-OOF-Vergleich](../reports/editorial_figures/verlustdeviance_merkmalspaare.png)


## Ergänzung: Zusatznutzen und Abhängigkeit

Die Frage nach redundanter Information wird nach Kenntnis des ursprünglichen Ergebnisses ergänzt. Der alte Vergleich setzte Alter voraus; er konnte weder den Zusatznutzen von Alter gegenüber Bonus-Malus allein noch Bonus-Malus + Area prüfen. Die [neue Spezifikation](../config/feature_ablation_protocol.json) ergänzt Alter, Bonus-Malus und Area jeweils allein sowie Bonus-Malus + Area, je als GLM und GAM. Die 40 zusätzlichen Frequenzfits verwenden dieselben fünf Entwicklungs-Folds, Kodierungen und Strafen. Sämtliche Regeln behalten das Trainingsclaim-Mittel als Schadenhöhe. Der Vergleich verwendet die vorhandenen OOF-Vorhersagen der ursprünglichen Modelle und beschränkt sich weiterhin auf die Entwicklung.

| Merkmalsmenge | GLM-Deviance | GAM-Deviance |
|---|---:|---:|
| Alter allein | 84,716678 | 84,317644 |
| Bonus-Malus allein | 83,239732 | 82,537206 |
| Area allein | 87,282513 | 87,279540 |
| Alter + Bonus-Malus | 82,620512 | 82,027856 |
| Alter + Area | 84,620291 | 84,181046 |
| Bonus-Malus + Area | 83,389222 | 82,620964 |
| Alter + log(Density) | 84,503809 | 84,057302 |
| Alter + Region | 84,667702 | 84,216007 |

Kleinere exponierungsgewichtete Jahresraten-Tweedie-Deviance ist besser. [Scores](../reports/s6a/ablation/scores.json), [Fold-Scores](../reports/s6a/ablation/fold_scores.json).

Im erweiterten Kreis hat **Alter + Bonus-Malus (GAM)** mit **82,027856** das kleinste Punktergebnis. Gegenüber der vollständigen OOF-Regel entspricht dies 3,58 % mehr Deviance. Bonus-Malus + Area (GAM) minus ursprüngliches Alter-BM-GAM beträgt 0,593108 mit bedingtem 95%-Intervall [-0,143607; 1,803305]. Das Intervall umfasst null. Bei festen Vorhersagen bleibt damit unsicher, welches der beiden Paare besser abschneidet. Nur das GAM mit Alter und Bonus-Malus liegt innerhalb von 0,5 % des Minimums. Diese Rangfolge gilt für den explorativ untersuchten Kandidatenkreis.

Alter zusätzlich zu Bonus-Malus verbessert den Hauptscore um 0,509350 (rund 0,62 % gegenüber Bonus-Malus allein); das bedingte Intervall für allein minus Paar [0,114879; 1,103115] liegt über null. Die gemessene Abhängigkeit macht Alter damit nicht vollständig redundant. Bonus-Malus + Area verbessert gegenüber Bonus-Malus allein die Frequenzdeviance (0,461053 statt 0,462314), aber nicht den primären Verlustscore (82,620964 statt 82,537206). Area enthält also durchaus Prognoseinformation; ihre Hinzunahme garantiert bei konstanter Schadenhöhe keinen besseren Schadenbedarfsscore.

Die Einzelmerkmale machen den zusätzlichen Prognosewert sichtbar. Positive Differenzen in der folgenden Tabelle bedeuten einen besseren Score des erweiterten Punktsiegers:

| Vergleich | Deviance-Differenz | bedingtes 95%-Intervall |
|---|---:|---:|
| Bonus-Malus allein (GAM) minus Punktsieger | 0,509350 | [0,114879; 1,103115] |
| Alter allein (GAM) minus Punktsieger | 2,289788 | [1,785603; 2,812738] |
| Area allein (GAM) minus Punktsieger | 5,251684 | [2,070311; 11,212510] |

[Alle gepaarten Vergleiche](../reports/s6a/ablation/bootstrap_two_BonusMalus_gam.json) verwenden 500 Ziehungen ganzer Verträge bei festen Vorhersagen. Sie enthalten weder Refit- noch nachträgliche Auswahlunsicherheit. Einzelmerkmal-Ablation zeigt einen Nutzen unter der jeweiligen Spezifikation, keinen kausalen Beitrag einer Variablen.

Alter und Bonus-Malus haben eine Pearson-Korrelation von -0,480, exponierungsgewichtet -0,444; die ungewichtete Spearman-Rangkorrelation beträgt -0,572. Bindungen und die Häufung bei BM=50 bleiben relevant. Die [Altersgruppen](../reports/s6a/ablation/dependence.json) zeigen dazu BM-Mittel und BM=50-Anteile. Area wird nominal behandelt: Die exponierungsgewichteten Korrelationsverhältnisse eta² betragen 0,00260 für Alter und 0,01672 für Bonus-Malus. Sie messen den Anteil der numerischen Streuung zwischen Area-Gruppen und sind keine unmittelbar vergleichbare Skala zur Alter-BM-Korrelation. Kleine Werte belegen keine statistische Unabhängigkeit.

Ein möglicher Vorteil weniger überlappender oder leichter erklärbarer Merkmale muss gegen die gemessene Prognoseleistung und die Stabilität der einzelnen Effekte abgewogen werden. Die Abhängigkeitsmaße und OOF-Ablationen prüfen noch nicht, wie stabil ein GAM das Risiko zwischen Alter und Bonus-Malus aufteilt. Dafür fehlen etwa Refit- und Concurvity-Diagnostik. Ein Stabilitätsvorteil von Area ist hier ebenfalls nicht nachgewiesen. Korrelation ist kein Ausschlusskriterium für Prognosen, aber ein Anlass, getrennte Effektinterpretationen vorsichtig zu formulieren.

Die folgenden Kalibrierungs- und Kostenbeispiele verwenden das GAM mit Alter und Bonus-Malus. Für die übrigen Merkmalsmengen gibt es [separate Segmenttabellen](../reports/s6a/ablation/segments.json). Die [Spezifikation und Ergebnisdateien der Ablation](../reports/s6a/ablation/report.json) dokumentieren den Vergleich.

## Beobachtung und adjustierte Assoziation

Die vorab festgelegten geografischen Gruppen heißen AB, CD und EF; ursprüngliche Area-Stufen, Dichteklassen, Regionen und Alterskombinationen bleiben in den Tabellen erhalten. Median/p90/p99 betreffen positive Schadenzeilen; Frequenz und Schadenbedarf alle Verträge. Schäden ab 10.000 und 100.000 EUR zeigen den Tail-Einfluss. Gekappte Beträge sind ausdrücklich alternative Sensitivitätsziele.

Die gemeinsame Unterstützung kombiniert Altersgruppe, BM-Gruppe, Fahrzeugalter und Kraftstoff. Die Daueranalyse ergänzt Gruppen bis 0,25, über 0,25 bis 0,75 und über 0,75 Jahre. Jede behaltene Zelle benötigt in **jeder** geografischen Gruppe mindestens 100 Vertragsjahre und zehn positive Schadenzeilen. Alle Gruppen erhalten dieselben gepoolten Exponierungsgewichte der behaltenen Zellen. Area und Density bleiben auf beobachteten Verträgen gemeinsam erhalten. Das belegt grobe gemeinsame Unterstützung, keine kontinuierliche Positivität. Mit und ohne Dauer werden unterschiedliche unterstützte Populationen untersucht; ihre Differenz ist kein isolierter Daueradjustierungseffekt.

Die vollständigen GLM-/GAM-Entwicklungsrefits liefern auf diesen beobachteten Verträgen ergänzende standardisierte Frequenzprofile: adjustierte Modellassoziationen, keine OOF-Gütescores. Ihre Intervalle halten Unterstützung, Referenzmix und Modelle fest. Dauer kann mit Vertragsbeendigung, Schadenmeldung oder ungemessener Nutzung zusammenhängen. Sie wird diagnostisch untersucht und nicht als drittes Tarifmerkmal eingeführt.

Standardisiert werden die genannten groben Zellen. Region, VehBrand, VehPower und die exakten Merkmalsverteilungen innerhalb einer Zelle bleiben gruppenspezifisch. Die Profile sind daher standardisierte vorhergesagte Gruppenraten und keine isolierten Area-Partialeffekte bei ansonsten identischem vollständigem Merkmalsmix.

## Welche Mechanismen lassen sich unterscheiden?

### Beobachtete Geografie und gemeinsame Unterstützung

| Area-Gruppe | Verträge | Vertragsjahre | Claims | Frequenz/Jahr | Claimmittel EUR | Median EUR | Schadenbedarf EUR/Jahr | Betragsanteil ab 100.000 EUR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AB | 143587 | 83.974,02 | 4806 | 0,05723 | 2.878,60 | 1.172,00 | 164,75 | 43,15 % |
| CD | 274750 | 145.362,17 | 10805 | 0,07433 | 2.078,73 | 1.172,00 | 154,52 | 18,24 % |
| EF | 124055 | 57.544,62 | 5455 | 0,09480 | 2.050,50 | 1.172,00 | 194,38 | 15,99 % |

Die drei Gruppen haben denselben Median von 1.172 EUR, aber verschiedene Mittelwerte und Tail-Anteile. Ein kleinerer Mittelwert ist damit kein Beleg für viele kleine Stadtunfälle. Feste Zahlbeträge sind mit der dokumentierten Konventionsabrechnung vereinbar; einzelne 1.172-EUR-Zeilen werden hier keiner Konvention nachweislich zugeordnet. [Gesamte Verteilungen und Sensitivitäten](../reports/s6a/geo.json), [Area × Alter](../reports/s6a/Area_age.json), [Wohnortdichte](../reports/s6a/density.json), [Regionen](../reports/s6a/Region.json).

| Area-Gruppe | Alter¹ | BM¹ | Fahrzeugalter¹ | Diesel¹ | B12¹ | mittlere Dauer² | Dauer bis 0,25 Jahre² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AB | 47,89 | 55,91 | 8,39 | 56,07 % | 13,27 % | 0,585 | 27,84 % |
| CD | 47,18 | 57,69 | 7,55 | 46,61 % | 17,38 % | 0,529 | 32,99 % |
| EF | 46,12 | 60,77 | 6,78 | 37,70 % | 26,60 % | 0,464 | 38,29 % |

¹ Exponierungsgewichtete Merkmalsmittel beziehungsweise Exponierungsanteile; ² Vertragsmittel beziehungsweise Vertragsanteile. B12 ist eine unbekannte Markengruppe und kein beobachteter Fahrzeugpreis oder Hinweis auf eine bestimmte Fahrzeugklasse.

Gemeinsame Unterstützung ohne Dauer: 48 gemeinsame Zellen, 501.612 Verträge, 271.366,20 Vertragsjahre; mit Dauer: 79 gemeinsame Zellen, 389.468 Verträge, 246.801,56 Vertragsjahre. [Unterstützung ohne Dauer](../reports/s6a/support_without_duration.json), [mit Dauer](../reports/s6a/support_with_duration.json). Die auf diesen tatsächlich behaltenen Verträgen ausgewiesenen [Verteilungen ohne Dauer](../reports/s6a/common_geo_without_duration.json) und [mit Dauer](../reports/s6a/common_geo_with_duration.json) ergänzen Claims, Quantile und Tail-Anteile; [Alter × Geografie](../reports/s6a/common_geo_age_with_duration.json) macht ihre Unterstützung sichtbar.

| Standardisierung | Gruppe | Frequenz/Jahr | Schadenbedarf EUR/Jahr | implizites Claimmittel EUR | GLM-Frequenzprofil | GAM-Frequenzprofil |
| --- | --- | --- | --- | --- | --- | --- |
| ohne Dauer | AB | 0,05958 | 177,15 | 2.973,20 | 0,05849 | 0,05833 |
| ohne Dauer | CD | 0,07176 | 149,13 | 2.078,05 | 0,07184 | 0,07160 |
| ohne Dauer | EF | 0,08647 | 177,83 | 2.056,55 | 0,08474 | 0,08409 |
| mit Dauer | AB | 0,05643 | 112,51 | 1.993,73 | 0,05708 | 0,05691 |
| mit Dauer | CD | 0,06732 | 137,59 | 2.043,98 | 0,07011 | 0,06981 |
| mit Dauer | EF | 0,08051 | 154,70 | 1.921,53 | 0,08316 | 0,08237 |

Das implizite Claimmittel ist standardisierter Schadenbedarf geteilt durch standardisierte Frequenz, kein einfacher Mittelwert der Gruppenschadenmittel. Beobachtete und vorhergesagte Summen werden je Zelle zunächst durch deren Exponierung geteilt, danach mit identischen Referenzgewichten aggregiert. [Standardisierte Werte ohne Dauer](../reports/s6a/standardized_without_duration.json), [mit Dauer](../reports/s6a/standardized_with_duration.json).

| Unterstützung | Kontrast EF minus AB | Punktwert | bedingtes 95%-Intervall |
| --- | --- | --- | --- |
| ohne Dauer | Frequenz | 0,02689 | [0,02341; 0,03046] |
| ohne Dauer | Schadenbedarf EUR/Jahr | 0,68 | [-146,67; 90,02] |
| mit Dauer | Frequenz | 0,02408 | [0,02097; 0,02718] |
| mit Dauer | Schadenbedarf EUR/Jahr | 42,19 | [7,78; 87,20] |

Die Kontrastintervalle sind direkt aus gepaarten Vertragsziehungen berechnet; ein Vergleich überlappender Einzelintervalle wäre etwas anderes. [Intervalle ohne Dauer](../reports/s6a/standardized_intervals_without_duration.json), [mit Dauer](../reports/s6a/standardized_intervals_with_duration.json). Unterstützung wurde auch anhand beobachteter Claimzahlen gewählt; die Unsicherheit hält diese Zellen und ihren Referenzmix fest. Unterschiede zwischen den beiden Populationen dürfen nicht als reine Dauerwirkung bezeichnet werden.

![Beobachtete Geografie mit Vertragsintervallen](../reports/s6a/figures/geografie.png)

![Standardisierung mit und ohne Dauergruppen](../reports/s6a/figures/mix_dauer.png)

| Hypothese und erwartete Beobachtung | Diagnose und tatsächlicher Befund | Konkurrenz / Status | Benötigte Zusatzdaten |
|---|---|---|---|
| Dichteres Wohnumfeld geht über andere Fahrten oder mehr Verkehrskontakte mit mehr erfassten Schäden je Jahr einher. Erwartet: höhere Frequenz in EF auch bei ähnlichem beobachtetem Mix. | EF liegt im gemeinsamen Mix über AB; auch mit Dauergruppen bleibt der Frequenzkontrast positiv. Die vollständigen GLM-/GAM-Profile zeigen ebenfalls einen Gradienten. | Das **Muster** ist unterstützt; Kontaktmechanismus, Kilometerleistung, Meldeverhalten und Bestandsselektion werden nicht unterschieden. | Kilometer/Nutzungszweck, Fahrten, Unfallort/-art, Meldewahrscheinlichkeit und Deckung. |
| Höhere außerörtliche Geschwindigkeit könnte die physische Schwere erhöhen. Erwartet: schwerere Verletzungen bei entsprechenden Unfallorten; ein stabiler Versicherungsbetragsgradient wäre nur indirekt kompatibel. | AB hat einen höheren rohen Claimmittelwert; alle Gruppen denselben Median. Ohne Dauer liegt das standardisierte AB-Mittel bei etwa 2.973 EUR, mit Dauer bei etwa 1.994 EUR auf anderer Unterstützung; EF bei etwa 2.057 beziehungsweise 1.922 EUR. | Ein stabiler allgemeiner Schadenhöhenmechanismus wird durch diese Sensitivität **abgeschwächt**. Geschwindigkeit bleibt extern plausibel, im Portfolio **nicht identifiziert**. Tail, Abrechnung und Auswahl sind Alternativen. | Unfallort/Straße, Geschwindigkeit, Personenschaden, Haftungs-/Regulierungsdaten. |
| Dokumentiertes Alter bildet Erfahrung oder Nutzung ab. Erwartet bei einer reinen monotonen Erfahrungsgeschichte: nach ähnlichem Mix stetig sinkende Frequenz. | Modellierte Altersprofile auf gemeinsam unterstützten Altersbereichen zeigen im GAM bei zugewiesenem Alter 35/45/55 die Raten 0,052486/0,077730/0,074113; GLM 0,064506/0,081771/0,072271. 291.234 Referenzverträge mit 163.403,34 Vertragsjahren. Die bestehenden Kontrastintervalle benachbarter mittlerer Altersgruppen sind breit. | Eine einfache monotone Erfahrungserklärung reicht **nicht aus**. Adjustierte Altersassoziation ist vorhanden, Unfallfahrer, Erfahrung und Nutzung sind nicht getrennt beobachtet. | Tatsächlicher Unfallfahrer, Führerscheindauer, Kilometer, Nutzung und zeitliche Schadenhistorie. |
| Unterschiedliche Vertragsdauer und Selektion erzeugen Teile der Geografiemuster. Erwartet: kurzer Dauerbestand mit anderem Mix und abweichenden Jahresraten. | Mittlere Dauer sinkt von AB über CD nach EF; Dauersegment-Kalibrierung und gemeinsame Dauerzellen werden separat gezeigt. Der Frequenzgradient bleibt, Verlust- und Claimmittelrelationen reagieren stärker. | Dauer-/Selektionsalternativen sind **relevant**, erklären aber nicht automatisch den gesamten verbleibenden Gradienten. Wechselnde Unterstützung verhindert eine kausale Zerlegung. | Beginn/Ende und Beendigungsgrund, Unfall-/Meldedatum, Nutzung, mehrjährige Bestandsverläufe. |

Die Altersprofile stammen aus Fits auf der vollständigen Entwicklung. Die Referenz hält Bonus-Malus, Fahrzeug, Kraftstoff und Region in gemeinsam besetzten Gruppen fest. Die Profile beschreiben Zusammenhänge im Modell. [Altersprofile](../reports/s6a/existing_age_profiles.json), [bestehende Kontrastintervalle](../reports/s6a/existing_age_contrasts.json), [Dauer-/Geografiekalibrierung](../reports/s6a/geography_calibration.json). Diese Befunde liefern keine Unfallartgeschichte. Auch Bonus-Malus als nützliches Prognosemerkmal belegt weder einen Verhaltenseffekt der Tarifierung noch seine generelle Marktunverzichtbarkeit.


Ein kausaler Vergleich benötigt eine definierte Intervention sowie begründete Konsistenz-, Austauschbarkeits- und Unterstützungsannahmen; [Hernán und Robins (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2652882/) erläutern diese Unterscheidung. Eine Geschwindigkeitsregel ist anders definierbar als „dieselbe Person älter machen“ oder „ländlicher wohnen“, ohne veränderte Fahrten und Lebensumstände festzulegen. Eine denkbare Kette wäre Wohnumfeld → Fahrten/Geschwindigkeit → Unfall → Betrag. Fahrleistung, Fahrzeugwahl und Vertragsselektion könnten mehrere Größen zugleich beeinflussen. Diese Pfeile sind Annahmen. Bonus-Malus kann Erfahrung und vergangene Schäden abbilden, auf einem früheren Wirkungspfad liegen oder mit Selektion zusammenhängen. Seine Rolle hängt von der zeitlichen Frage ab; er ist nicht pauschal ein Confounder. Die Beschränkung der Schadenhöhenanalyse auf positive Claims kann weitere Selektion erzeugen.

### Externe Evidenz und ihre Übertragungsgrenzen

Die [ONISR-Statistik 2024](https://www.onisr.securite-routiere.gouv.fr/etat-de-linsecurite-routiere/bilans-annuels-de-la-securite-routiere/bilan-2024-de-la-securite-routiere) ordnet 60 % der Verkehrstoten und 47 % der Schwerverletzten Straßen außerorts zu. Das sind Anteile ohne passenden Kilometer-/Vertragsnenner, keine Schadenmittel ländlich wohnender Versicherter. Personen aller Verkehrsarten in Frankreich métropolitaine 2024 sind ein anderer Bestand und Zeitraum. Der [Bericht für 2012](https://www.onisr.securite-routiere.gouv.fr/etat-de-l-insecurite-routiere/bilans-annuels-de-la-securite-routiere/bilan-2012-de-la-securite-routiere) liegt zeitlich näher, misst aber ebenfalls Personenschadenunfälle. Widersprüchliche Gesamtzahlen auf seiner Übersichtsseite werden nicht numerisch übernommen.

Die originäre [Cerema-Evaluation der 80-km/h-Regel](https://www.cerema.fr/en/actualites/lowering-maximum-authorised-speed-80-kmh-final-assessment) vergleicht betroffene französische Straßen vor und nach Juli 2018 mit dem übrigen Straßennetz als Kontrolle, hauptsächlich bis Dezember 2019. Berichtet werden geringere gefahrene Geschwindigkeiten und geringere Mortalität. Das stützt Geschwindigkeit als plausible Einflussgröße **physischer** Unfallschwere. Es identifiziert weder den Geschwindigkeitsmix der freMTPL2-Verträge noch die Ursache ihrer Zahlbeträge. Es handelt sich um eine beobachtende Maßnahmenevaluation mit eigenen Vergleichsannahmen, kein randomisiertes Experiment im Versicherungsbestand.

Benötigt werden Unfallort/Straßentyp, Unfallart, Geschwindigkeit, Kilometer/Nutzungszweck, tatsächlicher Unfallfahrer, Personenschadenindikator sowie Unfall-, Melde- und Regulierungszeitpunkte. Deckung, Haftungsanteile und Konventionsabrechnung müssten dazu passen. Diese Verknüpfung könnte konkurrierende Erklärungen gezielt prüfen. Die [Quellenübersicht](../reports/s6a/sources.json) hält Gegenstand und Grenzen fest.

## Vom Schadenbedarf zur Prämie

Beide Szenarien setzen ausdrücklich jährliche Kosten an. Ohne weitere Lasten und Steuern: `P_fix=q+50` und aus `P_var=q+0.15*P_var` daher `P_var=q/0.85`. Ein Kostenanteil an der finalen Prämie ist kein 15%-Aufschlag auf q. Der Schnittpunkt beträgt `q=850/3=283,33… EUR/Jahr`. Darunter ist das variable Szenario günstiger, darüber die Pauschale. Dies ist eine Preisrelation unter angenommenen Kostenregeln, kein Profitabilitätsnachweis.

| Fahreralter | Bonus-Malus | Verträge | Vertragsjahre | q EUR/Jahr | P_fix EUR/Jahr | P_var EUR/Jahr |
| --- | --- | --- | --- | --- | --- | --- |
| 25–34 | 50 | 14590 | 8.746,32 | 86,98 | 136,98 | 102,33 |
| ab 65 | 51–75 | 6671 | 3.967,17 | 187,43 | 237,43 | 220,51 |
| ab 65 | 76–100 | 1976 | 990,34 | 337,72 | 387,72 | 397,32 |
| 35–49 | ab 101 | 1207 | 577,94 | 760,11 | 810,11 | 894,25 |

Die Beispiele verteilen sich über die unterstützten berechneten Risikogruppen; alle Gruppen stehen in [costs.json](../reports/s6a/costs.json). Die Gruppenwerte sind durchschnittliche OOF-Risikoprämien, keine unabhängig geschätzten beobachteten Zellmittel.


Die Gruppen verwenden exponierungsgewichtete OOF-Vertragsprodukte des ursprünglich ausgewählten Alter-BM-GAM. Schäden und Schadenregulierungskosten sind abzugrenzen, damit Aufwand nicht doppelt geladen wird. Mit zusätzlich definierten jährlichen festen Kosten F, Risiko-/Kapitallast R, Gewinnbetrag G und variablem Anteil v wäre eine mögliche Nettopreisregel `P=(q+F+R+G)/(1-v)`. Ein angenommener Steuersatz t würde bei passender Bemessungsbasis `P*(1+t)` ergeben. Hier werden keine tatsächlichen Kosten, Kapitallasten oder Steuersätze geschätzt.

Bei gleichem q und tatsächlich niedrigeren zurechenbaren Kanalvollkosten könnte ein Direktpreis denselben erwarteten Deckungsbeitrag wie ein höherer Agenturpreis erreichen. Digitale Akquisitionskosten und ungünstige Selektion könnten dies umkehren. Benötigt werden zurechenbare Akquisitions-/Betreuungskosten, Provisionen, Konversion, Vertragsdauer, Storno und Schadenverläufe nach Kanal und vergleichbarer Deckung. freMTPL2 belegt keine Kanal-, Preiselastizitäts- oder Wettbewerbseffekte. Daraus folgt keine generelle Notwendigkeit verschiedener Preise.

Für Deutschland fehlen ein aktueller Zielbestand, vergleichbare Deckungs-/Haftungsdefinitionen, Schadeninflation, Regulierungs-/Bewertungspraxis, Nutzungs-/Merkmalsmix und externe zeitliche Validierung. Französische BM-Werte sind nicht einfach deutsche SF-Klassen. Regewichtung kann beobachteten Mix angleichen, aber fehlende Zielgrößen oder ungemessene Nutzung nicht ersetzen. Ein niedrigerer Preis belegt weder Kostendeckung noch Profitabilität oder Markterfolg.

## Anwendung der Ergebnisse


**1. Welche Daten und Auffälligkeiten?** Ein historischer französischer Haftpflichtbestand mit abgeglichenen positiven Schadenzeilen. Nullschäden, lange Exponierungen und große Beträge gehen in die Hauptanalyse ein. Sensitivitäten zeigen, wie stark wenige Schäden die Gruppenmittel beeinflussen. [Daten und Einheiten](#daten-ziel-und-einheiten).

**2. Wie werden Merkmale und Aufbereitung beurteilt?** OOF-Scores messen die gemeinsame Vorhersageleistung. GLM-Gruppen sind einfach zu erläutern, Splines bilden glatte nichtlineare Verläufe ab. Die Vorverarbeitung wird pro Trainingsfold bestimmt. Verfügbarkeit beim Angebot und erklärbare Kodierung sind zusätzliche Kriterien. [Modellvergleich](#der-begrenzte-modellvergleich).

**3. Warum Frequenz mal Schadenhöhe?** Die Rate wird mit Vertragsjahren zur erwarteten Anzahl; ein Trainingsclaim-Mittel liefert die Schadenkomponente. Die Komponenten werden je Vertrag multipliziert und anschließend aggregiert. Die bedingte Mittelwertannahme ist eine Annahme, kein Ergebnis des Zeilenabgleichs. [Zerlegung](#daten-ziel-und-einheiten).

**4. Wo passt das Modell?** Gesamtbalance ist gut, Segmente weichen deutlich ab. Beim ursprünglich ausgewählten Alter-BM-GAM zeigen die folgenden OOF-Werte Grenzen, die vor einem Tarif weitere Arbeit verlangen:

| Dimension | Gruppe | beobachtet EUR/Jahr | vorhergesagt EUR/Jahr | A/P |
| --- | --- | --- | --- | --- |
| geo | AB | 164,75 | 154,22 | 1,068 |
| geo | CD | 154,52 | 165,14 | 0,936 |
| geo | EF | 194,38 | 182,80 | 1,063 |
| duration | 0 | 594,18 | 187,70 | 3,166 |
| duration | 1 | 200,47 | 182,71 | 1,097 |
| duration | 2 | 100,47 | 154,53 | 0,650 |

   Dauergruppen 0/1/2 bedeuten bis 0,25 / über 0,25 bis 0,75 / über 0,75 Jahre. Die Gruppen enthalten jeweils sämtliche Nullschäden. [Vollständige Kalibrierung](../reports/s6a/geography_calibration.json).

**5. Was sagen die geografischen Unterschiede?** Höhere Frequenz und höhere mittlere Schadenhöhe sind verschiedene Befunde. Wohngebiet ist nicht Unfallort; die höhere Frequenz in EF bleibt im gemeinsamen Mix, der Schadenhöhenunterschied ist tail- und populationsabhängig. [Mechanismen](#welche-mechanismen-lassen-sich-unterscheiden).

**6. Welche zwei Tarifmerkmale?** Der ursprüngliche Vergleich bevorzugte Alter + Bonus-Malus. Im erweiterten Kreis liegt Alter + Bonus-Malus (GAM) im Punktergebnis vorn (82,028); nahe Alternativen und die bedingte Unsicherheit sind mitzunennen. Prognoseleistung, Erklärbarkeit und Effektstabilität sind getrennte Kriterien. Vollständige Stabilitätsdiagnostik fehlt; die Evaluation wird nicht neu zur Auswahl benutzt. [Aktualisierte Einordnung](#ergänzung-zusatznutzen-und-abhängigkeit).

**7. Was kommt zum Schadenbedarf?** Abgegrenzte Schadenregulierung, Abschluss/Verwaltung, Risiko/Kapital, Gewinn und gegebenenfalls definierte Steuern. Die Kostenbeispiele verwenden ausdrücklich angenommene Größen. [Prämienbestandteile](#vom-schadenbedarf-zur-prämie).

**8. 50 EUR oder 15 %?** Bei 15 % der finalen Prämie ergibt sich q geteilt durch 0,85. Der Schnittpunkt liegt bei 283,33 EUR/Jahr; darunter ist dieses Szenario günstiger, darüber die Pauschale. Die tatsächliche Kostenstruktur entscheidet über Kostendeckung. [Gruppenbeispiele](../reports/s6a/costs.json).

**9. Direktpreis versus Agenturpreis?** Unterschiede sind nur bedingt durch Kosten, Nachfrage und Selektion begründbar. Ein Kanalvergleich erfordert zurechenbare Vollkosten und vergleichbare Deckungsbeiträge. Solche Angaben fehlen im Bestand. [Produktfragen](#vom-schadenbedarf-zur-prämie).

**10. Was fehlt für Deutschland?** Aktuelle deutsche Zielbestands-/Schadendaten, vergleichbare Deckung und Bewertung sowie externe zeitliche Validierung. Ein französischer Zufallssplit beantwortet das nicht; BM ist keine automatisch übertragbare SF-Klasse. [Transfer](#vom-schadenbedarf-zur-prämie).

Das [Paper](../paper/begleitpaper.pdf) und der [Vortrag](../slides/hauptvortrag.pdf) vergleichen die Versicherungsbefunde mit ONISR und Cerema nach Population, Zeitraum, Ort und Zielgröße.

## Ausführen und nachlesen

Nach der [Vorbereitung und Kernanalyse](reproduktion.md):

```powershell
python -m src.interpretation
python scripts/s6a_diagnostics.py
python scripts/verify_interpretation.py
python -m src.feature_ablation
python scripts/verify_feature_ablation.py
python -m unittest discover -s tests -q
python scripts/execute_notebook.py notebooks/06_actuarial_interpretation.ipynb
```

Die erste Ausführung registriert den Stand vor den Fits; Wiederholungen prüfen unveränderliche Caches. Jeder neue Lauf speichert dabei seinen registrierten Quellstand unveränderlich als `executed_source.py`, bevor Trainingsclaim-Mittel oder Frequenzfits berechnet werden. Wiederholungen prüfen die exakten Bytes; ein anderer Quellstand oder eine veränderte Snapshotdatei wird abgelehnt. [report.json](../reports/s6a/report.json) nennt Lauf, Quelle, Split, Foldmengen, Trainingsclaim-Mittel und Manifest-Hashes. Vollständige keyed OOF-Vorhersagen liegen lokal unter `data/interpretation/<run_id>/results/`. Öffentliche kleine JSON-Tabellen und Grafiken zeigen die Befunde; [das reguläre Notebook](../notebooks/06_actuarial_interpretation.ipynb) enthält ausgeführte Ergebnisse.

Der [Herkunftsnachweis](../reports/s6a/release.json) dokumentiert die wiederverwendeten Frequenzfits, die zugehörigen Quellstände und den Abgleich der bereits berechneten Altersprofile.
