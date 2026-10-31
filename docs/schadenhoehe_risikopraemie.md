# Schadenhöhe und reine Risikoprämie

Die Erweiterung prüft, ob eine modellierte Schadenhöhe gegenüber einem arithmetischen Trainingsmittel zusätzlichen Vorhersagenutzen liefert. Alle Hauptbeträge bleiben ungekappt. Die Entwicklungsentscheidung entsteht aus echten Vorhersagen außerhalb der jeweiligen Trainingsfalte; die gemeinsame Evaluation von Frequenz, Schadenhöhe und Verlustkosten wird erst anschließend geöffnet. Ein fehlender Mehrwert der Schadenhöhenmodelle ist ein zulässiges Ergebnis.

## Zielgröße, Gewichtung und Vorhersagezeitpunkt

`ClaimAmount` ist der historische Eurobetrag einer erfassten positiven Schadenzeile. Jede dieser Zeilen hat Gewicht eins. Mehrere Zeilen eines Vertrags bleiben zusammen in dessen festgelegter Entwicklungsfalte beziehungsweise Evaluation. Gleiche Vertragsnummer und gleicher Betrag sind ohne echte Schaden-ID kein Beleg einer Dublette.

Die Schadenhöhe erhält **keine Exponierungsgewichte und keinen Exponierungs-Offset**. Exponierung gehört zur Häufigkeit beziehungsweise zu erwarteten Periodenverlusten. Eine Anpassung an einer Schadenzeile ist außerdem nicht generell dasselbe wie eine logarithmische Anpassung an einem Vertragsdurchschnitt. In dieser Studie werden durchgehend einzelne Schadenzeilen angepasst.

Die neun Merkmale bleiben identisch zu den prospektiven Frequenzmerkmalen. Schadenanzahl, Verlust, Vertragskennung, Schadenzeilenkennung, Fold und aktuelle Schadenrealisierung sind keine Prädiktoren. Nach der Anpassung wird dieselbe Funktion für jeden späteren Vertrag verwendet, einschließlich aller Verträge ohne beobachteten Schaden. Erst für die Bewertung werden die vorhergesagten Eurobeträge über die Schlüssel mit den tatsächlich gehaltenen Schadenzeilen verknüpft. Es gibt keinen anderen Prädiktor und keinen Mittelwert-Fallback abhängig vom späteren Schadenstatus.

Die [Spezifikation](../config/severity_protocol.json) legt vor der ersten Anpassung drei Kandidaten, ihre gemeinsame Kodierung, den Verlustscore und die Auswahlregel fest. Die Kategorien, Mittelwerte, Skalierungen und Rangreduktionen werden ausschließlich aus Trainingsschadenzeilen gelernt. Unbekannte spätere Kategorien verwenden die jeweilige Trainingsreferenz und werden gezählt. B12 und die ungekapp­ten Ausgangswerte bleiben erhalten.

## Drei gezielte Kandidaten

**Arithmetischer Mittelwert.** `sum(Y_train)/n_claims_train` ist das konstante Modell für Euro je Schadenzeile. In jeder der fünf Entwicklungsfalten werden alle Schäden der anderen vier Falten verwendet. Es ist weder ein geometrischer Mittelwert noch der Mittelwert von Vertragsdurchschnitten. Multipliziert mit der vorhergesagten Jahresfrequenz ergibt es eine vollständig prospektive Basis.

**Gamma-GLM mit Log-Link.** Die additive Spezifikation enthält feste Alters-, Fahrzeugalters- und Bonus-Malus-Gruppen, log-Dichte sowie Referenzfaktoren für Leistung, Marke, Kraftstoff, Area und Region. Sie prüft bei denselben sinnvollen Merkmalen wie im Frequenzmodell unmittelbar den positiven bedingten Schadenmittelwert. Die Kategorien und Rangprüfung betreffen die Trainingsschäden. Das Modell wird ohne Regularisierung angepasst. Die [verwendete Implementierung](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html) verwendet einen Log-Link; Konvergenz und volle Rangidentifikation werden geprüft.

Die relevante Zielfunktion ist bis auf additive Konstanten `mean(Y/exp(eta) + eta)`. Ihre Interceptgleichung lautet `mean(Y/mu)=1`. Sie erzwingt **nicht** `sum(Y)=sum(mu)`. Deshalb werden Euro-A/P und der relative Score getrennt ausgewiesen. Ein Trainings-A/P ungleich eins ist kein Beweis einer fehlerhaften Gamma-Optimierung. Es erfolgt keine nachträgliche Skalierung, die diese Differenz versteckt. Die Gamma-Verteilung wird nicht als wahr vorausgesetzt; die dokumentierte Pearson-Dispersion ist eine Diagnose unter der Varianzform `phi*mu²`, keine zugesicherte Verteilungsanpassung.

**Log-OLS mit globalem Smearing.** Der dritte Kandidat prüft gezielt die Rücktransformationsidee der historischen Analyse unter korrigierten Trainingsgrenzen. OLS modelliert `log(Y)`, ausdrücklich nicht `log(1+Y)`. Die Eurovorhersage lautet

`m_hat(x) = exp(eta_hat(x)) * mean_train(exp(log(Y)-eta_hat(X)))`.

Es wird kein Eins subtrahiert. Aus `Y=exp(eta(x)+epsilon)` folgt `E[Y|x]=exp(eta(x))*E[exp(epsilon)|x]`. Die Ersetzung des zweiten Faktors durch einen globalen Trainingsresidualmittelwert ist eine Modellierungsannahme, wenn dieser Faktor von den Merkmalen abhängt. Nullmittelwert der Log-Residuen oder eine angenommene konstante Residuenvarianz beweisen die erforderliche Bedingung nicht. [Duan (1983), *Smearing Estimate: A Nonparametric Retransformation Method*](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017) ist die methodische Referenz; hier wird keine bedingte Unverzerrtheit behauptet.

Trainingsweise werden Log-Residualvarianz und mittlere exponentierte Residuen nach Alters-, BM- und Kraftstoffgruppen sowie der Anteil des größten exponentierten Residuums ausgewiesen. Gruppenspezifische Faktoren werden diagnostiziert und nicht nachträglich als neue Vorhersagekorrektur verwendet. Alle Rücktransformationsgrößen stammen ausschließlich aus den jeweiligen Trainingsschäden.

## Frequenz, Schadenhöhe und erwarteter Aufwand

Für die Jahresfrequenz `r_i`, Schadenhöhe `m_i` und Exponierung `e_i` gelten:

| Größe | Definition und Einheit |
|---|---|
| Jahresfrequenz | `r_i`, Schäden je Vertragsjahr |
| Schadenhöhe | `m_i`, EUR je erfasster positiver Schadenzeile |
| Jährliche reine Risikoprämie | `p_i=r_i*m_i`, EUR je Vertragsjahr |
| Erwartete Periodenkosten | `c_i=e_i*p_i`, EUR je tatsächlicher Vertragsperiode |
| Erwartete jährliche Portfoliokosten | `sum(c_i)/sum(e_i)` |
| Beobachtete jährliche Portfoliokosten | `sum(S_i)/sum(e_i)`, einschließlich schadenfreier Verträge |

Eine hinreichende Mittelwertannahme ist bei geeigneten endlichen Momenten `E[S|N,x,e]=N*m(x)` zusammen mit `E[N|x,e]=e*r(x)`. Dann liefert iteriertes Erwarten `E[S|x,e]=e*r(x)*m(x)`. Unabhängigkeit der vollständigen Häufigkeits- und Schadenhöhenverteilungen ist dafür nicht erforderlich. Der buchhalterische Zeilenabgleich belegt die Mittelwertannahme nicht. Eine Eurokalibrierung und Verlustbewertung der Produktvorhersagen ist deshalb unverzichtbar.

Im Allgemeinen gilt `mean_e(r*m) != mean_e(r)*mean_e(m)`. Die Alterszerlegung weist deshalb beobachtete Verluste, den exponierungsgewichteten Mittelwert des Produkts und das Produkt separater Komponentenmittel getrennt aus. Der prognostizierte schadenanzahlgewichtete Schadenhöhenmittelwert verwendet Gewichte `e_i*r_i`, nicht beobachtete künftige Anzahlen. Marginale Altersprofile des tatsächlichen Bestands sind wiederum keine bei unverändertem Merkmalsmix standardisierten Alterswirkungen.

## Gemeinsame Bewertung und Auswahlgrenze

Die Schadenhöhe wird auf allen gehaltenen positiven Schadenzeilen gehaltener Verträge bewertet: Gamma-Deviance mit Gewicht eins, MAE und RMSE in EUR sowie A/P mit Euro- und Zeilensummen. Derartige Claim-Scores beurteilen eine andere Population als die Risikoprämie auf allen Verträgen.

Für Verlustkosten ist vorab **Tweedie-Power 1,5** festgelegt. Dies ist eine gemeinsame Scoringkonvention für Nullen und positive Verluste, kein aus der Evaluation geschätzter wahrer Verteilungsparameter. Der primäre Score lautet `sum e_i*d_1.5(S_i/e_i,p_i)/sum e_i`. Bei Power 1,5 lässt sich die Zeilen-Deviance stabil als `4*(sqrt(y)-sqrt(mu))²/sqrt(mu)` berechnen. Die Umsetzung wird unabhängig gegen die [offizielle Metrik](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html) geprüft.

Die getrennt ausgewiesene mittlere Perioden-Deviance `mean d_1.5(S_i,e_i*p_i)` ist bei wechselnder Exponierung **nicht identisch**. Aus der Homogenität folgt `d_p(e*y,e*mu)=e^(2-p)*d_p(y,mu)`. Die aus dem Poisson-Fall bekannte einfache Identität darf nicht ungeprüft auf Tweedie übertragen werden. Keiner dieser Scores ist ein Eurofehler oder ein Prozentwert.

Die feste S4-Frequenz `hgb_l15` wird mit den drei Schadenhöhenvarianten kombiniert. Eine vierte Basis kombiniert Trainingsinterceptfrequenz und Trainingsschadenmittel; ihre konstante Jahresprämie entspricht der Trainingsverlustsumme geteilt durch Trainingsexponierung. Sie ersetzt nicht nachträglich die bereits festgelegte Frequenzauswahl. Die Entwicklungsentscheidung zur Schadenhöhe verwendet die kleinste gemeinsame OOF-Verlustdeviance bei derselben HGB-Frequenz; liegt die konstante Schadenhöhe innerhalb von 0,5 % des Minimums, wird sie vorgezogen.

500 gepaarte Vertragsziehungen halten alle festen Vorhersagen gemeinsam; Nullen bleiben enthalten und alle Schadenzeilen werden über den aggregierten Vertragsverlust mitgeführt. Score-Differenzen und A/P-Perzentile verwenden pro Ziehung denselben Vertragsindex und den gezogenen Exponierungsnenner. Die Intervalle enthalten weder erneutes Fitten, Auswahlunsicherheit, unbekannte Haushaltsabhängigkeit noch Schutz gegen unbeobachtete extreme Schäden. Die fünf überlappenden Trainingsfits sind keine fünf unabhängigen Wiederholungen für einen Standardfehler.

Nach Auswahl werden alle drei Schadenhöhenmodelle ausschließlich auf den vollständigen Entwicklungsschäden angepasst. Ihre Dateien, die S4-Frequenzrefits, Trainingsintercept, Konfiguration, Quellstand und Entwicklungsartefakte werden in der [Entwicklungsentscheidung](../reports/s5/development_decision.json) gesperrt. Erst anschließend öffnet der zweite Befehl die gemeinsame Evaluation. Dort werden alle drei Schadenhöhenvarianten, die vier Verlustkombinationen und die unveränderten Frequenzfamilien GLM/GAM/HGB mit Intercept auf denselben Evaluationsverträgen geprüft. Die Auswahl bleibt auch bei abweichender Evaluationsrangfolge unverändert. Die frühere vollständige Exploration bleibt offengelegt; das ist keine externe oder buchstäblich nie zuvor gesehene Validierung.

## Tails, Hebel und Einfluss

Der Entwicklungsnachweis enthält tatsächliche Schadenquantile, Verlustkonzentration und die Euroauswirkung der bereits in der EDA untersuchten hypothetischen Kappung bei 25.000 EUR. Diese Kappung ändert die Zielgröße; sie wird nicht als Datenbereinigung und nicht zur Hauptanpassung verwendet. Alle Hauptzeilen und ihre Beträge bleiben erhalten.

Für eine begrenzte Einflussprüfung wird ausschließlich beim zusätzlichen Entwicklungsrefit der Vertrag mit dem größten aggregierten Entwicklungsverlust samt allen seinen Schadenzeilen entfernt. Die geänderten Modelle prognostizieren denselben Entwicklungsbestand; erwartete Portfoliosummen und Vorhersageverhältnisse werden verglichen. Dies ist eine Trainingssensitivität, kein weiterer OOF-Kandidat und keine nachträgliche Auswahl auf einem veränderten Schadenbestand.

Designhebel und wirtschaftlicher Einfluss sind verschiedene Größen. Für das Gamma-Modell mit Log-Link und Varianzform `phi*mu²` entspricht der erwartete Fisher-Designhebel hier dem ungewichteten OLS-Designhebel. Er hängt von den Merkmalen ab. Ein großer Schaden kann trotz mäßigem Designhebel erheblichen Einfluss auf erwartete Eurosummen und Smearing haben. Die zusätzliche Vertragslöschung misst eine solche tatsächliche Anpassungsänderung. Sie garantiert keine Robustheit gegen bisher unbeobachtete Tail-Schäden.

## Ausführen und Nachweise

Nach der vorhandenen Datenvorbereitung und S4-Berechnung:

```powershell
.\.venv\Scripts\python.exe scripts/verify_severity_inputs.py
.\.venv\Scripts\python.exe -m src.severity develop
# Entwicklungsentscheidung und Modellnachweise sind jetzt gespeichert.
.\.venv\Scripts\python.exe -m src.severity evaluate
.\.venv\Scripts\python.exe -m src.severity_figures
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/05_severity_pure_premium.ipynb --output data/notebooks/05_severity_pure_premium.executed.ipynb
```

Der neue [Eingangsnachweis](../reports/s5/input_snapshot.json) überbrückt den unabhängig geprüften lokalen Wechsel von CPython 3.12.14 auf 3.12.10. Quelle, Aufteilung und numerische S4-Vorhersagen sind identisch; Laufkennungen ändern sich durch die dokumentierte Softwareidentität. Die Ausführung dieses bereits vorhandenen Wiederholungslaufs wurde nicht nachträglich als beobachtetes Experiment ausgegeben. Historische Nachweise bleiben ihrer damaligen Identität zugeordnet. Vorhandene Notebook-Ausgaben und lokale Ergebnisdateien wurden erhalten.

Unveränderliche Vertragsvorhersagen und Modellartefakte liegen lokal unter `data/severity/<Laufkennung>/`. Öffentliche JSON-Dateien enthalten aggregierte Scores, Segmente, Komponentenzerlegung, bedingte Bootstrap-Intervalle und Herkunftsnachweise. Die [Entwicklungsergebnisse](../reports/s5/development_report.json) und [Evaluationsergebnisse](../reports/s5/evaluation_report.json) tragen dieselbe Modell- und Konfigurationsidentität. Die gespeicherte Freigabe verhindert eine neue Entwicklungsidentität unter derselben geöffneten Evaluation. Wiederholungen lesen geprüfte Caches und erzeugen keine neue Auswahl. Nur Windows x64 mit den aufgezeichneten Python- und Paketversionen ist hier geprüft.

## Ergebnisse und Grenzen der gemeinsamen Evaluation

Lauf `s5sev-2c4bc0d8ad1fd1c1` verwendet den unveränderten Quellstand `cas-744e13a80283a07a`, Vorbereitung `s1-6b71086d8adc0e3c` und Split `s2split-fe11d414168572dd`. Die Entscheidung `s5decision-df9a82872ec16be6` wurde vor Einsicht in die neuen Evaluationsscores gesperrt. Sie wählt `pp_hgb_mean`, also die eingefrorene HGB-Frequenz und die arithmetische Trainingsschadenhöhe. Quelle, Einheiten und die frühe vollständige Analysegeschichte bleiben offengelegt.

### Schadenhöhenvergleich

| Kandidat | Entwicklungs-OOF: Gamma-Deviance je Schadenzeile | Evaluation: Gamma-Deviance je Schadenzeile | Evaluation: Schaden-A/P |
|---|---:|---:|---:|
| Arithmetisches Schadenmittel | 1,773504 | 1,806204 | 1,0253 |
| Gamma-GLM | 1,888679 | 1,733921 | 1,0518 |
| Log-OLS mit Smearing | 1,808287 | 1,808367 | 1,0032 |

Die konstante Schadenhöhe ist in der Entwicklung sowohl auf dem Schaden- als auch dem primären Verlustscore das tatsächliche Minimum. In der Evaluation hat das Gamma-GLM dagegen die niedrigste Schaden-Gamma-Deviance. Eine Behauptung, dass die Konstante jeden gehaltenen Score gewinnt oder Merkmale grundsätzlich keinen Schadenhöhennutzen hätten, wäre falsch. Die Studie zeigt den begrenzten und wechselnden Zusatznutzen dieser registrierten Spezifikationen. Ihre Merkmale und Rücktransformation werden nach der Evaluation nicht verändert.

### Risikoprämienvergleich auf allen Verträgen

| Produkt | Entwicklungs-OOF: Jahresraten-Tweedie-Deviance | Evaluation: Jahresraten-Tweedie-Deviance | Evaluation: Verlust-A/P |
|---|---:|---:|---:|
| Interceptfrequenz × Schadenmittel | 87,233849 | 88,917790 | 1,0488 |
| HGB × Schadenmittel (vorab gewählt) | 79,194844 | 80,206534 | 1,0481 |
| HGB × Gamma | 80,386955 | 80,063797 | 1,0807 |
| HGB × Log-Smearing | 79,370343 | 80,026128 | 1,0308 |

Alle Produkte erhalten dieselben 135,599 Evaluationsverträge einschließlich 130,499 Verträgen ohne Verlust. Ihre 5,378 positiven Schadenzeilen werden vollständig erhalten. Keine Vertrags- oder Schadenzeile und kein Eurobetrag wird in der Hauptanalyse ausgeschlossen. Die Evaluation enthält 71.602,016636 Vertragsjahre und 12.428.461,85 EUR beobachtete Verluste.

Das vorab gewählte Produkt prognostiziert 11.857.523,74 EUR Periodenverlust, entsprechend **165,60 EUR je Vertragsjahr**. Beobachtet werden **173,58 EUR je Vertragsjahr**. Verlust-A/P ist 1,048150. Der bedingte 95%-Vertragsbootstrap-Bereich für A/P beträgt ungefähr [0,849; 1,287]. Er ist breit und enthält keine unbekannte Tail- oder Auswahlunsicherheit.

Gegenüber der vollständigen Interceptbasis sinkt die Evaluations-Verlustdeviance um 9,80 %. Die gepaarte Differenz Basis minus gewähltes Produkt beträgt etwa 8,711, mit bedingtem Perzentilbereich [4,258; 15,396]. Diese Aussage betrifft den vorab festgelegten Score; sie belegt weder einen marktfähigen Preis noch ausreichende Segmentkalibrierung.

Die Produkte mit Gamma und Log-Smearing haben in der Evaluation leicht niedrigere Verlustscores als das ausgewählte Produkt. Ihre Differenzen Kandidat minus ausgewählte Konstante betragen ungefähr -0,143 beziehungsweise -0,180; die bedingten Perzentilintervalle [-2,101; 1,353] und [-0,723; 0,275] enthalten null. Das rechtfertigt keinen Wechsel der gesperrten Entwicklungsentscheidung. Es wird kein erfolgreicher Schadenhöhengewinn aus der Evaluation heraus ausgesucht.

Die getrennte mittlere Perioden-Tweedie-Deviance des gewählten Produkts beträgt 63,701881. Sie hat wegen der anderen Exponierungsskalierung nicht denselben Zahlenwert wie der primäre Score 80,206534. Die Eurojahresprämie wird als Quotient erwarteter Periodenkostensumme und Exponierungssumme berechnet, nicht als ungewichteter Vertragsmittelwert.

### Frequenzvergleich nach unveränderter Auswahl

Die gleichzeitig geöffneten Frequenzscores betragen 0,484115 für den Intercept, 0,461123 für das GLM-Familienminimum, 0,457762 für das GAM und 0,451766 für das gewählte HGB. Die HGB-Frequenzentscheidung bleibt damit auch in dieser Auswertung nachvollziehbar. GLM und GAM wurden vollständig als substantielle Kandidaten berücksichtigt; ihre Rangfolge wird nicht aus dem Modellnamen abgeleitet.

### Segmentabweichungen und Tails begrenzen die Anwendung

Beim ausgewählten Produkt beträgt das Evaluations-Verlust-A/P ungefähr 3,057 für sieben Tage bis unter 0,1 Vertragsjahre, 2,152 für 0,1 bis unter 0,5 Jahre und 0,607 ab einem Jahr. Die Gruppe ein bis unter sieben Tage hat A/P 14,94, aber lediglich 18,71 Vertragsjahre und 16 Schäden; sie verfehlt die Unterstützungsschwelle. Diese Unterschiede bleiben als gemessene Vorhersageabweichungen stehen. Die Daten trennen unvollständige Mittelwertmodellierung, Nutzung, Risikomix und Beobachtungsende nicht. Sie begründen keine automatische Quellkorrektur oder Exponierungskappung.

Auch Alters- und andere Segmente sind nicht hinreichend kalibriert. Beispielsweise stehen bei den 18- bis 24-Jährigen 2,857 Mio. EUR beobachtete Verluste rund 0,904 Mio. EUR Erwartung gegenüber, A/P etwa 3,160, bei 436 Schadenzeilen und 2.503 Vertragsjahren. Die Altersgruppe 25–29 hat dagegen A/P etwa 0,657. Die Schwellen 100 Schadenzeilen beziehungsweise 500 Vertragsjahre sind sichtbare Unterstützung, keine Garantie präziser Euroerwartungen. Diese tatsächlichen Altersgruppenprofile hängen vom beobachteten Mix und schweren Schäden ab; sie sind keine standardisierten oder kausalen Alterswirkungen. Ein erfolgreiches Frequenzmodell heilt diese Schadenhöhen- und Verlustprobleme nicht. **Ein unmittelbar einsetzbarer Tarif wird nicht behauptet.**

Im Entwicklungsbestand beträgt der größte Schaden 4.075.400,56 EUR beziehungsweise 8,58 % der Verlustsumme; die größten aufgerundeten 1 % der Schadenzeilen tragen 38,04 %. Das Entfernen des größten Verlustvertrags nur für den zusätzlichen Einflussrefit verändert die erwartete Entwicklungs-Portfoliosumme um −8,58 % beim Schadenmittel, −5,75 % beim Gamma-Modell und −8,86 % beim Log-Smearing. Der größte Schaden hat im Gamma-Design lediglich Hebel 0,00242; der maximale Designhebel beträgt etwa 0,03789. Mäßiger Designhebel schützt nicht vor großem wirtschaftlichem Einfluss.

Der globale Entwicklungs-Smearingfaktor beträgt 2,4045. Für die Trainingsgruppe unter 25 Jahren liegt das mittlere exponentierte Residuum bei 6,0586, also rund 2,52-mal so hoch; ein Residuum trägt dort 47,17 % der Summe exponentierter Residuen. Das zeigt die Gruppen- und Tail-Empfindlichkeit der globalen Rücktransformation. Es beweist nicht isoliert einen bestimmten bedingten Bias oder eine Ursache. Gruppenfaktoren werden nicht nach Einsicht als neue Korrektur eingeführt.

Die Kappung bei 25.000 EUR je Entwicklungs-Schadenzeile würde 13.227.184,49 EUR entfernen. Sie wird allein als bereits spezifizierte Zieländerung bilanziert. Der vollständige ungekappte Schadenaufwand bleibt das Hauptziel.

Die vier [Abbildungen mit Herkunftsnachweis](../reports/s5/figures/manifest.json), das [ausführbare Notebook](../notebooks/05_severity_pure_premium.ipynb) und der [Reproduzierbarkeitsnachweis](../reports/s5/reproducibility.json) verbinden diese Aussagen mit den tatsächlich berechneten Artefakten. Der nächste Schritt ist die integrierte Reproduktion und öffentliche Konsistenzprüfung; aus der geöffneten Evaluation wird keine weitere Auswahl abgeleitet.
