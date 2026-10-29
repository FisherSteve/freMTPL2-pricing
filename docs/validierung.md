# Messgrößen und Validierungsgrenzen

Die Modellierungsstudie vergleicht Vorhersagen für die Anzahl erfasster Schadenzeilen mit positivem Betrag. Eine gute Rangfolge allein reicht dafür nicht: Die Vorhersagen müssen auf denselben Verträgen bewertet werden, ihre Einheiten müssen stimmen und die Modellwahl muss von der späteren Evaluation getrennt bleiben. Das [maschinenlesbare Protokoll](../config/validation_protocol.json) legt diese Regeln vor den neuen Modellvergleichen fest.

## Jahresrate und erwartete Anzahl

Für Vertrag $i$ bezeichnet $N_i$ die beobachtete Schadenanzahl und $e_i>0$ die Exponierung in Vertragsjahren. Eine vorhergesagte Jahresrate $r_i$ hat die Einheit Schäden je Vertragsjahr. Die erwartete Anzahl in der tatsächlichen Vertragsperiode ist

$$
\mu_i=e_i r_i.
$$

Beispielsweise entsprechen einer Jahresrate von 0,1 bei einer Exponierung von 0,25 Jahren erwartete 0,025 Schäden in dieser Periode. Jahresrate und Periodenanzahl dürfen weder bei der Bewertung noch bei der Gruppierung verwechselt werden. Die Größen beziehen sich auf die festgelegte Population positiver Schadenzeilen, nicht auf sämtliche gemeldeten Schäden.

Die aktuelle Rate-API verwendet Exponierung als Skalierung und Trainingsgewicht, nicht als zusätzlichen Prädiktor. Die proportionale Skalierung der erwarteten Anzahl bei sonst gleichen Merkmalen ist eine Modellannahme. Kurze Vertragsperioden und mögliche Unterschiede ihrer Zusammensetzung müssen in der Entwicklungsanalyse untersucht werden; der Datenabgleich bestätigt diese Annahme nicht.

Für eine beobachtete Anzahl $n$ und eine strikt positive erwartete Anzahl $\mu$ wird die Poisson-Deviance als

$$
d(n,\mu)=2\{n\log(n/\mu)-n+\mu\}
$$

definiert. Bei $n=0$ ist der logarithmische Term null, also $d(0,\mu)=2\mu$. Der primäre Score lautet

$$
D=\frac{\sum_i d(N_i,e_i r_i)}{\sum_i e_i}.
$$

Aus $N_i=e_i(N_i/e_i)$ folgt $d(N_i,e_i r_i)=e_i d(N_i/e_i,r_i)$. Deshalb entspricht $D$ der mit Exponierung gewichteten mittleren Deviance von beobachteter und vorhergesagter Jahresrate. Die [scikit-learn-Referenzfunktion](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html) wird in Tests für diesen Vergleich verwendet. Werden bereits erwartete Anzahlen bewertet, kommt **kein weiteres Exponierungsgewicht** hinzu. Der frühere Score verletzte diese Gleichheit.

Die Deviance ist kein Prozentwert und keine Wahrscheinlichkeit. Die relative Verbesserung $1-D_{\text{Modell}}/D_{\text{Basis}}$ ist eine andere Größe und braucht ein ausdrücklich benanntes Basismodell. Die mittlere Anzahl-Deviance $\sum d/n_{\text{Verträge}}$ wird getrennt ausgewiesen.

## Kalibrierung und Konzentration

Die Aggregatkalibrierung verwendet die Richtung **A/P**, tatsächlich durch vorhergesagt:

$$
\mathrm{A/P}=\frac{\sum_i N_i}{\sum_i\mu_i}.
$$

Ein Wert über eins bedeutet mehr beobachtete als vorhergesagte Schäden. Die Anzahlen und die Exponierung werden daneben gezeigt. Eine niedrige vorhergesagte Frequenz ist für sich kein Beleg für gute Kalibrierung.

Kalibrierungsgruppen werden nach der vorhergesagten **Jahresrate** gebildet. Der Rangmittelpunkt jedes Blocks exakt gleicher Raten bestimmt seine nominale Vertragsquantilgruppe. Gleichstände bleiben vollständig zusammen; es können daher weniger als zehn oder deutlich ungleich große Gruppen entstehen. Die Gruppierung verwendet keine beobachteten Schäden und keine ergebnisabhängige Auflösung von Gleichständen. Je Gruppe werden Vertragszahl, Exponierung, beobachtete und erwartete Anzahl sowie $\sum N/\sum e$ und $\sum\mu/\sum e$ ausgegeben. Spätere Segmentprüfungen nach Alter, Bonus-Malus, Kraftstoff und Region zeigen ihre Unterstützung; die festgelegte Schwelle von 500 Vertragsjahren ist eine Kennzeichnung geringer Unterstützung, keine automatische Ausschlussregel.

Die Frequenz-Konzentrationskurve sortiert die Jahresraten absteigend. Auf der x-Achse steht der kumulierte Exponierungsanteil, auf der y-Achse der kumulierte Anteil beobachteter Schäden. Jede Modellkurve erhält ihre eigene Sortierung und ihre eigenen Exponierungskoordinaten. Der Ursprung gehört dazu; exakt gleiche Raten bilden ungeteilte Blöcke mit linearer Verbindung. Der **rohe** Gini ist $G=2\,\mathrm{AUC}-1$, ohne Normalisierung an einer nach beobachteten Schäden sortierten Idealkurve. Ein überall gleicher Score ergibt damit null. Ohne beobachtete Schäden ist die Konzentration nicht definiert. Diese Rangkennzahl misst weder Kalibrierung noch Kausalität und bezieht sich auf Anzahl, nicht auf Schadenbetrag.

## Aufteilung auf Vertragsebene

Der festgelegte CAS-Datenstand wird anhand seiner eindeutigen `IDpol`-Werte aufgeteilt. Der Algorithmus sortiert zunächst die Schlüssel, permutiert sie mit NumPy-PCG64 und Seed `20261029` und weist die ersten `ceil(0.2 * n)` Verträge der Evaluation zu. Die übrigen Verträge bilden den Entwicklungsbestand. In der verbleibenden Permutationsreihenfolge werden sie zyklisch auf fünf Falten verteilt. Weder Schadenanzahl, Schadenbetrag, Exponierung noch Merkmale bestimmen die Zuordnung.

Alle Schadenzeilen übernehmen Split und Falte ihres Vertrags. Die technische `claim_row_id` hält die Verknüpfung nachprüfbar; sie ist keine echte Schaden-ID. Es gibt keine verlässliche Haushalts- oder Zeitkennung. Deshalb wird hier keine Aussage über die Generalisierung auf neue Haushalte oder spätere Kalenderperioden gemacht. Die numerische Reihenfolge von `IDpol` wird nicht als Zeitachse interpretiert.

Die Splitkennung umfasst Quellidentität, sortierte Schlüsselmenge und Aufteilungsregel. Code- und Protokollprüfsummen werden zusätzlich festgehalten; eine spätere Textkorrektur darf die Verträge nicht neu aufteilen. Vollständige Zuordnungen bleiben lokal in Parquet-Dateien, die öffentlichen Nachweise enthalten ihre Prüfsummen. Die lokale Sperrdatei verhindert einen versehentlichen Wechsel von Seed oder Aufteilungsregel.

**Die Analysegeschichte bleibt relevant.** Die ursprüngliche Arbeit explorierte den vollständigen Datenstand und verwendete die damalige Falte 5 zur Entwicklung. Auch die neue Quellenwahl und der deterministische Zielabgleich erfolgten retrospektiv über alle Quellzeilen. Eine neue Zufallsaufteilung macht dies nicht rückgängig. Die Evaluation ist deshalb keine externe und keine buchstäblich nie zuvor betrachtete Validierung. Ab der jetzigen Festlegung werden ihre Beobachtungen jedoch nicht für neue Exploration, Merkmalswahl, Transformationen, Hyperparameter oder Frühstopp verwendet.

## Echte OOF-Vorhersagen und Lernverfahren

In jedem der fünf Entwicklungsläufe wird auf vier Falten trainiert und ausschließlich die ausgelassene Falte vorhergesagt. Jeder Entwicklungsvertrag erhält genau eine zu ihm passende Out-of-Fold-Vorhersage. Gelernte Vorverarbeitung wird zusammen mit dem Schätzer für jede Trainingsfalte neu geklont und angepasst. Dies folgt dem Grundsatz der [ausschließlich auf Trainingsdaten angepassten Vorverarbeitung](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Unbekannte Validierungskategorien müssen ausdrücklich behandelt und dokumentiert werden, beispielsweise mit `OneHotEncoder(handle_unknown="ignore")`.

Der aktuelle reale Lauf passt ausschließlich das Exponierungs-Basismodell an:

$$
\widehat r_{-k}=\frac{\sum_{i\in\mathcal D\setminus\mathcal F_k}N_i}{\sum_{i\in\mathcal D\setminus\mathcal F_k}e_i}.
$$

Dabei bezeichnet $\mathcal D$ ausschließlich den Entwicklungsbestand und $\mathcal F_k$ dessen ausgelassene Falte. Das liefert fünf trainingsspezifische Raten. Der gepoolte Gini kann deshalb von null abweichen; daraus folgt keine sinnvolle Risikodifferenzierung durch ein Interceptmodell. Die generische OOF-Infrastruktur wird außerdem mit künstlichen Daten auf Trainingsgrenzen und schlüsselbasierte Ausrichtung geprüft. Diese Prüfungen sind keine Portfoliomodellvergleiche.

Der Gesamtscore wird aus allen OOF-Beobachtungen berechnet. Gleichwertig ist das mit Faltenexponierung gewichtete Mittel der fünf Faltenscores. Ein ungewichtetes Mittel von fünf Quotienten wäre bei unterschiedlichen Exponierungen eine andere Größe. Die Dokumentation zu [`cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) unterscheidet entsprechend die vollständige OOF-Vorhersage von der Interpretation aggregierter Bewertungsgrößen.

Vorhersagen enthalten `IDpol`, Split, Falte, Modell- und Laufkennung, Jahresrate und erwartete Periodenanzahl. Ihre Manifeste nennen Quelle, Vorbereitung, Split, Protokoll, Code und Software. Modelle werden anhand von Schlüsseln zusammengeführt, niemals allein anhand ihrer Zeilenposition oder einer wiederverwendeten Variablen wie `test_pred`.

## Vorab begrenzter späterer Vergleich

Die inhaltlichen Merkmale und Terme werden nach der Entwicklungs-EDA begründet und vor dem jeweiligen Fit registriert. Das Protokoll setzt dafür eine Obergrenze:

| Kandidat | Budget und Grenzen |
|---|---|
| Exponierungs-Basismodell | Eine Konfiguration; Rate aus Trainingssummen |
| Poisson-GLM | Höchstens zwei explizite Kodierungen und drei Regularisierungsstufen, insgesamt höchstens sechs Konfigurationen |
| Penalisiertes Poisson-GAM | Basisgrößen 10/20 und Glättungsstrafen 0,1/1/10, insgesamt höchstens sechs Konfigurationen |
| Histogramm-Poisson-Boosting | Blattzahl 15/31 und L2-Strafe 0/1, insgesamt vier Konfigurationen; Lernrate 0,05, 300 Iterationen, `early_stopping=False` |

Alle Kandidaten verwenden dieselben fünf Entwicklungsfalten. Strafparameter verschiedener Bibliotheken sind ohne Prüfung ihrer Zielfunktionen und Skalierung nicht gleichzusetzen. Konvergenzprobleme und begrenzte Reparaturen werden dokumentiert; sie eröffnen keine unbegrenzte Suche. Die [Boosting-API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) besitzt automatische Frühstoppregeln; die ausdrückliche Deaktivierung vermeidet hier eine zusätzliche implizite Validierungsaufteilung.

Primär zählt die OOF-Deviance, ergänzt durch Aggregat- und Segmentkalibrierung, Stabilität, Konvergenz, Interpretierbarkeit und Rechenzeit. Liegen Kandidaten innerhalb von 0,5 % relativer Deviance des besten Werts, kann das einfachere stabile Modell vorgezogen werden. Diese Toleranz ist eine praktische Konvention, kein statistischer Signifikanznachweis. Ein unterlegenes GAM oder ein geringer Boosting-Zusatznutzen ist ein zulässiges Ergebnis.

OOF-Werte, die zur Auswahl verwendet werden, sind Selektionsschätzungen. Sie bewerten nicht unverzerrt das gesamte abgestimmte Verfahren. Die [scikit-learn-Darstellung zu verschachtelter Validierung](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) erläutert diese Unterscheidung; eine verschachtelte Validierung wird hier nicht behauptet.

## Stabilität und spätere Evaluation

Geplant sind gepaarte Differenzen der Faltenscores sowie 500 gepaarte Bootstrap-Ziehungen von Entwicklungsvertragszeilen mit Seed `20261030`. In jeder Ziehung werden dieselben Vertragsindizes für alle Kandidaten verwendet und die Deviance-Differenz durch die gezogene Exponierung normalisiert. Die 2,5-/97,5-Perzentile beschreiben Stichprobenstabilität **bei festen Vorhersagen**. Sie enthalten weder erneutes Fitten noch die Unsicherheit der Modellwahl und beheben keine unbeobachtete Haushaltsabhängigkeit. Die fünf Fits mit überlappenden Trainingsmengen sind keine unabhängigen Wiederholungen für einen naiven Standardfehler. Diese Vergleichs- und Bootstrap-Auswertungen sind vorab geplant, noch nicht als empirische Ergebnisse ausgeführt.

Die Evaluation wird erst nach dokumentierter Entwicklungsentscheidung und festgehaltener Modellkonfiguration für Frequenz und spätere Schadenhöhe geöffnet. Der aktuelle Ablauf erzeugt keine Evaluationsvorhersagen und keine Evaluationsscores. Ihre Ergebnisse dürfen nicht zur nachträglichen Abstimmung zurückgeführt werden. Für ein mögliches späteres Experiment mit synthetischen Trainingsdaten ist keine weitere unabhängige Reserve festgelegt; ohne neue Reserve oder unabhängige Daten bleibt es eine explorative Entwicklungsanalyse.

## Ausführen und nachprüfen

Nach der Datenvorbereitung im Projektstamm:

```powershell
.\.venv\Scripts\python.exe -m src.validation freeze
.\.venv\Scripts\python.exe -m src.validation oof-baseline
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/02_validation_contracts.ipynb --output data/notebooks/02_validation_contracts.executed.ipynb
```

Die erste Aktion speichert die Aufteilung und ihren [öffentlichen Nachweis](../reports/s2/split_manifest.json). Die zweite schreibt tatsächliche OOF-Vorhersagen und einen [berechneten Basismodellnachweis](../reports/s2/baseline_oof_report.json). Die vollständigen Tabellen bleiben lokal unter `data/validation/`. Das [Notebook](../notebooks/02_validation_contracts.ipynb) liest ausschließlich diese bereits erzeugten Artefakte und ihre Nachweise, prüft die Zuordnung und zeigt Entwicklungswerte. Es passt kein Modell an und teilt keine Verträge neu auf.
