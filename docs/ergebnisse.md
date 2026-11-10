# Vorhersage des jährlichen Schadenbedarfs

Vier Produkte aus Jahresfrequenz und erwarteter Schadenhöhe werden je Bestand auf denselben Verträgen verglichen. Die Entwicklungsspalte enthält Out-of-Fold-Vorhersagen. A/P ist die beobachtete geteilt durch die erwartete Verlustsumme in der Evaluation.

| Produkt | Verlustdeviance Entwicklungs-OOF | Verlustdeviance Evaluation | Verlust-A/P Evaluation |
|---|---:|---:|---:|
| Intercept × Schadenmittel | 87,233849 | 88,917790 | 1,0488 |
| HGB × Schadenmittel (gewählt) | 79,194844 | 80,206534 | 1,0481 |
| HGB × Gamma | 80,386955 | 80,063797 | 1,0807 |
| HGB × Log-Smearing | 79,370343 | 80,026128 | 1,0308 |

Die Tabelle verwendet die exponierungsgewichtete Tweedie-Deviance der Jahresraten mit Power 1,5.

In der Evaluation senkt das zuvor gewählte Produkt die Verlustdeviance gegenüber der Interceptbasis um 9,80 %. Es prognostiziert 165,60 EUR je Vertragsjahr gegenüber beobachteten 173,58 EUR.

Die Gesamtsummen verdecken Abweichungen in Dauer- und Altersgruppen. Einzelne Großschäden prägen die beobachteten Verluste. [Segmentbefunde und Unsicherheit](schadenhoehe_risikopraemie.md) · [Zahlen und Herkunft](../reports/s6/results.json)
