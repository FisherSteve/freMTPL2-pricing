# Berechnete Kernergebnisse

Automatisch aus den [numerischen Nachweisen](../reports/s6/results.json) erzeugt; keine neue Modellwahl.

Quelle `cas-744e13a80283a07a`, Split `s2split-fe11d414168572dd`, Schadenhöhenlauf `s5sev-2c4bc0d8ad1fd1c1`.

| Produkt | OOF-Verlustdeviance | Evaluations-Verlustdeviance | Verlust-A/P |
|---|---:|---:|---:|
| Intercept × Schadenmittel | 87,233849 | 88,917790 | 1,0488 |
| HGB × Schadenmittel (gewählt) | 79,194844 | 80,206534 | 1,0481 |
| HGB × Gamma | 80,386955 | 80,063797 | 1,0807 |
| HGB × Log-Smearing | 79,370343 | 80,026128 | 1,0308 |

Score: exponierungsgewichtete Jahresraten-Tweedie-Deviance mit fester Power 1,5.

Das gewählte Produkt senkt sie gegenüber der Interceptbasis um 9,80 %. Die vorhergesagten Jahreskosten betragen 165,60 EUR, beobachtet werden 173,58 EUR.

Die Auswahl bleibt vor der Evaluation festgelegt. Abweichungen in Dauer- und Altersgruppen sowie seltene große Schäden begrenzen die Anwendung. Ein guter Gesamtscore belegt keinen unmittelbar verwendbaren Tarif. Einzelheiten: [Methodik und Grenzen](schadenhoehe_risikopraemie.md).
