# freMTPL2 — Pricing (Frequency)
GLM/GAM mit OOF-Validierung, Kalibrierung und schlanker Governance

Ein kleines, reproduzierbares Pricing-Setup auf dem freMTPL2-Datensatz. 
Fokus: saubere Frequenz-Baselines (GLM/GAM), OOF-Vergleich, Kalibrierung und möglichst wenig "Magie".

---

## TL;DR – Ergebnisse (aktueller Lauf)
| Modell | Test-PDW ↓ | Gini ↑ | Kalibrierung (Act/Pred CF) |
|:--|--:|--:|--:|
| INT (Intercept) | 31.26 % | –    | – |
| GLM1            | 29.55 % | 0.299 | 7.38 % / 7.36 % |
| GLM2            | 29.52 % | 0.301 | 7.38 % / 7.36 % |
| **GAM**         | **29.16 %** | **0.299** | **7.38 % / 7.36 %** |

- **Dispersion (GLM2, Pearson χ²/df):** ~1.x → Poisson ausreichend; NegBin als Sensitivitäts-Check im Notebook.
- **Interpretierbarkeit:** Splines (z. B. Driver Age) mit Konfidenzbändern statt Black-Box.

> Hinweis: Zahlen stammen aus dem letzten Run. Seeds/Folds konstant (5-Fold OOF, Seed 42), Test nur 1×.


## Explorative Analyse (EDA)
Zur Orientierung: ein paar klassische Muster aus dem freMTPL2-Datensatz.

| Fahrer-/Fahrzeugvariablen | Beispielplots |
|:--|:--|
| Bonus-Malus & Fahrer-Alter | ![Driver Age × Bonus-Malus](reports/figs/D_violin_drivage_claims_by_bm.png) |
| Fahrzeugalter & Bonus-Malus | ![VehAge × BM](reports/figs/B_vehage_bm_lowess.png) |
| Brand & Fuel | ![Brand/Fuel](reports/figs/A_brand_fuel.png) |

> Diese EDA-Grafiken sind im Ordner `reports/figs/` reproduzierbar.  
> Ich nutze sie, um Feature-Binnings (z. B. `VehAge`, `DrivAge`) im GLM zu begründen.

---

## Modellplots (GLM/GAM)
| Thema | Plot |
|:--|:--|
| **Kalibrierung (Deciles)** | ![Calibration GLM2](figures/calibration_glm2.png) |
| **Partial Effect – Driver Age** | ![GAM Driver Age](figures/gam_spline_drivage.png) |
| **Partial Effect – VehAge** | ![GAM VehAge](figures/gam_spline_vehage.png) |
| **Lorenz-/Gini-Vergleich (GLM1/2 vs GAM)** | ![Lorenz](figures/lorenz_glm_gam.png) |

> Diese vier Grafiken stammen aus dem Notebook `03_glm_gam_boosting.ipynb`.  
> Sie werden beim letzten Cell-Run automatisch in `figures/` gespeichert.



---

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab

```
