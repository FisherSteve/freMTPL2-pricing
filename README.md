# freMTPL2 — Pricing (Frequency)
GLM/GAM mit OOF-Validierung, Kalibrierung und schlanker Governance

Ein kleines, reproduzierbares Pricing-Setup auf dem freMTPL2-Datensatz. 
Fokus: saubere Frequenz-Baselines (GLM/GAM), OOF-Vergleich, Kalibrierung.

---


### Modell-Überblick (Test)
![Dashboard](figures/perf_dashboard.png)

- PDW ↓: Poisson-Deviance (gewichtete PDW).  
- Gini ↑: Discriminative Power (Lorenz-basiert).  
- Calibration: Actual vs. Predicted Claim Frequency (gesamt).  
- Pure Premium: Erwartete €/Exposure (E[N]×E[X]) aus GLM/GAM + Severity.



## TL;DR – Ergebnisse (aktueller Lauf)
| Modell | Test-PDW ↓ | Gini ↑ | Kalibrierung (Act/Pred CF) |
|:--|--:|--:|--:|
| INT (Intercept) | 31.26 % | –    | – |
| GLM1            | 29.55 % | 0.299 | 7.38 % / 7.36 % |
| GLM2            | 29.52 % | 0.301 | 7.38 % / 7.36 % |
| **GAM**         | **29.16 %** | **0.299** | **7.38 % / 7.36 %** |


- **Interpretierbarkeit:** Splines (z. B. Driver Age, s.u.) mit Konfidenzbändern statt "Black-Box".

> Hinweis: Zahlen stammen aus dem letzten Run. Seeds/Folds konstant (5-Fold OOF, Seed 42).

- **Performance Summary (PDW/Gini + Calibration)**  
  ![Performance Summary](figures/perf_summary.png)



## Explorative Analyse (EDA)
Zur Orientierung: ein paar klassische Muster aus dem freMTPL2-Datensatz.

| Fahrer-/Fahrzeugvariablen | Beispielplots |
|:--|:--|
| Bonus-Malus & Fahrer-Alter | ![Driver Age × Bonus-Malus](reports/figs/D_violin_drivage_claims_by_bm.png) |
| Fahrzeugalter & Bonus-Malus | ![VehAge × BM](reports/figs/B_vehage_bm_lowess.png) |
| Brand & Fuel | ![Brand/Fuel](reports/figs/A_brand_fuel.png) |

> Diese Grafiken stammen aus dem Notebook `02_eda_overview.ipynb`.  



---

## Modellplots (GLM/GAM)
| Thema | Plot |
|:--|:--|
| **Kalibrierung (Deciles)** | ![Calibration GLM2](figures/calibration_glm2.png) |
| **Partial Effect – Driver Age** | ![GAM Driver Age](figures/gam_spline_drivage.png) |
| **Partial Effect – VehAge** | ![GAM VehAge](figures/gam_spline_vehage.png) |
| **Lorenz-/Gini-Vergleich (GLM1/2 vs GAM)** | ![Lorenz](figures/lorenz_glm_gam.png) |

> Diese vier Grafiken stammen aus dem Notebook `03_glm_gam_boosting.ipynb`.  
---

- **Pure Premium – Verteilung / gegen DrivAge**  
  ![PP Hist](figures/pure_premium_hist.png)  
  ![PP vs. DrivAge](figures/pure_premium_drivage.png)


## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab #für Notebooks

```
