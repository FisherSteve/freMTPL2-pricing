# src/metrics.py
from __future__ import annotations
import numpy as np

_EPS = 1e-12

def pdw(pred_counts, obs_counts, exposure) -> float:
    """Weighted Poisson deviance in Prozent (wie im R-Notebook).
    PDW = 200 * sum_ex[ ex * (pred - obs + obs*log(obs/pred)) ] / sum(ex)
    Konvention: obs*log(obs/pred)=0 für obs=0. pred wird unten auf >0 geclippt.
    """
    pred = np.asarray(pred_counts, dtype=float)
    obs  = np.asarray(obs_counts,  dtype=float)
    ex   = np.asarray(exposure,    dtype=float)

    pred = np.clip(pred, _EPS, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(obs > 0, obs * np.log(obs / pred), 0.0)
    val = np.sum(ex * (pred - obs + term))
    return 200.0 * val / np.sum(ex)

def pdw2(label: str, l_pred, l_obs, l_ex, t_pred, t_obs, t_ex) -> str:
    return f"{label}, Learn/Test: {pdw(l_pred, l_obs, l_ex):.2f}% / {pdw(t_pred, t_obs, t_ex):.2f}%"

def cf2(label: str, l_obs, l_ex, t_obs, t_ex) -> str:
    l = float(np.sum(l_obs) / np.sum(l_ex))
    t = float(np.sum(t_obs) / np.sum(t_ex))
    return f"{label}: {100*l:.2f}% / {100*t:.2f}%"

def make_folds(n_rows: int, k: int = 5, seed: int = 42):
    """Gibt einen Array der Länge n_rows mit Werten 1..k zurück (reproduzierbar)."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, k + 1, size=n_rows)
