import numpy as np
import pandas as pd

def build_glm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Repliziert das Preprocessing nach Schelldorfer & Wüthrich (2019) für ein Poisson-GLM."""
    out = df.copy()

    # Area: als Kategorie (keine Integer-Codes als kontinuierliche Zahlen!)
    if "Area" in out:
        out["Area"] = out["Area"].astype("category")
        out["AreaGLM"] = out["Area"].cat.codes.astype("int64") + 1

    # VehPowerGLM: cap bei 9, als Faktor
    if "VehPower" in out:
        vp = np.minimum(out["VehPower"], 9)
        out["VehPowerGLM"] = pd.Categorical(vp.astype("int"), ordered=True)

    # --- VehAgeGLM: 0 -> Gruppe '1', 1..10 -> '2', >=11 -> '3'; Referenz in der Formel = '2'
    if "VehAge" in out:
        va = out["VehAge"].astype(int)
        va_grp = np.where(va == 0, "1",
                  np.where(va <= 10, "2", "3"))
        out["VehAgeGLM"] = pd.Categorical(va_grp, categories=["2","1","3"], ordered=False)
        # Kategorienreihenfolge so gewählt, dass '2' (1–10 J.) bequem als Reference gesetzt werden kann

    # DrivAgeGLM (18–20, 21–25, 26–30, 31–40, 41–50, 51–70, 71–100)
    if "DrivAge" in out:
        bins = [18, 21, 26, 31, 41, 51, 71, 101]   # rechts-offen; deckt 18..100 ab
        labels = ["1","2","3","4","5","6","7"]
        dag = pd.cut(out["DrivAge"].astype(int), bins=bins, right=False, labels=labels, include_lowest=True)
        # Referenz laut R: '5' (41–50)
        out["DrivAgeGLM"] = pd.Categorical(dag, categories=["5","1","2","3","4","6","7"], ordered=False)

    # BonusMalusGLM: cap bei 150, als numerischer Prädiktor
    if "BonusMalus" in out:
        out["BonusMalusGLM"] = np.minimum(out["BonusMalus"], 150).astype("int")

    # DensityGLM: log-Transform
    if "Density" in out:
        out["DensityGLM"] = np.log(out["Density"].astype("float"))

    # Region: Kategorie; Referenz, kein R24 bei mir, daher später angepasst
    if "Region" in out:
        out["Region"] = out["Region"].astype("category")
        if "Centre" in list(out["Region"].cat.categories):
            cats = ["Centre"] + [c for c in out["Region"].cat.categories if c != "Centre"]
            out["Region"] = out["Region"].cat.reorder_categories(cats, ordered=True)

    return out