"""Development-only actuarial EDA; no candidate fitting or evaluation scoring."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.data import (DataContractError, PROJECT_ROOT, json_bytes, load_prepared,
                      logical_hash, root_path, sha256, text_sha256)
from src.validation import (identity_id, open_split, prepared_location, protocol,
                            propagate_claims, read_artifacts, write_immutable)

CONFIG = PROJECT_ROOT / "config/eda_protocol.json"
N, E, L = "recorded_positive_count", "Exposure", "recorded_loss_eur"


def configuration() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def development_only(policies: pd.DataFrame, claims: pd.DataFrame,
                     frames: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take keys first, then validate development targets; never summarize evaluation."""
    membership = frames["policy_membership"]
    ids = membership.loc[membership["split"] == "development", "IDpol"]
    if not policies.IDpol.is_unique or set(policies.IDpol) != set(membership.IDpol):
        raise DataContractError("EDA policy universe differs from the frozen split")
    expected_claims = frames["claim_membership"]
    inherited = propagate_claims(claims[["claim_row_id", "IDpol"]], membership)
    if not inherited.equals(expected_claims):
        raise DataContractError("EDA claim keys/lineage differ from the frozen split")
    d = policies.loc[policies.IDpol.isin(ids)].sort_values("IDpol").reset_index(drop=True)
    c = claims.loc[claims.IDpol.isin(ids)].sort_values("claim_row_id").reset_index(drop=True)
    check_development(d, c)
    return d, c


def check_development(d: pd.DataFrame, c: pd.DataFrame) -> None:
    if d.empty or not d.IDpol.is_unique or not c.claim_row_id.is_unique:
        raise DataContractError("EDA requires nonempty unique policies and unique technical claim rows")
    values = d[[E, N, L, "ClaimNb"]].to_numpy(dtype=float)
    amounts = c.ClaimAmount.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or not np.isfinite(amounts).all()
            or (values[:, 0] <= 0).any() or (values[:, 1:] < 0).any()
            or (amounts <= 0).any() or not np.equal(values[:, 1], np.floor(values[:, 1])).all()
            or not d[N].eq(d.ClaimNb).all() or not set(c.IDpol).issubset(set(d.IDpol))):
        raise DataContractError("Invalid development targets/exposure/claim linkage")
    linked = c.groupby("IDpol").agg(count=("ClaimAmount", "size"), loss=("ClaimAmount", "sum"))
    observed = d.set_index("IDpol")[[N, L]].join(linked).fillna(0)
    if (not observed[N].eq(observed["count"]).all()
            or not np.allclose(observed[L], observed["loss"], rtol=1e-12, atol=1e-7)):
        raise DataContractError("Development count/loss conservation failed")


def bins(values: pd.Series, edges: list, labels: list) -> pd.Series:
    result = pd.cut(values, edges, labels=labels, right=False)
    if result.isna().any():
        raise DataContractError("A source value lies outside the explicit EDA bins")
    return result.astype(str)


def add_groups(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = d.copy()
    for name, column, prefix in [("age_band", "DrivAge", "age"),
                                 ("age10_band", "DrivAge", "age10"),
                                 ("bm_band", "BonusMalus", "bm"),
                                 ("vehicle_band", "VehAge", "vehicle"),
                                 ("density_band", "Density", "density")]:
        d[name] = bins(d[column], cfg[f"{prefix}_edges"], cfg[f"{prefix}_labels"])
    d["exposure_band"] = bins(d[E], [0, 1/365, 7/365, .1, .5, 1, np.inf],
                              ["<1 Tag", "1–<7 Tage", "7 Tage–<0,1 J.", "0,1–<0,5 J.", "0,5–<1 J.", "1+ J."])
    d["brand_new"] = np.where(d.VehAge.eq(0), "VehAge=0", "VehAge>0")
    d["brand12"] = np.where(d.VehBrand.eq("B12"), "B12", "andere Marken")
    return d


def summary(d: pd.DataFrame) -> dict:
    if d.empty:
        raise DataContractError("A summary requires a nonempty policy population")
    exposure, counts, loss = float(d[E].sum()), int(d[N].sum()), float(d[L].sum())
    return {"policies": len(d), "exposure": exposure, "claims": counts, "loss_eur": loss,
            "frequency": counts/exposure, "severity_eur": loss/counts if counts else None,
            "loss_cost_eur_per_year": loss/exposure}


def grouped(d: pd.DataFrame, columns: list[str], support: float = 500) -> pd.DataFrame:
    """Ratio of sums; Poisson intervals are assumption-based reference intervals."""
    t = d.groupby(columns, observed=True, sort=True).agg(
        policies=("IDpol", "size"), exposure=(E, "sum"), claims=(N, "sum"), loss_eur=(L, "sum")).reset_index()
    t["frequency"] = t.claims/t.exposure
    t["severity_eur"] = t.loss_eur/t.claims.replace(0, np.nan)
    t["loss_cost_eur_per_year"] = t.loss_eur/t.exposure
    t["poisson_lower"] = np.where(t.claims > 0, chi2.ppf(.025, 2*t.claims)/(2*t.exposure), 0)
    t["poisson_upper"] = chi2.ppf(.975, 2*(t.claims+1))/(2*t.exposure)
    t["supported"] = t.exposure.ge(support)
    return t


def bootstrap_ratios(d: pd.DataFrame, replicates: int, rng: np.random.Generator) -> np.ndarray:
    """Keep whole policies paired; returned columns are frequency, severity, loss cost."""
    if d.empty or replicates < 1:
        raise DataContractError("Bootstrap requires observations and positive replicate count")
    a = d[[E, N, L]].to_numpy(dtype=float)
    result = np.empty((replicates, 3))
    for b in range(replicates):
        ex, count, loss = a[rng.integers(0, len(a), size=len(a))].sum(axis=0)
        result[b] = count/ex, loss/count if count else np.nan, loss/ex
    return result


def bootstrap_table(d: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap"]["seed"]))
    rows, contrasts, draws = [], [], {}
    groups = [("Gesamt", d)] + [(label, d.loc[d.age_band.eq(label)]) for label in cfg["age_labels"]]
    for label, part in groups:
        if part.empty:
            continue
        boot = bootstrap_ratios(part, cfg["bootstrap"]["replicates"], rng)
        draws[label] = boot
        row = {"age_band": label, **summary(part), "replicates": len(boot),
               "undefined_severity_draws": int(np.isnan(boot[:, 1]).sum())}
        for j, metric in enumerate(["frequency", "severity_eur", "loss_cost_eur_per_year"]):
            finite = boot[:, j][np.isfinite(boot[:, j])]
            bounds = np.percentile(finite, cfg["bootstrap"]["percentiles"]) if len(finite) else [np.nan, np.nan]
            row[metric+"_lower"], row[metric+"_upper"] = bounds
        rows.append(row)
    # Fixed, adjacent comparisons; not a search for the largest observed contrast.
    for left, right in [("40–44", "35–39"), ("45–49", "40–44"), ("50–54", "45–49")]:
        if left not in draws or right not in draws:
            continue
        a, b = draws[left], draws[right]
        for j, metric in enumerate(["frequency", "severity_eur", "loss_cost_eur_per_year"]):
            delta = a[:, j]-b[:, j]
            finite = delta[np.isfinite(delta)]
            bounds = np.percentile(finite, cfg["bootstrap"]["percentiles"]) if len(finite) else [np.nan, np.nan]
            left_value = summary(d.loc[d.age_band.eq(left)])[metric]
            right_value = summary(d.loc[d.age_band.eq(right)])[metric]
            difference = left_value-right_value if left_value is not None and right_value is not None else np.nan
            contrasts.append({"left_age": left, "right_age": right, "metric": metric,
                              "difference": difference,
                              "lower": bounds[0], "upper": bounds[1], "finite_draws": len(finite)})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def standardize_bm(d: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Direct descriptive standardization on common supported BM cells, fixed mix."""
    ages = cfg["standardization"]["age_groups"]
    sub = d.loc[d.age10_band.isin(ages)]
    cells = grouped(sub, ["age10_band", "bm_band"], cfg["support_policy_years"])
    support = cells.pivot(index="bm_band", columns="age10_band", values="exposure").reindex(columns=ages)
    common = support.index[support.ge(cfg["support_policy_years"]).all(axis=1)].tolist()
    if not common:
        raise DataContractError("No common supported BM strata for descriptive standardization")
    cells["common_support"] = cells.bm_band.isin(common)
    pooled = cells.loc[cells.common_support].groupby("bm_band").exposure.sum()
    weights = (pooled/pooled.sum()).to_dict()
    cells["reference_weight"] = cells.bm_band.map(weights).fillna(0.)
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap"]["seed"]+1))
    rows, samples = [], {}
    for age in ages:
        all_age = sub.loc[sub.age10_band.eq(age)]
        kept = all_age.loc[all_age.bm_band.isin(common)]
        score, draws = 0., np.zeros(cfg["bootstrap"]["replicates"])
        for bm in common:
            part = kept.loc[kept.bm_band.eq(bm)]
            score += weights[bm]*summary(part)["frequency"]
            draws += weights[bm]*bootstrap_ratios(part, len(draws), rng)[:, 0]
        samples[age] = draws
        lower, upper = np.percentile(draws, cfg["bootstrap"]["percentiles"])
        rows.append({"age10_band": age, **summary(kept), "all_age_frequency": summary(all_age)["frequency"],
                     "excluded_policies": len(all_age)-len(kept),
                     "excluded_exposure": float(all_age[E].sum()-kept[E].sum()),
                     "excluded_claims": int(all_age[N].sum()-kept[N].sum()),
                     "excluded_loss_eur": float(all_age[L].sum()-kept[L].sum()),
                     "standardized_frequency": score, "lower": lower, "upper": upper})
    contrasts = []
    for left, right in [(ages[1], ages[0]), (ages[2], ages[1])]:
        lower, upper = np.percentile(samples[left]-samples[right], cfg["bootstrap"]["percentiles"])
        by_age = {row["age10_band"]: row for row in rows}
        contrasts.append({"left_age": left, "right_age": right,
                          "difference": by_age[left]["standardized_frequency"]-by_age[right]["standardized_frequency"],
                          "lower": lower, "upper": upper})
    return cells, pd.DataFrame(rows), pd.DataFrame(contrasts)


def sensitivity(d: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    base = summary(d)
    rows = []
    exclusions = {"ungekappt": np.ones(len(d), dtype=bool), "ohne Exposure < 1/365": d[E].ge(1/365),
                  "ohne Exposure < 0,1": d[E].ge(.1), "ohne Exposure > 1": d[E].le(1),
                  "ohne B12": d.VehBrand.ne("B12"),
                  "ohne B12 und VehAge=0": ~(d.VehBrand.eq("B12") & d.VehAge.eq(0)),
                  "ohne DrivAge >= 90 oder VehAge >= 100": d.DrivAge.lt(90) & d.VehAge.lt(100)}
    for label, keep in exclusions.items():
        if not np.any(keep):
            raise DataContractError(f"Sensitivity leaves no policies: {label}")
        rows.append({"scenario": label, "kind": "Basis" if label == "ungekappt" else "Populationsänderung", **summary(d.loc[keep])})
    cap = d.copy()
    cap[E] = cap[E].clip(upper=1)
    rows.append({"scenario": "Exposure bei 1 gekappt", "kind": "Nenneränderung", **summary(cap)})
    cap = d.copy()
    cap[N] = cap[N].clip(upper=4)
    row = {"scenario": "Anzahl bei 4 gekappt", "kind": "Anzahlziel geändert; unveränderte Schadenzeilen passen nicht mehr", **summary(cap)}
    row["severity_eur"] = np.nan
    rows.append(row)
    cap = d.copy()
    losses = c.assign(capped=c.ClaimAmount.clip(upper=25000)).groupby("IDpol").capped.sum()
    cap[L] = cap.IDpol.map(losses).fillna(0.)
    rows.append({"scenario": "Schadenzeilenbetrag bei EUR 25000 gekappt", "kind": "Betragszieländerung", **summary(cap)})
    result = pd.DataFrame(rows)
    for metric in ["policies", "exposure", "claims", "loss_eur"]:
        result["removed_"+metric] = base[metric]-result[metric]
    result["frequency_change_pct"] = 100*(result.frequency/base["frequency"]-1)
    result["loss_cost_change_pct"] = 100*(result.loss_cost_eur_per_year/base["loss_cost_eur_per_year"]-1)
    return result


def tail_tables(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = c.sort_values(["ClaimAmount", "claim_row_id"], ascending=[False, True])
    total = float(c.ClaimAmount.sum())
    sizes = sorted(set([1, 5, 10, 100, int(np.ceil(.01*len(c))), len(c)]))
    rows = []
    for k in sizes:
        part = ordered.head(k)
        loss = float(part.ClaimAmount.sum())
        rows.append({"top_claim_rows": len(part), "claim_row_share": len(part)/len(c),
                     "policies": part.IDpol.nunique(), "loss_eur": loss, "loss_share": loss/total})
    quantiles = pd.DataFrame({"quantile": [.5, .75, .9, .95, .99, .995, .999, 1.]})
    quantiles["claim_amount_eur"] = c.ClaimAmount.quantile(quantiles["quantile"].to_numpy(), interpolation="linear").to_numpy()
    return pd.DataFrame(rows), quantiles


def analyze(d: pd.DataFrame, c: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    check_development(d, c)
    d = add_groups(d, cfg)
    support = cfg["support_policy_years"]
    tables = {}
    for name in ["age_band", "age10_band", "DrivAge", "vehicle_band", "bm_band", "VehGas", "VehBrand", "Area", "Region", "density_band", "exposure_band"]:
        tables[name] = grouped(d, [name], support)
    pieces = []
    for name in ["bm_band", "vehicle_band", "VehGas", "VehBrand", "Area", "Region", "density_band"]:
        t = grouped(d, ["age_band", name], support).rename(columns={name: "level"})
        t["feature"] = name
        t["exposure_share_within_age"] = t.exposure/t.groupby("age_band").exposure.transform("sum")
        pieces.append(t)
    tables["age_composition"] = pd.concat(pieces, ignore_index=True)
    tables["b12_new"] = grouped(d, ["brand12", "brand_new"], support)
    tables["age_b12_new"] = grouped(d, ["age_band", "brand12", "brand_new"], support)
    tables["exposure_bm"] = grouped(d, ["exposure_band", "bm_band"], support)
    window = []
    half = cfg["age_window_half_width"]
    for age in sorted(d.DrivAge.unique()):
        part = d.loc[d.DrivAge.between(age-half, age+half)]
        window.append({"DrivAge": age, "age_min": int(part.DrivAge.min()), "age_max": int(part.DrivAge.max()), **summary(part)})
    tables["age_windows"] = pd.DataFrame(window)
    tables["age_bootstrap"], tables["age_contrasts"] = bootstrap_table(d, cfg)
    tables["bm_common_cells"], tables["bm_standardized"], tables["bm_standardized_contrasts"] = standardize_bm(d, cfg)
    tables["sensitivities"] = sensitivity(d, c)
    capped = d.copy()
    capped_losses = c.assign(capped=c.ClaimAmount.clip(upper=25000)).groupby("IDpol").capped.sum()
    capped[L] = capped.IDpol.map(capped_losses).fillna(0.)
    raw_age, capped_age = grouped(d, ["age_band"], support), grouped(capped, ["age_band"], support)
    raw_age["amount_target"], capped_age["amount_target"] = "ungekappt", "Schadenzeile bei EUR 25000 gekappt"
    tables["age_loss_sensitivity"] = pd.concat([raw_age, capped_age], ignore_index=True)
    age_sensitivities = []
    for label, subset in [("ungekappt", d), ("ohne B12", d.loc[d.VehBrand.ne("B12")]),
                          ("ohne Exposure < 0,1", d.loc[d[E].ge(.1)]),
                          ("Schadenzeile bei EUR 25000 gekappt", capped)]:
        part = grouped(subset, ["age_band"], support)
        part["scenario"] = label
        age_sensitivities.append(part)
    tables["age_sensitivity"] = pd.concat(age_sensitivities, ignore_index=True)
    tables["tail_concentration"], tables["claim_quantiles"] = tail_tables(c)
    top = c.nlargest(10, "ClaimAmount").merge(d[["IDpol", "DrivAge", "age_band"]], on="IDpol", validate="many_to_one")
    # Public aggregates omit policy and technical row keys.
    tables["top_loss_age_support"] = top.groupby(["DrivAge", "age_band"], observed=True).agg(
        top10_claim_rows=("ClaimAmount", "size"), top10_loss_eur=("ClaimAmount", "sum")).reset_index()
    # Observed ratio identity is not a frequency-severity prediction assumption.
    age = tables["age_band"]
    if not np.allclose(age.frequency*age.severity_eur.fillna(0), age.loss_cost_eur_per_year):
        raise DataContractError("Observed frequency-severity aggregation identity failed")
    report = {"development": summary(d), "development_claim_rows": len(c),
              "supplied_reconstructed_count_differences": int(d.ClaimNb.ne(d[N]).sum()),
              "zero_claim_policies": int(d[N].eq(0).sum()),
              "exact_repeated_policy_amount_rows_retained": int(c.duplicated(["IDpol", "ClaimAmount"]).sum()),
              "evaluation_rows_analyzed": 0, "models_fitted": 0,
              "age_heaping_counts": {str(k): int(d.DrivAge.mod(10).eq(k).sum()) for k in range(10)},
              "max_driver_age": int(d.DrivAge.max()), "max_vehicle_age": int(d.VehAge.max()),
              "age_100_policies": int(d.DrivAge.eq(100).sum()),
              "uncertainty": {"bootstrap": cfg["bootstrap"], "poisson": cfg["poisson_intervals"]},
              "scope": cfg["population"], "target": cfg["target"],
              "units": {"exposure": "Vertragsjahre", "frequency": "positive Schadenzeilen / Vertragsjahr",
                        "severity_eur": "historische EUR / positive Schadenzeile",
                        "loss_cost_eur_per_year": "historische erfasste EUR / Vertragsjahr"},
              "predictions": "Keine; beobachtete Verlustkosten sind kein vorhergesagter Tarif oder Marktpreis."}
    return tables, report


def render_figures(tables: dict, output: Path, cfg: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    paths = []

    def save(fig, name, note):
        fig.text(.01, .01, note, fontsize=8)
        fig.tight_layout(rect=(0, .055, 1, .95))
        path = output / (name+".png")
        fig.savefig(path, dpi=140, metadata={"Software": "freMTPL2 S3"})
        plt.close(fig)
        paths.append(path)

    ages = cfg["age_labels"]
    a = tables["age_bootstrap"].set_index("age_band").reindex(ages)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, label in zip(axes.flat[:3], ["frequency", "severity_eur", "loss_cost_eur_per_year"],
                                ["Schadenfrequenz / Vertragsjahr", "Mittlere Schadenhöhe (EUR / Schaden)", "Beobachtete Verlustkosten (EUR / Vertragsjahr)"]):
        y = a[metric].to_numpy()
        ax.plot(ages, y, "o-")
        ax.vlines(np.arange(len(ages)), a[metric+"_lower"], a[metric+"_upper"])
        ax.set_ylabel(label)
    axes[1, 1].bar(ages, a.exposure/1000, label="Exponierung (Tsd. Jahre)")
    axes[1, 1].set_ylabel("Tausend Vertragsjahre")
    for i, (_, row) in enumerate(a.iterrows()):
        axes[1, 1].text(i, row.exposure/1000, str(int(row.claims)), ha="center", va="bottom", fontsize=8)
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("DrivAge (Jahre)")
    fig.suptitle("Alter: Frequenz, Schadenhöhe und Verlustkosten im Entwicklungsbestand")
    save(fig, "alter_beobachtet", "Punktweise 95%-Perzentilintervalle aus 500 Vertragsziehungen. Zahlen über Balken: Schäden. Keine Kausal- oder Tarifschätzung.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    exact, win = tables["DrivAge"], tables["age_windows"]
    axes[0].plot(exact.DrivAge, exact.frequency, ".", color=".6", label="Einzelalter (ohne Intervall)")
    axes[0].plot(win.DrivAge, win.frequency, "-", label="Überlappendes 5-Jahres-Fenster, Quotient der Summen")
    centers = [21, 27, 32, 37, 42, 47, 52, 57, 62, 69.5, 82]
    axes[0].plot(centers, a.frequency, "o--", label="Feste Altersgruppen (siehe Hauptgrafik)")
    ten = tables["age10_band"].set_index("age10_band").reindex(cfg["age10_labels"])
    axes[0].plot([23.5, 34.5, 44.5, 54.5, 67, 82], ten.frequency, "s:", label="Breitere Gruppen")
    axes[0].set(xlim=(25, 65), ylabel="Schäden / Vertragsjahr")
    axes[0].legend(fontsize=8)
    axes[1].bar(exact.DrivAge, exact.exposure/1000, color=".5")
    axes[1].set(xlim=(18, 100), ylabel="Tsd. Vertragsjahre", xlabel="DrivAge (Jahre)")
    fig.suptitle("Altersprofil: Einzelalter, Gruppen und Exponierungsunterstützung")
    save(fig, "alter_robustheit", "Fenster überlappen und sind keine unabhängigen Schätzungen; keine Konfidenzbänder für eine Modellglättung. Randalter bleiben sichtbar.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    comp = tables["age_composition"]
    for bm in cfg["bm_labels"]:
        part = comp.loc[comp.feature.eq("bm_band") & comp.level.eq(bm)].set_index("age_band").reindex(ages)
        line, = axes[0, 0].plot(ages, part.frequency.where(part.supported), "o-", label="BM "+bm)
        sparse = ~part.supported.to_numpy(dtype=bool)
        if sparse.any():
            axes[0, 0].scatter(np.arange(len(ages))[sparse], part.frequency.to_numpy()[sparse], marker="x", color=line.get_color())
    axes[0, 0].set_ylabel("Schäden / Vertragsjahr")
    axes[0, 0].legend(fontsize=8)
    mix = comp.loc[comp.feature.eq("bm_band")].pivot(index="age_band", columns="level", values="exposure_share_within_age").reindex(ages)
    mix.reindex(columns=cfg["bm_labels"]).plot.bar(stacked=True, ax=axes[0, 1], legend=True, width=.85)
    axes[0, 1].legend(title="Bonus-Malus", fontsize=8)
    axes[0, 1].set_ylabel("Exponierungsanteil")
    std = tables["bm_standardized"]
    axes[1, 0].plot(std.age10_band, std.frequency, "o--", label="Marginal, gemeinsame Unterstützung")
    axes[1, 0].plot(std.age10_band, std.standardized_frequency, "o-", label="Gemeinsame BM-Mischung")
    axes[1, 0].vlines(np.arange(len(std)), std.lower, std.upper)
    axes[1, 0].set_ylabel("Schäden / Vertragsjahr")
    axes[1, 0].legend(fontsize=8)
    for feature, level, label in [("VehGas", "Diesel", "Diesel"), ("VehBrand", "B12", "B12"), ("Area", "F", "Area F")]:
        part = comp.loc[comp.feature.eq(feature) & comp.level.eq(level)].set_index("age_band").reindex(ages)
        axes[1, 1].plot(ages, part.exposure_share_within_age, "o-", label=label)
    axes[1, 1].set_ylabel("Exponierungsanteil")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("DrivAge (Jahre)")
    fig.suptitle("Alter und Zusammensetzung: deskriptive BM-Standardisierung")
    save(fig, "alter_zusammensetzung", "×: unter 500 Vertragsjahre. Standardisierung: nur gemeinsame BM-Unterstützung, feste Gewichte, 95%-Vertragsbootstrap; keine volle Adjustierung.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    focus = ["30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–64"]
    sens = tables["age_sensitivity"]
    for scenario in ["ungekappt", "ohne B12", "ohne Exposure < 0,1"]:
        part = sens.loc[sens.scenario.eq(scenario)].set_index("age_band").reindex(focus)
        axes[0].plot(focus, part.frequency, "o-", label=scenario)
    for scenario in ["ungekappt", "Schadenzeile bei EUR 25000 gekappt"]:
        part = sens.loc[sens.scenario.eq(scenario)].set_index("age_band").reindex(focus)
        axes[1].plot(focus, part.loss_cost_eur_per_year, "o-", label=scenario)
    axes[0].set_ylabel("Schäden / Vertragsjahr")
    axes[1].set_ylabel("Beobachtete EUR / Vertragsjahr")
    for ax in axes:
        ax.set_xlabel("DrivAge (Jahre)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle("Mittleres Alter: getrennte Populations- und Betragszieländerungen")
    save(fig, "alter_sensitivitaet", "Deskriptive Sensitivität ohne Intervalle; vollständige Unterstützung in age_sensitivity.json. Jüngste/älteste Gruppen in der Hauptgrafik.")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    specs = [("vehicle_band", "Fahrzeugalter", cfg["vehicle_labels"]), ("bm_band", "Bonus-Malus", cfg["bm_labels"]),
             ("VehGas", "Kraftstoff", None), ("VehBrand", "Anonymisierte Marke", None), ("Area", "Gebietsklasse", None), ("density_band", "Dichte", cfg["density_labels"])]
    for ax, (key, label, order) in zip(axes.flat, specs):
        part = tables[key].set_index(key)
        if order:
            part = part.reindex(order)
        y = part.frequency.to_numpy()
        ax.errorbar(part.index, y, yerr=[y-part.poisson_lower, part.poisson_upper-y], fmt="o", capsize=3)
        ax.set(xlabel=label, ylabel="Schäden / Vertragsjahr")
        ax.tick_params(axis="x", rotation=45)
        for i, (_, row) in enumerate(part.iterrows()):
            ax.annotate(f"{row.exposure/1000:.1f}k", (i, row.poisson_upper), fontsize=7, xytext=(0, 3), textcoords="offset points", ha="center")
    fig.suptitle("Portfoliomerkmale: beobachtete Raten mit Unterstützung")
    save(fig, "portfolio_raten", "Zahlen: Tsd. Vertragsjahre. Punktweise 95%-Poisson-Referenzintervalle; Überdispersion/Abhängigkeit nicht abgedeckt. Region: vollständige Ergebnistabelle.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ex_order = ["<1 Tag", "1–<7 Tage", "7 Tage–<0,1 J.", "0,1–<0,5 J.", "0,5–<1 J.", "1+ J."]
    ex = tables["exposure_band"].set_index("exposure_band").reindex(ex_order)
    axes[0].bar(ex.index, ex.frequency)
    axes[0].set(xlabel="Beobachtete Exponierung", ylabel="Schäden / Vertragsjahr")
    b12 = tables["b12_new"]
    for brand in ["B12", "andere Marken"]:
        part = b12.loc[b12.brand12.eq(brand)].set_index("brand_new").reindex(["VehAge=0", "VehAge>0"])
        axes[1].plot(part.index, part.frequency, "o-", label=brand)
    axes[1].set(xlabel="Fahrzeugalter", ylabel="Schäden / Vertragsjahr")
    axes[1].legend()
    tail = tables["tail_concentration"]
    axes[2].plot(tail.claim_row_share*100, tail.loss_share*100, "o-")
    axes[2].set(xscale="log", xlabel="Größte Schadenzeilen (%; logarithmisch)", ylabel="Anteil Schadenbetrag (%)")
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Exponierung, B12 und Verlustkonzentration: Prüfgruppen bleiben erhalten")
    save(fig, "pruefgruppen_verluste", "Nur Entwicklung; Einzelheiten/Unterstützung und getrennte Ziel-/Populationsänderungen in den Ergebnistabellen. Keine automatische Bereinigung.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--split-dir", type=Path, default=Path("data/validation"))
    parser.add_argument("--output", type=Path, default=Path("data/eda"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/s3"))
    args = parser.parse_args()
    policies, claims, prepared = load_prepared(prepared_location(args.prepared))
    frames, split = open_split(root_path(args.split_dir), prepared, protocol())
    d, c = development_only(policies, claims, frames)
    del policies, claims  # Analysis functions see development rows only.
    cfg = configuration()
    identity = {"source_id": prepared["source_id"], "preparation_id": prepared["preparation_id"],
                "split_id": split["split_id"], "eda_config_sha256": text_sha256(CONFIG),
                "eda_code_sha256": text_sha256(Path(__file__)),
                "development_policy_sha256": logical_hash(d), "development_claim_sha256": logical_hash(c),
                "software": {"python": platform.python_version(), **{p: version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "pyarrow"]}}}
    run_id = identity_id("s3eda-", identity)
    directory = root_path(args.output) / run_id
    tables, report = analyze(d, c, cfg)
    saved = write_immutable(directory, tables, {"run_id": run_id, "identity": identity, **report})
    reports = root_path(args.report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    figures = render_figures(tables, reports / "figures", cfg)
    saved["figures"] = {p.name: {"sha256": sha256(p)} for p in figures}
    # Public aggregate JSON mirrors: visible without downloading individual policy data.
    saved["public_tables"] = {}
    for name, table in tables.items():
        records = json.loads(table.to_json(orient="records", double_precision=15, force_ascii=False))
        path = reports / (name+".json")
        path.write_bytes(json_bytes({"run_id": run_id, "rows": records}))
        saved["public_tables"][path.name] = {"sha256": sha256(path)}
    (reports / "eda_report.json").write_bytes(json_bytes(saved))
    print(f"{run_id}: {len(d)} development policies; {len(tables)} tables; zero evaluation rows analyzed")


def load_report(report_dir: Path | str = "reports/s3", output: Path | str = "data/eda") -> tuple[dict, dict]:
    """Verify current code/config, immutable tables and every public mirror/figure."""
    report_dir = root_path(report_dir)
    saved = json.loads((report_dir / "eda_report.json").read_text(encoding="utf-8"))
    if (saved["run_id"] != identity_id("s3eda-", saved["identity"])
            or saved["identity"]["eda_code_sha256"] != text_sha256(Path(__file__))
            or saved["identity"]["eda_config_sha256"] != text_sha256(CONFIG)):
        raise DataContractError("EDA report identity/current code/config mismatch")
    tables, local = read_artifacts(root_path(output) / saved["run_id"])
    if local != {k: v for k, v in saved.items() if k not in {"figures", "public_tables"}}:
        raise DataContractError("EDA public and immutable local manifests differ")
    expected_mirrors = {name+".json" for name in tables}
    if set(saved["public_tables"]) != expected_mirrors:
        raise DataContractError("EDA public table inventory mismatch")
    for name, table in tables.items():
        path = report_dir / (name+".json")
        records = json.loads(table.to_json(orient="records", double_precision=15, force_ascii=False))
        if (sha256(path) != saved["public_tables"][path.name]["sha256"]
                or json.loads(path.read_text(encoding="utf-8")) != {"run_id": saved["run_id"], "rows": records}):
            raise DataContractError(f"EDA public table mismatch: {path}")
    for name, spec in saved["figures"].items():
        if Path(name).name != name or sha256(report_dir / "figures" / name) != spec["sha256"]:
            raise DataContractError("EDA figure identity mismatch")
    return tables, saved


if __name__ == "__main__":
    main()
