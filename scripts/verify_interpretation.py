"""Verify keyed S6a scores, units, fold claim means and published table hashes."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import json_bytes, sha256, text_sha256
from src.interpretation import N, E, L, cost_scenarios, feature_budget, load_inputs


def verify(check=False):
    reports = ROOT/"reports/s6a"
    if not check:
        # Match committed LF bytes on Windows without altering JSON values.
        for path in reports.glob("*.json"):
            raw = path.read_bytes()
            raw.decode("utf-8")
            if b"\r\n" in raw:
                path.write_bytes(raw.replace(b"\r\n", b"\n"))
    r = json.loads((reports/"report.json").read_text(encoding="utf-8"))
    for path, expected in r["identity"]["source_sha256"].items():
        assert text_sha256(ROOT/path) == expected, path
    assert text_sha256(ROOT/"config/interpretation_protocol.json") == r["identity"]["config_sha256"]
    for path, expected in r["identity"]["inputs_sha256"].items():
        assert sha256(ROOT/path) == expected, path
    d, c, membership, _, _, _ = load_inputs()
    directory = ROOT/"data/interpretation"/r["run_id"]/"results"
    assert sha256(directory.parent/"executed_source.py") == r["identity"]["source_sha256"]["src/interpretation.py"]
    assert sha256(directory/"manifest.json") == r["results_manifest_sha256"]
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    ppath = directory/"pp_predictions.parquet"
    assert sha256(ppath) == manifest["artifacts"]["pp_predictions"]["sha256"]
    p = pd.read_parquet(ppath)
    scores = pd.read_json(reports/"scores.json").set_index("model_id")
    expected = membership.query("split == 'development'")[["IDpol", "fold"]].sort_values("IDpol").reset_index(drop=True)
    checked = []
    for model, part in p.groupby("model_id"):
        assert part.IDpol.is_unique and part.run_id.eq(r["run_id"]).all()
        pd.testing.assert_frame_equal(part[["IDpol", "fold"]].sort_values("IDpol").reset_index(drop=True), expected)
        a = d[["IDpol", E, N, L]].merge(part, on="IDpol", validate="one_to_one")
        np.testing.assert_allclose(a.annual_pure_premium, a.annual_rate*a.severity_eur, rtol=1e-12)
        np.testing.assert_allclose(a.expected_period_loss, a.annual_pure_premium*a[E], rtol=1e-12)
        assert (a[["annual_rate", "severity_eur", "annual_pure_premium", "expected_period_loss"]].to_numpy() > 0).all()
        e, y, mu = a[E].to_numpy(), a[L].to_numpy()/a[E].to_numpy(), a.annual_pure_premium.to_numpy()
        # Independent algebra for p=1.5, including zero observed loss.
        deviance = np.dot(e, 4*(np.sqrt(y)-np.sqrt(mu))**2/np.sqrt(mu))/e.sum()
        ap = a[L].sum()/a.expected_period_loss.sum()
        np.testing.assert_allclose([deviance, ap], scores.loc[model, ["tweedie_deviance_per_exposure", "actual_over_predicted"]].to_numpy(float), rtol=1e-12)
        if model.startswith("two_"):
            second = model.split("_")[1]
            feature_budget(["DrivAge", second], [])
            for fold in range(1, 6):
                train_ids = set(membership.query("split == 'development' and fold != @fold").IDpol)
                claims = c.loc[c.IDpol.isin(train_ids)]
                held = part.query("fold == @fold")
                np.testing.assert_allclose(held.severity_eur, claims.ClaimAmount.mean(), rtol=1e-12)
        checked.append({"model_id": model, "deviance": float(deviance), "actual_over_predicted": float(ap),
                        "zero_loss_policies": int(a[N].eq(0).sum())})
    for name in ["Area", "geo", "Region", "density", "age", "Area_age", "geo_duration"]:
        t = pd.read_json(reports/f"{name}.json")
        np.testing.assert_allclose([t.policies.sum(), t.exposure_years.sum(), t.claims.sum(), t.loss_eur.sum()],
                                  [len(d), d[E].sum(), len(c), c.ClaimAmount.sum()], rtol=1e-12)
    costs = pd.read_json(reports/"costs.json")
    fixed, variable = cost_scenarios(costs.q_eur_per_year)
    np.testing.assert_allclose(costs.P_fix, fixed, rtol=1e-12)
    np.testing.assert_allclose(costs.P_var, variable, rtol=1e-12)
    np.testing.assert_allclose(costs.P_var-costs.q_eur_per_year, .15*costs.P_var, rtol=1e-12)
    files = [p for p in reports.glob("*.json") if p.name not in {"evidence.json", "release.json"}]
    files += list((reports/"figures").glob("*.png"))
    files += [ROOT/"scripts/s6a_diagnostics.py", Path(__file__), ROOT/"tests/test_interpretation.py"]
    evidence = {"run_id": r["run_id"], "models_recomputed": checked,
        "checks": ["exact development keys/folds and zeros", "independent p=1.5 algebra and A/P",
                   "40 training-claim mean comparisons", "contract products and annual/period units",
                   "7 complete partition totals", "cost shares and crossover unit tests"],
        "files_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in sorted(files)}}
    target = reports/"evidence.json"
    if check:
        assert json.loads(target.read_text(encoding="utf-8")) == evidence, "Published evidence differs"
    else:
        target.write_bytes(json_bytes(evidence))
    print(json.dumps({"run_id": r["run_id"], "models": len(checked), "files": len(files), "check": check}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    verify(parser.parse_args().check)
