"""Small common-support/tail/calibration additions; never fit or read evaluation predictions."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import json_bytes, sha256
from src.interpretation import (N, E, L, configuration, describe_groups, groups,
    load_inputs, support_cells)
from src.validation import read_artifacts


def run():
    reports = ROOT/"reports/s6a"
    report = json.loads((reports/"report.json").read_text(encoding="utf-8"))
    directory = ROOT/"data/interpretation"/report["run_id"]/"results"
    if sha256(directory/"manifest.json") != report["results_manifest_sha256"]:
        raise ValueError("S6a result manifest differs")
    frames, manifest = read_artifacts(directory)
    d, c, membership, prepared, frozen, _ = load_inputs()
    d = groups(d)
    tables = {}
    for duration in [False, True]:
        retained, cells, strata = support_cells(d, duration)
        tag = "with_duration" if duration else "without_duration"
        tables[f"common_geo_{tag}"] = describe_groups(retained, c.loc[c.IDpol.isin(retained.IDpol)], ["geo"]) if len(retained) else pd.DataFrame()
        # Inspect only the actual jointly supported population, including zero losses.
        tables[f"common_geo_age_{tag}"] = describe_groups(retained, c.loc[c.IDpol.isin(retained.IDpol)], ["geo", "age"]) if len(retained) else pd.DataFrame()
    rows = []
    for model, pred in frames["pp_predictions"].groupby("model_id"):
        joined = d.merge(pred[["IDpol", "expected_period_loss", "annual_rate"]], on="IDpol", validate="one_to_one")
        joined["expected_count"] = joined[E]*joined.annual_rate
        for dimension in ["geo", "Area", "age", "duration"]:
            for level, p in joined.groupby(dimension, observed=True):
                e = p[E].sum()
                rows.append({"model_id": model, "dimension": dimension, "level": level,
                    "policies": len(p), "exposure_years": e, "claims": int(p[N].sum()),
                    "observed_frequency": p[N].sum()/e, "predicted_frequency": p.expected_count.sum()/e,
                    "observed_annual_loss_eur": p[L].sum()/e,
                    "predicted_annual_loss_eur": p.expected_period_loss.sum()/e,
                    "loss_actual_over_predicted": p[L].sum()/p.expected_period_loss.sum(),
                    "supported": bool(e >= 500 and p[N].sum() >= 100)})
    tables["geography_calibration"] = pd.DataFrame(rows)
    provenance = {"run_id": report["run_id"], "script_sha256": sha256(Path(__file__)),
        "result_manifest_sha256": sha256(directory/"manifest.json"),
        "source_id": prepared["source_id"], "split_id": frozen["split_id"], "inputs": {}}
    # Reuse existing uncertainty and supported full-model age profiles, with their lineage.
    for name, source in [("existing_age_contrasts", "reports/s3/age_contrasts.json"),
                         ("existing_age_profiles", "reports/s4/standardized_age.json")]:
        content = json.loads((ROOT/source).read_text(encoding="utf-8"))
        provenance["inputs"][source] = sha256(ROOT/source)
        if isinstance(content, dict) and "rows" in content:
            content = content["rows"]
        elif isinstance(content, list) and content and "rows" in content[0]:
            content = [record["rows"] for record in content]
        tables[name] = pd.DataFrame(content)
    for name, t in tables.items():
        (reports/f"{name}.json").write_bytes((t.to_json(orient="records", indent=2, force_ascii=False, double_precision=15)+"\n").encode("utf-8"))
    provenance["tables_sha256"] = {name: sha256(reports/f"{name}.json") for name in tables}
    (reports/"diagnostics.json").write_bytes(json_bytes(provenance))
    print(json.dumps({"run_id": report["run_id"], "tables": list(tables)}))


if __name__ == "__main__":
    run()
