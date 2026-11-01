"""Additional, post-registration joint-support diagnostics; never alter fits/rows.

Added after inspecting the first generator's output. These checks explain
fidelity limitations and are not a revised rejection or model-selection rule.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.data import json_bytes, sha256


def main():
    report = json.loads((ROOT / "reports/o2/report.json").read_text(encoding="utf-8"))
    base = ROOT / "data/o2" / report["run_id"]
    rows = []
    discrete = report["identity"]["config"]["discrete_columns"]
    for audit in report["generator_audits"]:
        folder = base / f"fold{audit['fold']}_seed{audit['seed']}"
        real = pd.read_parquet(folder / "training.parquet")
        syn = pd.read_parquet(folder / "synthetic.parquet")
        bounds = real.groupby("Area").Density.agg(["min", "max"])
        joined = syn[["Area", "Density"]].join(bounds, on="Area")
        outside = ~joined.Density.between(joined["min"], joined["max"])
        by_area = [{"Area": str(a), "training_density_min": float(b['min']),
            "training_density_max": float(b['max']), "synthetic_rows": int(syn.Area.eq(a).sum()),
            "outside_conditional_range": int((outside & syn.Area.eq(a)).sum())} for a,b in bounds.iterrows()]
        # For an exact discrete signature, compare both remaining continuous
        # coordinates using training-range-scaled logDensity and Exposure.
        center = np.array([np.log(real.Density.min()), real.Exposure.min()])
        scale = np.array([np.log(real.Density.max())-center[0], real.Exposure.max()-center[1]])
        scale = np.where(scale>0,scale,1.)
        numeric = lambda frame: (np.column_stack([np.log(frame.Density), frame.Exposure])-center)/scale
        reference = {}
        values = numeric(real)
        for key, indices in real.groupby(discrete, observed=True, sort=False).indices.items():
            reference[key] = values[indices]
        distances = []
        syn_values = numeric(syn)
        matched_rows = 0
        for key, indices in syn.groupby(discrete, observed=True, sort=False).indices.items():
            if key not in reference:
                continue
            matched_rows += len(indices)
            for i in indices:
                distances.append(float(np.abs(reference[key]-syn_values[i]).max(axis=1).min()))
        rows.append({"fold":audit["fold"],"seed":audit["seed"],"synthetic_rows":len(syn),
            "ratio_summaries":[{"ratio":ratio,"synthetic_rows":int(len(real)*ratio),
                "real_exposure":float(real.Exposure.sum()),
                "synthetic_exposure":float(syn.head(int(len(real)*ratio)).Exposure.sum()),
                "synthetic_claims":int(syn.head(int(len(real)*ratio)).recorded_positive_count.sum()),
                "synthetic_zero_fraction":float(syn.head(int(len(real)*ratio)).recorded_positive_count.eq(0).mean())}
                for ratio in report["identity"]["config"]["ratios"]],
            "outside_training_Area_density_range":int(outside.sum()),
            "outside_training_Area_density_fraction":float(outside.mean()), "by_area":by_area,
            "exact_discrete_signature_in_training_rows":matched_rows,
            "near_training_copies_at_tolerance_0_001":int((np.asarray(distances)<=.001).sum()),
            "continuous_distance_quantiles_on_matched_signatures":{str(q):float(np.quantile(distances,q)) for q in (0,.5,.99,1)} if distances else {},
            "input_sha256":{p.name:sha256(p) for p in [folder/'training.parquet',folder/'synthetic.parquet']}})
    output={"run_id":report["run_id"],"post_registration_diagnostic":True,
        "scope":"Additional explanatory diagnostics only; no row exclusion, refit, tuning or revised decision",
        "interpretation":"Area-density ranges are empirical training-subset support, not a proof of physical impossibility. Near copies require identical nine discrete values and max training-range-scaled distance <=0.001 in logDensity/Exposure; this is an operational diagnostic, not a privacy guarantee.",
        "rows":rows}
    (ROOT / "reports/o2/joint_support.json").write_bytes(json_bytes(output))
    print(f"Joint support checked for {len(rows)} generators")


if __name__=="__main__":
    main()
