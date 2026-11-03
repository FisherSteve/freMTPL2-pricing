"""Independently reconcile the extended S6a scores, lineage and fold boundaries."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.data import sha256, text_sha256
from src.interpretation import load_inputs, E, N, L
from src.validation import read_artifacts


def main():
    target = ROOT/"reports/s6a/ablation"
    report = json.loads((target/"report.json").read_text(encoding="utf-8"))
    identity = report["identity"]
    assert sha256(ROOT/"config/feature_ablation_protocol.json") == identity["config_sha256"]
    for path, digest in identity["source_sha256"].items():
        assert text_sha256(ROOT/path) == digest, path
    for name, digest in report["files_sha256"].items():
        assert sha256(target/name) == digest, name
    directory = ROOT/"data/interpretation"/report["run_id"]
    assert sha256(directory/"executed_source.py") == identity["source_sha256"]["src/feature_ablation.py"]
    assert sha256(directory/"results/manifest.json") == report["results_manifest_sha256"]
    old = ROOT/"data/interpretation"/report["specification"]["old_reference_run"]
    registration = json.loads((old/"registration.json").read_text(encoding="utf-8"))
    assert sha256(old/"registration.json") == identity["original_registration_sha256"]
    assert sha256(old/"results/manifest.json") == identity["original_manifest_sha256"]
    for path, digest in registration["identity"]["source_sha256"].items():
        assert text_sha256(ROOT/path) == digest, path
    frames, _ = read_artifacts(directory/"results")
    original, _ = read_artifacts(old/"results")
    p, oldp = frames["pp_predictions"], original["pp_predictions"]
    inherited = p.loc[p.model_id.isin(oldp.model_id)].copy()
    # The combined run identity is the only changed inherited field.
    pd.testing.assert_frame_equal(inherited.drop(columns="run_id").sort_values(["model_id", "IDpol"]).reset_index(drop=True),
        oldp.drop(columns="run_id").sort_values(["model_id", "IDpol"]).reset_index(drop=True))
    d, c, membership, _, _, _ = load_inputs()
    expected = membership.loc[membership.split.eq("development"), ["IDpol", "fold"]].sort_values("IDpol").reset_index(drop=True)
    scores = pd.read_json(target/"scores.json").set_index("model_id")
    training_means = {fold: c.loc[c.IDpol.isin(membership.loc[
        membership.split.eq("development") & membership.fold.ne(fold), "IDpol"])].ClaimAmount.mean()
        for fold in range(1, 6)}
    for model, part in p.groupby("model_id"):
        assert part.IDpol.is_unique and part.run_id.eq(report["run_id"]).all()
        pd.testing.assert_frame_equal(part[["IDpol", "fold"]].sort_values("IDpol").reset_index(drop=True), expected)
        aligned = d[["IDpol", E, N, L]].merge(part, on="IDpol", validate="one_to_one")
        e, y, q = aligned[E].to_numpy(), (aligned[L]/aligned[E]).to_numpy(), aligned.annual_pure_premium.to_numpy()
        independent = np.dot(e, 4*(np.sqrt(y)-np.sqrt(q))**2/np.sqrt(q))/e.sum()
        np.testing.assert_allclose(independent, scores.loc[model, "tweedie_deviance_per_exposure"], rtol=1e-12)
        np.testing.assert_allclose(aligned.expected_period_loss, e*q, rtol=1e-12)
        np.testing.assert_allclose(q, aligned.annual_rate*aligned.severity_eur, rtol=1e-12)
        for fold in range(1, 6):
            np.testing.assert_allclose(part.loc[part.fold.eq(fold), "severity_eur"], training_means[fold], rtol=1e-12)
    for audit in report["frequency_audit"]:
        assert audit["training_policies"]+audit["validation_policies"] == len(d)
        assert not audit["warnings"], audit
    for name in [p for p in frames if p.startswith("bootstrap_")]:
        boot = frames[name]
        reference = str(boot.reference_model.iloc[0])
        for row in boot.itertuples():
            np.testing.assert_allclose(row.deviance_difference,
                scores.loc[row.model_id, "tweedie_deviance_per_exposure"]-scores.loc[reference, "tweedie_deviance_per_exposure"], atol=1e-11)
        assert boot.replicates.eq(500).all()
    result = {"run_id": report["run_id"], "models": p.model_id.nunique(), "new_fits": len(report["frequency_audit"]),
        "development_policies_per_model": len(d), "evaluation_used": False,
        "original_predictions_exactly_preserved": True, "independent_score_and_units_check": True,
        "training_claim_means_checked": 5*p.model_id.nunique(), "unresolved_findings": []}
    local = ROOT/"data/s6a/ablation"
    local.mkdir(parents=True, exist_ok=True)
    (local/"verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
