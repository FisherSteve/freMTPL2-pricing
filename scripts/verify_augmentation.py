"""Independent arithmetic/lineage audit of a completed O2 pilot; no refits."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import xlogy
from src.data import sha256, text_sha256, logical_hash, load_prepared, json_bytes
from src.validation import prepared_location, open_split, protocol, identity_id


def verify():
    r = json.loads((ROOT / "reports/o2/report.json").read_text(encoding="utf-8"))
    identity = r["identity"]
    cfg = identity["config"]
    run = ROOT / "data/o2" / r["run_id"]
    assert identity_id("o2-", identity) == r["run_id"]
    assert json.loads((run / "registration.json").read_text(encoding="utf-8"))["identity"] == identity
    assert text_sha256(run / "executed_source.py") == identity["code"]["src/augmentation.py"]
    for name, digest in identity["code"].items():
        assert text_sha256(ROOT / name) == digest, name
    assert text_sha256(ROOT / "config/augmentation_protocol.json") == identity["config_sha256"]
    assert not r["evaluation_used"] and r["exploratory"]
    policies, _, prepared = load_prepared(prepared_location(None))
    frames, split = open_split(ROOT / "data/validation", prepared, protocol())
    membership = frames["policy_membership"]
    m = membership.loc[membership.split.eq("development")]
    eval_ids = set(membership.loc[membership.split.eq("evaluation"), "IDpol"])
    d = m.merge(policies, on="IDpol", validate="one_to_one").sort_values("IDpol").reset_index(drop=True)
    assert split["split_id"] == identity["split_id"]
    oof = pd.read_parquet(run / "oof.parquet")
    assert sha256(run / "oof.parquet") == r["local_oof_sha256"]
    predictions = {}
    count = d.recorded_positive_count.to_numpy(float)
    e = d.Exposure.to_numpy(float)
    losses = {}
    for s in r["scores"]:
        key = (s["method"], s["ratio"], s["seed"])
        p = oof.loc[oof.method.eq(key[0]) & oof.ratio.eq(key[1]) & oof.seed.eq(key[2])].sort_values("IDpol").reset_index(drop=True)
        assert p.IDpol.equals(d.IDpol) and p.fold.equals(d.fold)
        assert not set(p.IDpol) & eval_ids
        rate = p.annual_rate.to_numpy()
        mu = e*rate
        assert (rate > 0).all() and np.isfinite(rate).all()
        np.testing.assert_allclose(p.expected_count, mu, rtol=1e-12)
        # Direct mathematical count-deviance expression, independent of src.metrics.
        loss = 2*(xlogy(count, count)-xlogy(count, mu)-count+mu)
        np.testing.assert_allclose(loss.sum()/e.sum(), s["poisson_deviance_per_exposure"], rtol=1e-12)
        np.testing.assert_allclose(count.sum()/mu.sum(), s["actual_over_predicted"], rtol=1e-12)
        losses[key], predictions[key] = loss, p
        for fold in range(1, 6):
            f = next(f for f in r["fold_scores"] if (f["method"],f["ratio"],f["seed"],f["fold"]) == (*key, fold))
            mask = d.fold.eq(fold)
            np.testing.assert_allclose(loss[mask].sum()/e[mask].sum(), f["poisson_deviance_per_exposure"], rtol=1e-12)
    for s in r["scores"]:
        key = (s["method"], s["ratio"], s["seed"])
        for comparison, ref in [("vs_real", ("real", 0., 0)), ("vs_bootstrap", ("bootstrap", s["ratio"], s["seed"]))]:
            if comparison not in s:
                continue
            diff = losses[key]-losses[ref]
            delta = diff.sum()/e.sum()
            se = np.sqrt(len(e)*np.var(diff-delta*e, ddof=1))/e.sum()
            np.testing.assert_allclose(delta, s[comparison]["difference"], atol=2e-15)
            np.testing.assert_allclose(se, s[comparison]["standard_error"], atol=2e-15)
            np.testing.assert_allclose([delta-1.959963984540054*se, delta+1.959963984540054*se],
                                       s[comparison]["conditional_95_interval"], atol=2e-15)
    for fold in range(1, 6):
        ids = pd.read_parquet(run / f"training_ids_fold{fold}.parquet")
        assert ids.IDpol.is_unique and len(ids) == cfg["training_policies_per_fold"]
        assert not set(ids.IDpol) & (eval_ids | set(d.loc[d.fold.eq(fold), "IDpol"]))
        # Independently reconstruct the outcome-free subset recipe.
        eligible = d.loc[d.fold.ne(fold)].sort_values("IDpol").reset_index(drop=True)
        rng = np.random.default_rng(cfg["training_subset_seed"]+fold)
        idx = np.sort(rng.choice(len(eligible), len(ids), replace=False))
        assert ids.IDpol.equals(eligible.iloc[idx].IDpol.reset_index(drop=True))
        for seed in cfg["generator_seeds"]:
            folder = run / f"fold{fold}_seed{seed}"
            if not (folder / "synthetic.parquet").exists():
                continue
            train = pd.read_parquet(folder / "training.parquet")
            synthetic = pd.read_parquet(folder / "synthetic.parquet")
            worker = json.loads((folder / "worker.json").read_text(encoding="utf-8"))
            assert worker["device"] == "cpu" and worker["seed"] == seed
            assert logical_hash(train) == worker["training_joint_sha256"]
            assert list(train) == ["VehPower","VehAge","DrivAge","BonusMalus","VehBrand","VehGas","Area","Density","Region","Exposure","recorded_positive_count"]
            expected_train = eligible.iloc[idx][list(train)].reset_index(drop=True).copy()
            expected_train["Density"] = expected_train.Density.astype(float)
            pd.testing.assert_frame_equal(train, expected_train)
            assert synthetic.synthetic_id.is_unique and not synthetic.synthetic_id.astype(str).isin(ids.IDpol.astype(str)).any()
            assert synthetic.training_fold_excluded.eq(fold).all() and synthetic.generator_seed.eq(seed).all()
            assert synthetic.training_id_sha256.eq(logical_hash(ids)).all()
            assert synthetic.origin_run.eq(r["run_id"]).all()
            for name, digest in worker["files"].items():
                assert sha256(folder / name) == digest, name
            for col in cfg["discrete_columns"]:
                assert synthetic[col].isin(train[col]).all()
            for col in cfg["continuous_columns"]:
                assert np.isfinite(synthetic[col]).all() and synthetic[col].gt(0).all()
                assert synthetic[col].between(train[col].min(),train[col].max()).all()
            candidates = pd.read_parquet(folder / "candidates.parquet")
            good = candidates.notna().all(axis=1)
            for col in cfg["discrete_columns"]:
                good &= candidates[col].isin(train[col])
            for col in cfg["continuous_columns"]:
                good &= np.isfinite(candidates[col]) & candidates[col].gt(0) & candidates[col].between(train[col].min(),train[col].max())
            pd.testing.assert_frame_equal(synthetic[list(train)], candidates.loc[good].head(len(train)).reset_index(drop=True))
            quality = next(a for a in r["generator_audits"] if a["fold"]==fold and a["seed"]==seed)
            assert quality["validity"]["rejected"] == int((~good).sum())
            assert quality["fidelity"]["synthetic"]["claims"] == int(synthetic.recorded_positive_count.sum())
            # Confirm natural sampling probabilities in the actual fitted generator.
            from ctgan import CTGAN
            # Only load this run's locally generated, hash-verified weights.
            generator = CTGAN.load(str(folder / "generator.pkl"))
            for j, indices in enumerate(generator._data_sampler._rid_by_cat_cols):
                counts = np.array([len(x) for x in indices])
                np.testing.assert_allclose(generator._data_sampler._discrete_column_category_prob[j,:len(counts)], counts/counts.sum())
            del generator
    expected = 1+2*len(cfg["ratios"])*len(cfg["generator_seeds"])
    if not r["failures"]:
        assert len(r["scores"]) == expected and len(r["generator_audits"]) == 15
    ct = [s for s in r["scores"] if s["method"]=="ctgan"]
    real = next(s for s in r["scores"] if s["method"]=="real")
    keep = not r["failures"] and len(ct)==6 and all(s["vs_real"]["difference"]<0 and s["vs_bootstrap"]["difference"]<0
        and abs(s["actual_over_predicted"]-1) <= abs(real["actual_over_predicted"]-1)+.02 for s in ct)
    assert r["decision"] == ("further_independent_study" if keep else "discard_for_this_pilot")
    result = {"run_id": r["run_id"], "real_policies":len(d), "complete_arms":len(r["scores"]),
              "verified_generators":len(r["generator_audits"]), "decision":r["decision"], "passed":True}
    (run / "verification.json").write_bytes(json_bytes(result))
    print(json.dumps(result))


if __name__ == "__main__":
    verify()
