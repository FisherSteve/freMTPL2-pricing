"""Bounded CTGAN frequency pilot; generators receive training rows only.

Run: python -m src.augmentation. Core evaluation is never scored or selected.
The optional worker runs in a separate process so its time budget is enforceable.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data import (PROJECT_ROOT, DataContractError, json_bytes, load_prepared,
                      logical_hash, sha256, text_sha256)
from src.frequency_models import FEATURES, FrequencyGLM
from src.metrics import frequency_summary, poisson_count_deviance
from src.validation import open_split, prepared_location, protocol, identity_id

N, E = "recorded_positive_count", "Exposure"
JOINT = [*FEATURES, E, N]
CONFIG = PROJECT_ROOT / "config/augmentation_protocol.json"
CODE = ["src/augmentation.py", "src/frequency_models.py", "src/metrics.py",
        "src/data.py", "src/validation.py", "requirements-o2.txt", "requirements-o2.lock.txt", "requirements.lock.txt"]


def configuration():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (set(cfg["discrete_columns"]) | set(cfg["continuous_columns"]) != set(JOINT)
            or set(cfg["discrete_columns"]) & set(cfg["continuous_columns"])
            or N not in cfg["discrete_columns"] or cfg["generator"]["enable_gpu"]
            or cfg["ratios"] != [0.25, 1.0] or len(set(cfg["generator_seeds"])) != 3):
        raise DataContractError("Invalid bounded CPU pilot configuration")
    return cfg


def joint_frame(frame):
    """Strict allowlist: IDs, losses, memberships and predictions never reach CTGAN."""
    if list(frame) != JOINT or frame.empty or frame.isna().any().any():
        raise DataContractError("Generator requires exactly prospective features, exposure and count")
    numeric = frame[["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", E, N]].to_numpy(float)
    if (not np.isfinite(numeric).all() or (frame[E] <= 0).any() or (frame.Density <= 0).any()
            or (frame[N] < 0).any() or (frame[N] % 1 != 0).any()):
        raise DataContractError("Invalid training frequency units")
    out = frame.copy()
    # Density is a continuous measurement here; avoid inverse integer casts by
    # the generator's transformer. Actual ages/BM/counts are categorical.
    out["Density"] = out["Density"].astype(float)
    out[E] = out[E].astype(float)
    return out


def fold_training(development, membership, fold, cfg):
    expected = membership.loc[membership.split.eq("development"), ["IDpol", "fold"]]
    if (not development.IDpol.is_unique or set(development.IDpol) != set(expected.IDpol)
            or list(development) != ["IDpol", *JOINT] or fold not in range(1, 6)):
        raise DataContractError("Expected only the complete real development population")
    joined = expected.merge(development, on="IDpol", validate="one_to_one").sort_values("IDpol")
    eligible = joined.loc[joined.fold.ne(fold)].reset_index(drop=True)
    valid = joined.loc[joined.fold.eq(fold)].reset_index(drop=True)
    size = min(cfg["training_policies_per_fold"], len(eligible))
    idx = np.random.default_rng(cfg["training_subset_seed"] + fold).choice(len(eligible), size, replace=False)
    train = eligible.iloc[np.sort(idx)].reset_index(drop=True)
    if set(train.IDpol) & set(valid.IDpol) or train.empty or valid.empty:
        raise DataContractError("Training/validation IDs overlap or are empty")
    return train[["IDpol", *JOINT]], valid[["IDpol", *JOINT]]


def valid_synthetic(candidate, training, cfg):
    """Return auditable union mask; reason counts overlap. No repair by rounding."""
    if list(candidate) != JOINT:
        raise DataContractError("Unexpected synthetic columns")
    reasons = {"missing": candidate.isna().any(axis=1).to_numpy()}
    for c in cfg["discrete_columns"]:
        reasons[f"unsupported_{c}"] = (~candidate[c].isin(training[c].unique())).to_numpy()
    for c in cfg["continuous_columns"]:
        values = pd.to_numeric(candidate[c], errors="coerce").to_numpy(float)
        reasons[f"invalid_{c}"] = (~np.isfinite(values) | (values <= 0)
            | (values < training[c].min()) | (values > training[c].max()))
    counts = pd.to_numeric(candidate[N], errors="coerce").to_numpy(float)
    reasons["invalid_count"] = ~np.isfinite(counts) | (counts < 0) | (counts % 1 != 0)
    rejected = np.logical_or.reduce(list(reasons.values()))
    return ~rejected, {"attempted": len(candidate), "accepted": int((~rejected).sum()),
                       "rejected": int(rejected.sum()), "reasons": {k: int(v.sum()) for k, v in reasons.items()}}


def fidelity(training, synthetic):
    training, synthetic = joint_frame(training), joint_frame(synthetic)
    def describe(d):
        result = {"rows": len(d), "exposure": float(d[E].sum()), "claims": int(d[N].sum()),
                  "annual_rate": float(d[N].sum()/d[E].sum()),
                  "zero_fraction": float(d[N].eq(0).mean()),
                  "count_ge2_fraction": float(d[N].ge(2).mean()),
                  "count_max": int(d[N].max())}
        for c in (E, N, "DrivAge", "BonusMalus", "Density"):
            result[c + "_quantiles"] = {str(q): float(d[c].quantile(q)) for q in (0, .01, .5, .99, 1)}
        result["annual_policy_rate_quantiles"] = {str(q): float((d[N]/d[E]).quantile(q)) for q in (.5, .99, 1)}
        result["exposure_by_count"] = [{"count": int(n), "rows": len(g), "exposure": float(g[E].sum()),
            "mean_exposure": float(g[E].mean())} for n, g in d.groupby(N)]
        return result
    pairs = {}
    for cols in (("Area", "Region"), ("DrivAge", "BonusMalus"), ("VehBrand", "VehGas")):
        support = pd.MultiIndex.from_frame(training[list(cols)]).unique()
        pairs["/".join(cols)] = int((~pd.MultiIndex.from_frame(synthetic[list(cols)]).isin(support)).sum())
    # Exact complete-row matches are an operational memorization diagnostic, not privacy certification.
    real_hash = pd.util.hash_pandas_object(training[JOINT], index=False)
    syn_hash = pd.util.hash_pandas_object(synthetic[JOINT], index=False)
    marginal = {}
    for c in configuration()["discrete_columns"]:
        support = sorted(training[c].unique().tolist())
        marginal[c] = [{"value": str(v), "real_fraction": float(training[c].eq(v).mean()),
                        "synthetic_fraction": float(synthetic[c].eq(v).mean())} for v in support]
    return {"real": describe(training), "synthetic": describe(synthetic), "categorical_marginals": marginal,
            "unsupported_pairs": pairs, "exact_training_row_matches": int(syn_hash.isin(real_hash).sum()),
            "within_synthetic_duplicates": int(synthetic.duplicated(JOINT).sum())}


def augmented_training(real, extra):
    real, extra = joint_frame(real), joint_frame(extra)
    combined = pd.concat([real, extra], ignore_index=True)
    weights = combined[E].to_numpy(float) * (real[E].sum()/combined[E].sum())
    if not np.isclose(weights.sum(), real[E].sum(), rtol=1e-12):
        raise DataContractError("Effective exposure normalization failed")
    return combined, weights


def fit_predict(training, weights, validation_features, cfg):
    model = FrequencyGLM(**cfg["learner"])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=cfg["threads"]):
        warnings.simplefilter("always")
        model.fit(training[list(FEATURES)], training[N]/training[E], sample_weight=weights)
        pred = np.asarray(model.predict(validation_features), float)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise DataContractError("Unresolved learner convergence")
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise DataContractError("Invalid held-out annual rate")
    return pred, {"fit": model.diagnostics_, "unknown_validation_categories": model.last_unknown_categories_,
                  "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}


def paired_difference(counts, exposure, candidate, reference):
    e = np.asarray(exposure, float)
    diff = (poisson_count_deviance(counts, e*np.asarray(candidate))
            - poisson_count_deviance(counts, e*np.asarray(reference)))
    delta = float(diff.sum()/e.sum())
    centered = diff - delta*e
    se = float(np.sqrt(len(e)*np.var(centered, ddof=1))/e.sum())
    z = float(norm.ppf(.975))
    return {"difference": delta, "standard_error": se, "conditional_95_interval": [delta-z*se, delta+z*se]}


def restore_natural_sampling(generator):
    """0.11.1 sample_original_condvec reuses log-frequency probabilities.

    Keep log-frequency *training*. After fitting, set sampling probabilities to
    empirical training frequencies using the stored per-category row indices.
    Each discrete column retains total probability one, hence uniform column
    selection after flattening. No validation values or outcome reweighting.
    """
    sampler = generator._data_sampler
    old = sampler._discrete_column_category_prob.copy()
    natural = np.zeros_like(old)
    for col, indices in enumerate(sampler._rid_by_cat_cols):
        counts = np.asarray([len(rows) for rows in indices], float)
        if counts.sum() != sampler._data_length:
            raise DataContractError("CTGAN internal category counts differ from training size")
        natural[col, :len(counts)] = counts/counts.sum()
    sampler._discrete_column_category_prob = natural
    return {"training_condition_probabilities": old.tolist(),
            "sampling_condition_probabilities": natural.tolist(),
            "correction": "empirical category counts in pinned CTGAN 0.11.1, after fit only"}


def generator_worker(input_path, destination, seed, cfg):
    import torch
    from ctgan import CTGAN
    if version("ctgan") != cfg["ctgan_version"]:
        raise DataContractError("Unpinned CTGAN implementation")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train = joint_frame(pd.read_parquet(input_path))
    started = time.perf_counter()
    params = dict(cfg["generator"])
    params["generator_dim"] = tuple(params["generator_dim"])
    params["discriminator_dim"] = tuple(params["discriminator_dim"])
    generator = CTGAN(**params)
    generator.set_random_state(seed)
    with threadpool_limits(limits=cfg["threads"]):
        generator.fit(train, discrete_columns=cfg["discrete_columns"])
        fit_seconds = time.perf_counter()-started
        sampling = restore_natural_sampling(generator)
        target = int(len(train)*max(cfg["ratios"]))
        # One fixed candidate budget; rejection is not silently replaced indefinitely.
        candidate = generator.sample(target*cfg["maximum_sampling_multiplier"])[JOINT]
    candidate.to_parquet(destination / "candidates.parquet", index=False)
    generator.save(str(destination / "generator.pkl"))
    generator.loss_values.to_json(destination / "loss.json", orient="records")
    (destination / "worker.json").write_bytes(json_bytes({"fit_seconds": fit_seconds,
        "fit_and_sample_seconds": time.perf_counter()-started, "device": str(generator._device),
        "seed": seed, "training_joint_sha256": logical_hash(train), "sampling": sampling,
        "files": {p.name: sha256(p) for p in destination.iterdir() if p.is_file() and p.suffix != ".log"}}))


def generate(train, destination, seed, cfg):
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "training.parquet"
    joint_frame(train).to_parquet(path, index=False)
    (destination / "config.json").write_bytes(json_bytes(cfg))
    with (destination / "worker.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m", "src.augmentation", "--worker", str(path),
                        "--destination", str(destination), "--seed", str(seed)],
                       cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                       timeout=cfg["generator_timeout_seconds"], check=True)
    candidate = pd.read_parquet(destination / "candidates.parquet")
    accepted, audit = valid_synthetic(candidate, train, cfg)
    syn = candidate.loc[accepted].head(len(train)).reset_index(drop=True)
    (destination / "validity.json").write_bytes(json_bytes(audit))
    if len(syn) != len(train):
        raise DataContractError("Too few valid rows within fixed sampling budget")
    return syn, audit, json.loads((destination / "worker.json").read_text(encoding="utf-8"))


def summarize(development, membership, predictions):
    truth = membership.loc[membership.split.eq("development"), ["IDpol", "fold"]].merge(
        development, on="IDpol", validate="one_to_one").sort_values("IDpol").reset_index(drop=True)
    keys = predictions[["method", "ratio", "seed"]].drop_duplicates().sort_values(["method", "ratio", "seed"])
    aligned = {}
    for key in keys.itertuples(index=False, name=None):
        method, ratio, seed = key
        p = predictions.loc[predictions.method.eq(method) & predictions.ratio.eq(ratio) & predictions.seed.eq(seed)]
        if (not p.IDpol.is_unique or set(p.IDpol) != set(truth.IDpol)
                or not p.sort_values("IDpol").fold.reset_index(drop=True).equals(truth.fold)):
            raise DataContractError("Incomplete or misaligned real OOF predictions")
        aligned[key] = truth[["IDpol"]].merge(p, on="IDpol", validate="one_to_one").annual_rate.to_numpy()
    rows, folds, segments = [], [], []
    baseline = aligned[("real", 0.0, 0)]
    for key, rates in aligned.items():
        method, ratio, seed = key
        row = {"method": method, "ratio": ratio, "seed": seed,
               **frequency_summary(truth[N], rates, truth[E]),
               "vs_real": paired_difference(truth[N], truth[E], rates, baseline)}
        if method == "ctgan":
            row["vs_bootstrap"] = paired_difference(truth[N], truth[E], rates, aligned[("bootstrap", ratio, seed)])
        rows.append(row)
        for fold in range(1, 6):
            mask = truth.fold.eq(fold).to_numpy()
            folds.append({"method": method, "ratio": ratio, "seed": seed, "fold": fold,
                **frequency_summary(truth.loc[mask, N], rates[mask], truth.loc[mask, E]),
                "vs_real": paired_difference(truth.loc[mask, N], truth.loc[mask, E], rates[mask], baseline[mask])})
        groups = {"Area": truth.Area.astype(str), "age_band": pd.cut(truth.DrivAge, [0, 25, 35, 50, 65, 120], right=False).astype(str),
                  "bm_band": pd.cut(truth.BonusMalus, [0, 51, 76, 101, 1000], right=False).astype(str),
                  "exposure_band": pd.cut(truth[E], [0, .1, .5, 1, np.inf], right=False).astype(str)}
        for feature, labels in groups.items():
            if labels.eq("nan").any():
                raise DataContractError("Uncovered real diagnostic support")
            for value in sorted(labels.unique()):
                mask = labels.eq(value).to_numpy()
                segments.append({"method": method, "ratio": ratio, "seed": seed, "feature": feature, "value": value,
                    **frequency_summary(truth.loc[mask, N], rates[mask], truth.loc[mask, E])})
    reference = rows[[r["method"] for r in rows].index("real")]
    ct = [r for r in rows if r["method"] == "ctgan"]
    keep = len(ct) == 6 and all(r["vs_real"]["difference"] < 0 and r["vs_bootstrap"]["difference"] < 0
        and abs(r["actual_over_predicted"]-1) <= abs(reference["actual_over_predicted"]-1)+.02 for r in ct)
    return rows, folds, segments, "further_independent_study" if keep else "discard_for_this_pilot"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    if args.worker:
        cfg = json.loads((args.destination / "config.json").read_text(encoding="utf-8"))
        generator_worker(args.worker, args.destination, args.seed, cfg)
        return
    cfg = configuration()
    policies, _, prepared = load_prepared(prepared_location(args.prepared))
    frames, split = open_split(PROJECT_ROOT / "data/validation", prepared, protocol())
    membership = frames["policy_membership"]
    ids = membership.loc[membership.split.eq("development"), "IDpol"]
    development = policies.loc[policies.IDpol.isin(ids), ["IDpol", *JOINT]].sort_values("IDpol").reset_index(drop=True)
    joint_frame(development[JOINT])
    deps = {p: version(p) for p in ("ctgan", "torch", "rdt", "numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib", "threadpoolctl")}
    if deps["ctgan"] != cfg["ctgan_version"]:
        raise DataContractError("CTGAN release differs from protocol")
    identity = {"config": cfg, "code": {p: text_sha256(PROJECT_ROOT / p) for p in CODE},
        "config_sha256": text_sha256(CONFIG), "dependencies": deps,
        "preparation_id": prepared["preparation_id"], "source_id": prepared["source_id"],
        "split_id": split["split_id"], "prepared_tables": prepared["analysis_tables"],
        "development_sha256": logical_hash(development), "python": platform.python_version()}
    run_id = identity_id("o2-", identity)
    destination = PROJECT_ROOT / "data/o2" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "registration.json").write_bytes(json_bytes({"run_id": run_id, "identity": identity}))
    (destination / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    predictions, audits, failures = [], [], []
    started = time.perf_counter()

    def save_prediction(valid, rates, method, ratio, seed, fold):
        p = valid[["IDpol"]].copy()
        p["fold"], p["method"], p["ratio"], p["seed"] = fold, method, ratio, seed
        p["annual_rate"], p["expected_count"] = rates, rates*valid[E].to_numpy()
        predictions.append(p)
        pd.concat(predictions, ignore_index=True).to_parquet(destination / "oof.parquet", index=False)

    for fold in range(1, 6):
        train, valid = fold_training(development, membership, fold, cfg)
        base = train[JOINT]
        pd.DataFrame({"IDpol": train.IDpol}).to_parquet(destination / f"training_ids_fold{fold}.parquet", index=False)
        rates, fit = fit_predict(base, base[E].to_numpy(), valid[list(FEATURES)], cfg)
        save_prediction(valid, rates, "real", 0.0, 0, fold)
        audits.append({"fold": fold, "method": "real", "training_policies": len(train),
            "training_id_sha256": logical_hash(train[["IDpol"]]), "validation_id_sha256": logical_hash(valid[["IDpol"]]), **fit})
        print(f"{run_id} fold {fold}: real baseline done", flush=True)
        for seed in cfg["generator_seeds"]:
            worker_dir = destination / f"fold{fold}_seed{seed}"
            if time.perf_counter()-started > cfg["total_timeout_seconds"]:
                failures.append({"fold": fold, "seed": seed, "error": "total runtime budget exhausted"})
                continue
            try:
                syn, validity, worker = generate(base, worker_dir, seed, cfg)
                lineage = syn.copy()
                lineage.insert(0, "synthetic_id", [f"{run_id}/f{fold}/s{seed}/{i}" for i in range(len(syn))])
                lineage["origin_run"], lineage["training_fold_excluded"], lineage["generator_seed"] = run_id, fold, seed
                lineage["training_id_sha256"] = logical_hash(train[["IDpol"]])
                lineage.to_parquet(worker_dir / "synthetic.parquet", index=False)
                diagnostic = fidelity(base, syn)
                (worker_dir / "fidelity.json").write_bytes(json_bytes(diagnostic))
                audits.append({"fold": fold, "seed": seed, "method": "ctgan_generator", "validity": validity,
                               "worker": worker, "fidelity": diagnostic})
                for ratio in cfg["ratios"]:
                    n = int(len(base)*ratio)
                    rng = np.random.default_rng(seed*100+fold)
                    duplicate = base.iloc[rng.integers(0, len(base), n)].reset_index(drop=True)
                    for method, extra in (("bootstrap", duplicate), ("ctgan", syn.head(n))):
                        combined, weights = augmented_training(base, extra)
                        rates, fit = fit_predict(combined, weights, valid[list(FEATURES)], cfg)
                        save_prediction(valid, rates, method, ratio, seed, fold)
                        audits.append({"fold": fold, "seed": seed, "ratio": ratio, "method": method,
                            "additional_rows": n, "effective_exposure": float(weights.sum()),
                            "raw_real_exposure": float(base[E].sum()), "raw_extra_exposure": float(extra[E].sum()), **fit})
                print(f"fold {fold}, seed {seed}: generator {worker['fit_seconds']:.1f}s; all controls done", flush=True)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, DataContractError) as error:
                failures.append({"fold": fold, "seed": seed, "error": str(error)})
                print(f"fold {fold}, seed {seed}: FAILED {error}", flush=True)
            (destination / "audit.json").write_bytes(json_bytes(audits))
            (destination / "failures.json").write_bytes(json_bytes(failures))
    all_predictions = pd.concat(predictions, ignore_index=True)
    # Only publish complete OOF arms. Partial records remain in the local cache.
    complete = all_predictions.groupby(["method", "ratio", "seed"]).filter(lambda x: len(x) == len(development))
    rows, folds, segments, decision = summarize(development, membership, complete)
    if failures:
        decision = "discard_for_this_pilot"
    public = PROJECT_ROOT / "reports/o2"
    public.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "identity": identity, "exploratory": True, "evaluation_used": False,
        "training_regime": "20,000 fixed real policies per fold, same subset for all seeds and controls",
        "decision": decision, "failures": failures, "seconds": time.perf_counter()-started,
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "scores": rows, "fold_scores": folds, "generator_audits": [a for a in audits if a["method"] == "ctgan_generator"],
        "local_oof_sha256": sha256(destination / "oof.parquet"),
        "uncertainty": cfg["uncertainty"], "limitations": ["Exploratory after core evaluation; no confirmatory reserve",
            "Small fixed training subset; not full-data augmentation", "30 epochs and one architecture only; no convergence claim",
            "Conditional fixed-prediction intervals; folds and generator seeds are not independent portfolios",
            "Unknown household dependence; no causal or privacy guarantee"]}
    (public / "report.json").write_bytes(json_bytes(report))
    (public / "segments.json").write_bytes(json_bytes(segments))
    (destination / "report.json").write_bytes(json_bytes(report))
    print(f"{run_id}: {decision}; {len(rows)} complete OOF arms; {len(failures)} failures", flush=True)


if __name__ == "__main__":
    main()
