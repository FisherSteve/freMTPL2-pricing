"""Bounded S0 source audit, not the S1 analysis ingestion pipeline.

All joins use validated policy keys. No modeling, eligibility removal, caps, or
geographic recoding is applied. Only the four pinned raw data files are needed
to repeat the comparison; provider metadata is snapshotted in the manifest.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import urllib.request

import numpy as np
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("Area", "VehBrand", "VehGas", "Region")
FREQ_COLUMNS = ("IDpol", "ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
                "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_file(cache: Path, spec: dict, download: bool) -> Path:
    path = cache / spec["filename"]
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; use --download to retrieve pinned bytes")
        cache.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "freMTPL2-S0-audit"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch downloading {path.name}; cache not written")
        path.write_bytes(data)
    if digest(path) != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}; preserve/investigate the cache")
    if spec.get("provider_md5") and digest(path, "md5") != spec["provider_md5"]:
        raise ValueError(f"Provider MD5 mismatch: {path}")
    return path


def read_arff(path: Path) -> pd.DataFrame:
    """Read these pinned dense ARFFs; reject unsupported ARFF syntax.

    Numeric and single-quoted/string fields only. Not a general ARFF reader.
    The original headers and hashes remain available in the raw cache.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    attributes = []
    for position, line in enumerate(lines):
        line = line.strip()
        if line.lower() == "@data":
            break
        if line.lower().startswith("@attribute"):
            match = re.fullmatch(r"@attribute\s+(\w+)\s+(.+)", line, re.IGNORECASE)
            if not match:
                raise ValueError("Unsupported ARFF attribute")
            name, kind = match.groups()
            if kind.lower() not in {"numeric", "string"} and not kind.startswith("{"):
                raise ValueError(f"Unsupported ARFF type: {kind}")
            attributes.append((name, kind))
    else:
        raise ValueError("ARFF @data marker missing")
    if not attributes or len(set(n for n, _ in attributes)) != len(attributes):
        raise ValueError("ARFF attribute names missing or duplicated")
    frame = pd.read_csv(io.StringIO("\n".join(lines[position + 1:])),
                        names=[n for n, _ in attributes], quotechar="'", dtype="str")
    for name, kind in attributes:
        if kind.lower() == "numeric" and name != "IDpol":
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        elif kind.startswith("{"):
            levels = [s.strip().strip("'") for s in kind[1:-1].split(",")]
            if not frame[name].isin(levels).all():
                raise ValueError(f"Unexpected ARFF category in {name}")
    return frame


def validate(frame: pd.DataFrame, severity: bool) -> pd.DataFrame:
    required = {"IDpol", "ClaimAmount"} if severity else set(FREQ_COLUMNS)
    if set(frame) != required:
        raise ValueError(f"Unexpected columns: {list(frame)}")
    out = frame.copy().reset_index(drop=True)
    if out.isna().any().any():
        raise ValueError("Null values found")
    # One source factor level is '1e+05'. Parse that value exactly, never round.
    ids = out["IDpol"].astype(str)
    exceptional = ~ids.str.fullmatch(r"[1-9][0-9]*")
    replacements = {}
    for value in ids.loc[exceptional].unique():
        try:
            number = Decimal(value)
            if not number.is_finite() or number <= 0 or number != number.to_integral_value():
                raise ValueError("Policy keys must be finite positive integers; no rounding")
            if number > np.iinfo(np.int64).max:
                raise ValueError("Policy key exceeds int64")
            replacements[value] = str(int(number))
        except InvalidOperation as exc:
            raise ValueError("Invalid policy key") from exc
    ids.loc[exceptional] = ids.loc[exceptional].map(replacements)
    out["IDpol"] = ids.astype("int64")
    if not severity and out["IDpol"].duplicated().any():
        raise ValueError("Frequency policy keys must be unique")
    for name in required - set(CATEGORIES) - {"IDpol"}:
        out[name] = pd.to_numeric(out[name], errors="raise")
        if not np.isfinite(out[name].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {name}")
    if severity:
        if (out["ClaimAmount"] <= 0).any():
            raise ValueError("Severity target requires positive amounts")
    else:
        for name in ("ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"):
            if (out[name] < 0).any() or (out[name] % 1 != 0).any():
                raise ValueError(f"Nonnegative integral {name} required")
        if (out["Exposure"] <= 0).any() or (out["Density"] <= 0).any():
            raise ValueError("Exposure and density must be positive")
        for name in CATEGORIES:
            out[name] = out[name].astype(str)
    return out


def policy_totals(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    claims = sev.groupby("IDpol").agg(recorded_positive_count=("ClaimAmount", "size"),
                                     recorded_loss_eur=("ClaimAmount", "sum"))
    out = freq.set_index("IDpol").join(claims, how="left", validate="one_to_one")
    out["recorded_positive_count"] = out["recorded_positive_count"].fillna(0).astype("int64")
    out["recorded_loss_eur"] = out["recorded_loss_eur"].fillna(0)
    return out


def block_totals(frame: pd.DataFrame) -> dict:
    return {"policies": len(frame), "exposure_years": float(frame["Exposure"].sum()),
            "supplied_claim_count": int(frame["ClaimNb"].sum()),
            "recorded_positive_count": int(frame["recorded_positive_count"].sum()),
            "recorded_loss_eur": float(frame["recorded_loss_eur"].sum())}


def summarize(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    linked = sev["IDpol"].isin(freq["IDpol"])
    policies = policy_totals(freq, sev)
    unmatched = sev.loc[~linked].groupby("IDpol").agg(
        rows=("ClaimAmount", "size"), loss_eur=("ClaimAmount", "sum"))
    mismatch = policies["ClaimNb"] != policies["recorded_positive_count"]
    pairs = policies.groupby(["ClaimNb", "recorded_positive_count"]).size()
    return {
        "frequency_rows": len(freq), "severity_rows": len(sev),
        "frequency_schema": {c: str(freq[c].dtype) for c in freq},
        "severity_schema": {c: str(sev[c].dtype) for c in sev},
        "categories": {c: sorted(freq[c].unique().tolist()) for c in CATEGORIES},
        "totals": block_totals(policies),
        "raw_severity_loss_eur": float(sev["ClaimAmount"].sum()),
        "severity_unique_policy_keys": int(sev["IDpol"].nunique()),
        "severity_repeated_policy_rows_beyond_first": int(sev["IDpol"].duplicated().sum()),
        "severity_exact_repeat_rows_beyond_first": int(sev.duplicated().sum()),
        "unmatched_severity": [{"IDpol": int(i), "rows": int(r.rows), "loss_eur": float(r.loss_eur)}
                               for i, r in unmatched.iterrows()],
        "policies_without_severity_rows": int((policies["recorded_positive_count"] == 0).sum()),
        "supplied_vs_recorded_mismatch": block_totals(policies.loc[mismatch]),
        "count_cross_table": [{"supplied": int(a), "recorded": int(b), "policies": int(n)}
                              for (a, b), n in pairs.items()],
        "exposure_above_one": block_totals(policies.loc[policies["Exposure"] > 1]),
        "ranges": {c: [float(freq[c].min()), float(freq[c].max())]
                   for c in ("Exposure", "ClaimNb", "DrivAge", "VehAge", "Density")},
        "severity_range_eur": [float(sev["ClaimAmount"].min()), float(sev["ClaimAmount"].max())],
        "count_cap_four_removed_counts": int((policies["recorded_positive_count"] - 4).clip(lower=0).sum()),
    }


def compare(cas_f: pd.DataFrame, cas_s: pd.DataFrame,
            oml_f: pd.DataFrame, oml_s: pd.DataFrame) -> dict:
    cas = policy_totals(cas_f, cas_s)
    oml = policy_totals(oml_f, oml_s)
    common = cas.index.intersection(oml.index).sort_values()
    only_cas = cas.index.difference(oml.index).sort_values()
    only_oml = oml.index.difference(cas.index).sort_values()
    a, b = cas.loc[common], oml.loc[common]
    changes = {}
    for col in FREQ_COLUMNS[1:]:
        different = a[col] != b[col]  # exact equality, not an undocumented tolerance
        entry = {"changed_policies": int(different.sum())}
        if col in CATEGORIES:
            pairs = pd.DataFrame({"cas": a[col], "openml": b[col]}).groupby(["openml", "cas"]).size()
            entry["label_cross_table"] = [{"openml": x, "cas": y, "policies": int(n)}
                                          for (x, y), n in pairs.items()]
        elif different.any():
            delta = a.loc[different, col] - b.loc[different, col]
            entry["cas_minus_openml_sum"] = float(delta.sum())
            entry["delta_range"] = [float(delta.min()), float(delta.max())]
        changes[col] = entry
    # Claim IDs are absent: compare MULTISETS, keeping equal-amount repeat rows.
    sa = cas_s.groupby(["IDpol", "ClaimAmount"]).size().rename("cas")
    sb = oml_s.groupby(["IDpol", "ClaimAmount"]).size().rename("openml")
    claims = pd.concat([sa, sb], axis=1).fillna(0).astype("int64")
    difference = claims["cas"] - claims["openml"]
    diff_by_policy = {}
    for (key, amount), delta in difference[difference != 0].items():
        entry = diff_by_policy.setdefault(int(key), {"IDpol": int(key),
            "cas_minus_openml_rows": 0, "cas_minus_openml_loss_eur": 0.0})
        entry["cas_minus_openml_rows"] += int(delta)
        entry["cas_minus_openml_loss_eur"] += float(amount * delta)
    return {
        "common_policies": len(common),
        "cas_only": {"IDs": only_cas.tolist(), **block_totals(cas.loc[only_cas])},
        "openml_only": {"IDs": only_oml.tolist(), **block_totals(oml.loc[only_oml])},
        "common_column_changes": changes,
        "severity_multiset_common_rows": int(claims.min(axis=1).sum()),
        "severity_multiset_cas_only_rows": int(difference.clip(lower=0).sum()),
        "severity_multiset_openml_only_rows": int((-difference).clip(lower=0).sum()),
        "severity_multiset_differences_by_policy": list(diff_by_policy.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch absent pinned raw files")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/s0/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/s0/source_comparison.json")
    args = parser.parse_args()
    manifest_path = ROOT / "reports/s0/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames, raw_schema = {}, {}
    for key, spec in manifest["data_files"].items():
        path = checked_file(args.cache, spec, args.download)
        if path.suffix == ".rda":
            objects = pyreadr.read_r(str(path))
            if list(objects) != [spec["object"]]:
                raise ValueError(f"Unexpected R objects in {path.name}")
            raw = objects[spec["object"]]
        else:
            raw = read_arff(path)
        raw_schema[key] = {"rows": len(raw), "columns": {c: str(raw[c].dtype) for c in raw}}
        frames[key] = validate(raw, severity=key.endswith("sev"))
    software = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "pyreadr": pyreadr.__version__}
    identity = {k: v["sha256"] for k, v in manifest["data_files"].items()}
    run_id = "s0-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    result = {"run_id": run_id,
              "identity_scope": "run_id identifies raw-source hashes; script, manifest and software identify the comparison procedure",
              "script_sha256": digest(Path(__file__)), "manifest_sha256": digest(manifest_path),
              "software": software, "raw_parsed_schema": raw_schema,
              "comparison_units": {"Exposure": "policy-years", "ClaimNb": "claims per policy period",
                                   "ClaimAmount": "recorded EUR per claim row"},
              "normalization": ["IDs validated as exact positive integers, including scientific notation; no rounding",
                                "R factor labels retained as strings; row names discarded",
                                "Numeric values parsed without caps; no Region conversion",
                                "No full analysis tables exported in S0"],
              "cas": summarize(frames["cas_freq"], frames["cas_sev"]),
              "openml": summarize(frames["openml_freq"], frames["openml_sev"]),
              "between_sources": compare(frames["cas_freq"], frames["cas_sev"],
                                         frames["openml_freq"], frames["openml_sev"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{run_id}: wrote {args.output}")


if __name__ == "__main__":
    main()
