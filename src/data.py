"""Pinned CAS acquisition and loss-aligned, uncapped analysis tables (S1).

No split, learned preprocessing, feature encoding, or model is fitted here.
Relative CLI paths are always resolved against PROJECT_ROOT, never the kernel.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd
import pyreadr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT_ROOT / "reports/s0/source_manifest.json"
CONTRACT_PATH = PROJECT_ROOT / "config/data_contract.json"
FREQ_COLUMNS = ("IDpol", "ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
                "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")
CATEGORIES = ("Area", "VehBrand", "VehGas", "Region")
TARGET_COLUMNS = ("ClaimNb", "recorded_positive_count", "recorded_loss_eur")


class DataContractError(ValueError):
    """Input cannot support the declared target; diagnostics are not exclusions."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


def root_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(path: Path) -> str:
    raw = path.read_bytes()
    raw.decode("utf-8")
    if b"\r" in raw:
        raise DataContractError(f"Expected exact UTF-8/LF provenance bytes: {path}")
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def verify_raw(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise DataContractError(f"Raw checksum/size mismatch: {path}; preserve and investigate cache")
    # CAS's snapshotted MD5 is additional local provenance, not a provider signature.
    if hashlib.md5(path.read_bytes()).hexdigest() != spec["md5"]:
        raise DataContractError(f"Raw MD5 mismatch: {path}")


def acquire(cache: Path, spec: dict, download: bool = False) -> Path:
    """Verify existing bytes, or download missing bytes to an exclusive cache entry."""
    path = cache / spec["filename"]
    if path.exists():
        verify_raw(path, spec)
        return path
    if not download:
        raise FileNotFoundError(f"Missing pinned source {path}; run with --download")
    cache.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "freMTPL2-reproducible-study"})
    with tempfile.TemporaryDirectory(dir=cache, prefix=".download-") as temporary:
        candidate = Path(temporary) / spec["filename"]
        with urllib.request.urlopen(request, timeout=60) as response, candidate.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        verify_raw(candidate, spec)
        # Exclusive creation works without Windows hard-link privileges. An interrupted
        # copy remains detectable by verify_raw; never repair/replace it silently.
        try:
            with path.open("xb") as target, candidate.open("rb") as verified:
                for block in iter(lambda: verified.read(1024 * 1024), b""):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
        except FileExistsError:
            verify_raw(path, spec)
    return path


def exact_ids(values: pd.Series) -> pd.Series:
    if values.isna().any() or values.map(lambda x: isinstance(x, (bool, np.bool_))).any():
        raise DataContractError("IDpol must contain exact positive integer keys, without nulls/bools")
    # A rounded binary float at this magnitude cannot certify the source integer.
    unsafe_float = values.map(lambda x: isinstance(x, (float, np.floating)) and abs(x) >= 2**53)
    if unsafe_float.any():
        raise DataContractError("IDpol float is outside safe exact-integer range; preserve source tokens")
    labels = values.astype(str)
    exceptional = ~labels.str.fullmatch(r"[1-9][0-9]*")
    replacements = {}
    for label in labels[exceptional].unique():
        try:
            value = Decimal(label)
            if not value.is_finite() or value <= 0 or value != value.to_integral_value():
                raise DataContractError("IDpol must be an exact positive integer; no rounding")
            if value > np.iinfo(np.int64).max:
                raise DataContractError("IDpol exceeds int64")
            replacements[label] = str(int(value))
        except InvalidOperation as error:
            raise DataContractError("Invalid IDpol token") from error
    labels.loc[exceptional] = labels.loc[exceptional].map(replacements)
    limit = str(np.iinfo(np.int64).max)
    if ((labels.str.len() > len(limit)) |
            ((labels.str.len() == len(limit)) & (labels > limit))).any():
        raise DataContractError("IDpol exceeds int64")
    return labels.astype("int64")


def validate(frame: pd.DataFrame, severity: bool, contract: dict | None = None) -> pd.DataFrame:
    contract = load_contract() if contract is None else contract
    columns = ("IDpol", "ClaimAmount") if severity else FREQ_COLUMNS
    if len(frame.columns) != len(columns) or set(frame.columns) != set(columns):
        raise DataContractError(f"Unexpected {'severity' if severity else 'policy'} columns: {list(frame)}")
    if frame.empty:
        raise DataContractError("Source table is empty")
    out = frame.loc[:, list(columns)].copy().reset_index(drop=True)
    if out.isna().any().any():
        raise DataContractError("Source contains null values")
    out["IDpol"] = exact_ids(out["IDpol"])
    if not severity and out["IDpol"].duplicated().any():
        raise DataContractError("Policy IDpol must be unique after exact normalization")
    numeric = [c for c in columns if c not in (*CATEGORIES, "IDpol")]
    for name in numeric:
        if out[name].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise DataContractError(f"Boolean numeric field: {name}")
        out[name] = pd.to_numeric(out[name], errors="raise")
        if not np.isfinite(out[name].to_numpy(dtype=float)).all():
            raise DataContractError(f"Nonfinite {name}")
    if severity:
        if (out["ClaimAmount"] <= 0).any():
            raise DataContractError("Recorded positive-loss target requires ClaimAmount > 0")
        out["ClaimAmount"] = out["ClaimAmount"].astype("float64")
    else:
        for name in numeric:
            if name == "Exposure":
                if (out[name] <= 0).any():
                    raise DataContractError("Exposure must be positive, in policy-years")
                out[name] = out[name].astype("float64")
                continue
            lower, upper = contract["integer_domains"][name]
            if ((out[name] % 1 != 0) | (out[name] < lower) | (out[name] > upper)).any():
                raise DataContractError(f"Integral {name} outside declared domain [{lower}, {upper}]")
            out[name] = out[name].astype("int64")
        for name in CATEGORIES:
            if not out[name].isin(contract["category_levels"][name]).all():
                raise DataContractError(f"Unknown source category in {name}")
            # Retain source labels; ordinal meaning does not imply equal spacing.
            out[name] = out[name].astype("str")
    return out


def policy_totals(policies: pd.DataFrame) -> dict:
    return {"policies": len(policies), "exposure_years": float(policies["Exposure"].sum()),
            "supplied_claims": int(policies["ClaimNb"].sum()),
            "recorded_positive_claims": int(policies["recorded_positive_count"].sum()),
            "recorded_loss_eur": float(policies["recorded_loss_eur"].sum())}


def claim_totals(claims: pd.DataFrame) -> dict:
    return {"claim_rows": len(claims), "policies": int(claims["IDpol"].nunique()),
            "recorded_loss_eur": float(claims["ClaimAmount"].sum())}


def build_tables(freq: pd.DataFrame, sev: pd.DataFrame, severity_sha: str,
                 contract: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate and link; stop on orphan rows/count discrepancies, never discard them."""
    freq, sev = validate(freq, False, contract), validate(sev, True, contract)
    claims = sev.copy()
    claims.insert(0, "claim_row_id", [f"cas-{severity_sha}:{i:06d}" for i in range(len(sev))])
    counts = claims.groupby("IDpol", sort=True).agg(
        recorded_positive_count=("ClaimAmount", "size"), recorded_loss_eur=("ClaimAmount", "sum"))
    policies = freq.merge(counts, on="IDpol", how="left", sort=False, validate="one_to_one")
    policies["recorded_positive_count"] = policies["recorded_positive_count"].fillna(0).astype("int64")
    policies["recorded_loss_eur"] = policies["recorded_loss_eur"].fillna(0).astype("float64")
    policies = policies.sort_values("IDpol", kind="stable").reset_index(drop=True)
    linked = claims["IDpol"].isin(policies["IDpol"])
    without = ~policies["IDpol"].isin(claims["IDpol"])
    mismatch = policies["ClaimNb"] != policies["recorded_positive_count"]
    report = {
        "raw": {"policy_rows": len(freq), "claim_rows": len(sev),
                "claim_policy_ids": int(sev["IDpol"].nunique()),
                "recorded_loss_eur": float(sev["ClaimAmount"].sum())},
        "retained": policy_totals(policies),
        "excluded": {"policies": 0, "claim_rows": 0, "exposure_years": 0.0, "recorded_loss_eur": 0.0},
        "exclusion_reason": "Keine Ausschlüsse; Vertragszeilen ohne Schaden bleiben erhalten.",
        "anti_joins": {
            "policy_without_claim_rows": policy_totals(policies.loc[without]),
            "claim_rows_without_policy": claim_totals(claims.loc[~linked]),
            "orphan_policy_ids": sorted(int(x) for x in claims.loc[~linked, "IDpol"].unique()),
        },
        "supplied_vs_derived_mismatch": policy_totals(policies.loc[mismatch]),
        "count_cross_table": [
            {"supplied": int(a), "derived": int(b), "policies": int(n)}
            for (a, b), n in policies.groupby(["ClaimNb", "recorded_positive_count"]).size().items()],
        "repeat_claim_policy_rows_beyond_first": int(sev["IDpol"].duplicated().sum()),
        "exact_repeat_claim_rows_beyond_first_retained": int(sev.duplicated().sum()),
    }
    if not linked.all() or mismatch.any():
        raise DataContractError("Canonical source linkage/count contract failed; no rows were excluded", report)
    if policies["recorded_positive_count"].sum() != len(claims) or not np.isclose(
            policies["recorded_loss_eur"].sum(), claims["ClaimAmount"].sum(), rtol=0, atol=1e-6):
        raise DataContractError("Count/loss conservation failed", report)
    return policies, claims, report


def impact_report(policies: pd.DataFrame, claims: pd.DataFrame) -> dict:
    flags = {
        "exposure_above_one": policies["Exposure"] > 1,
        "exposure_below_one_month": policies["Exposure"] < 1 / 12,
        "driver_age_above_90": policies["DrivAge"] > 90,
        "vehicle_age_above_40": policies["VehAge"] > 40,
        "driver_age_100": policies["DrivAge"] == 100,
        "vehicle_age_100": policies["VehAge"] == 100,
        "count_above_four": policies["recorded_positive_count"] > 4,
        "bonus_malus_above_150": policies["BonusMalus"] > 150,
        "brand_B12": policies["VehBrand"] == "B12",
        "brand_B12_new_vehicle": (policies["VehBrand"] == "B12") & (policies["VehAge"] == 0),
    }
    total_loss = float(claims["ClaimAmount"].sum())
    return {
        "interpretation": "Überlappende Prüfgruppen, keine Ausschlüsse oder Kappungen; Ursachen ungeklärt.",
        "policy_flags": {name: policy_totals(policies.loc[mask]) for name, mask in flags.items()},
        "ranges": {name: {"min": float(policies[name].min()), "max": float(policies[name].max())}
                   for name in ("Exposure", "ClaimNb", "VehAge", "DrivAge", "BonusMalus", "Density")},
        "extreme_age_support": {
            name: [{"age": int(age), **policy_totals(group)} for age, group in
                   policies.loc[policies[name] > bound].groupby(name, sort=True)]
            for name, bound in (("VehAge", 40), ("DrivAge", 90))},
        "hypothetical_exposure_cap_one": {
            "removed_policy_years": float((policies["Exposure"] - 1).clip(lower=0).sum()),
            "remaining_policy_years": float(policies["Exposure"].clip(upper=1).sum()),
            "unchanged_claims": len(claims), "unchanged_loss_eur": total_loss},
        "hypothetical_count_cap_four": {
            "removed_counts": int((policies["recorded_positive_count"] - 4).clip(lower=0).sum()),
            "remaining_counts": int(policies["recorded_positive_count"].clip(upper=4).sum()),
            "loss_allocation": "Nicht definiert: keine willkürliche Auswahl zu entfernender Schadenzeilen."},
        "hypothetical_claim_amount_caps_eur": {
            str(cap): {**claim_totals(claims.loc[claims["ClaimAmount"] > cap]),
                       "removed_loss_eur": float((claims["ClaimAmount"] - cap).clip(lower=0).sum()),
                       "remaining_loss_eur": float(claims["ClaimAmount"].clip(upper=cap).sum())}
            for cap in (25000, 100000, 1000000)},
        "claim_amount_quantiles_eur": {str(q): float(claims["ClaimAmount"].quantile(q))
                                       for q in (0, .5, .9, .95, .99, .999, 1)},
        "top_claim_loss_shares": {str(n): float(claims["ClaimAmount"].nlargest(n).sum() / total_loss)
                                  for n in (1, 10, 100)},
    }


def logical_hash(frame: pd.DataFrame) -> str:
    """Stable declared-order CSV fingerprint; Parquet hashes additionally certify bytes."""
    raw = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prepare(cache: Path | str | None = None, output: Path | str = "data/prepared",
            download: bool = False) -> tuple[dict, Path]:
    """Rebuild deterministically, publish versioned artifacts, verify cached repetitions."""
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    contract = load_contract()
    cache = root_path(cache or f"data/raw/cas-{manifest['cas_commit']}")
    output = root_path(output)
    selected = {key: manifest["data_files"][key] for key in ("cas_freq", "cas_sev")}
    source_identity = {key: spec["sha256"] for key, spec in selected.items()}
    source_id = "cas-" + hashlib.sha256(json_bytes(source_identity)).hexdigest()[:16]
    software = {name: version(name) for name in ("numpy", "pandas", "pyreadr", "pyarrow")}
    software["python"] = platform.python_version()
    identity = {"source_id": source_id, "source_manifest_sha256": text_sha256(SOURCE_MANIFEST),
                "contract_sha256": text_sha256(CONTRACT_PATH), "code_sha256": text_sha256(Path(__file__)),
                "software": software}
    preparation_id = "s1-" + hashlib.sha256(json_bytes(identity)).hexdigest()[:16]
    frames, raw_schema = {}, {}
    for key, spec in selected.items():
        path = acquire(cache, spec, download)
        objects = pyreadr.read_r(str(path))
        if list(objects) != [spec["object"]]:
            raise DataContractError(f"Unexpected R object in {path.name}: {list(objects)}")
        frames[key] = objects[spec["object"]]
        raw_schema[key] = {"rows": len(frames[key]), "columns": {c: str(frames[key][c].dtype) for c in frames[key]}}
    policies, claims, reconciliation = build_tables(frames["cas_freq"], frames["cas_sev"],
                                                   selected["cas_sev"]["sha256"], contract)
    report = {"preparation_id": preparation_id, "source_id": source_id, "identity": identity,
              "identity_scope": "Source ID: CAS raw hashes. Preparation ID: source, manifest, contract, code and software. Output hashes: exact files and declared-order CSV content.",
              "source": {"version": manifest["cas_version"], "commit": manifest["cas_commit"],
                         "citation": manifest["citation"], "declared_terms": manifest["cas_declared_terms"],
                         "files": selected, "provider_metadata": manifest["provenance_files"]},
              "contract": contract, "raw_parsed_schema": raw_schema,
              "reconciliation": reconciliation, "impact_audit": impact_report(policies, claims),
              "analysis_tables": {}}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / preparation_id
    with tempfile.TemporaryDirectory(dir=output, prefix=".prepare-") as temporary:
        stage = Path(temporary) / preparation_id
        stage.mkdir()
        for name, frame in (("policies", policies), ("claims", claims)):
            path = stage / f"{name}.parquet"
            frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
            restored = pd.read_parquet(path, engine="pyarrow")
            pd.testing.assert_frame_equal(frame, restored, check_dtype=True)
            report["analysis_tables"][name] = {"filename": path.name, "rows": len(frame),
                "columns": {c: str(frame[c].dtype) for c in frame},
                "sha256": sha256(path), "logical_csv_sha256": logical_hash(frame)}
        (stage / "manifest.json").write_bytes(json_bytes(report))
        if destination.exists():
            for path in stage.iterdir():
                existing = destination / path.name
                if not existing.exists() or sha256(existing) != sha256(path):
                    raise DataContractError(f"Prepared artifact differs: {existing}; preserve and investigate")
        else:
            stage.rename(destination)
    return report, destination


def load_prepared(directory: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read one identified run, verify file hashes and logical content before use."""
    directory = root_path(directory)
    report = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_id = "s1-" + hashlib.sha256(json_bytes(report["identity"])).hexdigest()[:16]
    if report["preparation_id"] != expected_id or set(report["analysis_tables"]) != {"policies", "claims"}:
        raise DataContractError(f"Prepared manifest identity/table contract mismatch: {directory}")
    frames = {}
    for name, spec in report["analysis_tables"].items():
        if spec["filename"] != f"{name}.parquet":
            raise DataContractError(f"Unexpected prepared table filename: {spec['filename']}")
        path = directory / spec["filename"]
        if sha256(path) != spec["sha256"]:
            raise DataContractError(f"Prepared artifact checksum mismatch: {path}")
        frame = pd.read_parquet(path, engine="pyarrow")
        if {c: str(frame[c].dtype) for c in frame} != spec["columns"]:
            raise DataContractError(f"Prepared schema mismatch: {path}")
        if len(frame) != spec["rows"] or logical_hash(frame) != spec["logical_csv_sha256"]:
            raise DataContractError(f"Prepared content mismatch: {path}")
        frames[name] = frame
    return frames["policies"], frames["claims"], report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing pinned CAS files only")
    parser.add_argument("--cache", type=Path, help="Raw cache (relative paths use project root)")
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--report", type=Path, default=Path("reports/s1/preparation_report.json"))
    args = parser.parse_args()
    try:
        report, directory = prepare(args.cache, args.output, args.download)
        report_path = root_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(json_bytes(report))
        print(f"{report['preparation_id']}: {directory}")
        print(f"Reconciliation/manifest: {report_path}")
    except (DataContractError, FileNotFoundError, urllib.error.URLError) as error:
        print(str(error), file=sys.stderr)
        if getattr(error, "diagnostics", None):
            print(json_bytes(error.diagnostics).decode("utf-8"), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
