"""Compare a completed isolated execution with published scientific evidence.

Run from the reference checkout. Existing reference artifacts are only read.
Large keyed prediction comparisons are optional when local caches are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce import commands
from scripts.build_results import collect


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific(value):
    """Only run labels may differ in otherwise identical scientific tables."""
    if isinstance(value, dict):
        return {key: scientific(item) for key, item in value.items() if key != "run_id"}
    if isinstance(value, list):
        return [scientific(item) for item in value]
    return value


def verify(run: Path, revision: str) -> dict:
    def published(name):
        return subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)

    def old(name):
        return json.loads(published(name))

    def current(name):
        return json.loads((run/name).read_text(encoding="utf-8"))

    execution = current("data/reproduction/execution.json")
    if (not execution["completed"] or not execution["fresh_cache_check"]
            or execution["initial_analytical_caches"]
            or [s["arguments"] for s in execution["steps"]] != commands()
            or any(s["exit_code"] for s in execution["steps"])):
        raise ValueError("A successful complete cache-free execution is required")
    if sha(run/"scripts/reproduce.py") != execution["producer_sha256"]:
        raise ValueError("Execution producer has changed")
    tracked = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision], cwd=ROOT, text=True).splitlines())
    reviewed = {p.relative_to(ROOT).as_posix() for pattern in
                ["src/*.py", "config/*.json", "requirements*.txt", "scripts/*.py", "notebooks/*.ipynb"]
                for p in ROOT.glob(pattern) if p.name != "features_glm.py"}
    if set(execution["workflow_source_sha256"]) != reviewed:
        raise ValueError("Execution source inventory differs from reviewed workflow")
    for name, digest in execution["workflow_source_sha256"].items():
        if sha(run/name) != digest:
            raise ValueError(f"Workflow source changed after start: {name}")
        reference_digest = hashlib.sha256(published(name)).hexdigest() if name in tracked else sha(ROOT/name)
        if digest != reference_digest:
            raise ValueError(f"Execution source differs from reviewed version: {name}")
    for step in execution["steps"]:
        if sha(run/step["log"]) != step["log_sha256"]:
            raise ValueError("Execution log changed")
    # The historical source manifest stays pinned; the clean run downloads its
    # raw files instead of inheriting this checkout's analytical caches.
    source = current("reports/s0/source_manifest.json")
    if source != old("reports/s0/source_manifest.json"):
        raise ValueError("Source manifest differs from reference")
    for name in ["cas_freq", "cas_sev"]:
        spec = source["data_files"][name]
        raw = run/"data/raw"/f"cas-{source['cas_commit']}"/spec["filename"]
        if sha(raw) != spec["sha256"]:
            raise ValueError("Raw download checksum differs")
    prep = current("reports/s1/preparation_report.json")
    original_prep = old("reports/s1/preparation_report.json")
    if any(prep[key] != original_prep[key] for key in ["reconciliation", "impact_audit", "contract"]):
        raise ValueError("Preparation accounting differs")
    split = current("reports/s2/split_manifest.json")
    reference_split = old("reports/s2/split_manifest.json")
    if any(split[key] != reference_split[key] for key in
           ["split_id", "identity", "artifacts", "membership_sizes", "development_fold_sizes",
            "protocol_sha256_at_freeze", "split_code_sha256_at_freeze"]):
        raise ValueError("Frozen membership differs from reference")
    frequency = current("reports/s4/frequency_report.json")
    severity = current("reports/s5/development_report.json")
    summary = collect(run)  # Checks phase identities, release hashes and local/public loss mirrors.
    original_decision = old("reports/s5/development_decision.json")
    decision = current("reports/s5/development_decision.json")
    if any(decision[key] != original_decision[key] for key in
           ["selected_pp_model_id", "selected_severity_model_id", "frequency_selected_model_id", "evaluation_power"]):
        raise ValueError("Reproduction changed the original modeling decision")
    if frequency["decision"]["selected_frequency_model"] != old("reports/s4/development_decision.json")["selected_frequency_model"]:
        raise ValueError("Frequency selection differs from the original decision")
    baseline = current("reports/s2/baseline_oof_report.json")
    reference_baseline = old("reports/s2/baseline_oof_report.json")
    if any(baseline[key] != reference_baseline[key] for key in ["fold_diagnostics", "summary", "units"]):
        raise ValueError("Baseline fold fits differ from reference")
    tables = []
    tables += [f"reports/s3/{name}" for name in old("reports/s3/eda_report.json")["public_tables"]]
    tables += [f"reports/s4/{spec['filename']}" for spec in old("reports/s4/frequency_report.json")["public_tables"].values()]
    for phase in ["development", "evaluation"]:
        report = current(f"reports/s5/{phase}_report.json")
        reference_report = old(f"reports/s5/{phase}_report.json")
        if set(report["artifacts"]) != set(reference_report["artifacts"]):
            raise ValueError("Complete original aggregate/prediction artifact inventory is required")
        tables += [f"reports/s5/{phase}_{name}.json" for name in report["artifacts"] if not name.endswith("predictions")]
    comparisons = {}
    for name in tables:
        before = published(name)
        if scientific(json.loads(before)) != scientific(current(name)):
            raise ValueError(f"Scientific result differs: {name}")
        comparisons[name] = {"reference_sha256": hashlib.sha256(before).hexdigest(),
                             "recomputed_sha256": sha(run/name),
                             "equal_excluding_only_run_id": True}
    images = {}
    expected_images = {
        "reports/s3/figures": set(old("reports/s3/eda_report.json")["figures"]),
        "reports/s4/figures": {v["filename"] for v in old("reports/s4/frequency_report.json")["figures"].values()},
        "reports/s5/figures": {v["filename"] for v in old("reports/s5/figures/manifest.json")["figures"].values()}}
    for folder, expected in expected_images.items():
        if {p.name for p in (run/folder).glob("*.png")} != expected:
            raise ValueError(f"Expected complete figure set: {folder}")
        for path in sorted((run/folder).glob("*.png")):
            name = path.relative_to(run).as_posix()
            reference_hash = hashlib.sha256(published(name)).hexdigest()
            if sha(path) != reference_hash:
                raise ValueError(f"Figure differs: {name}")
            images[name] = reference_hash
    predictions = {}
    import pandas as pd
    original_frequency = old("reports/s4/frequency_report.json")["run_id"]
    original_severity = old("reports/s5/development_report.json")["run_id"]
    pairs = [(f"frequency/{original_frequency}/{m}/oof_predictions.parquet",
              f"frequency/{frequency['run_id']}/{m}/oof_predictions.parquet") for m in frequency["models"]]
    for phase in ["development", "evaluation"]:
        report = current(f"reports/s5/{phase}_report.json")
        pairs += [(f"severity/{original_severity}/{phase}/{name}.parquet",
                   f"severity/{severity['run_id']}/{phase}/{name}.parquet")
                  for name in report["artifacts"] if name.endswith("predictions")]
    for before, after in pairs:
        old_path, new_path = ROOT/"data"/before, run/"data"/after
        if not old_path.exists():
            predictions[after] = {"reference_cache_available": False}
            continue
        a = pd.read_parquet(old_path).drop(columns="run_id")
        b = pd.read_parquet(new_path).drop(columns="run_id")
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        predictions[after] = {"reference_cache_available": True, "exactly_equal_excluding_only_run_id": True,
                              "rows": len(b), "reference_sha256": sha(old_path), "recomputed_sha256": sha(new_path)}
    notebooks = {}
    expected_notebooks = {args[1].split("/")[-1].replace(".ipynb", ".executed.ipynb")
                          for args in commands() if args[0] == "scripts/execute_notebook.py"}
    if {p.name for p in (run/"data/notebooks").glob("*.executed.ipynb")} != expected_notebooks:
        raise ValueError("Expected exactly the five active executed notebooks")
    for path in sorted((run/"data/notebooks").glob("*.executed.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        code = [cell for cell in cells if cell["cell_type"] == "code"]
        source_cells = current("notebooks/"+path.name.replace(".executed.ipynb", ".ipynb"))["cells"]
        if [cell["source"] for cell in code] != [cell["source"] for cell in source_cells if cell["cell_type"] == "code"]:
            raise ValueError("Executed code differs from the reviewed source notebook")
        if any(cell["execution_count"] is None or any(o["output_type"] == "error" for o in cell["outputs"]) for cell in code):
            raise ValueError("Notebook did not execute successfully")
        notebooks[path.name] = {"sha256": sha(path), "executed_code_cells": len(code), "cell_errors": 0}
    if len(notebooks) != 5:
        raise ValueError("Five executed active notebooks are required")
    return {"status": "passed", "reference_commit": revision, "execution_checkout_commit": execution["checkout_commit"],
            "verifier_sha256": sha(Path(__file__)), "execution": execution,
            "source_id": prep["source_id"], "preparation_id": prep["preparation_id"],
            "split_id": split["split_id"], "frequency_run_id": frequency["run_id"],
            "severity_run_id": severity["run_id"],
            "decision_id": current("reports/s5/development_decision.json")["decision_id"],
            "split_comparison": "Identical split identity and exact membership file/content hashes; preparation/software-at-freeze execution metadata may differ.",
            "baseline_summary_and_fold_diagnostics_equal": True,
            "locked_original_model_choices_preserved": True,
            "selected_evaluation": summary["selected_evaluation"],
            "aggregate_table_comparisons": comparisons, "byte_identical_figures": images,
            "prediction_comparisons": predictions, "notebooks": notebooks,
            "reference_scope": "Committed historical evidence; existing modified reference files are neither overwritten nor treated as the published baseline.",
            "limitations": ["Windows x64 / CPython 3.12.10 only.", "Sampled process-tree RSS is not an exact peak or unique physical memory.",
                            "Same-data deterministic reproduction is not new external validation or a new model-selection experiment."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-revision", default="HEAD")
    parser.add_argument("--output", type=Path, default=Path("reports/s6/reproduction.json"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.reference_revision], cwd=ROOT, text=True).strip()
    result = verify(args.run_root.resolve(), revision)
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(f"Verified {len(result['aggregate_table_comparisons'])} aggregate tables, {len(result['byte_identical_figures'])} figures and five fresh notebooks: {output}")


if __name__ == "__main__":
    main()
