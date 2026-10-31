"""Run the supported workflow sequentially; record logs, wall time and sampled RSS.

Use a separate clone: the scientific commands publish reports in this checkout.
No existing data are deleted and no model specification is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def commands() -> list[list[str]]:
    steps = [
        ["-m", "pip", "check"],
        ["scripts/check_environment.py", "--output", "data/reproduction/environment.json"],
        ["-m", "unittest", "discover", "-s", "tests", "-q"],
        ["-m", "src.data", "--download"],
        ["-m", "src.validation", "freeze"],
        ["-m", "src.validation", "oof-baseline"],
        ["-m", "src.eda"],
        ["-m", "src.frequency"],
        ["scripts/check_frequency_interaction.py"],
        ["scripts/verify_severity_inputs.py"],
        ["-m", "src.severity", "develop"],
        ["-m", "src.severity", "evaluate"],
        ["-m", "src.severity_figures"],
        ["scripts/build_results.py"],
    ]
    for name in ["01_data_preparation", "02_validation_contracts", "03_development_eda",
                 "04_frequency_models", "05_severity_pure_premium"]:
        steps.append(["scripts/execute_notebook.py", f"notebooks/{name}.ipynb",
                      "--output", f"data/notebooks/{name}.executed.ipynb"])
    return steps


def measured(command: list[str], log: Path) -> dict:
    start = time.perf_counter()
    peak = 0
    with log.open("wb") as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                members = [process, *process.children(recursive=True)]
            except psutil.Error:
                members = []
            rss = 0
            for member in members:
                try:
                    rss += member.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss)
            time.sleep(.2)
    return {"arguments": command[1:], "exit_code": child.returncode,
            "wall_seconds": time.perf_counter()-start,
            "sampled_peak_tree_rss_bytes": peak,
            "log": log.relative_to(ROOT).as_posix(),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true",
                        help="Require absent analytical caches before execution; never deletes anything")
    args = parser.parse_args()
    cache_paths = ["raw", "prepared", "validation", "eda", "frequency", "severity", "notebooks"]
    present = [name for name in cache_paths if (ROOT / "data" / name).exists()]
    if args.fresh and present:
        parser.error(f"Fresh run requires absent caches; found {present}. Use another clone.")
    destination = ROOT / "data/reproduction"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "execution.json").exists():
        parser.error("Execution evidence already exists; use a separate clone for another run.")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "platform": platform.platform(), "python": sys.version,
              "executable": sys.executable, "fresh_cache_check": args.fresh,
              "initial_analytical_caches": present,
              "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "workflow_source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for pattern in ["src/*.py", "config/*.json", "requirements*.txt", "scripts/*.py", "notebooks/*.ipynb"]
                  for p in sorted(ROOT.glob(pattern)) if p.name != "features_glm.py"},
              "measurement": "Wall time includes each child process. RSS is the maximum sampled sum over its process tree at 0.2-second intervals, not exact peak or unique physical memory; shared pages may be counted twice. Installation and this monitor are excluded.",
              "steps": [], "completed": False}
    started = time.perf_counter()
    for index, arguments in enumerate(commands(), 1):
        print(f"[{index}] {' '.join(arguments)}", flush=True)
        step = measured([sys.executable, *arguments], destination / f"{index:02d}.log")
        report["steps"].append(step)
        report["wall_seconds"] = time.perf_counter()-started
        (destination / "execution.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        if step["exit_code"]:
            raise SystemExit(f"Step {index} failed; inspect {step['log']}. Evidence retained.")
    report["completed"] = True
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (destination / "execution.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Completed {len(report['steps'])} commands in {report['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
