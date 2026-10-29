"""Check the supported interpreter, installed dependencies and tiny API fixtures.

The fixtures exercise compatibility only. They are not portfolio experiments or
evidence about comparative model quality; no freMTPL2 outcomes are accessed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import struct
import sys
import warnings


ROOT = Path(__file__).resolve().parents[1]


def check_environment() -> dict:
    if (platform.python_implementation() != "CPython"
            or sys.version_info[:2] != (3, 12) or struct.calcsize("P") != 8):
        raise RuntimeError("Supported workflow requires 64-bit CPython 3.12")

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyreadr
    import matplotlib
    import nbclient
    import nbformat
    import ipykernel
    import statsmodels.api as sm
    from pygam import PoissonGAM, s
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Nonconstant deterministic counts and varying exposure in a small fixture.
    # This is an API smoke check, not a model comparison or statistical test.
    x = np.linspace(0.0, 1.0, 120).reshape(-1, 1)
    exposure = np.tile([0.25, 0.5, 1.0, 1.5], 30)
    counts = np.tile([0, 0, 1, 2, 0, 1], 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        glm = sm.GLM(counts, sm.add_constant(x), family=sm.families.Poisson(),
                     offset=np.log(exposure)).fit()
        gam = PoissonGAM(s(0, n_splines=6), max_iter=100).fit(
            x, counts / exposure, weights=exposure)
        hgb = HistGradientBoostingRegressor(
            loss="poisson", max_iter=5, min_samples_leaf=5,
            early_stopping=False, random_state=17).fit(
                x, counts / exposure, sample_weight=exposure)
    predictions = {
        "poisson_glm_expected_period_counts": glm.predict(
            sm.add_constant(x), offset=np.log(exposure)),
        "poisson_gam_annual_rate": gam.predict(x),
        "histogram_poisson_annual_rate": hgb.predict(x),
    }
    for name, values in predictions.items():
        if values.shape != (len(x),) or not np.isfinite(values).all() or (values <= 0).any():
            raise RuntimeError(f"Invalid API smoke predictions: {name}")
    if not glm.converged or gam.logs_["diffs"][-1] >= gam.tol:
        raise RuntimeError("Tiny GLM/GAM compatibility fixture did not converge")
    # pyGAM 0.12.0 uses ~add_scale in its UBRE summary (pygam.py:1223).
    # CPython 3.12 warns about bool inversion. Record this upstream limitation;
    # fitting/prediction is exercised here, and no UBRE claim is made or used.
    # Any other warning remains an error, including convergence/numeric warnings.
    known_warnings = []
    for warning in caught:
        message = str(warning.message)
        if (warning.category is DeprecationWarning
                and Path(warning.filename).name == "pygam.py"
                and message.startswith("Bitwise inversion '~' on bool is deprecated")):
            known_warnings.append({"category": warning.category.__name__,
                                   "package": "pygam", "message": message,
                                   "source": "pygam/pygam.py:1223"})
        else:
            raise RuntimeError(f"Unexpected compatibility fixture warning: {message}")

    buffer = io.BytesIO()
    expected = pd.DataFrame({"IDpol": [1, 2], "Exposure": [0.25, 1.5]})
    pq.write_table(pa.Table.from_pandas(expected, preserve_index=False), buffer)
    actual = pq.read_table(io.BytesIO(buffer.getvalue())).to_pandas()
    pd.testing.assert_frame_equal(actual, expected)

    installed = {dist.metadata["Name"]: dist.version
                 for dist in importlib.metadata.distributions()}
    direct = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, required = line.split("==")
        version = importlib.metadata.version(name)
        if version != required:
            raise RuntimeError(f"Expected {name}=={required}; found {version}")
        direct[name] = version
    return {
        "scope": "Dependency and API fixture checks only; no portfolio model fitted",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture_bits": struct.calcsize("P") * 8,
        "direct_dependencies": direct,
        "package_metadata_sources": {
            name: f"https://pypi.org/pypi/{name}/{version}/json"
            for name, version in direct.items()},
        "installed_distributions": dict(sorted(installed.items(), key=lambda x: x[0].lower())),
        "checks": {"imports": "passed", "poisson_glm_fixture": "passed",
                   "poisson_gam_fixture": "passed", "histogram_poisson_fixture": "passed",
                   "parquet_round_trip": "passed", "fixture_warnings": known_warnings},
        "limitations": [
            "pyGAM 0.12.0 emits a CPython bool-inversion deprecation in its UBRE summary; "
            "UBRE values are not validated by this fixture and must not be treated as "
            "portfolio model-selection evidence. No third-party source is patched.",
            "Only Windows x64 / CPython 3.12 is verified; importing/fitting tiny fixtures "
            "does not validate the future portfolio modeling workflow."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_environment()
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        path = args.output if args.output.is_absolute() else ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"Environment and API fixtures passed; wrote {path}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
