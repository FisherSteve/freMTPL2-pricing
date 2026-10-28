"""Bounded checks for S0 provenance, exact keys, and conservation contracts.

Run with the temporary S0 parser environment; these are synthetic fixtures, not
model tests or a claim that the future S1 ingestion pipeline has been built.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("s0_audit", ROOT / "scripts/s0_compare_sources.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def frequency(keys=(1, 2)):
    return pd.DataFrame([
        {"IDpol": key, "ClaimNb": 3 if j == 0 else 0,
         "Exposure": 0.25 if j == 0 else 0.75, "VehPower": 5,
         "VehAge": 2, "DrivAge": 42, "BonusMalus": 50,
         "VehBrand": "B1", "VehGas": "Regular", "Area": "A",
         "Density": 10, "Region": "R11"}
        for j, key in enumerate(keys)
    ])


def severity(keys=(1, 1, 3), amounts=(100.0, 100.0, 7.0)):
    return pd.DataFrame({"IDpol": keys, "ClaimAmount": amounts})


@contextmanager
def workspace_tempdir():
    # Use the workspace because the Windows sandbox's default TEMP can deny writes.
    parent = ROOT / "data/s0"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="unit-s0-", dir=parent)
    path = Path(temporary.name).resolve()
    if ROOT not in path.parents:
        raise AssertionError("Temporary fixture must remain within the workspace")
    try:
        yield path
    finally:
        temporary.cleanup()


class KeyAndSchemaTests(unittest.TestCase):
    def test_scientific_factor_key_is_exact_and_does_not_change_amount(self):
        frame = severity(keys=("1e+05", "100001.0"), amounts=(1.25, 8.75))
        result = audit.validate(frame, severity=True)
        self.assertEqual(result["IDpol"].tolist(), [100000, 100001])
        self.assertEqual(result["ClaimAmount"].tolist(), [1.25, 8.75])

    def test_invalid_policy_keys_are_rejected_without_rounding(self):
        for bad in ("0", "-1", "1.0000000000000001", "NaN", "Infinity",
                    "abc", "9223372036854775808.0"):
            with self.subTest(key=bad), self.assertRaises((ValueError, OverflowError)):
                audit.validate(severity(keys=(bad,), amounts=(1.0,)), severity=True)

    def test_duplicate_frequency_keys_detected_after_normalization(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            audit.validate(frequency(keys=(1, "1e+00")), severity=False)

    def test_null_nonfinite_and_nonpositive_claim_amounts_fail(self):
        for amount in (None, np.nan, np.inf, -1.0, 0.0):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                audit.validate(severity(keys=(1,), amounts=(amount,)), severity=True)

    def test_invalid_frequency_units_fail(self):
        for column, bad in (("Exposure", 0), ("Exposure", -0.1),
                            ("Exposure", np.inf), ("ClaimNb", 0.5),
                            ("Density", 0)):
            frame = frequency()
            frame[column] = frame[column].astype(float)
            frame.loc[0, column] = bad
            with self.subTest(column=column, value=bad), self.assertRaises(ValueError):
                audit.validate(frame, severity=False)

    def test_exposure_above_one_and_uncapped_claim_counts_are_retained(self):
        frame = frequency()
        frame.loc[0, "Exposure"] = 1.5
        frame.loc[0, "ClaimNb"] = 8
        actual = audit.validate(frame, severity=False)
        self.assertEqual(actual.loc[0, "Exposure"], 1.5)
        self.assertEqual(actual.loc[0, "ClaimNb"], 8)

    def test_missing_and_unexpected_columns_fail(self):
        for frame in (severity().drop(columns="ClaimAmount"),
                      severity().assign(fold=1)):
            with self.subTest(columns=list(frame)), self.assertRaisesRegex(ValueError, "columns"):
                audit.validate(frame, severity=True)


class ArffTests(unittest.TestCase):
    def read(self, text):
        with workspace_tempdir() as directory:
            path = directory / "fixture.arff"
            path.write_text(text, encoding="utf-8")
            return audit.read_arff(path)

    def test_numeric_id_avoids_lossy_float_round_trip(self):
        # This exact integer is above binary64's consecutive-integer range.
        frame = self.read("@relation fixture\n@attribute IDpol numeric\n"
                          "@attribute ClaimAmount numeric\n@data\n"
                          "9007199254740993.0,10\n")
        self.assertEqual(audit.validate(frame, True).loc[0, "IDpol"], 9007199254740993)

    def test_dense_nominal_labels_and_quoted_values(self):
        frame = self.read("@relation fixture\n@attribute IDpol numeric\n"
                          "@attribute Area {'A','F'}\n@data\n1,'F'\n")
        self.assertEqual(frame["Area"].tolist(), ["F"])

    def test_unknown_nominal_value_and_missing_data_marker_fail(self):
        examples = ("@attribute Area {'A','F'}\n@data\n'Z'\n",
                    "@attribute IDpol numeric\n1\n",
                    "@attribute IDpol date\n@data\n1\n")
        for text in examples:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)


class LinkageAndComparisonTests(unittest.TestCase):
    def setUp(self):
        self.freq = audit.validate(frequency(), False)
        self.sev = audit.validate(severity(), True)

    def test_exact_repeated_claim_rows_remain_distinct_claims(self):
        policies = audit.policy_totals(self.freq, self.sev)
        self.assertEqual(policies.loc[1, "recorded_positive_count"], 2)
        self.assertEqual(policies.loc[1, "recorded_loss_eur"], 200.0)
        self.assertEqual(policies.loc[2, "recorded_positive_count"], 0)
        self.assertEqual(policies.loc[2, "recorded_loss_eur"], 0.0)
        # The supplied response is preserved rather than silently overwritten.
        self.assertEqual(policies.loc[1, "ClaimNb"], 3)

    def test_anti_join_and_loss_conservation_include_unmatched_severity(self):
        result = audit.summarize(self.freq, self.sev)
        self.assertEqual(result["unmatched_severity"],
                         [{"IDpol": 3, "rows": 1, "loss_eur": 7.0}])
        self.assertEqual(result["policies_without_severity_rows"], 1)
        self.assertEqual(result["severity_repeated_policy_rows_beyond_first"], 1)
        self.assertEqual(result["severity_exact_repeat_rows_beyond_first"], 1)
        self.assertEqual(result["totals"]["recorded_positive_count"] +
                         sum(x["rows"] for x in result["unmatched_severity"]), 3)
        self.assertEqual(result["totals"]["recorded_loss_eur"] +
                         sum(x["loss_eur"] for x in result["unmatched_severity"]),
                         result["raw_severity_loss_eur"])
        self.assertEqual(result["totals"]["exposure_years"], 1.0)
        self.assertEqual(result["supplied_vs_recorded_mismatch"]["policies"], 1)

    def test_comparison_is_invariant_to_independent_row_order(self):
        expected = audit.compare(self.freq, self.sev, self.freq, self.sev)
        actual = audit.compare(self.freq.iloc[::-1], self.sev,
                               self.freq, self.sev.iloc[::-1])
        self.assertEqual(expected, actual)
        self.assertTrue(all(x["changed_policies"] == 0
                            for x in actual["common_column_changes"].values()))
        self.assertEqual(actual["severity_multiset_common_rows"], 3)

    def test_comparison_uses_claim_multiset_not_set_or_row_positions(self):
        other = audit.validate(severity(keys=(3, 2, 3, 1),
                                        amounts=(7.0, 8.0, 7.0, 100.0)), True)
        result = audit.compare(self.freq, self.sev, self.freq.iloc[::-1], other)
        self.assertEqual(result["severity_multiset_common_rows"], 2)
        self.assertEqual(result["severity_multiset_cas_only_rows"], 1)
        self.assertEqual(result["severity_multiset_openml_only_rows"], 2)
        self.assertEqual(sorted(result["severity_multiset_differences_by_policy"],
                                key=lambda row: row["IDpol"]), [
            {"IDpol": 1, "cas_minus_openml_rows": 1, "cas_minus_openml_loss_eur": 100.0},
            {"IDpol": 2, "cas_minus_openml_rows": -1, "cas_minus_openml_loss_eur": -8.0},
            {"IDpol": 3, "cas_minus_openml_rows": -1, "cas_minus_openml_loss_eur": -7.0},
        ])

    def test_frequency_anti_joins_preserve_source_specific_totals(self):
        other = audit.validate(frequency(keys=(2, 4)), False)
        result = audit.compare(self.freq, self.sev, other, self.sev)
        self.assertEqual(result["common_policies"], 1)
        self.assertEqual(result["cas_only"]["IDs"], [1])
        self.assertEqual(result["cas_only"]["recorded_loss_eur"], 200.0)
        self.assertEqual(result["openml_only"]["IDs"], [4])
        self.assertEqual(result["openml_only"]["recorded_loss_eur"], 0.0)


class ProvenanceChecks(unittest.TestCase):
    def test_bad_cached_sha_is_rejected_and_preserved(self):
        with workspace_tempdir() as directory:
            path = directory / "raw.bin"
            path.write_bytes(b"historic bytes")
            spec = {"filename": path.name, "sha256": hashlib.sha256(b"different").hexdigest()}
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                audit.checked_file(directory, spec, download=False)
            self.assertEqual(path.read_bytes(), b"historic bytes")

    def test_provider_md5_is_independent_of_local_sha(self):
        with workspace_tempdir() as directory:
            path = directory / "raw.bin"
            path.write_bytes(b"bytes")
            spec = {"filename": path.name, "sha256": audit.digest(path), "provider_md5": "0" * 32}
            with self.assertRaisesRegex(ValueError, "Provider MD5 mismatch"):
                audit.checked_file(directory, spec, download=False)

    def test_failed_download_hash_does_not_create_cache_file(self):
        with workspace_tempdir() as directory:
            spec = {"filename": "raw.bin", "url": "https://example.invalid/pinned",
                    "sha256": hashlib.sha256(b"expected").hexdigest()}
            with patch.object(audit.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "cache not written"):
                    audit.checked_file(directory, spec, download=True)
            self.assertFalse((directory / "raw.bin").exists())


if __name__ == "__main__":
    unittest.main()
