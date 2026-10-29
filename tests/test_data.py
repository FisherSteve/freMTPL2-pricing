"""Failure-risk and statistical-contract checks for the permanent data layer."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data


@contextmanager
def workspace_tempdir():
    base = data.PROJECT_ROOT / "data/test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as path:
        yield Path(path)


def frequency():
    return pd.DataFrame({
        "IDpol": ["1e+00", "2", "3"], "ClaimNb": [2, 0, 1],
        "Exposure": [1.5, .1, .5], "VehPower": [4, 6, 8], "VehAge": [100, 0, 5],
        "DrivAge": [100, 18, 42], "BonusMalus": [50, 100, 150],
        "VehBrand": ["B1", "B12", "B2"], "VehGas": ["Regular", "Diesel", "Regular"],
        "Area": ["A", "F", "C"], "Density": [1, 27000, 100],
        "Region": ["Centre", "Corse", "Alsace"],
    })


def severity():
    return pd.DataFrame({"IDpol": [1, 1, 3], "ClaimAmount": [100., 100., 1000000.]})


class SchemaAndKeys(unittest.TestCase):
    def test_ids_exact_scientific_and_large_string_without_rounding(self):
        result = data.exact_ids(pd.Series(["1e+05", "9007199254740993.0", "9223372036854775807"]))
        self.assertEqual(result.tolist(), [100000, 9007199254740993, 9223372036854775807])

    def test_fractional_null_infinite_boolean_out_of_range_ids_fail(self):
        for value in ("1.00000000001", "0", "-1", None, np.inf, True,
                      "9223372036854775808", float(2**53), "not-a-key"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                data.exact_ids(pd.Series([value]))

    def test_duplicate_policy_after_normalization_fails(self):
        frame = frequency()
        frame["IDpol"] = ["1", "1.0", "3"]
        with self.assertRaisesRegex(data.DataContractError, "unique"):
            data.validate(frame, False)

    def test_invalid_numerics_and_domains_fail_without_cleaning(self):
        for column, bad in (("ClaimNb", .5), ("ClaimNb", -1), ("Exposure", 0),
                            ("Exposure", np.inf), ("Density", 0), ("DrivAge", 17),
                            ("BonusMalus", 351), ("VehAge", -.1), ("VehPower", 0),
                            ("ClaimNb", True), ("VehAge", np.nan)):
            frame = frequency().astype({column: "object"})
            frame.loc[0, column] = bad
            with self.subTest(column=column, bad=bad), self.assertRaises(ValueError):
                data.validate(frame, False)
        for bad in (0, -1, np.inf, np.nan, None, True):
            frame = severity().astype({"ClaimAmount": "object"})
            frame.loc[0, "ClaimAmount"] = bad
            with self.subTest(amount=bad), self.assertRaises(ValueError):
                data.validate(frame, True)

    def test_category_labels_are_source_semantics(self):
        for column, value in (("Area", "G"), ("Region", "R24"),
                              ("VehGas", "Petrol"), ("VehBrand", "BMW")):
            frame = frequency()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "category"):
                data.validate(frame, False)

    def test_missing_extra_duplicated_columns_and_empty_table_fail(self):
        frame = frequency()
        cases = [frame.drop(columns="Exposure"), frame.assign(fold=1), frame.iloc[:0],
                 pd.concat([frame, frame[["Exposure"]]], axis=1)]
        for bad in cases:
            with self.subTest(columns=list(bad)), self.assertRaises(ValueError):
                data.validate(bad, False)


class LinkageAndImpacts(unittest.TestCase):
    def setUp(self):
        self.policies, self.claims, self.report = data.build_tables(frequency(), severity(), "a" * 64)

    def test_zero_policies_repeated_claims_uncapped_values_and_lineage_survive(self):
        self.assertEqual(self.policies["ClaimNb"].tolist(), [2, 0, 1])
        self.assertEqual(self.policies["recorded_positive_count"].tolist(), [2, 0, 1])
        self.assertEqual(self.policies["recorded_loss_eur"].tolist(), [200., 0., 1000000.])
        self.assertEqual(self.policies["Exposure"].tolist(), [1.5, .1, .5])
        self.assertEqual(self.policies["VehAge"].max(), 100)
        self.assertEqual(self.claims["ClaimAmount"].max(), 1000000)
        self.assertTrue(self.claims["claim_row_id"].is_unique)
        self.assertTrue(self.claims["claim_row_id"].str.contains("a" * 64, regex=False).all())
        self.assertEqual(self.report["exact_repeat_claim_rows_beyond_first_retained"], 1)
        self.assertEqual(self.report["anti_joins"]["policy_without_claim_rows"]["policies"], 1)
        self.assertEqual(self.report["retained"]["recorded_loss_eur"], self.claims["ClaimAmount"].sum())
        self.assertEqual(self.report["raw"]["policy_rows"], 3)
        self.assertEqual(self.report["raw"]["claim_policy_ids"], 2)

    def test_orphan_and_count_mismatch_fail_with_quantified_diagnostics(self):
        orphan = pd.concat([severity(), pd.DataFrame({"IDpol": [99], "ClaimAmount": [7.]})])
        with self.assertRaises(data.DataContractError) as caught:
            data.build_tables(frequency(), orphan, "a" * 64)
        diagnostic = caught.exception.diagnostics
        self.assertEqual(diagnostic["anti_joins"]["orphan_policy_ids"], [99])
        self.assertEqual(diagnostic["anti_joins"]["claim_rows_without_policy"]["recorded_loss_eur"], 7.)
        freq = frequency()
        freq.loc[0, "ClaimNb"] = 3
        with self.assertRaises(data.DataContractError) as caught:
            data.build_tables(freq, severity(), "a" * 64)
        self.assertEqual(caught.exception.diagnostics["supplied_vs_derived_mismatch"]["policies"], 1)
        self.assertEqual(caught.exception.diagnostics["supplied_vs_derived_mismatch"]["exposure_years"], 1.5)

    def test_policy_alignment_is_independent_of_input_order(self):
        actual, claims, _ = data.build_tables(frequency().iloc[::-1], severity().iloc[::-1], "a" * 64)
        pd.testing.assert_frame_equal(actual, self.policies)
        self.assertEqual(claims["IDpol"].tolist(), [3, 1, 1])

    def test_hypothetical_caps_quantify_impact_without_changing_targets(self):
        before_policy = self.policies.copy(deep=True)
        before_claim = self.claims.copy(deep=True)
        report = data.impact_report(self.policies, self.claims)
        self.assertEqual(report["hypothetical_exposure_cap_one"]["removed_policy_years"], .5)
        self.assertEqual(report["hypothetical_claim_amount_caps_eur"]["25000"]["removed_loss_eur"], 975000.)
        self.assertEqual(report["policy_flags"]["vehicle_age_100"]["recorded_positive_claims"], 2)
        self.assertAlmostEqual(report["top_claim_loss_shares"]["1"], 1000000 / 1000200)
        pd.testing.assert_frame_equal(self.policies, before_policy)
        pd.testing.assert_frame_equal(self.claims, before_claim)

    def test_count_cap_does_not_fabricate_loss_allocation(self):
        freq = frequency().iloc[:1].copy()
        freq["ClaimNb"] = 6
        sev = pd.DataFrame({"IDpol": [1] * 6, "ClaimAmount": [10., 20., 30., 40., 50., 60.]})
        policies, claims, _ = data.build_tables(freq, sev, "a" * 64)
        impact = data.impact_report(policies, claims)
        self.assertEqual(impact["hypothetical_count_cap_four"]["removed_counts"], 2)
        self.assertNotIn("removed_loss_eur", impact["hypothetical_count_cap_four"])
        self.assertEqual(policies["recorded_loss_eur"].sum(), 210.)


class CacheAndArtifacts(unittest.TestCase):
    def spec(self, raw):
        return {"filename": "fixture.rda", "url": "https://example.invalid/fixed",
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "md5": hashlib.md5(raw).hexdigest()}

    def test_cache_mismatch_is_never_overwritten_or_downloaded(self):
        with workspace_tempdir() as directory:
            path = directory / "fixture.rda"
            path.write_bytes(b"historical")
            with patch.object(data.urllib.request, "urlopen") as request:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    data.acquire(directory, self.spec(b"correct"), True)
                request.assert_not_called()
            self.assertEqual(path.read_bytes(), b"historical")

    def test_missing_and_bad_download_never_publish_raw(self):
        with workspace_tempdir() as directory:
            spec = self.spec(b"expected")
            with self.assertRaisesRegex(FileNotFoundError, "--download"):
                data.acquire(directory, spec)
            with patch.object(data.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(ValueError):
                    data.acquire(directory, spec, True)
            self.assertFalse((directory / "fixture.rda").exists())

    def test_verified_download_and_cached_repeat_keep_bytes(self):
        with workspace_tempdir() as directory:
            raw = b"raw\r\nprovider\x00bytes"
            with patch.object(data.urllib.request, "urlopen", return_value=io.BytesIO(raw)):
                path = data.acquire(directory, self.spec(raw), True)
            with patch.object(data.urllib.request, "urlopen") as request:
                self.assertEqual(data.acquire(directory, self.spec(raw)), path)
                request.assert_not_called()
            self.assertEqual(path.read_bytes(), raw)

    def test_prepare_repeat_round_trip_and_tamper_detection(self):
        with workspace_tempdir() as directory:
            manifest = json.loads(data.SOURCE_MANIFEST.read_text(encoding="utf-8"))
            for key in ("cas_freq", "cas_sev"):
                raw = key.encode()
                spec = self.spec(raw)
                spec.update(filename=f"{key}.rda", object=key)
                manifest["data_files"][key] = spec
                (directory / spec["filename"]).write_bytes(raw)
            manifest_path = directory / "source_manifest.json"
            manifest_path.write_bytes(data.json_bytes(manifest))

            def read_r(path):
                return {"cas_freq": frequency()} if Path(path).stem == "cas_freq" else {"cas_sev": severity()}

            with patch.object(data, "SOURCE_MANIFEST", manifest_path), \
                    patch.object(data.pyreadr, "read_r", side_effect=read_r):
                first, location = data.prepare(directory, directory / "prepared")
                second, repeated_location = data.prepare(directory, directory / "prepared")
                self.assertEqual(first, second)
                self.assertEqual(location, repeated_location)
                policies, claims, saved = data.load_prepared(location)
                self.assertEqual(saved, first)
                self.assertEqual(policies["recorded_positive_count"].sum(), len(claims))
                manifest_output = location / "manifest.json"
                inconsistent = json.loads(data.json_bytes(saved))
                inconsistent["identity"]["source_id"] = "changed-source"
                manifest_output.write_bytes(data.json_bytes(inconsistent))
                with self.assertRaisesRegex(ValueError, "identity/table contract"):
                    data.load_prepared(location)
                inconsistent = json.loads(data.json_bytes(saved))
                inconsistent["analysis_tables"]["policies"]["columns"]["Exposure"] = "int64"
                manifest_output.write_bytes(data.json_bytes(inconsistent))
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    data.load_prepared(location)
                manifest_output.write_bytes(data.json_bytes(saved))
                table_path = location / "claims.parquet"
                table_path.write_bytes(b"corrupt historical output")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    data.load_prepared(location)
                with self.assertRaisesRegex(ValueError, "artifact differs"):
                    data.prepare(directory, directory / "prepared")
                self.assertEqual(table_path.read_bytes(), b"corrupt historical output")

    def test_relative_paths_do_not_follow_current_working_directory(self):
        self.assertEqual(data.root_path("data/raw"), data.PROJECT_ROOT / "data/raw")

    def test_provenance_requires_lf_without_altering_raw_bytes(self):
        with workspace_tempdir() as directory:
            path = directory / "config.json"
            path.write_bytes(b"{}\r\n")
            with self.assertRaisesRegex(ValueError, "UTF-8/LF"):
                data.text_sha256(path)


if __name__ == "__main__":
    unittest.main()
