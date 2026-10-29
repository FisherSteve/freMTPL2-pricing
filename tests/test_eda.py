"""Statistical and evaluation-boundary contracts for development-only EDA."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from statsmodels.stats.rates import confint_poisson

from src import eda
from src.data import DataContractError, json_bytes, sha256, text_sha256
from src.validation import identity_id, propagate_claims, write_immutable


def fixture():
    """Real repeated amounts, short/long exposure, zero policies and evaluation."""
    amounts = [[100.], [], [200., 200., 200., 200., 30000.], [50000.],
               [800.], [], [1000., 1000.], [], [2000.], [3000.], [], [], [700.]]
    count = [len(a) for a in amounts]
    policies = pd.DataFrame({
        "IDpol": np.arange(1, 14),
        "Exposure": [.001, .25, 2., 1., .8, .5, .4, 1., .7, 1., 1., 1., .5],
        "recorded_positive_count": count, "ClaimNb": count,
        "recorded_loss_eur": [sum(a) for a in amounts],
        "DrivAge": [35, 36, 40, 41, 45, 46, 50, 51, 100, 52, 42, 43, 60],
        "VehAge": [0, 0, 1, 5, 2, 4, 10, 12, 100, 3, 1, 2, 0],
        "VehBrand": ["B12", "B12", "B12", "B1", "B1", "B1", "B1", "B1", "B1", "B1", "B1", "B1", "B1"],
        "VehGas": ["Regular"] * 13, "Area": ["A"] * 13,
        "Region": ["Alsace"] * 13, "Density": [100] * 13,
        "BonusMalus": [50, 60, 80, 100, 50, 60, 80, 100, 50, 60, 80, 100, 50],
    })
    claims = pd.DataFrame([{"claim_row_id": f"row-{i:02d}-{j}", "IDpol": i,
                            "ClaimAmount": amount}
                           for i, values in enumerate(amounts, 1)
                           for j, amount in enumerate(values)])
    membership = pd.DataFrame({"IDpol": policies.IDpol,
                               "split": ["development"] * 12 + ["evaluation"],
                               "fold": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 0]})
    frames = {"policy_membership": membership,
              "claim_membership": propagate_claims(claims, membership)}
    return policies, claims, frames


def dev_fixture():
    return eda.development_only(*fixture())


class BoundaryAndLinkage(unittest.TestCase):
    def test_poison_evaluation_outcomes_cannot_change_development(self):
        p, c, frames = fixture()
        expected_p, expected_c = eda.development_only(p, c, frames)
        poison_p, poison_c = p.copy(), c.copy()
        for column in [eda.E, eda.N, eda.L, "ClaimNb", "DrivAge", "Density"]:
            poison_p[column] = poison_p[column].astype(float)
            poison_p.loc[poison_p.IDpol.eq(13), column] = np.nan
        poison_c.loc[poison_c.IDpol.eq(13), "ClaimAmount"] = -np.inf
        actual_p, actual_c = eda.development_only(poison_p.iloc[::-1], poison_c.iloc[::-1], frames)
        pd.testing.assert_frame_equal(actual_p, expected_p, check_dtype=False)
        pd.testing.assert_frame_equal(actual_c, expected_c)
        self.assertNotIn(13, actual_p.IDpol.tolist())
        self.assertNotIn(13, actual_c.IDpol.tolist())
        self.assertEqual(eda.summary(actual_p), eda.summary(expected_p))

    def test_wrong_policy_universe_or_claim_lineage_fails(self):
        p, c, frames = fixture()
        for bad_p in [p.iloc[1:], pd.concat([p, p.iloc[[0]]])]:
            with self.assertRaisesRegex(DataContractError, "universe"):
                eda.development_only(bad_p, c, frames)
        bad_c = c.copy()
        bad_c.loc[0, "IDpol"] = 2
        with self.assertRaisesRegex(DataContractError, "lineage"):
            eda.development_only(p, bad_c, frames)
        with self.assertRaisesRegex(DataContractError, "lineage"):
            eda.development_only(p, c.iloc[1:], frames)

    def test_zero_policies_and_repeated_claim_rows_are_preserved(self):
        d, c = dev_fixture()
        self.assertEqual(len(d), 12)
        self.assertEqual(len(c), 12)
        self.assertEqual(c.duplicated(["IDpol", "ClaimAmount"]).sum(), 4)
        self.assertEqual(d[eda.N].sum(), len(c))
        self.assertEqual(d[eda.L].sum(), c.ClaimAmount.sum())
        self.assertEqual(d.loc[d.IDpol.eq(2), eda.N].item(), 0)
        self.assertEqual(d.loc[d.IDpol.eq(2), eda.L].item(), 0)
        eda.check_development(d.iloc[::-1], c.iloc[::-1])

    def test_corrupt_development_targets_fail_without_repair(self):
        d, c = dev_fixture()
        for column, value in [(eda.E, 0), (eda.N, .5), (eda.L, np.nan), ("ClaimNb", 99)]:
            bad = d.copy()
            bad[column] = bad[column].astype(float)
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(DataContractError):
                eda.check_development(bad, c)
        bad = d.copy()
        bad.loc[0, eda.L] += 1
        with self.assertRaisesRegex(DataContractError, "conservation"):
            eda.check_development(bad, c)
        bad_c = c.copy()
        bad_c.loc[0, "ClaimAmount"] = 0
        with self.assertRaises(DataContractError):
            eda.check_development(d, bad_c)


class RatesAndUncertainty(unittest.TestCase):
    def test_ratio_of_sums_and_observed_decomposition(self):
        d = pd.DataFrame({"IDpol": [1, 2, 3], eda.E: [.1, 1., .5], eda.N: [1, 0, 2],
                          eda.L: [100., 0., 600.], "group": ["a", "a", "b"]})
        total = eda.summary(d)
        self.assertAlmostEqual(total["frequency"], 3/1.6)
        self.assertNotAlmostEqual(total["frequency"], np.mean(d[eda.N]/d[eda.E]))
        grouped = eda.grouped(d, ["group"], support=1)
        a = grouped.set_index("group").loc["a"]
        self.assertAlmostEqual(a.frequency, 1/1.1)
        self.assertAlmostEqual(a.severity_eur, 100)
        np.testing.assert_allclose(grouped.frequency*grouped.severity_eur,
                                   grouped.loss_cost_eur_per_year)
        self.assertAlmostEqual(total["frequency"]*total["severity_eur"], total["loss_cost_eur_per_year"])
        self.assertTrue(a.supported)
        self.assertFalse(grouped.set_index("group").loc["b", "supported"])

    def test_poisson_intervals_match_independent_reference_including_zero(self):
        d = pd.DataFrame({"IDpol": [1, 2, 3], eda.E: [.1, 2., .5], eda.N: [0, 1, 10],
                          eda.L: [0., 10., 500.], "group": ["a", "b", "c"]})
        result = eda.grouped(d, ["group"])
        lower, upper = confint_poisson(result.claims.to_numpy(), result.exposure.to_numpy(), method="exact-c")
        np.testing.assert_allclose(result.poisson_lower, lower, rtol=1e-13, atol=0)
        np.testing.assert_allclose(result.poisson_upper, upper, rtol=1e-13, atol=0)
        self.assertEqual(result.loc[0, "poisson_lower"], 0)
        self.assertGreater(result.loc[0, "poisson_upper"], 0)
        self.assertTrue(np.isnan(result.loc[0, "severity_eur"]))

    def test_bootstrap_draws_whole_policy_tuples_and_keeps_zero_claim_draws(self):
        class PrescribedDraws:
            def __init__(self):
                self.draws = iter([[0, 0], [1, 1], [0, 1]])

            def integers(self, low, high, size):
                assert (low, high, size) == (0, 2, 2)
                return np.asarray(next(self.draws))

        d = pd.DataFrame({eda.E: [.1, 1.], eda.N: [2, 0], eda.L: [300., 0.]})
        actual = eda.bootstrap_ratios(d, 3, PrescribedDraws())
        expected = np.asarray([[20, 150, 3000], [0, np.nan, 0], [2/1.1, 150, 300/1.1]])
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        valid = np.isfinite(actual[:, 1])
        np.testing.assert_allclose(actual[valid, 0]*actual[valid, 1], actual[valid, 2])

    def test_bootstrap_seed_repeat_and_undefined_severity(self):
        d, _ = dev_fixture()
        a = eda.bootstrap_ratios(d, 50, np.random.Generator(np.random.PCG64(123)))
        b = eda.bootstrap_ratios(d, 50, np.random.Generator(np.random.PCG64(123)))
        np.testing.assert_array_equal(a, b)
        zeros = d.loc[d[eda.N].eq(0)]
        z = eda.bootstrap_ratios(zeros, 20, np.random.default_rng(123))
        np.testing.assert_array_equal(z[:, [0, 2]], np.zeros((20, 2)))
        self.assertTrue(np.isnan(z[:, 1]).all())
        with self.assertRaises(DataContractError):
            eda.bootstrap_ratios(d.iloc[:0], 10, np.random.default_rng(123))

    def test_zero_claim_age_contrast_has_no_fabricated_severity(self):
        d = pd.DataFrame({"IDpol": [1, 2], eda.E: [1., 1.], eda.N: [0, 1],
                          eda.L: [0., 100.], "age_band": ["35–39", "40–44"]})
        cfg = copy.deepcopy(eda.configuration())
        cfg["bootstrap"]["replicates"] = 10
        intervals, contrasts = eda.bootstrap_table(d, cfg)
        zero = intervals.set_index("age_band").loc["35–39"]
        self.assertEqual(zero.undefined_severity_draws, 10)
        self.assertTrue(pd.isna(zero.severity_eur_lower))
        severity = contrasts.loc[contrasts.metric.eq("severity_eur")].iloc[0]
        self.assertEqual(severity.finite_draws, 0)
        self.assertTrue(pd.isna(severity["difference"]))
        self.assertTrue(pd.isna(severity.lower))


class StandardizationAndSensitivity(unittest.TestCase):
    def test_common_support_fixed_exposure_weights_and_excluded_totals(self):
        d = pd.DataFrame({
            "IDpol": np.arange(1, 9),
            "age10_band": ["30–39", "30–39", "40–49", "40–49", "50–59", "50–59", "30–39", "40–49"],
            "bm_band": ["50", "51–75", "50", "51–75", "50", "51–75", "101+", "101+"],
            eda.E: [1., 3., 2., 1., 1., 2., 3., 4.], eda.N: [1, 6, 0, 4, 4, 2, 30, 40],
            eda.L: [100., 600., 0., 400., 400., 200., 3000., 4000.],
        })
        cfg = copy.deepcopy(eda.configuration())
        cfg["support_policy_years"] = .5
        cfg["bootstrap"]["replicates"] = 20
        cells, result, contrasts = eda.standardize_bm(d, cfg)
        common = cells.loc[cells.common_support]
        self.assertEqual(set(common.bm_band), {"50", "51–75"})
        np.testing.assert_allclose(common.groupby("bm_band").reference_weight.first().sort_index(), [.4, .6])
        np.testing.assert_allclose(result.standardized_frequency, [1.6, 2.4, 2.2])
        np.testing.assert_allclose(result.excluded_exposure, [3., 4., 0.])
        np.testing.assert_array_equal(result.excluded_claims, [30, 40, 0])
        np.testing.assert_allclose(result.excluded_loss_eur, [3000., 4000., 0.])
        self.assertAlmostEqual(contrasts.iloc[0]["difference"], .8)
        cells2, result2, contrasts2 = eda.standardize_bm(d.iloc[::-1], cfg)
        pd.testing.assert_frame_equal(cells, cells2)
        # Single-policy strata make bootstrap invariant to row order as well.
        pd.testing.assert_frame_equal(result, result2)
        pd.testing.assert_frame_equal(contrasts, contrasts2)

    def test_no_common_support_fails_instead_of_extrapolating(self):
        d = pd.DataFrame({"IDpol": [1, 2], "age10_band": ["30–39", "40–49"],
                          "bm_band": ["50", "51–75"], eda.E: [1000., 1000.],
                          eda.N: [1, 2], eda.L: [100., 200.]})
        with self.assertRaisesRegex(DataContractError, "common supported"):
            eda.standardize_bm(d, eda.configuration())

    def test_caps_and_exclusions_have_separate_actuarial_impacts(self):
        d, c = dev_fixture()
        original_d, original_c = d.copy(deep=True), c.copy(deep=True)
        result = eda.sensitivity(d, c).set_index("scenario")
        self.assertEqual(result.loc["ungekappt", "removed_policies"], 0)
        self.assertAlmostEqual(result.loc["Exposure bei 1 gekappt", "removed_exposure"], 1.)
        self.assertEqual(result.loc["Exposure bei 1 gekappt", "removed_claims"], 0)
        self.assertEqual(result.loc["Exposure bei 1 gekappt", "removed_loss_eur"], 0)
        capped_count = result.loc["Anzahl bei 4 gekappt"]
        self.assertEqual(capped_count.removed_claims, 1)
        self.assertEqual(capped_count.removed_loss_eur, 0)
        self.assertTrue(pd.isna(capped_count.severity_eur))
        loss_cap = result.loc["Schadenzeilenbetrag bei EUR 25000 gekappt"]
        self.assertEqual(loss_cap.removed_loss_eur, 30000.)
        self.assertEqual(loss_cap.removed_claims, 0)
        b12_all = result.loc["ohne B12"]
        b12_new = result.loc["ohne B12 und VehAge=0"]
        self.assertEqual(b12_all.removed_policies, 3)
        self.assertEqual(b12_all.removed_claims, 6)
        self.assertEqual(b12_new.removed_policies, 2)
        self.assertEqual(b12_new.removed_claims, 1)
        self.assertEqual(result.loc["ohne Exposure < 1/365", "removed_claims"], 1)
        self.assertEqual(result.loc["ohne Exposure > 1", "removed_claims"], 5)
        pd.testing.assert_frame_equal(d, original_d)
        pd.testing.assert_frame_equal(c, original_c)

    def test_explicit_bin_boundaries_and_uncovered_values(self):
        cfg = eda.configuration()
        result = eda.bins(pd.Series([18, 39, 40, 44, 45, 100]), cfg["age_edges"], cfg["age_labels"])
        self.assertEqual(result.tolist(), ["18–24", "35–39", "40–44", "40–44", "45–49", "75+"])
        with self.assertRaisesRegex(DataContractError, "outside"):
            eda.bins(pd.Series([17]), cfg["age_edges"], cfg["age_labels"])

    def test_all_descriptive_tables_conserve_support_and_amount_caps(self):
        d, c = dev_fixture()
        d["BonusMalus"] = 50  # Common support exists in all three age decades.
        cfg = copy.deepcopy(eda.configuration())
        cfg["support_policy_years"] = .001
        cfg["bootstrap"]["replicates"] = 5
        tables, report = eda.analyze(d, c, cfg)
        self.assertEqual(report["evaluation_rows_analyzed"], 0)
        self.assertEqual(report["models_fitted"], 0)
        self.assertEqual(report["exact_repeated_policy_amount_rows_retained"], 4)
        for name in ["age_band", "age10_band", "DrivAge", "vehicle_band", "bm_band", "VehGas",
                     "VehBrand", "Area", "Region", "density_band", "exposure_band", "b12_new"]:
            with self.subTest(table=name):
                t = tables[name]
                self.assertEqual(t.policies.sum(), len(d))
                self.assertEqual(t.claims.sum(), len(c))
                self.assertAlmostEqual(t.exposure.sum(), d[eda.E].sum())
                self.assertAlmostEqual(t.loss_eur.sum(), c.ClaimAmount.sum())
        shares = tables["age_composition"].groupby(["age_band", "feature"]).exposure_share_within_age.sum()
        np.testing.assert_allclose(shares, 1)
        cap = tables["age_loss_sensitivity"].groupby("amount_target").agg(
            claims=("claims", "sum"), exposure=("exposure", "sum"), loss=("loss_eur", "sum"))
        self.assertEqual(cap.claims.nunique(), 1)
        self.assertEqual(cap.exposure.nunique(), 1)
        self.assertAlmostEqual(cap.loc["ungekappt", "loss"]-cap.loc["Schadenzeile bei EUR 25000 gekappt", "loss"], 30000.)

    def test_empty_summary_rejected_as_undefined_population(self):
        d, _ = dev_fixture()
        with self.assertRaisesRegex(DataContractError, "nonempty"):
            eda.summary(d.iloc[:0])


class PublishedEvidence(unittest.TestCase):
    def test_report_rejects_changed_mirror_even_with_updated_hash_and_stale_code_config(self):
        base = eda.PROJECT_ROOT / "data/test-tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root = Path(temp)
            reports, output = root / "reports", root / "tables"
            reports.mkdir()
            identity = {"eda_code_sha256": text_sha256(Path(eda.__file__)),
                        "eda_config_sha256": text_sha256(eda.CONFIG),
                        "source_id": "small-artificial-fixture"}
            run_id = identity_id("s3eda-", identity)
            table = pd.DataFrame({"exposure": [.1, 1.], "claims": [1, 0],
                                  "loss_eur": [100., 0.]})
            saved = write_immutable(output / run_id, {"support": table},
                                    {"run_id": run_id, "identity": identity})
            mirror = reports / "support.json"
            expected_mirror = {"run_id": run_id, "rows": json.loads(table.to_json(orient="records", double_precision=15))}
            mirror.write_bytes(json_bytes(expected_mirror))
            figure = reports / "figures" / "fixture.png"
            figure.parent.mkdir()
            figure.write_bytes(b"image-content-checksum-fixture")
            saved["public_tables"] = {mirror.name: {"sha256": sha256(mirror)}}
            saved["figures"] = {figure.name: {"sha256": sha256(figure)}}
            report_path = reports / "eda_report.json"
            report_path.write_bytes(json_bytes(saved))
            loaded, actual_report = eda.load_report(reports, output)
            pd.testing.assert_frame_equal(loaded["support"], table)
            self.assertEqual(actual_report, saved)

            changed_mirror = copy.deepcopy(expected_mirror)
            changed_mirror["rows"][0]["claims"] = 2
            mirror.write_bytes(json_bytes(changed_mirror))
            # Updating a checksum must not let changed statistical content pass.
            forged = copy.deepcopy(saved)
            forged["public_tables"][mirror.name]["sha256"] = sha256(mirror)
            report_path.write_bytes(json_bytes(forged))
            with self.assertRaisesRegex(DataContractError, "public table mismatch"):
                eda.load_report(reports, output)
            mirror.write_bytes(json_bytes(expected_mirror))

            for field in ["eda_code_sha256", "eda_config_sha256"]:
                stale = copy.deepcopy(saved)
                stale["identity"][field] = "0" * 64
                # Internally consistent ID, but not the currently running source.
                stale["run_id"] = identity_id("s3eda-", stale["identity"])
                report_path.write_bytes(json_bytes(stale))
                with self.subTest(identity=field), self.assertRaisesRegex(DataContractError, "current code/config"):
                    eda.load_report(reports, output)

            report_path.write_bytes(json_bytes(saved))
            figure.write_bytes(b"changed-image-content")
            with self.assertRaisesRegex(DataContractError, "figure identity"):
                eda.load_report(reports, output)


if __name__ == "__main__":
    unittest.main()
