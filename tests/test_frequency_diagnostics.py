"""Statistical boundary, alignment and resampling risks in frequency diagnostics."""
import unittest

import numpy as np
import pandas as pd

from src.data import DataContractError
from src.frequency_diagnostics import age_standardization, paired_bootstrap, score_predictions
from src.metrics import poisson_count_deviance


def fixture():
    d = pd.DataFrame({"IDpol": np.arange(1, 7), "Exposure": [.1, 1., 2., .5, 1., 1.5],
                      "recorded_positive_count": [1, 0, 0, 1, 0, 2],
                      "DrivAge": [35, 45, 55, 35, 45, 55], "VehAge": [3]*6,
                      "BonusMalus": [50]*6, "VehGas": ["Diesel"]*6,
                      "VehBrand": ["B12"]*6, "Region": ["R24"]*6,
                      "Density": [100]*6})
    parts = []
    for model, rates in [("base", [.2]*6), ("candidate", [.3, .1, .1, .4, .1, .5])]:
        p = d[["IDpol"]].copy()
        p["split"], p["fold"] = "development", [1, 2, 3, 4, 5, 1]
        p["model_id"], p["run_id"] = model, "run1"
        p["annual_rate"] = rates
        p["expected_count"] = p.annual_rate*d.Exposure
        parts.append(p)
    return d, pd.concat(parts, ignore_index=True)


class DiagnosticContracts(unittest.TestCase):
    def test_keyed_reordering_preserves_scores_segments_and_fold_totals(self):
        d, p = fixture()
        actual = score_predictions(d, p, support=1)
        reordered = score_predictions(d.sample(frac=1, random_state=2), p.sample(frac=1, random_state=3), support=1)
        for left, right in zip(actual, reordered):
            pd.testing.assert_frame_equal(left, right)
        summary, folds, segments, quantiles = actual
        for model, group in folds.groupby("model_id"):
            total = summary.set_index("model_id").loc[model]
            self.assertAlmostEqual(group.expected_claims.sum(), total.expected_claims)
            self.assertAlmostEqual(np.average(group.poisson_deviance_per_exposure, weights=group.exposure_years), total.poisson_deviance_per_exposure)
            for _, part in segments.loc[segments.model_id.eq(model)].groupby("dimension"):
                self.assertEqual(part.policies.sum(), len(d))
                self.assertAlmostEqual(part.exposure_years.sum(), d.Exposure.sum())
        self.assertTrue((quantiles.expected_claims > 0).all())

    def test_pearson_diagnostic_uses_expected_period_counts_and_n_denominator(self):
        d, p = fixture()
        summary = score_predictions(d, p)[0].set_index("model_id")
        mu = d.Exposure.to_numpy()*.2
        expected = np.mean((d.recorded_positive_count.to_numpy()-mu)**2/mu)
        self.assertAlmostEqual(summary.loc["base", "pearson_mean_quadratic"], expected)

    def test_missing_duplicate_units_and_evaluation_rows_fail(self):
        d, p = fixture()
        invalid = [p.iloc[1:].copy(), pd.concat([p, p.iloc[:1]])]
        q = p.copy(); q.loc[0, "split"] = "evaluation"; invalid.append(q)
        q = p.copy(); q.loc[0, "expected_count"] *= 2; invalid.append(q)
        q = p.copy(); q.loc[0, "fold"] = 0; invalid.append(q)
        q = p.copy(); q.loc[0, "run_id"] = "another-run"; invalid.append(q)
        for q in invalid:
            with self.subTest(case=len(q)), self.assertRaises(DataContractError):
                score_predictions(d, q)

    def test_different_model_fold_assignment_fails(self):
        d, p = fixture()
        p.loc[p.model_id.eq("candidate") & p.IDpol.eq(1), "fold"] = 2
        with self.assertRaisesRegex(DataContractError, "same policy-fold"):
            paired_bootstrap(d, p, "base", replicates=10)

    def test_bootstrap_matches_shared_index_reference_and_drawn_exposure(self):
        d, p = fixture()
        result = paired_bootstrap(d, p, "base", replicates=37, seed=19).set_index("model_id")
        e, n = d.Exposure.to_numpy(), d.recorded_positive_count.to_numpy()
        left = p.loc[p.model_id.eq("candidate"), "expected_count"].to_numpy()
        right = p.loc[p.model_id.eq("base"), "expected_count"].to_numpy()
        delta = poisson_count_deviance(n, left)-poisson_count_deviance(n, right)
        rng = np.random.Generator(np.random.PCG64(19))
        draws = []
        wrong_denominator = []
        for _ in range(37):
            index = rng.integers(0, len(e), size=len(e))
            draws.append(delta[index].sum()/e[index].sum())
            wrong_denominator.append(delta[index].sum()/e.sum())
        np.testing.assert_allclose(result.loc["candidate", ["lower", "upper"]].to_numpy(dtype=float), np.percentile(draws, [2.5, 97.5]), rtol=1e-14)
        self.assertNotAlmostEqual(result.loc["candidate", "upper"], np.percentile(wrong_denominator, 97.5))
        np.testing.assert_array_equal(result.loc["base", ["deviance_difference", "lower", "upper"]].to_numpy(dtype=float), np.zeros(3))

    def test_bootstrap_order_reproducible_and_invalid_reference_replicates_fail(self):
        d, p = fixture()
        a = paired_bootstrap(d, p, "base", replicates=12)
        b = paired_bootstrap(d.iloc[::-1], p.iloc[::-1], "base", replicates=12)
        pd.testing.assert_frame_equal(a, b)
        for replicas in [0, -1, True, 1.5]:
            with self.assertRaises(DataContractError):
                paired_bootstrap(d, p, "base", replicates=replicas)
        with self.assertRaisesRegex(DataContractError, "absent"):
            paired_bootstrap(d, p, "missing", replicates=2)


class KnownAgeModel:
    def predict(self, x):
        return np.exp(.02*x.DrivAge.to_numpy() + .005*x.VehAge.to_numpy())


class AgeProfileContracts(unittest.TestCase):
    def test_same_exposure_weighted_reference_and_additive_log_relativity(self):
        d, _ = fixture()
        # One unsupported Region appears only at one age: must be excluded.
        extra = d.iloc[[0]].copy(); extra["IDpol"] = 999; extra["Region"] = "R99"; extra["Exposure"] = 20
        d = pd.concat([d, extra], ignore_index=True)
        profile, cells, report = age_standardization(d, {"known": KnownAgeModel()}, ["DrivAge", "VehAge"], support=.1)
        self.assertEqual(report["common_cells"], 1)
        self.assertEqual(report["retained_policies"], 6)
        self.assertEqual(report["excluded_policies"], 1)
        self.assertEqual(report["excluded_exposure_years"], 20.)
        np.testing.assert_allclose(profile.standardized_annual_rate, np.exp(.02*np.array([35, 45, 55])+.005*3))
        np.testing.assert_allclose(profile.relativity_to_first_age, np.exp(.02*np.array([0, 10, 20])))
        self.assertEqual(len(cells.loc[~cells.common_support & cells.exposure_years.eq(0)]), 2)

    def test_support_uses_exposure_and_covariates_without_outcome_selection(self):
        d, _ = fixture()
        first = age_standardization(d, {"known": KnownAgeModel()}, ["DrivAge", "VehAge"], support=.1)
        d["recorded_positive_count"] = np.arange(len(d))*100
        second = age_standardization(d, {"known": KnownAgeModel()}, ["DrivAge", "VehAge"], support=.1)
        pd.testing.assert_frame_equal(first[0], second[0])
        self.assertEqual(first[2]["retained_policies"], second[2]["retained_policies"])

    def test_empty_common_support_returns_documented_empty_profile(self):
        d, _ = fixture()
        result, cells, report = age_standardization(d, {"known": KnownAgeModel()}, ["DrivAge", "VehAge"], support=1000)
        self.assertTrue(result.empty)
        self.assertFalse(cells.common_support.any())
        self.assertEqual(report["excluded_policies"], len(d))
        self.assertEqual(report["retained_exposure_years"], 0.)

    def test_outcome_exposure_predictors_and_outside_reference_ages_fail(self):
        d, _ = fixture()
        for features, ages in [(["DrivAge", "Exposure"], (35, 45, 55)), (["DrivAge", "recorded_positive_count"], (35, 45, 55)), (["DrivAge"], (18, 45, 55)), (["DrivAge"], (35, 35, 55))]:
            with self.subTest(features=features, ages=ages), self.assertRaises(DataContractError):
                age_standardization(d, {"known": KnownAgeModel()}, features, support=.1, ages=ages)


if __name__ == "__main__":
    unittest.main()
