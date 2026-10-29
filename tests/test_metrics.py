"""Numerical references and actuarial failure-risk tests for S2 metrics."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from src import metrics


class PoissonContracts(unittest.TestCase):
    def test_zero_counts_and_hand_calculated_variable_exposure_example(self):
        n = np.array([1, 0])
        e = np.array([.1, 1.])
        rates = np.array([1., 1.])
        expected = np.array([2 * (np.log(10) - 1 + .1), 2.])
        actual = metrics.poisson_count_deviance(n, e * rates)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        self.assertAlmostEqual(metrics.poisson_deviance_per_exposure(n, rates, e), expected.sum() / 1.1)
        self.assertAlmostEqual(metrics.mean_count_deviance(n, e * rates), expected.mean())
        legacy_extra_weight = float((e * expected).sum() / e.sum())
        self.assertNotAlmostEqual(metrics.poisson_deviance_per_exposure(n, rates, e), legacy_extra_weight)

    def test_count_and_weighted_rate_identity_against_sklearn(self):
        rng = np.random.default_rng(729)
        n = rng.poisson(.3, 200)
        e = np.exp(rng.uniform(-5, .7, 200))
        rates = np.exp(rng.normal(-2, .6, 200))
        reference = mean_poisson_deviance(n / e, rates, sample_weight=e)
        actual = metrics.poisson_deviance_per_exposure(n, rates, e)
        self.assertAlmostEqual(actual, reference, places=12)
        self.assertAlmostEqual(metrics.mean_count_deviance(n, e * rates),
                               mean_poisson_deviance(n, e * rates), places=12)

    def test_near_equality_has_small_nonnegative_deviance(self):
        n = np.array([1., 2., 15.])
        mu = n * np.array([1 + 1e-12, 1 - 1e-12, 1.])
        actual = metrics.poisson_count_deviance(n, mu)
        self.assertTrue((actual >= 0).all())
        self.assertTrue((actual[:2] > 0).all())
        self.assertLess(actual.max(), 1e-20)
        self.assertEqual(actual[-1], 0.)

    def test_finite_tiny_prediction_avoids_ratio_overflow(self):
        actual = metrics.poisson_count_deviance([1], [1e-320])
        self.assertTrue(np.isfinite(actual).all())
        self.assertAlmostEqual(actual[0], 2 * (-np.log(1e-320) - 1))

    def test_invalid_counts_expectations_dimensions_and_alignment_fail(self):
        examples = [([], []), ([[1]], [[1.]]), ([1, 0], [1.]),
                    ([-1], [1.]), ([.5], [1.]), ([np.inf], [1.]),
                    ([1], [0.]), ([1], [-1.]), ([1], [np.nan]), ([True], [1.])]
        for n, mu in examples:
            with self.subTest(n=n, mu=mu), self.assertRaises(ValueError):
                metrics.poisson_count_deviance(n, mu)

    def test_invalid_rate_exposure_and_product_fail_without_clipping(self):
        for rates, exposure in (([0.], [1.]), ([1.], [0.]), ([1.], [-1.]),
                                ([np.inf], [1.]), ([1.], [np.nan]),
                                ([1e308], [1e308]), ([1e-320], [1e-320]),
                                ([1., 2.], [1.]), ([1.], [[1.]])):
            with self.subTest(rates=rates, exposure=exposure), self.assertRaises(ValueError):
                metrics.poisson_deviance_per_exposure([0], rates, exposure)

    def test_summary_has_explicit_units_totals_and_AP_direction(self):
        result = metrics.frequency_summary([0, 1, 0, 2], [.1, .1, .2, .4], [.5, 1.5, 1., 2.])
        self.assertEqual(result["policies"], 4)
        self.assertEqual(result["exposure_years"], 5.)
        self.assertEqual(result["observed_claims"], 3)
        self.assertAlmostEqual(result["expected_claims"], 1.2)
        self.assertAlmostEqual(result["observed_annual_rate"], .6)
        self.assertAlmostEqual(result["predicted_annual_rate"], .24)
        self.assertAlmostEqual(result["actual_over_predicted"], 2.5)

    def test_aggregate_overflow_is_rejected_before_group_calibration(self):
        with self.assertRaisesRegex(ValueError, "totals must be finite"):
            metrics.calibration_table([0, 1], [1e308, 1e308], [1., 1.])


class CalibrationContracts(unittest.TestCase):
    def test_known_group_totals_use_annual_rate_and_correct_denominators(self):
        result = metrics.calibration_table([0, 1, 0, 2], [.1, .1, .2, .4], [.5, 1.5, 1., 2.], 2)
        self.assertEqual(result["policies"].tolist(), [2, 2])
        self.assertEqual(result["observed_claims"].tolist(), [1, 2])
        np.testing.assert_allclose(result["exposure_years"], [2., 3.])
        np.testing.assert_allclose(result["expected_claims"], [.2, 1.])
        np.testing.assert_allclose(result["observed_annual_rate"], [.5, 2 / 3])
        np.testing.assert_allclose(result["predicted_annual_rate"], [.1, 1 / 3])
        np.testing.assert_allclose(result["actual_over_predicted"], [5., 2.])

    def test_ties_are_indivisible_even_if_fewer_groups_result(self):
        rates = [.1] * 11 + [.2]
        result = metrics.calibration_table([0] * 11 + [1], rates, [1.] * 12, 10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["policies"].tolist(), [11, 1])
        self.assertEqual(result.attrs["requested_groups"], 10)
        self.assertEqual(result.attrs["actual_groups"], 2)
        constant = metrics.calibration_table([0, 5], [.1, .1], [.5, 5.], 10)
        self.assertEqual(len(constant), 1)
        self.assertEqual(constant.iloc[0]["policies"], 2)

    def test_group_membership_does_not_depend_on_observed_outcomes_or_row_order(self):
        n = np.array([0, 1, 0, 2])
        rates = np.array([.1, .1, .2, .4])
        e = np.array([.5, 1.5, 1., 2.])
        original = metrics.calibration_table(n, rates, e, 2)
        changed = metrics.calibration_table(n[::-1], rates, e, 2)
        keys = ["group", "min_annual_rate", "max_annual_rate", "policies", "exposure_years", "expected_claims"]
        pd.testing.assert_frame_equal(original[keys], changed[keys])
        permutation = [2, 0, 3, 1]
        reordered = metrics.calibration_table(n[permutation], rates[permutation], e[permutation], 2)
        pd.testing.assert_frame_equal(original, reordered)

    def test_model_specific_rate_changes_are_actually_used(self):
        n, e = [1, 0, 2, 0], [1., 1., 1., 1.]
        a = metrics.calibration_table(n, [.1, .2, .3, .4], e, 2)
        b = metrics.calibration_table(n, [.4, .3, .2, .1], e, 2)
        self.assertEqual(a["observed_claims"].tolist(), [1, 2])
        self.assertEqual(b["observed_claims"].tolist(), [2, 1])

    def test_invalid_group_count_fails(self):
        for value in (0, -1, 2.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metrics.calibration_table([0], [.1], [1.], value)


class ConcentrationContracts(unittest.TestCase):
    def test_each_model_uses_its_own_order_and_exposure_coordinates(self):
        n, e = [2, 0, 1], [.1, .6, .3]
        a = metrics.concentration_curve(n, [9., 2., 1.], e)
        b = metrics.concentration_curve(n, [1., 2., 9.], e)
        np.testing.assert_allclose(a["cumulative_exposure_fraction"], [0, .1, .7, 1])
        np.testing.assert_allclose(b["cumulative_exposure_fraction"], [0, .3, .9, 1])
        np.testing.assert_allclose(a["cumulative_claim_fraction"], [0, 2 / 3, 2 / 3, 1])
        np.testing.assert_allclose(b["cumulative_claim_fraction"], [0, 1 / 3, 1 / 3, 1])
        self.assertAlmostEqual(a.attrs["raw_gini"], 11 / 30)
        self.assertAlmostEqual(b.attrs["raw_gini"], -11 / 30)

    def test_constant_scores_have_zero_gini_despite_outcome_and_exposure_order(self):
        result = metrics.concentration_curve([9, 0, 1], [.2, .2, .2], [.1, 5., .9])
        self.assertEqual(result.to_dict("list"), {"cumulative_exposure_fraction": [0., 1.],
                                                 "cumulative_claim_fraction": [0., 1.]})
        self.assertEqual(result.attrs["raw_gini"], 0.)
        self.assertEqual(metrics.frequency_gini([9, 0, 1], [.2, .2, .2], [.1, 5., .9]), 0.)

    def test_exact_tied_blocks_make_curve_permutation_invariant(self):
        n = np.array([0, 4, 1, 1])
        r = np.array([.1, .1, .3, .3])
        e = np.array([.5, 1.5, .5, 1.5])
        a = metrics.concentration_curve(n, r, e)
        order = [3, 0, 2, 1]
        b = metrics.concentration_curve(n[order], r[order], e[order])
        self.assertEqual(len(a), 3)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.attrs["raw_gini"], b.attrs["raw_gini"])

    def test_zero_claim_portfolio_has_no_defined_concentration(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            metrics.concentration_curve([0, 0], [.1, .2], [.5, 1.])
        # Deviance and calibration remain defined for the same zero outcomes.
        self.assertGreater(metrics.poisson_deviance_per_exposure([0, 0], [.1, .2], [.5, 1.]), 0)
        self.assertEqual(metrics.frequency_summary([0, 0], [.1, .2], [.5, 1.])["actual_over_predicted"], 0.)


if __name__ == "__main__":
    unittest.main()
