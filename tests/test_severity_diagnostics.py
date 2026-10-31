"""Claim/population boundaries, loss units and policy bootstrap risks for S5."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_tweedie_deviance

from src.data import DataContractError
from src.severity_diagnostics import (compose_pure_premium, gamma_deviance_rows,
    paired_policy_bootstrap, score_pure_premium, score_severity,
    tweedie_deviance_rows)


def fixture():
    membership = pd.DataFrame({"IDpol": np.arange(1, 9),
        "split": ["development"]*6+["evaluation"]*2,
        "fold": [1, 2, 3, 4, 5, 1, 0, 0]})
    policies = pd.DataFrame({"IDpol": np.arange(1, 7),
        "Exposure": [.1, 1., 2., .5, 1., 1.5],
        "recorded_positive_count": [2, 0, 1, 1, 0, 1],
        "recorded_loss_eur": [1000., 0., 200., 100., 0., 5000.],
        "DrivAge": [35, 45, 55, 35, 45, 55], "VehAge": [3]*6,
        "BonusMalus": [50, 50, 75, 100, 50, 101],
        "VehGas": ["Diesel", "Regular"]*3,
        "VehBrand": ["B12"]*6, "Region": ["R24"]*4+["R99", "R24"],
        "Density": [100]*6})
    claims = pd.DataFrame({"IDpol": [1, 1, 3, 4, 6],
        "claim_row_id": ["c1", "c2", "c3", "c4", "c5"],
        "ClaimAmount": [100., 900., 200., 100., 5000.]})
    frequencies, severities = [], []
    for model, rates in [("intercept", [.2]*6), ("hgb_l15", [.3, .1, .2, .5, .1, .4])]:
        part = membership.iloc[:6].copy()
        part["model_id"], part["run_id"] = model, "freq-run"
        part["annual_rate"] = rates
        part["expected_count"] = part.annual_rate*policies.Exposure
        frequencies.append(part)
    for model, amounts in [("mean", [1200.]*6), ("gamma", [1000., 2000., 1000., 2000., 3000., 4000.])]:
        part = membership.iloc[:6].copy()
        part["model_id"], part["run_id"] = model, "severity-run"
        part["severity_eur"] = amounts
        severities.append(part)
    pairs = [dict(model_id="pure_intercept", frequency_model_id="intercept", severity_model_id="mean"),
             dict(model_id="hgb_mean", frequency_model_id="hgb_l15", severity_model_id="mean"),
             dict(model_id="hgb_gamma", frequency_model_id="hgb_l15", severity_model_id="gamma")]
    return policies, claims, membership, pd.concat(frequencies, ignore_index=True), pd.concat(severities, ignore_index=True), pairs


def premium_fixture():
    d, c, m, f, s, pairs = fixture()
    predictions = compose_pure_premium(d, m, f, s, pairs, "combined-run")
    return d, c, m, predictions


class LossMetricContracts(unittest.TestCase):
    def test_gamma_rows_match_sklearn_and_are_nonnegative_near_equality(self):
        y = np.array([10., 100., 1000., 40000.])
        mu = np.array([40., 80., 3000., 100.])
        self.assertAlmostEqual(gamma_deviance_rows(y, mu).mean(), mean_gamma_deviance(y, mu), places=12)
        near = gamma_deviance_rows([1., 3., 1000.], np.array([1., 3., 1000.])*np.array([1+1e-12, 1-1e-12, 1.]))
        self.assertTrue((near >= 0).all())
        self.assertTrue((near[:2] > 0).all())
        self.assertEqual(near[2], 0.)

    def test_tweedie_zero_losses_reference_and_exposure_homogeneity(self):
        loss = np.array([0., 100., 7000., 0.])
        e = np.array([.01, .1, .5, 2.])
        premium = np.array([80., 200., 1000., 50.])
        annual = tweedie_deviance_rows(loss/e, premium)
        period = tweedie_deviance_rows(loss, e*premium)
        self.assertAlmostEqual(np.average(annual, weights=e), mean_tweedie_deviance(loss/e, premium, sample_weight=e, power=1.5), places=11)
        np.testing.assert_allclose(e*annual, e**.5*period, rtol=1e-14)
        self.assertNotAlmostEqual(np.average(annual, weights=e), period.mean())
        self.assertEqual(tweedie_deviance_rows([0.], [100.])[0], 40.)

    def test_invalid_amounts_predictions_broadcasting_and_unregistered_power_fail(self):
        for function, y, mu in [(gamma_deviance_rows, [0], [1]), (gamma_deviance_rows, [1], [0]),
                                (gamma_deviance_rows, [1, 2], [1]), (tweedie_deviance_rows, [-1], [1]),
                                (tweedie_deviance_rows, [1], [np.nan]), (tweedie_deviance_rows, [1], [0]),
                                (tweedie_deviance_rows, [True], [1])]:
            with self.subTest(function=function.__name__, y=y, mu=mu), self.assertRaises(DataContractError):
                function(y, mu)
        with self.assertRaisesRegex(DataContractError, "power at 1.5"):
            tweedie_deviance_rows([0], [1], power=1.9)


class PredictionAndClaimContracts(unittest.TestCase):
    def test_composition_is_keyed_outcome_free_and_retains_zero_claim_policies(self):
        d, _, m, f, s, pairs = fixture()
        p = compose_pure_premium(d, m, f, s, pairs, "combined-run")
        q = compose_pure_premium(d.drop(columns=["recorded_positive_count", "recorded_loss_eur"]).iloc[::-1], m,
                                 f.sample(frac=1, random_state=4), s.sample(frac=1, random_state=3), pairs, "combined-run")
        pd.testing.assert_frame_equal(p, q)
        self.assertEqual(len(p), 3*len(d))
        self.assertTrue(p.groupby("model_id").IDpol.nunique().eq(len(d)).all())
        self.assertTrue(p.loc[p.IDpol.isin([2, 5]), "severity_eur"].gt(0).all())
        self.assertTrue(p.run_id.eq("combined-run").all())
        self.assertTrue(p.frequency_run_id.eq("freq-run").all())
        joined = p.merge(d[["IDpol", "Exposure"]], on="IDpol", validate="many_to_one")
        np.testing.assert_allclose(joined.annual_pure_premium, joined.annual_rate*joined.severity_eur)
        np.testing.assert_allclose(joined.expected_period_loss, joined.Exposure*joined.annual_pure_premium)

    def test_claim_scores_have_unit_weight_per_row_not_policy_or_exposure_weights(self):
        d, c, m, _, s, _ = fixture()
        summary, segments, folds = score_severity(d, c, s, m, support_claims=2)
        row = summary.set_index("model_id").loc["mean"]
        expected = np.full(len(c), 1200.)
        self.assertEqual(row.claims, 5)
        self.assertEqual(row.policies_predicted, 6)
        self.assertEqual(row.observed_claim_mean_eur, 1260.)
        self.assertEqual(row.predicted_total_for_observed_claims_eur, 6000.)
        self.assertAlmostEqual(row.mean_gamma_deviance, mean_gamma_deviance(c.ClaimAmount, expected))
        self.assertAlmostEqual(row.claim_mae_eur, np.abs(c.ClaimAmount-expected).mean())
        self.assertAlmostEqual(row.claim_rmse_eur, np.sqrt(np.mean((c.ClaimAmount-expected)**2)))
        exposure = c.merge(d[["IDpol", "Exposure"]], on="IDpol", validate="many_to_one").Exposure
        self.assertNotAlmostEqual(row.mean_gamma_deviance, mean_gamma_deviance(c.ClaimAmount, expected, sample_weight=exposure))
        zero_group = segments.loc[segments.model_id.eq("mean") & segments.dimension.eq("Region") & segments.level.eq("R99")].iloc[0]
        self.assertEqual(zero_group.claims, 0)
        self.assertFalse(zero_group.supported_claims)
        self.assertTrue(pd.isna(zero_group.mean_gamma_deviance))
        self.assertEqual(folds.loc[folds.model_id.eq("mean"), "claims"].sum(), 5)

    def test_repeated_policy_and_equal_amount_rows_are_preserved_but_lineage_duplicates_fail(self):
        d, c, m, _, s, _ = fixture()
        c.loc[c.claim_row_id.eq("c2"), "ClaimAmount"] = 100.
        d.loc[d.IDpol.eq(1), "recorded_loss_eur"] = 200.
        result = score_severity(d, c, s, m)[0]
        self.assertTrue(result.claims.eq(5).all())
        invalid = pd.concat([c, c.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(DataContractError, "Claim rows must be unique"):
            score_severity(d, invalid, s, m)

    def test_wrong_split_claims_and_mismatched_count_loss_reconciliation_fail(self):
        d, c, m, _, s, _ = fixture()
        foreign = c.copy(); foreign.loc[0, "IDpol"] = 7; foreign.loc[0, "ClaimAmount"] = np.nan
        with self.assertRaisesRegex(DataContractError, "within the declared policy split"):
            score_severity(d, foreign, s, m)
        wrong = c.copy(); wrong.loc[0, "ClaimAmount"] += 1.
        with self.assertRaisesRegex(DataContractError, "do not reconcile"):
            score_severity(d, wrong, s, m)

    def test_claim_fold_metadata_is_verified_before_joining_predictions(self):
        d, c, m, _, s, _ = fixture()
        labeled = c.merge(m, on="IDpol", validate="many_to_one")
        score_severity(d, labeled.iloc[::-1], s, m)
        labeled.loc[0, "fold"] = 5
        with self.assertRaisesRegex(DataContractError, "Claim split/fold"):
            score_severity(d, labeled, s, m)

    def test_no_observed_claims_keeps_predictions_but_claim_scores_are_undefined(self):
        d, c, m, _, s, _ = fixture()
        d["recorded_positive_count"], d["recorded_loss_eur"] = 0, 0.
        summary, segments, folds = score_severity(d, c.iloc[:0], s, m)
        self.assertTrue(summary.policies_predicted.eq(len(d)).all())
        self.assertTrue(summary.claims.eq(0).all())
        self.assertTrue(summary.mean_gamma_deviance.isna().all())
        self.assertTrue(folds.mean_gamma_deviance.isna().all())
        self.assertFalse(segments.supported_claims.any())

    def test_key_fold_run_pair_and_period_unit_failures_are_rejected(self):
        d, c, m, p = premium_fixture()
        invalid = [p.iloc[1:].copy(), pd.concat([p, p.iloc[:1]], ignore_index=True)]
        q = p.copy(); q.loc[0, "fold"] = 2; invalid.append(q)
        q = p.copy(); q.loc[0, "split"] = "evaluation"; invalid.append(q)
        q = p.copy(); q.loc[0, "run_id"] = "another-run"; invalid.append(q)
        q = p.copy(); q.loc[0, "severity_model_id"] = "different-model"; invalid.append(q)
        q = p.copy(); q.loc[0, "expected_period_loss"] *= 10; invalid.append(q)
        q = p.copy(); q.loc[0, "annual_pure_premium"] *= 2; invalid.append(q)
        for q in invalid:
            with self.subTest(case=len(q)), self.assertRaises(DataContractError):
                score_pure_premium(d, c, q, m)

    def test_evaluation_is_an_explicit_separate_key_contract(self):
        d, c, m, f, s, pairs = fixture()
        d = d.iloc[:2].copy(); d["IDpol"] = [7, 8]
        d["recorded_positive_count"], d["recorded_loss_eur"] = [1, 0], [100., 0.]
        claims = c.iloc[:1].copy(); claims["IDpol"] = 7
        f = f.groupby("model_id", sort=False).head(2).copy()
        s = s.groupby("model_id", sort=False).head(2).copy()
        for frame in [f, s]:
            frame["IDpol"] = [7, 8]*2
            frame["split"], frame["fold"] = "evaluation", 0
        with self.assertRaisesRegex(DataContractError, "exactly the declared frozen"):
            compose_pure_premium(d, m, f, s, pairs, "eval-run")
        p = compose_pure_premium(d, m, f, s, pairs, "eval-run", split="evaluation")
        scored = score_pure_premium(d, claims, p, m, split="evaluation")[0]
        self.assertTrue(scored.policies.eq(2).all())
        self.assertTrue(scored.zero_loss_policies.eq(1).all())
        self.assertTrue(p.fold.eq(0).all())


class PremiumAndBootstrapContracts(unittest.TestCase):
    def test_all_policy_score_matches_reference_and_correct_product_aggregation(self):
        d, c, m, p = premium_fixture()
        summary, segments, folds = score_pure_premium(d, c, p, m, support_exposure=1.)
        row = summary.set_index("model_id").loc["hgb_gamma"]
        aligned = d.merge(p.loc[p.model_id.eq("hgb_gamma")], on="IDpol", validate="one_to_one")
        reference = mean_tweedie_deviance(aligned.recorded_loss_eur/aligned.Exposure, aligned.annual_pure_premium, sample_weight=aligned.Exposure, power=1.5)
        self.assertEqual(row.policies, len(d))
        self.assertEqual(row.zero_loss_policies, 2)
        self.assertEqual(row.observed_loss_eur, 6300.)
        self.assertAlmostEqual(row.tweedie_deviance_per_exposure, reference)
        self.assertAlmostEqual(row.predicted_annual_loss_cost, np.average(aligned.annual_rate*aligned.severity_eur, weights=aligned.Exposure))
        self.assertNotAlmostEqual(row.predicted_annual_loss_cost, row.product_of_exposure_weighted_component_means)
        self.assertAlmostEqual(row.predicted_annual_loss_cost, row.exposure_weighted_mean_annual_frequency*row.predicted_claim_weighted_mean_severity_eur)
        model_folds = folds.loc[folds.model_id.eq("hgb_gamma")]
        self.assertAlmostEqual(np.average(model_folds.tweedie_deviance_per_exposure, weights=model_folds.exposure_years), row.tweedie_deviance_per_exposure)
        for _, group in segments.loc[segments.model_id.eq("hgb_gamma")].groupby("dimension"):
            self.assertEqual(group.policies.sum(), len(d))
            self.assertAlmostEqual(group.observed_loss_eur.sum(), 6300.)

    def test_bootstrap_matches_paired_policy_reference_including_zero_policies(self):
        d, _, m, p = premium_fixture()
        result = paired_policy_bootstrap(d, p, m, "hgb_mean", replicates=41, seed=23).set_index("model_id")
        models = sorted(p.model_id.unique())
        means = np.column_stack([p.loc[p.model_id.eq(name)].sort_values("IDpol").annual_pure_premium for name in models])
        e, loss = d.Exposure.to_numpy(), d.recorded_loss_eur.to_numpy()
        rng = np.random.Generator(np.random.PCG64(23))
        ref = models.index("hgb_mean")
        scores, calibrations = [], []
        for _ in range(41):
            ix = rng.integers(0, len(d), size=len(d))
            score = np.array([mean_tweedie_deviance(loss[ix]/e[ix], means[ix, j], sample_weight=e[ix], power=1.5) for j in range(len(models))])
            scores.append(score-score[ref])
            calibrations.append(loss[ix].sum()/(e[ix, None]*means[ix]).sum(axis=0))
        scores, calibrations = np.asarray(scores), np.asarray(calibrations)
        for j, name in enumerate(models):
            bounds = np.percentile(scores[:, j], [2.5, 97.5])
            np.testing.assert_allclose(result.loc[name, ["lower", "upper"]].to_numpy(dtype=float), bounds, rtol=1e-13, atol=1e-12)
            np.testing.assert_allclose(result.loc[name, ["actual_over_predicted_lower", "actual_over_predicted_upper"]].to_numpy(dtype=float), np.percentile(calibrations[:, j], [2.5, 97.5]), rtol=1e-13)
            delta = calibrations[:, j]-calibrations[:, ref]
            np.testing.assert_allclose(result.loc[name, ["actual_over_predicted_difference_lower", "actual_over_predicted_difference_upper"]].to_numpy(dtype=float), np.percentile(delta, [2.5, 97.5]), rtol=1e-13, atol=1e-13)
        self.assertEqual(result.loc["hgb_mean", "deviance_difference"], 0.)
        self.assertEqual(result.loc["hgb_mean", "actual_over_predicted_difference_upper"], 0.)

    def test_bootstrap_is_key_order_invariant_and_zero_loss_population_remains_valid(self):
        d, c, m, p = premium_fixture()
        a = paired_policy_bootstrap(d, p, m, "hgb_mean", replicates=17)
        b = paired_policy_bootstrap(d.iloc[::-1], p.iloc[::-1], m, "hgb_mean", replicates=17)
        pd.testing.assert_frame_equal(a, b)
        d["recorded_positive_count"], d["recorded_loss_eur"] = 0, 0.
        empty_claims = c.iloc[:0]
        scores = score_pure_premium(d, empty_claims, p, m)[0]
        self.assertTrue(scores.zero_loss_policies.eq(len(d)).all())
        boot = paired_policy_bootstrap(d, p, m, "hgb_mean", replicates=17)
        for column in ["actual_over_predicted", "actual_over_predicted_lower", "actual_over_predicted_upper"]:
            self.assertTrue(boot[column].eq(0).all())

    def test_bootstrap_absent_reference_and_invalid_replicates_fail(self):
        d, _, m, p = premium_fixture()
        with self.assertRaisesRegex(DataContractError, "absent"):
            paired_policy_bootstrap(d, p, m, "missing", replicates=2)
        for value in [0, -1, True, 1.2]:
            with self.subTest(value=value), self.assertRaises(DataContractError):
                paired_policy_bootstrap(d, p, m, "hgb_mean", replicates=value)


if __name__ == "__main__":
    unittest.main()
