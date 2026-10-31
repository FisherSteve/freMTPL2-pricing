"""Severity arithmetic-mean, likelihood, retransformation and leakage contracts."""
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone

from src.data import DataContractError
from src.severity_models import (FEATURES, ClaimMeanSeverity, GammaSeverity,
    LogSmearingSeverity, SeverityDesign, make_severity_estimator)


def fixture(size=900):
    rng = np.random.default_rng(731)
    X = pd.DataFrame({"VehPower": rng.choice([4, 7, 10], size),
        "VehAge": rng.integers(0, 35, size), "DrivAge": rng.integers(18, 90, size),
        "BonusMalus": rng.integers(50, 180, size),
        "VehBrand": rng.choice(["B1", "B2", "B12"], size),
        "VehGas": rng.choice(["Diesel", "Regular"], size),
        "Area": rng.choice(["A", "B", "C"], size),
        "Density": np.exp(rng.uniform(1, 9, size)),
        "Region": rng.choice(["R1", "R2", "R3"], size)})
    mean = 1500 * np.exp(.02 * (X.DrivAge - 45) + .25 * X.VehGas.eq("Diesel"))
    y = rng.gamma(shape=2, scale=mean / 2)
    return X, y


def constant_features(size):
    X, _ = fixture(size)
    for column in X:
        X[column] = X[column].iloc[0]
    return X


class SeverityModelContracts(unittest.TestCase):
    def test_mean_and_intercept_gamma_are_arithmetic_claim_means(self):
        X = constant_features(3)
        amounts = np.array([1., 1., 1000.])
        for model in (ClaimMeanSeverity(), GammaSeverity()):
            with self.subTest(model=type(model).__name__):
                model.fit(X, amounts)
                np.testing.assert_allclose(model.predict(X), amounts.mean(), rtol=1e-14)
                self.assertNotAlmostEqual(float(model.predict(X)[0]), float(np.exp(np.log(amounts).mean())))
                self.assertAlmostEqual(model.diagnostics_["training_actual_over_predicted"], 1)

    def test_gamma_matches_independent_statsmodels_loglink(self):
        X, amounts = fixture()
        model = GammaSeverity().fit(X, amounts)
        design = model.design_.transform(X)
        independent = sm.GLM(amounts, design, family=sm.families.Gamma(link=sm.families.links.Log())).fit(maxiter=200, tol=1e-12)
        np.testing.assert_allclose(model.predict(X), independent.predict(design), rtol=2e-6, atol=1e-5)
        self.assertLess(model.diagnostics_["normalized_score_max"], 1e-7)
        # Gamma's intercept balances relative residuals, not raw euro totals.
        self.assertAlmostEqual(float(np.mean(amounts / model.predict(X))), 1., places=7)
        self.assertGreater(abs(float(amounts.sum() - model.predict(X).sum())), 1.)
        self.assertIsNotNone(model.diagnostics_["pearson_dispersion"])

    def test_log_response_and_global_smearing_recover_intercept_arithmetic_mean(self):
        X = constant_features(2)
        amounts = np.array([1., 100.])
        model = LogSmearingSeverity().fit(X, amounts)
        self.assertAlmostEqual(float(np.exp(model.coefficients_[0])), 10., places=10)
        self.assertAlmostEqual(model.smearing_, 5.05, places=10)
        np.testing.assert_allclose(model.predict(X), amounts.mean(), rtol=1e-14)
        self.assertEqual(model.diagnostics_["response_transform"], "log(Y)")
        # In particular, this is neither exp(eta) alone nor log1p retransformation.
        self.assertNotAlmostEqual(float(model.predict(X)[0]), 10.)

    def test_log_fit_matches_independent_ols_and_learned_smear(self):
        X, amounts = fixture()
        model = LogSmearingSeverity().fit(X, amounts)
        design = model.design_.transform(X)
        independent = sm.OLS(np.log(amounts), design).fit()
        expected_smear = float(np.exp(independent.resid).mean())
        self.assertAlmostEqual(model.smearing_, expected_smear, places=11)
        np.testing.assert_allclose(model.predict(X), np.exp(independent.predict(design)) * expected_smear, rtol=1e-10)
        self.assertLess(np.max(np.abs(design.T @ model.training_log_residual_)), 1e-7)

    def test_global_smearing_does_not_claim_conditional_unbiasedness(self):
        X = constant_features(12)
        X.loc[:5, "VehGas"] = "Diesel"
        X.loc[6:, "VehGas"] = "Regular"
        residuals = np.r_[np.tile([-.1, .1], 3), np.tile([-1.2, 1.2], 3)]
        amounts = 1000 * np.exp(residuals)
        model = LogSmearingSeverity().fit(X, amounts)
        groups = {g["group"]: g for g in model.diagnostics_["smearing_groups"] if g["feature"] == "VehGas"}
        self.assertLess(groups["Diesel"]["ratio_to_global_smearing"], 1)
        self.assertGreater(groups["Regular"]["ratio_to_global_smearing"], 1)
        self.assertGreater(groups["Regular"]["log_residual_variance"], groups["Diesel"]["log_residual_variance"])
        np.testing.assert_allclose(model.predict(X)[:6], model.predict(X)[6:], rtol=1e-13)
        self.assertGreater(abs(float(model.predict(X)[:6].mean() - amounts[:6].mean())), 100)
        self.assertIn("no conditional unbiasedness", model.diagnostics_["smearing_limits"])

    def test_design_preprocessing_is_outcome_free_and_unknown_categories_explicit(self):
        X, amounts = fixture()
        first = SeverityDesign().fit(X, amounts)
        second = SeverityDesign().fit(X, amounts[::-1] * 1000)
        self.assertEqual(first.diagnostics_, second.diagnostics_)
        future = X.iloc[:4].copy()
        future.loc[:, "VehBrand"] = "new-brand"
        future.loc[:, "Density"] = 1e12
        initial = dict(first.levels_)
        matrix = first.transform(future)
        self.assertEqual(first.last_unknown_categories_["VehBrand"], 4)
        self.assertEqual(initial, first.levels_)
        self.assertTrue(np.isfinite(matrix).all())
        for model in (GammaSeverity(), LogSmearingSeverity()):
            fitted = clone(model).fit(X, amounts)
            prediction = fitted.predict(future)
            self.assertEqual(prediction.shape, (4,))
            self.assertTrue((prediction > 0).all())
            self.assertEqual(fitted.last_unknown_categories_["VehBrand"], 4)

    def test_expected_fisher_and_ols_leverage_match_explicit_hat_matrix(self):
        X, amounts = fixture(240)
        for model in (GammaSeverity(), LogSmearingSeverity()):
            with self.subTest(model=type(model).__name__):
                model.fit(X, amounts)
                design = model.design_.transform(X)
                hat = design @ np.linalg.inv(design.T @ design) @ design.T
                np.testing.assert_allclose(model.training_leverage_, np.diag(hat), atol=1e-12)
                self.assertAlmostEqual(float(model.training_leverage_.sum()), design.shape[1], places=10)

    def test_rank_reduction_handles_constant_density_and_aliases(self):
        X, amounts = fixture()
        X["Density"] = 10
        X["Region"] = X.Area
        model = GammaSeverity().fit(X, amounts)
        diagnostic = model.diagnostics_["design"]
        self.assertIn("logDensity_standardized", diagnostic["dropped_covariate_columns"])
        self.assertTrue(any(x.startswith(("Region=", "Area=")) for x in diagnostic["dropped_covariate_columns"]))
        self.assertEqual(diagnostic["retained_design_rank"], diagnostic["retained_design_columns"])

    def test_weights_nonpositive_amounts_and_outcome_columns_rejected(self):
        X, amounts = fixture(120)
        for constructor in (ClaimMeanSeverity, GammaSeverity, LogSmearingSeverity):
            with self.subTest(model=constructor.__name__):
                with self.assertRaisesRegex(DataContractError, "unit claim-row"):
                    constructor().fit(X, amounts, sample_weight=np.ones(len(X)))
                for forbidden in ("Exposure", "IDpol", "ClaimNb", "ClaimAmount", "fold", "recorded_loss_eur"):
                    with self.subTest(forbidden=forbidden), self.assertRaisesRegex(DataContractError, "prospective"):
                        constructor().fit(X.assign(**{forbidden: 1}), amounts)
                for value in (0, -1, np.nan, np.inf):
                    invalid = amounts.copy()
                    invalid[0] = value
                    with self.assertRaisesRegex(DataContractError, "strictly positive"):
                        constructor().fit(X, invalid)

    def test_claim_rows_are_not_collapsed_to_policy_log_means(self):
        X = constant_features(3)
        amounts = np.array([10., 1000., 20.])
        # First two rows could belong to the same policy; their distinct logs
        # and unit weights remain two observations in the estimator contract.
        model = LogSmearingSeverity().fit(X, amounts)
        self.assertEqual(model.diagnostics_["training_claim_rows"], 3)
        self.assertAlmostEqual(float(model.predict(X)[0]), amounts.mean())
        collapsed_policy_average = np.array([505., 20.]).mean()
        self.assertNotAlmostEqual(float(model.predict(X)[0]), collapsed_policy_average)

    def test_factory_preserves_only_registered_gamma_parameters(self):
        model = make_severity_estimator({"family": "gamma", "parameters": {"max_iter": 200, "tol": 1e-8}})
        self.assertEqual(model.get_params()["max_iter"], 200)
        self.assertEqual(model.get_params()["tol"], 1e-8)
        self.assertIsInstance(make_severity_estimator({"family": "mean"}), ClaimMeanSeverity)
        self.assertIsInstance(make_severity_estimator({"family": "lognormal"}), LogSmearingSeverity)
        with self.assertRaisesRegex(DataContractError, "Unknown severity"):
            make_severity_estimator({"family": "unregistered"})


if __name__ == "__main__":
    unittest.main()
