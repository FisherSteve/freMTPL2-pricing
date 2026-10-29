"""Independent objective, identification and boundary tests for S4 candidates."""
import unittest

import numpy as np
import pandas as pd
from scipy import linalg, optimize
import statsmodels.api as sm
from sklearn.base import clone

from src.data import DataContractError
from src.frequency_models import (FrequencyGAM, FrequencyGLM, FrequencyHistogram,
                                  make_estimator, prospective_frame)


def fixture(size=600):
    rng = np.random.default_rng(384)
    frame = pd.DataFrame({"VehPower": rng.choice([4, 7, 10], size),
        "VehAge": rng.integers(0, 35, size), "DrivAge": rng.integers(18, 90, size),
        "BonusMalus": rng.integers(50, 180, size),
        "VehBrand": rng.choice(["B1", "B2", "B12"], size),
        "VehGas": rng.choice(["Diesel", "Regular"], size),
        "Area": rng.choice(["A", "B", "C"], size),
        "Density": np.exp(rng.uniform(1, 9, size)),
        "Region": rng.choice(["R1", "R2", "R3"], size)})
    e = rng.uniform(.1, 1.1, size)
    rates = np.exp(-1 + .02 * (frame.DrivAge - 40) + .3 * np.sin(frame.VehAge / 10)
                   + .002 * (frame.BonusMalus - 50))
    n = rng.poisson(e * rates)
    return frame, n / e, e


class FrequencyModelContracts(unittest.TestCase):
    def test_glm_rate_fit_matches_independent_count_offset_likelihood(self):
        X, y, e = fixture()
        model = FrequencyGLM(alpha=0).fit(X, y, sample_weight=e)
        design, _ = model._design(model._groups(prospective_frame(X)))
        design = design[:, model.keep_]
        independent = sm.GLM(y * e, design, family=sm.families.Poisson(),
                             offset=np.log(e)).fit(maxiter=100, tol=1e-11)
        np.testing.assert_allclose(model.predict(X) * e, independent.predict(design, offset=np.log(e)), rtol=1e-6)
        self.assertAlmostEqual(float(np.dot(e, model.predict(X))), float(np.dot(e, y)), places=5)
        self.assertEqual(model.diagnostics_["retained_design"]["weighted_design_rank"], design.shape[1])

    def test_glm_rank_reduction_handles_unsupported_interaction_cells(self):
        X, y, e = fixture()
        # Force absent cross-cells using covariates, without looking at outcomes.
        X.loc[X.DrivAge < 35, "BonusMalus"] = 50
        model = FrequencyGLM(alpha=.0001, encoding="age_bm_interaction").fit(X, y, sample_weight=e)
        self.assertTrue(model.diagnostics_["dropped_covariate_columns"])
        self.assertTrue(np.isfinite(model.predict(X)).all())
        self.assertEqual(model.diagnostics_["retained_design"]["weighted_design_rank"], len(model.keep_))

    def test_gam_smooths_are_exposure_centered_and_penalty_is_full_difference(self):
        X, y, e = fixture()
        model = FrequencyGAM(n_splines=6, lam=1).fit(X, y, sample_weight=e)
        frame = prospective_frame(X)
        design, _ = model._design(frame)
        for feature, index in model.slices_.items():
            self.assertLess(abs(float(np.average(design[:, index] @ model.coef_[index], weights=e))), 1e-10)
            full_beta = np.r_[model.coef_[index], 0.0]
            actual_penalty = model.coef_[index] @ model.penalty_[index, index] @ model.coef_[index]
            self.assertAlmostEqual(actual_penalty, float(np.diff(full_beta, n=2) @ np.diff(full_beta, n=2)), places=10)
        self.assertTrue(model.diagnostics_["converged"])
        self.assertLess(model.diagnostics_["normalized_gradient_max"], 1e-7)
        self.assertGreater(model.diagnostics_["effective_degrees_of_freedom"], 1)
        self.assertLess(model.diagnostics_["effective_degrees_of_freedom"], design.shape[1])

    def test_gam_matches_independent_minimizer_and_finite_difference_gradient(self):
        X, y, e = fixture(350)
        model = FrequencyGAM(n_splines=5, lam=3).fit(X, y, sample_weight=e)
        design, _ = model._design(prospective_frame(X))
        counts = y * e
        penalty = model.penalty_
        def objective(beta):
            eta = design @ beta
            return float(np.dot(e, np.exp(eta)) - np.dot(counts, eta) + .5 * beta @ penalty @ beta)
        def gradient(beta):
            return design.T @ (e * np.exp(design @ beta) - counts) + penalty @ beta
        probe = model.coef_ + np.linspace(-.01, .01, len(model.coef_))
        numerical = optimize._numdiff.approx_derivative(lambda beta: np.array([objective(beta)]), probe).ravel()
        np.testing.assert_allclose(gradient(probe), numerical, atol=1e-7, rtol=1e-6)
        independent = optimize.minimize(objective, np.zeros(design.shape[1]), jac=gradient,
                                        method="BFGS", options={"gtol": 1e-7, "maxiter": 500})
        self.assertLess(np.max(np.abs(independent.jac)), 1e-5)
        np.testing.assert_allclose(design @ independent.x, design @ model.coef_, atol=3e-6)

    def test_gam_unnormalized_penalty_scale_contract(self):
        X, y, e = fixture(350)
        model = FrequencyGAM(n_splines=5, lam=2).fit(X, y, sample_weight=e)
        # Scaling exposure and lambda together leaves the raw-sum objective
        # multiplied by the same constant. Scaling exposure alone does not.
        scaled = FrequencyGAM(n_splines=5, lam=6).fit(X, y, sample_weight=3 * e)
        np.testing.assert_allclose(model.predict(X), scaled.predict(X), atol=1e-8, rtol=1e-8)

    def test_gam_edf_matches_explicit_observation_hat_matrix(self):
        X, y, e = fixture(240)
        model = FrequencyGAM(n_splines=5, lam=4).fit(X, y, sample_weight=e)
        design, _ = model._design(prospective_frame(X))
        weighted_design = np.sqrt(e * model.predict(X))[:, None] * design
        penalized_information = weighted_design.T @ weighted_design + model.penalty_
        # Independently form the n x n observation-space hat matrix rather
        # than the model implementation's p x p parameter-space expression.
        hat = weighted_design @ linalg.solve(penalized_information, weighted_design.T)
        self.assertAlmostEqual(float(np.trace(hat)), model.diagnostics_["effective_degrees_of_freedom"], places=10)
        self.assertTrue(np.all(np.diag(hat) >= -1e-12))
        self.assertTrue(np.all(np.diag(hat) <= 1 + 1e-12))
        np.testing.assert_array_equal(model.penalty_[0], np.zeros(design.shape[1]))

    def test_substantive_gam_oof_predictions_exclude_own_fold_outcomes(self):
        from src.frequency import fit_oof
        from src.validation import make_membership, protocol
        X, y, e = fixture(240)
        X["IDpol"] = np.arange(1, len(X) + 1)
        X["Exposure"] = e
        X["recorded_positive_count"] = np.rint(y * e).astype(int) + 1
        membership = make_membership(X.IDpol, protocol()["split"])
        ids = membership.loc[membership.split.eq("development"), "IDpol"]
        development = X.loc[X.IDpol.isin(ids)].copy()
        config = {"family": "gam", "model_id": "fixture_gam", "n_splines": 5, "lam": 4}
        first, _, _ = fit_oof(development, membership, config, "fixture")
        changed = development.copy()
        held_ids = membership.loc[membership.fold.eq(3), "IDpol"]
        changed.loc[changed.IDpol.isin(held_ids), "recorded_positive_count"] += 2
        second, _, _ = fit_oof(changed.iloc[::-1], membership, config, "fixture")
        np.testing.assert_array_equal(first.loc[first.fold.eq(3), "annual_rate"], second.loc[second.fold.eq(3), "annual_rate"])
        self.assertGreater(np.max(np.abs(first.loc[first.fold.ne(3), "annual_rate"].to_numpy()
                                        - second.loc[second.fold.ne(3), "annual_rate"].to_numpy())), .01)

    def test_preprocessing_uses_only_training_rows_and_unknown_rules_are_explicit(self):
        X, y, e = fixture()
        for model in (FrequencyGLM(alpha=.0001), FrequencyGAM(n_splines=5), FrequencyHistogram(max_iter=5)):
            with self.subTest(model=type(model).__name__):
                fitted = clone(model).fit(X, y, sample_weight=e)
                initial_levels = dict(fitted.levels_)
                future = X.iloc[:4].copy()
                future.loc[:, "VehBrand"] = "new-brand"
                future.loc[:, "DrivAge"] = 120
                result = fitted.predict(future)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue((result > 0).all())
                self.assertEqual(fitted.last_unknown_categories_["VehBrand"], 4)
                self.assertEqual(initial_levels, fitted.levels_)
                if hasattr(fitted, "splines_"):
                    self.assertLess(fitted.diagnostics_["smooth_ranges"]["DrivAge"][1], 120)

    def test_forbidden_columns_invalid_targets_and_density_fail(self):
        X, y, e = fixture()
        for constructor in (FrequencyGLM, FrequencyGAM, FrequencyHistogram):
            with self.subTest(constructor=constructor.__name__):
                leaked = X.assign(recorded_positive_count=y * e)
                with self.assertRaisesRegex(DataContractError, "prospective"):
                    constructor().fit(leaked, y, sample_weight=e)
                invalid = X.copy()
                invalid.loc[0, "Density"] = 0
                with self.assertRaisesRegex(DataContractError, "density"):
                    constructor().fit(invalid, y, sample_weight=e)
                with self.assertRaisesRegex(DataContractError, "exposures"):
                    constructor().fit(X, y, sample_weight=np.zeros(len(X)))

    def test_factory_and_fixed_boosting_budget(self):
        X, y, e = fixture()
        model = make_estimator({"family": "histogram_poisson", "parameters": {"max_iter": 8}})
        model.fit(X, y, sample_weight=e)
        self.assertFalse(model.model_.early_stopping)
        self.assertEqual(model.model_.n_iter_, 8)
        with self.assertRaisesRegex(DataContractError, "Unknown"):
            make_estimator({"family": "unregistered"})


if __name__ == "__main__":
    unittest.main()
