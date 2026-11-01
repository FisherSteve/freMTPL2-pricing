"""Scientific contracts for the optional pilot, without requiring CTGAN/torch."""
import unittest
import importlib.util

import numpy as np
import pandas as pd

from src.augmentation import (JOINT, N, E, augmented_training, configuration,
                              fold_training, joint_frame, paired_difference,
                              valid_synthetic, fit_predict, restore_natural_sampling)
from src.data import DataContractError
from src.frequency_models import FEATURES


def fixture(n=150):
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"IDpol": np.arange(n), "VehPower": rng.choice([4, 6], n),
        "VehAge": rng.integers(0, 10, n), "DrivAge": rng.integers(18, 75, n),
        "BonusMalus": rng.choice([50, 65, 90, 110], n), "VehBrand": rng.choice(["B1", "B2"], n),
        "VehGas": rng.choice(["Diesel", "Regular"], n), "Area": rng.choice(["A", "B"], n),
        "Density": rng.uniform(10, 200, n), "Region": rng.choice(["R1", "R2"], n),
        E: rng.uniform(.01, 1, n), N: rng.poisson(.15, n)})
    membership = pd.DataFrame({"IDpol": d.IDpol, "split": "development", "fold": np.arange(n)%5+1})
    return d[["IDpol", *JOINT]], membership


class AugmentationContracts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("ctgan"), "Optional CTGAN environment")
    def test_actual_release_sampler_uses_corrected_frequencies(self):
        from collections import namedtuple
        from types import SimpleNamespace
        from ctgan.data_sampler import DataSampler
        Span = namedtuple("Span", "dim activation_fn")
        data = np.column_stack([np.arange(100)<99, np.arange(100)>=99,
                                np.arange(100)<70, np.arange(100)>=70]).astype(float)
        sampler = DataSampler(data, [[Span(2,"softmax")],[Span(2,"softmax")]], True)
        self.assertGreater(sampler._discrete_column_category_prob[0,1], .01)
        restore_natural_sampling(SimpleNamespace(_data_sampler=sampler))
        np.random.seed(55)
        sampled = sampler.sample_original_condvec(100000).mean(axis=0)
        # Uniform column selection, empirical category selection within column.
        np.testing.assert_allclose(sampled, [.495,.005,.35,.15], atol=.004)

    def test_generation_restores_natural_not_log_category_probabilities(self):
        from types import SimpleNamespace
        sampler = SimpleNamespace(_data_length=100,
            _rid_by_cat_cols=[[np.arange(99), np.arange(1)], [np.arange(70), np.arange(20), np.arange(10)]],
            _discrete_column_category_prob=np.array([[.87, .13, 0], [.4, .35, .25]]))
        old = sampler._discrete_column_category_prob.copy()
        evidence = restore_natural_sampling(SimpleNamespace(_data_sampler=sampler))
        np.testing.assert_allclose(sampler._discrete_column_category_prob, [[.99, .01, 0], [.7, .2, .1]])
        np.testing.assert_allclose(evidence['training_condition_probabilities'], old)
        self.assertFalse(np.array_equal(old, sampler._discrete_column_category_prob))

    def test_training_fold_is_invariant_to_own_validation_values(self):
        d, m = fixture()
        cfg = configuration()
        cfg["training_policies_per_fold"] = 60
        train, valid = fold_training(d, m, 2, cfg)
        changed = d.copy()
        changed.loc[m.fold.eq(2), [N, E, "Density", "DrivAge"]] = [999, 999, 99999, 999]
        again, _ = fold_training(changed, m, 2, cfg)
        pd.testing.assert_frame_equal(train, again)
        self.assertFalse(set(train.IDpol) & set(valid.IDpol))
        # Evaluation covariates/targets cannot accidentally be passed as development.
        m.loc[m.fold.eq(5), "split"] = "evaluation"
        with self.assertRaises(DataContractError):
            fold_training(d, m, 2, cfg)

    def test_generator_allowlist_rejects_ids_losses_and_fold(self):
        d, _ = fixture()
        base = d[JOINT]
        for forbidden in ("IDpol", "fold", "recorded_loss_eur", "annual_rate"):
            with self.assertRaises(DataContractError):
                joint_frame(base.assign(**{forbidden: 0}))

    def test_invalid_outputs_are_rejected_without_rounding(self):
        d, _ = fixture()
        base = d[JOINT]
        candidate = base.head(7).copy()
        candidate[N] = candidate[N].astype(float)
        candidate.loc[0, N] = .5
        candidate.loc[1, E] = 0
        candidate.loc[2, "Area"] = "unknown"
        candidate.loc[3, "Density"] = np.inf
        candidate.loc[4, "DrivAge"] = 999
        candidate.loc[5, E] = base[E].max()+1
        before = candidate.copy()
        valid, audit = valid_synthetic(candidate, base, configuration())
        self.assertEqual(valid.tolist(), [False]*6+[True])
        self.assertEqual(audit["rejected"], 6)
        pd.testing.assert_frame_equal(candidate, before)

    def test_duplication_does_not_change_normalized_penalty(self):
        d, _ = fixture(300)
        base = d[JOINT]
        combined, weights = augmented_training(base, base)
        self.assertAlmostEqual(weights.sum(), base[E].sum())
        cfg = configuration()
        a, _ = fit_predict(base, base[E].to_numpy(), base[list(FEATURES)], cfg)
        b, _ = fit_predict(combined, weights, base[list(FEATURES)], cfg)
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)

    def test_paired_score_uses_real_policy_count_loss_once(self):
        count, e = np.array([0, 1, 2, 0]), np.array([.1, 1., .5, .2])
        r, b = np.array([.2, .5, 1, .3]), np.full(4, .4)
        out = paired_difference(count, e, r, b)
        # Log-likelihood difference cancels the observed-only constants.
        difference = 2*(count*np.log(b/r)+e*(r-b))
        expected = difference.sum()/e.sum()
        self.assertAlmostEqual(out["difference"], expected)
        centered = difference-expected*e
        self.assertAlmostEqual(out["standard_error"], np.sqrt(4*np.var(centered, ddof=1))/e.sum())
        same = paired_difference(count, e, b, b)
        self.assertEqual(same["conditional_95_interval"], [0, 0])


if __name__ == "__main__":
    unittest.main()
