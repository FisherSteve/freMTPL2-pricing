"""Contracts linking registered candidates, policy folds and the S4 runner."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import DataContractError, PROJECT_ROOT
from src.frequency import configuration, estimator, fit_oof, choose_models
from src.validation import make_membership, protocol
from tests.test_frequency_models import fixture


class FrequencyRunnerContracts(unittest.TestCase):
    def test_registered_fixed_encodings_match_implemented_terms(self):
        from src.frequency_models import AGE_BREAKS, VEHICLE_BREAKS, BM_BREAKS, FACTORS, SMOOTHS, FEATURES
        cfg = configuration()
        self.assertEqual(cfg["glm_specification"]["age_edges"], list(AGE_BREAKS))
        self.assertEqual(cfg["glm_specification"]["vehicle_edges"], list(VEHICLE_BREAKS))
        self.assertEqual(cfg["glm_specification"]["bm_edges"], list(BM_BREAKS))
        self.assertEqual(cfg["gam_specification"]["factors"], list(FACTORS))
        self.assertEqual(cfg["glm_specification"]["factors"], list(FACTORS))
        self.assertEqual(cfg["gam_specification"]["smooths"], list(SMOOTHS))
        self.assertEqual(protocol()["prospective_features"], list(FEATURES))

    def test_registered_parameters_reach_every_estimator(self):
        for config in configuration()["configurations"]:
            model = estimator(config)
            for name, value in config.items():
                if name not in ("family", "model_id"):
                    self.assertEqual(model.get_params()[name], value)
            if config["family"] == "histogram_poisson":
                self.assertEqual(model.max_iter, 300)
                self.assertEqual(model.random_state, 20261029)

    def test_changed_search_budget_and_early_stopping_are_rejected(self):
        cfg = configuration()
        for field, value in [("early_stopping", True), ("max_iter", 301), ("learning_rate", .1)]:
            modified = copy.deepcopy(cfg)
            modified["histogram_poisson"][field] = value
            with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as temporary:
                path = Path(temporary) / "protocol.json"
                path.write_text(json.dumps(modified), encoding="utf-8")
                with patch("src.frequency.CONFIG", path), self.assertRaises(DataContractError):
                    configuration()

    def development(self):
        x, rates, exposure = fixture(150)
        x["IDpol"] = np.arange(150)+1
        x["Exposure"] = exposure
        x["recorded_positive_count"] = np.rint(rates*exposure).astype(int)+1
        membership = make_membership(x.IDpol, protocol()["split"])
        d = x.loc[x.IDpol.isin(membership.loc[membership.split.eq("development"), "IDpol"])].copy()
        return d, membership

    def test_actual_five_fits_have_exact_once_keyed_predictions_and_own_fold_isolation(self):
        d, membership = self.development()
        config = {"family": "intercept", "model_id": "intercept"}
        p, evidence, timing = fit_oof(d.sample(frac=1, random_state=3), membership, config, "fixture")
        self.assertEqual(len(p), len(d))
        self.assertTrue(p.IDpol.is_unique)
        self.assertEqual(len(evidence), 5)
        self.assertEqual(len(timing), 5)
        ids = membership.loc[membership.fold.eq(2), "IDpol"]
        changed = d.copy()
        changed.loc[changed.IDpol.isin(ids), "recorded_positive_count"] += 500
        q, _, _ = fit_oof(changed, membership, config, "fixture")
        np.testing.assert_array_equal(p.loc[p.fold.eq(2), "annual_rate"], q.loc[q.fold.eq(2), "annual_rate"])
        self.assertTrue((p.loc[p.fold.ne(2), "annual_rate"].to_numpy() != q.loc[q.fold.ne(2), "annual_rate"].to_numpy()).all())
        self.assertTrue(p["split"].eq("development").all())

    def test_evaluation_ids_or_input_fold_columns_never_enter_fitting(self):
        d, membership = self.development()
        for invalid in (d.assign(fold=1), d.assign(IDpol=np.arange(len(d))+10000)):
            with self.assertRaisesRegex(DataContractError, "exactly development"):
                fit_oof(invalid, membership, {"family": "intercept", "model_id": "intercept"}, "fixture")

    def test_selection_uses_common_registered_complete_scores(self):
        cfg = configuration()["configurations"]
        scores = pd.DataFrame({"model_id": [c["model_id"] for c in cfg],
                               "poisson_deviance_per_exposure": np.arange(len(cfg), 0, -1)/100})
        families, selected = choose_models(scores, cfg)
        self.assertEqual(selected, "hgb_l31")
        self.assertEqual(set(families), {"glm", "gam", "histogram_poisson"})
        scores.loc[scores.model_id.eq("intercept"), "poisson_deviance_per_exposure"] = .0001
        self.assertEqual(choose_models(scores, cfg)[1], "intercept")
        with self.assertRaisesRegex(DataContractError, "cover"):
            choose_models(scores.iloc[:-1], cfg)


if __name__ == "__main__":
    unittest.main()
