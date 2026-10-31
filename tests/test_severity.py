"""S5 runner contracts for claim grouping, prospective prediction and release."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import DataContractError, PROJECT_ROOT
from src.severity import (configuration, fit_oof, choose_severity, predict_severity,
                          development_boundary, evaluate)
from src.severity_models import ClaimMeanSeverity, FEATURES
from src.validation import make_membership, protocol
from tests.test_severity_models import fixture


class SeverityRunnerContracts(unittest.TestCase):
    def populations(self):
        x, y = fixture(250)
        x["IDpol"] = np.arange(len(x))+1
        x["Exposure"] = np.linspace(.05, 1.5, len(x))
        count = np.arange(len(x)) % 4
        rows = [{"IDpol": int(x.IDpol.iloc[i]), "claim_row_id": f"claim-{i:04d}-{j}",
                 "ClaimAmount": float(y[i]*(1+j))} for i in range(len(x)) for j in range(count[i])]
        c = pd.DataFrame(rows)
        x["recorded_positive_count"] = count
        x["recorded_loss_eur"] = x.IDpol.map(c.groupby("IDpol").ClaimAmount.sum()).fillna(0)
        membership = make_membership(x.IDpol, protocol()["split"])
        ids = membership.loc[membership["split"].eq("development"), "IDpol"]
        return x.loc[x.IDpol.isin(ids)].copy(), c.loc[c.IDpol.isin(ids)].copy(), membership

    def test_actual_oof_claim_grouping_and_own_fold_isolation_for_every_candidate(self):
        d, c, membership = self.populations()
        ids = membership.loc[membership.fold.eq(2), "IDpol"]
        changed = c.copy()
        changed.loc[changed.IDpol.isin(ids), "ClaimAmount"] *= 100
        for config in configuration()["configurations"]:
            with self.subTest(model=config["model_id"]):
                p, info, _ = fit_oof(d.sample(frac=1, random_state=4), c, membership, config, "fixture")
                q, _, _ = fit_oof(d, changed, membership, config, "fixture")
                self.assertEqual(len(p), len(d))
                self.assertTrue(p.IDpol.is_unique)
                self.assertTrue(p["split"].eq("development").all())
                np.testing.assert_array_equal(p.loc[p.fold.eq(2), "severity_eur"], q.loc[q.fold.eq(2), "severity_eur"])
                self.assertEqual(len(info), 5)
                self.assertEqual(info[1]["training_claim_rows"], int(c.loc[~c.IDpol.isin(ids)].shape[0]))

    def test_predicts_every_policy_without_reading_realized_counts_or_loss(self):
        d, c, membership = self.populations()
        model = ClaimMeanSeverity().fit(d[list(FEATURES)], np.full(len(d), 1234.))
        p = predict_severity(model, d, membership, "mean", "fixture", "development")
        altered = d.copy()
        altered["recorded_positive_count"] = 987
        altered["recorded_loss_eur"] = np.nan
        q = predict_severity(model, altered.iloc[::-1], membership, "mean", "fixture", "development")
        pd.testing.assert_frame_equal(p, q)
        self.assertEqual(len(p), len(d))
        np.testing.assert_array_equal(p.severity_eur, np.full(len(d), 1234.))

    def test_oof_rejects_evaluation_ids_and_cross_split_claims(self):
        d, c, membership = self.populations()
        bad = d.copy()
        bad.loc[bad.index[0], "IDpol"] = membership.loc[membership["split"].eq("evaluation"), "IDpol"].iloc[0]
        with self.assertRaisesRegex(DataContractError, "development"):
            fit_oof(bad, c, membership, configuration()["configurations"][0], "fixture")

    def test_selection_uses_fixed_frequency_and_practical_mean_tolerance(self):
        s = pd.DataFrame({"model_id": ["pp_hgb_mean", "pp_hgb_gamma", "pp_hgb_log_smear", "pp_intercept_mean"],
                          "tweedie_deviance_per_exposure": [100.4, 100., 100.2, 1.]})
        self.assertEqual(choose_severity(s), ("pp_hgb_mean", "severity_mean"))
        s.loc[s.model_id.eq("pp_hgb_mean"), "tweedie_deviance_per_exposure"] = 100.6
        self.assertEqual(choose_severity(s), ("pp_hgb_gamma", "severity_gamma"))
        with self.assertRaises(DataContractError):
            choose_severity(s.iloc[:2])

    def test_open_evaluation_rejects_new_identity_or_new_development_fitting(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/"data") as temporary:
            output = Path(temporary)
            self.assertFalse(development_boundary(output, "first"))
            (output/"evaluation_release.json").write_text(json.dumps({"run_id": "first"}), encoding="utf-8")
            with self.assertRaisesRegex(DataContractError, "new development"):
                development_boundary(output, "different")
            with self.assertRaisesRegex(DataContractError, "new fitting"):
                development_boundary(output, "first")
            (output/"first/development").mkdir(parents=True)
            (output/"first/development/manifest.json").write_text("{}", encoding="utf-8")
            self.assertTrue(development_boundary(output, "first"))

    def test_evaluation_without_locked_decision_does_not_release_or_inspect_targets(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/"data") as temporary:
            output = Path(temporary)
            with patch("src.severity.inputs", return_value=(object(), object(), {}, {}, {}, {})):
                with self.assertRaises(FileNotFoundError):
                    evaluate(output, output)
            self.assertFalse((output/"evaluation_release.json").exists())


if __name__ == "__main__":
    unittest.main()
