"""Statistical boundaries of the reduced tariff, support and expense arithmetic."""
import unittest
import tempfile
import hashlib
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.data import DataContractError
from src.interpretation import (TwoFeaturePoisson, cost_scenarios, feature_budget,
    fold_severity, groups, persist_source_snapshot, register_run, standardized,
    support_cells, verified_fit_reuse)
from src.severity_diagnostics import compose_pure_premium, paired_policy_bootstrap, tweedie_deviance_rows


class InterpretationTests(unittest.TestCase):
    def setUp(self):
        n = 100
        self.x = pd.DataFrame({"DrivAge": np.tile(np.arange(20, 70), 2), "Density": np.linspace(10, 10000, n)})
        self.e = np.linspace(.1, 1.1, n)
        self.y = np.tile([0., 1., 0., 0., 0.], 20)/self.e

    def test_complete_feature_budget_rejects_hidden_severity(self):
        self.assertEqual(feature_budget(["DrivAge", "Area"], []), ["Area", "DrivAge"])
        with self.assertRaises(DataContractError):
            feature_budget(["DrivAge", "Area"], ["VehAge"])
        with self.assertRaises(DataContractError):
            feature_budget(["logDensity", "DrivAge"], [])

    def test_exact_columns_and_positive_density(self):
        for family in ["glm", "gam"]:
            m = TwoFeaturePoisson("Density", family)
            with self.assertRaises(DataContractError):
                m.fit(self.x.assign(ClaimNb=0), self.y, self.e)
            with self.assertRaises(DataContractError):
                m.fit(self.x.assign(Density=0), self.y, self.e)

    def test_train_only_ranges_and_explicit_extrapolation(self):
        m = TwoFeaturePoisson("Density", "gam").fit(self.x, self.y, self.e)
        knots = m.splines_["Density"].bsplines_[0].t.copy()
        rates = m.predict(pd.DataFrame({"DrivAge": [100], "Density": [1e8]}))
        np.testing.assert_array_equal(knots, m.splines_["Density"].bsplines_[0].t)
        self.assertEqual(m.last_support_["outside_training_range"], {"DrivAge": 1, "Density": 1})
        self.assertTrue((rates > 0).all())

    def test_unknown_factor_does_not_learn_validation_level(self):
        x = self.x[["DrivAge"]].assign(Area=np.tile(["A", "B"], 50))
        for family in ["glm", "gam"]:
            model = TwoFeaturePoisson("Area", family).fit(x, self.y, self.e)
            before = list(model.levels_["Area"])
            predicted = model.predict(pd.DataFrame({"DrivAge": [35, 35], "Area": ["Z", "A"]}))
            self.assertEqual(model.levels_["Area"], before)
            self.assertEqual(model.last_support_["unknown_categories"]["Area"], 1)
            self.assertAlmostEqual(predicted[0], predicted[1])

    def test_rows_and_columns_do_not_change_fitted_rule(self):
        for family in ["glm", "gam"]:
            first = TwoFeaturePoisson("Density", family).fit(self.x, self.y, self.e)
            permutation = np.random.default_rng(3).permutation(len(self.x))
            other = TwoFeaturePoisson("Density", family).fit(self.x.iloc[permutation, ::-1], self.y[permutation], self.e[permutation])
            np.testing.assert_allclose(first.predict(self.x), other.predict(self.x), rtol=1e-9)

    @staticmethod
    def tiny():
        membership = pd.DataFrame({"IDpol": range(1, 12), "split": ["development"]*10+["evaluation"], "fold": list(range(1, 6))*2+[0]})
        d = membership.iloc[:10][["IDpol"]].assign(Exposure=[.1, .2, .4, .8, 1.]*2,
            recorded_positive_count=[1]*5+[0]*5, recorded_loss_eur=[10., 20., 30., 40., 100.]+[0.]*5)
        c = pd.DataFrame({"IDpol": range(1, 6), "claim_row_id": [f"c{i}" for i in range(5)], "ClaimAmount": [10., 20., 30., 40., 100.]})
        return d, c, membership

    def test_severity_training_and_claim_boundaries(self):
        d, c, membership = self.tiny()
        predictions, audit = fold_severity(d, c, membership, "test")
        self.assertEqual(audit[4]["severity_eur"], 25.)
        self.assertEqual(len(predictions), 10)
        self.assertTrue(predictions.loc[predictions.fold.eq(5), "severity_eur"].eq(25).all())
        with self.assertRaises(DataContractError):
            fold_severity(d, pd.concat([c, c.iloc[[0]].assign(IDpol=99, claim_row_id="bad")]), membership, "test")
        with self.assertRaises(DataContractError):
            fold_severity(d.iloc[:-1], c, membership, "test")

    def test_keyed_product_includes_zeros_and_correct_period_units(self):
        d, c, membership = self.tiny()
        s, _ = fold_severity(d, c, membership, "s")
        f = membership.iloc[:10].assign(model_id="f", run_id="f", annual_rate=np.linspace(.1, .5, 10))
        f["expected_count"] = f.annual_rate*d.Exposure
        pair = [{"model_id": "p", "frequency_model_id": "f", "severity_model_id": "severity_mean"}]
        p = compose_pure_premium(d, membership, f, s, pair, "p")
        other = compose_pure_premium(d.sample(frac=1), membership, f.sample(frac=1), s.sample(frac=1), pair, "p")
        pd.testing.assert_frame_equal(p, other)
        np.testing.assert_allclose(p.expected_period_loss, p.annual_pure_premium*d.Exposure)
        self.assertEqual(len(p), len(d))
        self.assertGreater(p.loc[d.recorded_positive_count.eq(0), "expected_period_loss"].sum(), 0)

    def test_paired_bootstrap_uses_drawn_exposure_denominator(self):
        d, _, membership = self.tiny()
        base = membership.iloc[:10].assign(model_id="a", run_id="p", annual_rate=.1, severity_eur=30.,
            annual_pure_premium=3., frequency_model_id="f", severity_model_id="s")
        base["expected_period_loss"] = base.annual_pure_premium*d.Exposure
        other = base.assign(model_id="b", annual_pure_premium=7., annual_rate=7/30)
        other["expected_period_loss"] = other.annual_pure_premium*d.Exposure
        p = pd.concat([base, other], ignore_index=True)
        out = paired_policy_bootstrap(d, p, membership, "a", replicates=37, seed=9).set_index("model_id")
        e, loss = d.Exposure.to_numpy(), d.recorded_loss_eur.to_numpy()
        diff = e*(tweedie_deviance_rows(loss/e, np.full(10, 7.))-tweedie_deviance_rows(loss/e, np.full(10, 3.)))
        rng = np.random.default_rng(9)
        draws = []
        for _ in range(37):
            i = rng.integers(0, 10, 10)
            draws.append(diff[i].sum()/e[i].sum())
        np.testing.assert_allclose([out.loc["b", "lower"], out.loc["b", "upper"]], np.percentile(draws, [2.5, 97.5]))

    def test_support_drops_empty_joint_cells_without_extrapolation(self):
        rows = []
        for g, area, density in [("AB", "A", 20), ("CD", "C", 500), ("EF", "F", 10000)]:
            for _ in range(120):
                rows.append({"IDpol": len(rows), "Area": area, "Density": density, "DrivAge": 40,
                    "BonusMalus": 50, "VehAge": 3, "VehGas": "Diesel", "Exposure": 1.,
                    "recorded_positive_count": 1, "recorded_loss_eur": 100.})
        d = groups(pd.DataFrame(rows))
        retained, cells, strata = support_cells(d)
        self.assertEqual(len(retained), len(d))
        self.assertEqual(set(zip(retained.Area, retained.Density)), {("A", 20), ("C", 500), ("F", 10000)})
        result = standardized(retained, strata, ["recorded_positive_count", "recorded_loss_eur"])
        np.testing.assert_allclose(result.recorded_loss_eur, 100.)
        reduced, _, _ = support_cells(d.loc[d.geo.ne("EF")])
        self.assertTrue(reduced.empty)
        self.assertAlmostEqual(cells.exposure.sum(), d.Exposure.sum())

    def test_cost_identity_crossover_and_order(self):
        q = np.array([0., 75., 850/3, 400., 900.])
        fixed, variable = cost_scenarios(q)
        np.testing.assert_allclose(variable-q, .15*variable)
        self.assertAlmostEqual(fixed[2], variable[2])
        self.assertLess(variable[1], fixed[1])
        self.assertGreater(variable[3], fixed[3])
        with self.assertRaises(DataContractError):
            cost_scenarios([-1])

    def test_partial_support_mask_survives_parquet_roundtrip(self):
        rows = []
        for area, density in [("A", 20), ("C", 500), ("F", 10000)]:
            for _ in range(120):
                rows.append({"IDpol": len(rows), "Area": area, "Density": density, "DrivAge": 40,
                    "BonusMalus": 50, "VehAge": 3, "VehGas": "Diesel", "Exposure": 1.,
                    "recorded_positive_count": 1, "recorded_loss_eur": 100.})
        rows.append({**rows[0], "IDpol": len(rows), "DrivAge": 80})
        _, cells, _ = support_cells(groups(pd.DataFrame(rows)))
        self.assertEqual(str(cells.retained.dtype), "bool")
        self.assertEqual(int((~cells.retained).sum()), 1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"data/test-tmp") as tmp:
            path = Path(tmp)/"support.parquet"
            cells.to_parquet(path, index=False)
            pd.testing.assert_frame_equal(cells, pd.read_parquet(path))

    def test_registration_utf8_repeat_preserves_origin_and_rejects_changes(self):
        import copy
        registration = {"identity": {"source": "fixed"}, "specification": {"scope": "geöffnet; Schadenhöhe"},
                        "verified_frequency_cache_origin": "earlier"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"data/test-tmp") as tmp:
            path = Path(tmp)/"registration.json"
            register_run(path, copy.deepcopy(registration))
            before = path.read_bytes()
            self.assertIn("Schadenhöhe".encode("utf-8"), before)
            repeated = {**copy.deepcopy(registration), "verified_frequency_cache_origin": None}
            self.assertEqual(register_run(path, repeated), registration)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(DataContractError):
                register_run(path, {**copy.deepcopy(registration), "specification": {"scope": "changed"}})
            with self.assertRaises(DataContractError):
                register_run(path, {**copy.deepcopy(registration), "verified_frequency_cache_origin": "other"})

    def test_registered_snapshot_can_be_reused_and_is_immutable(self):
        import src.interpretation as module
        source = Path(module.__file__)
        identity = {"source_sha256": {"src/interpretation.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"data/test-tmp") as tmp:
            root = Path(tmp)
            directory = root/"data/interpretation/previous"
            directory.mkdir(parents=True)
            register_run(directory/"registration.json", {"identity": identity,
                "specification": {"scope": "Schadenhöhe"}, "verified_frequency_cache_origin": None})
            snapshot = persist_source_snapshot(directory, identity, source)
            before = snapshot.stat().st_mtime_ns
            self.assertEqual(snapshot.read_bytes(), source.read_bytes())
            persist_source_snapshot(directory, identity, source)
            self.assertEqual(snapshot.stat().st_mtime_ns, before)
            with mock.patch.object(module, "ROOT", root):
                self.assertEqual(verified_fit_reuse("previous", identity), directory)
            snapshot.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(DataContractError, "Immutable"):
                persist_source_snapshot(directory, identity, source)
            self.assertEqual(snapshot.read_bytes(), b"tampered\n")
            with mock.patch.object(module, "ROOT", root), self.assertRaises(DataContractError):
                verified_fit_reuse("previous", identity)

    def test_source_change_after_registration_fails_before_snapshot_write(self):
        import src.interpretation as module
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"data/test-tmp") as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(DataContractError, "changed after registration"):
                persist_source_snapshot(directory, {"source_sha256": {"src/interpretation.py": "different"}}, Path(module.__file__))
            self.assertFalse((directory/"executed_source.py").exists())

    def test_standard_run_persists_snapshot_before_any_training(self):
        import src.interpretation as module
        class ReachedTraining(Exception):
            pass
        raw = Path(module.__file__).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"data/test-tmp") as tmp:
            root = Path(tmp)
            def before_training(*args):
                snapshots = list((root/"data/interpretation").glob("*/executed_source.py"))
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(snapshots[0].read_bytes(), raw)
                raise ReachedTraining()
            with mock.patch.object(module, "ROOT", root), \
                    mock.patch.object(module, "configuration", return_value={}), \
                    mock.patch.object(module, "load_inputs", return_value=(None, None, None,
                        {"source_id": "source", "preparation_id": "prep"}, {"split_id": "split"}, None)), \
                    mock.patch.object(module, "text_sha256", return_value=digest), \
                    mock.patch.object(module, "sha256", return_value="input"), \
                    mock.patch.object(module, "fold_severity", side_effect=before_training), \
                    mock.patch.object(module, "run_oof") as frequency_fit:
                with self.assertRaises(ReachedTraining):
                    module.run()
                frequency_fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
