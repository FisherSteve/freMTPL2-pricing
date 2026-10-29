"""Policy separation, preprocessing leakage, prediction alignment and freeze risks."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import validation as v
from src.data import DataContractError, json_bytes, logical_hash
from src.metrics import poisson_deviance_per_exposure


@contextmanager
def workspace_tempdir():
    base = v.PROJECT_ROOT / "data/test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        yield Path(directory)


def fixtures():
    ids = np.arange(1, 51)
    policies = pd.DataFrame({"IDpol": ids, "Exposure": .2 + ids % 5 * .2,
                             "recorded_positive_count": np.tile([0, 1, 0, 2, 1], 10),
                             "VehAge": ids % 20, "VehGas": ["common"] * len(ids)})
    membership = v.make_membership(policies["IDpol"], v.protocol()["split"])
    development = policies.loc[policies["IDpol"].isin(membership.loc[membership["split"] == "development", "IDpol"])].copy()
    claims = pd.DataFrame([{"claim_row_id": f"claim-{i}-{j}", "IDpol": i}
                           for i, n in zip(ids, policies["recorded_positive_count"]) for j in range(n)])
    prepared = {"source_id": "fixture-source", "preparation_id": "fixture-preparation",
                "analysis_tables": {"policies": {"sha256": "policy-hash"}, "claims": {"sha256": "claim-hash"}}}
    return policies, development, membership, claims, prepared


class SplitContracts(unittest.TestCase):
    def test_key_only_split_stable_after_reordering_and_exact_fold_coverage(self):
        policies, _, membership, _, _ = fixtures()
        reordered = v.make_membership(policies["IDpol"].iloc[::-1], v.protocol()["split"])
        pd.testing.assert_frame_equal(membership, reordered)
        self.assertEqual(membership["IDpol"].nunique(), 50)
        self.assertEqual((membership["split"] == "evaluation").sum(), 10)
        self.assertEqual(membership.groupby("fold").size().to_dict(), {0: 10, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8})
        v.validate_membership(membership)

    def test_all_claims_inherit_their_policy_split_even_with_repeated_keys(self):
        _, _, membership, claims, _ = fixtures()
        linked = v.propagate_claims(claims.iloc[::-1], membership.iloc[::-1])
        self.assertEqual(set(linked["claim_row_id"]), set(claims["claim_row_id"]))
        expected = membership.set_index("IDpol")
        for row in linked.itertuples():
            self.assertEqual(row.split, expected.loc[row.IDpol, "split"])
            self.assertEqual(row.fold, expected.loc[row.IDpol, "fold"])
        bad = claims.copy()
        bad.loc[0, "IDpol"] = 999
        with self.assertRaisesRegex(ValueError, "inherit"):
            v.propagate_claims(bad, membership)

    def test_invalid_universe_and_membership_fail(self):
        for ids in (pd.Series([1, 1] * 10), pd.Series([1, 2, 3]), pd.Series([1, 1.2] * 10)):
            with self.subTest(ids=ids.tolist()), self.assertRaises(ValueError):
                v.make_membership(ids, v.protocol()["split"])
        _, _, membership, _, _ = fixtures()
        bad = membership.copy()
        bad.loc[bad["split"] == "evaluation", "fold"] = 1
        with self.assertRaisesRegex(ValueError, "fold"):
            v.validate_membership(bad)

    def test_freeze_repeat_recovers_registry_and_forbids_seed_fishing(self):
        policies, _, _, claims, prepared = fixtures()
        with workspace_tempdir() as directory:
            first, path = v.freeze_split(policies["IDpol"], claims, prepared, directory, v.protocol())
            second, again = v.freeze_split(policies["IDpol"].iloc[::-1], claims.iloc[::-1], prepared, directory, v.protocol())
            self.assertEqual(first, second)
            self.assertEqual(path, again)
            (directory / "frozen_split.json").unlink()
            recovered, _ = v.freeze_split(policies["IDpol"], claims, prepared, directory, v.protocol())
            self.assertEqual(first, recovered)
            v.open_split(directory, prepared, v.protocol())
            changed = copy.deepcopy(v.protocol())
            changed["split"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "already frozen"):
                v.freeze_split(policies["IDpol"], claims, prepared, directory, changed)
            # Nonmembership protocol amendments cannot silently reshuffle IDs.
            changed = copy.deepcopy(v.protocol())
            changed["scope"] = "different prose"
            repeated, _ = v.freeze_split(policies["IDpol"], claims, prepared, directory, changed)
            self.assertEqual(first["split_id"], repeated["split_id"])

    def test_corrupt_registry_claim_lineage_and_artifact_bytes_fail(self):
        policies, _, _, claims, prepared = fixtures()
        with workspace_tempdir() as directory:
            saved, path = v.freeze_split(policies["IDpol"], claims, prepared, directory, v.protocol())
            registry = directory / "frozen_split.json"
            original = registry.read_bytes()
            wrong = json.loads(original)
            wrong["identity"]["source_id"] = "wrong"
            registry.write_bytes(json_bytes(wrong))
            with self.assertRaisesRegex(ValueError, "registry"):
                v.open_split(directory, prepared, v.protocol())
            registry.write_bytes(original)
            wrong_prepared = copy.deepcopy(prepared)
            wrong_prepared["analysis_tables"]["claims"]["sha256"] = "different claims"
            with self.assertRaisesRegex(ValueError, "Prepared data"):
                v.open_split(directory, wrong_prepared, v.protocol())
            table = path / "policy_membership.parquet"
            table.write_bytes(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                v.open_split(directory, prepared, v.protocol())
            self.assertEqual(table.read_bytes(), b"corruption")


class FitSpy(TransformerMixin, BaseEstimator):
    fits = []

    def fit(self, X, y=None):
        type(self).fits.append({"index": X.index.tolist(), "mean": float(X["VehAge"].mean()),
                               "ages": X["VehAge"].tolist()})
        return self

    def transform(self, X):
        return X


class OOFContracts(unittest.TestCase):
    def setUp(self):
        self.policies, self.dev, self.membership, _, _ = fixtures()

    def baseline(self, development=None):
        return v.run_oof(self.dev if development is None else development, self.membership,
                         v.ExposureIntercept, "baseline", "fixture-run", features=("VehAge",))

    def test_exactly_one_oof_prediction_each_and_training_only_intercept(self):
        predictions, diagnostics = self.baseline()
        self.assertEqual(len(predictions), len(self.dev))
        self.assertTrue(predictions["IDpol"].is_unique)
        self.assertTrue(predictions["split"].eq("development").all())
        dev = self.dev.merge(self.membership, on="IDpol")
        for fold in range(1, 6):
            train = dev.loc[dev["fold"] != fold]
            rate = train["recorded_positive_count"].sum() / train["Exposure"].sum()
            self.assertTrue(predictions.loc[predictions["fold"] == fold, "annual_rate"].eq(rate).all())
            log = diagnostics[fold - 1]
            self.assertEqual(log["training_id_sha256"], logical_hash(train[["IDpol"]].sort_values("IDpol")))
        aligned = self.dev.merge(predictions, on="IDpol")
        pooled = poisson_deviance_per_exposure(aligned["recorded_positive_count"], aligned["annual_rate"], aligned["Exposure"])
        weighted = sum(d["deviance_per_exposure"] * d["validation_exposure_years"] for d in diagnostics) / self.dev["Exposure"].sum()
        self.assertAlmostEqual(pooled, weighted, places=13)

    def test_own_validation_outcomes_cannot_change_own_predictions(self):
        before, _ = self.baseline()
        held_ids = self.membership.loc[self.membership["fold"] == 1, "IDpol"]
        changed = self.dev.copy()
        changed.loc[changed["IDpol"].isin(held_ids), "recorded_positive_count"] += 50
        after, _ = self.baseline(changed)
        pd.testing.assert_frame_equal(before.loc[before["fold"] == 1].reset_index(drop=True),
                                      after.loc[after["fold"] == 1].reset_index(drop=True))

    def test_eval_rows_missing_policies_target_features_and_reused_ids_rejected(self):
        for bad in (self.policies, self.dev.iloc[:-1], pd.concat([self.dev, self.dev.iloc[:1]])):
            with self.subTest(rows=len(bad)), self.assertRaisesRegex(ValueError, "exactly development"):
                self.baseline(bad)
        for features in (("ClaimNb",), ("IDpol",), ("fold",), ("Exposure",), ("VehAge", "VehAge")):
            with self.subTest(features=features), self.assertRaisesRegex(ValueError, "prospective"):
                v.run_oof(self.dev, self.membership, v.ExposureIntercept, "model", "run", features=features)

    def test_fresh_pipeline_preprocessing_fits_training_only_handles_unknown(self):
        FitSpy.fits = []
        dev = self.dev.copy()
        # A level appears only in one validation fold, requiring explicit unknown handling.
        fold1 = self.membership.loc[self.membership["fold"] == 1, "IDpol"]
        dev.loc[dev["IDpol"].isin(fold1), "VehGas"] = "unseen-in-fold1-training"
        def factory():
            preprocess = ColumnTransformer([
                ("age", StandardScaler(), ["VehAge"]),
                ("gas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["VehGas"])])
            return Pipeline([("spy", FitSpy()), ("preprocess", preprocess),
                             ("model", PoissonRegressor(alpha=1., max_iter=200, tol=1e-8))])
        prediction, diagnostics = v.run_oof(dev, self.membership, factory, "probe", "fixture", features=("VehAge", "VehGas"))
        self.assertEqual(len(FitSpy.fits), 5)
        aligned = dev.merge(self.membership, on="IDpol")
        for fold, logged in enumerate(FitSpy.fits, 1):
            self.assertEqual(logged["ages"], aligned.loc[aligned["fold"] != fold, "VehAge"].tolist())
        self.assertEqual(diagnostics[0]["unknown_validation_categories"]["VehGas"], len(fold1))
        self.assertEqual(len(prediction), len(dev))

    def test_prediction_alignment_reordering_and_wrong_model_units_fail(self):
        prediction, _ = self.baseline()
        actual = v.align_predictions(prediction.iloc[::-1], self.membership, self.dev[["IDpol", "Exposure"]].iloc[::-1], "baseline", "fixture-run")
        pd.testing.assert_frame_equal(actual, prediction)
        for column, value in (("model_id", "other"), ("run_id", "other"), ("split", "evaluation"),
                              ("fold", 0), ("annual_rate", 0.), ("expected_count", 99.)):
            bad = prediction.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                v.align_predictions(bad, self.membership, self.dev[["IDpol", "Exposure"]], "baseline", "fixture-run")
        for bad in (prediction.iloc[:-1], pd.concat([prediction, prediction.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, "ID/model/run"):
                v.align_predictions(bad, self.membership, self.dev[["IDpol", "Exposure"]], "baseline", "fixture-run")

    def test_invalid_response_or_no_training_claims_fail_without_fake_rates(self):
        for column, bad in (("Exposure", 0), ("recorded_positive_count", .5)):
            dev = self.dev.astype({column: float})
            dev.loc[dev.index[0], column] = bad
            with self.assertRaisesRegex(ValueError, "response/exposure"):
                self.baseline(dev)
        dev = self.dev.copy()
        dev["recorded_positive_count"] = 0
        with self.assertRaisesRegex(ValueError, "No positive training claim rate"):
            self.baseline(dev)


if __name__ == "__main__":
    unittest.main()
