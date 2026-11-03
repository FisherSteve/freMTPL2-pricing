"""Feature ablation must preserve the original model and training boundaries."""
import unittest
import numpy as np
import pandas as pd

from src.data import DataContractError
from src.feature_ablation import FeaturePoisson, weighted_corr, dependence
from src.interpretation import TwoFeaturePoisson


class FeatureAblationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = pd.DataFrame({"DrivAge": rng.integers(18, 85, 200),
            "BonusMalus": rng.integers(50, 180, 200), "Area": rng.choice(list("ABC"), 200)})
        self.w = rng.uniform(.1, 1.2, 200)
        self.y = rng.poisson(.2, 200)/self.w

    def test_exact_original_two_feature_predictions(self):
        for second in ("Area", "BonusMalus"):
            for family in ("glm", "gam"):
                x = self.x[["DrivAge", second]]
                original = TwoFeaturePoisson(second, family).fit(x, self.y, self.w)
                extended = FeaturePoisson(tuple(x), family).fit(x, self.y, self.w)
                np.testing.assert_allclose(original.predict(x), extended.predict(x), rtol=1e-12)

    def test_no_hidden_features_and_no_fitted_validation_ranges(self):
        for features in (("DrivAge",), ("BonusMalus",), ("Area",), ("BonusMalus", "Area")):
            for family in ("glm", "gam"):
                x = self.x[list(features)]
                model = FeaturePoisson(features, family).fit(x, self.y, self.w)
                with self.assertRaises(DataContractError):
                    model.predict(x.assign(ClaimNb=0))
                before = dict(model.ranges_)
                held = x.iloc[:2].copy()
                if "BonusMalus" in features:
                    held["BonusMalus"] = [1, 1000]
                if "Area" in features:
                    held["Area"] = ["Z", "A"]
                self.assertTrue(np.all(model.predict(held) > 0))
                self.assertEqual(before, model.ranges_)

    def test_unknown_area_is_training_reference(self):
        for family in ("glm", "gam"):
            model = FeaturePoisson(("Area",), family).fit(self.x[["Area"]], self.y, self.w)
            rates = model.predict(pd.DataFrame({"Area": ["Z", "A"]}))
            self.assertAlmostEqual(rates[0], rates[1])

    def test_weighted_dependence_is_not_numeric_area_coding(self):
        d = self.x.assign(Exposure=self.w)
        a = dependence(d)
        b = dependence(d.assign(Area=d.Area.map({"A": "Z", "B": "Y", "C": "X"})))
        self.assertAlmostEqual(a["Area_DrivAge_weighted_eta_squared"], b["Area_DrivAge_weighted_eta_squared"])
        self.assertAlmostEqual(weighted_corr([1, 2, 3], [3, 2, 1], [1, 2, 7]), -1)
        self.assertIsNone(weighted_corr([1, 1], [2, 3], [1, 1]))


if __name__ == "__main__":
    unittest.main()
