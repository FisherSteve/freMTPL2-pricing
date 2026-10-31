"""Public evidence must reject changed decisions, exclusions and duplicate scores."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.build_results import ROOT, collect, unique


class ReportingContracts(unittest.TestCase):
    def copy_evidence(self, destination):
        for relative in ["reports/s0/source_manifest.json", "reports/s4/scores.json", "reports/s4/frequency_report.json",
                "reports/s5/development_decision.json", "reports/s5/evaluation_release.json",
                "reports/s5/development_loss_scores.json", "reports/s5/evaluation_loss_scores.json",
                "reports/s5/evaluation_frequency_scores.json", "reports/s5/evaluation_report.json",
                "reports/s5/development_report.json", "reports/s5/reproducibility.json"]:
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)

    def test_duplicate_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            unique([{"model_id": "same"}, {"model_id": "same"}])

    def test_locked_chain_and_score_mutations_fail(self):
        mutations = [
            ("evaluation_report", lambda x: x.update(selection_changed_after_evaluation=True)),
            ("evaluation_report", lambda x: x["reconciliation"].update(excluded_claim_rows=1)),
            ("evaluation_release", lambda x: x.update(decision_sha256="changed")),
            ("evaluation_loss_scores", lambda x: x[0].update(tweedie_deviance_per_exposure=0)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=ROOT/"data") as directory:
                target = Path(directory)
                self.copy_evidence(target)
                path = target/f"reports/s5/{name}.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    collect(target)

    def test_summary_preserves_period_and_annual_units(self):
        result = collect(ROOT)
        row = result["selected_evaluation"]
        self.assertAlmostEqual(row["expected_loss_eur"]/row["exposure_years"],
                               row["predicted_annual_loss_cost"], places=11)
        self.assertAlmostEqual(row["observed_loss_eur"]/row["expected_loss_eur"],
                               row["actual_over_predicted"], places=12)
        self.assertEqual(result["selected_model_id"], "pp_hgb_mean")


if __name__ == "__main__":
    unittest.main()
