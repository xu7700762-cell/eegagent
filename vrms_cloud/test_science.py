"""Guard training-only reference statistics and calibrated deep-only decisions."""
import unittest
import numpy as np

from vrms_cloud.evaluate import metric
from vrms_cloud.improve import tangent_features
from vrms_cloud.supervisor import summarize_assessment


class ScientificContracts(unittest.TestCase):
    def test_other_heldout_path_cannot_change_reference_or_earlier_path(self):
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(6, 30, 30))
        cov = x @ x.swapaxes(1, 2) + np.eye(30)[None]
        features, reference = tangent_features(cov, [0, 1, 2])
        cov[5] *= 100
        changed, changed_reference = tangent_features(cov, [0, 1, 2])
        np.testing.assert_array_equal(reference, changed_reference)
        np.testing.assert_array_equal(features[:5], changed[:5])

    def test_explanation_and_api_failure_preserve_the_same_deep_probability(self):
        gate = dict(p_cal=.49, calibration_status="fitted", policy_status="validated",
                    signal_quality_bad=False, prediction_reliable=False, ood=False,
                    evidence_conflict=True)
        for response in (dict(success=False, seconds=1.), dict(success=True, seconds=1., prediction=dict(
                supporting_evidence=["deep score near threshold"], conflicting_evidence=["mixed cases"],
                missing_evidence=[], explanation="The deep score remains unresolved"))):
            result = summarize_assessment({"reliability": gate}, .55, response)
            self.assertEqual(result["high_probability"], .49)
            self.assertEqual(result["state"], "uncertain")
            self.assertEqual(result["effective_llm_weight"], 0)

    def test_policy_must_be_validated_before_a_calibrated_class_can_be_released(self):
        gate = dict(p_cal=.99, calibration_status="fitted", policy_status="unvalidated",
                    signal_quality_bad=False, prediction_reliable=True, ood=False,
                    evidence_conflict=False)
        result = summarize_assessment({"reliability": gate}, .9)
        self.assertEqual(result["state"], "uncertain")

    def test_failure_and_single_class_are_reported_without_inventing_bacc(self):
        summary = metric(np.array([0, 1, 0]), np.array([.1, np.nan, .2]))
        self.assertEqual(summary["total_paths"], 3)
        self.assertEqual(summary["scored_paths"], 2)
        self.assertEqual(summary["acc"], 1)
        self.assertIsNone(summary["bacc"])


if __name__ == "__main__":
    unittest.main()
