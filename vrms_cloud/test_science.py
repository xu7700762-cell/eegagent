"""Guard train-only normalization and validation-only blend selection."""
import unittest
import numpy as np

from vrms_cloud.evaluate import choose_blend, metric
from vrms_cloud.improve import tangent_features


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

    def test_cloud_failure_preserves_combined_decisions_and_prefers_no_weight(self):
        y = np.array([0, 1, 0, 1])
        numeric = np.array([.2, .8, .3, .7])
        selected = choose_blend(y, numeric, {"cloud": np.full(4, np.nan)})
        self.assertEqual(selected["weight"], 0)
        self.assertEqual(selected["mode"], "none")

    def test_cloud_weight_requires_actual_validation_improvement(self):
        y = np.array([0, 1, 0, 1])
        numeric = np.array([.7, .7, .3, .3])
        cloud = np.array([.1, .9, .1, .9])
        selected = choose_blend(y, numeric, {"cloud": cloud})
        self.assertGreater(selected["weight"], 0)
        self.assertEqual(selected["bacc"], 1)

    def test_failure_and_single_class_are_reported_without_inventing_bacc(self):
        summary = metric(np.array([0, 1, 0]), np.array([.1, np.nan, .2]))
        self.assertEqual(summary["total_paths"], 3)
        self.assertEqual(summary["scored_paths"], 2)
        self.assertEqual(summary["acc"], 1)
        self.assertIsNone(summary["bacc"])


if __name__ == "__main__":
    unittest.main()
