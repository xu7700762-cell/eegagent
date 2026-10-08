"""Meaningful stream, split, missing-evidence and lazy-execution checks."""
import unittest

import numpy as np

from vrms_pilot.data import CausalP0, covariance, covariance_features
from vrms_pilot.engine import adaptive_policy, consensus, signed_support, temporal_features
from vrms_pilot.experiment import make_splits, metrics


class Invariants(unittest.TestCase):
    def test_arbitrary_chunking_preserves_filter_and_decimation(self):
        x = np.random.default_rng(3).normal(size=(6001, 30))
        whole = CausalP0().process(x)
        stream = CausalP0()
        chunks = [x[:3], x[3:1004], x[1004:2019], x[2019:]]
        replay = np.concatenate([stream.process(c) for c in chunks])
        np.testing.assert_allclose(whole, replay, rtol=0, atol=1e-6)

    def test_future_changes_cannot_change_past_output(self):
        x = np.random.default_rng(7).normal(size=(4000, 30))
        future_changed = x.copy()
        future_changed[2000:] += 1000
        np.testing.assert_array_equal(CausalP0().process(x)[:500], CausalP0().process(future_changed)[:500])

    def test_scaled_covariance_is_unsigned_shift(self):
        x = np.random.default_rng(1).normal(size=(30, 1280))
        reference = covariance(x)
        same = covariance_features(x, reference)[-1]
        moved = covariance_features(2 * x, reference)[-1]
        self.assertLess(same, 1e-5)
        self.assertAlmostEqual(moved, np.sqrt(30) * np.log(4), places=4)

    def test_gap_breaks_positive_prediction_persistence(self):
        features = temporal_features([2.0, 2.0, 2.0], [0.0, 5.0, 20.0])
        self.assertAlmostEqual(features[-1], 1 / 12)

    def test_neutral_and_missing_are_not_conflict(self):
        neutral = consensus([.5, None, .5])
        self.assertIsNone(neutral["conflict_score"])
        self.assertEqual(neutral["status"], "insufficient_evidence")
        conflict = consensus([.99, .01])
        self.assertGreater(conflict["conflict_score"], .99)

    def test_threshold_centered_support_matches_class_direction(self):
        self.assertGreater(signed_support(.64, .62), 0)
        self.assertLess(signed_support(.60, .62), 0)
        self.assertEqual(signed_support(.62, .62), 0)

    def test_outer_and_three_inner_subject_sets_are_disjoint(self):
        paths = [dict(subject_key=s, label=y) for s in range(1, 25) for y in (0, 1)]
        splits = make_splits(paths)
        self.assertEqual(len(splits), 24)
        for split in splits:
            keys = ("base_train", "meta_calibration", "policy_validation", "outer_test")
            groups = [set(split[k]) for k in keys]
            self.assertEqual([len(g) for g in groups], [14, 5, 4, 1])
            self.assertEqual(len(set.union(*groups)), 24)
            self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]))

    def test_early_exit_does_not_request_extra_tools(self):
        called = []
        def score(name):
            called.append(name)
            return .9
        result = adaptive_policy(score, reliability=dict(p_cal=.9, prediction_reliable=True,
                                                        signal_quality_bad=False, ood=False))
        self.assertEqual(result["state"], "high")
        self.assertEqual(called, ["deep"])

    def test_neutral_evidence_cannot_publish_from_fusion_intercept(self):
        probabilities = {"deep": .5, "deep_temporal": .5, "biomarker": .5,
                         "covariance": .5, "deep_temporal_bio": .9, "full": .9}
        result = adaptive_policy(probabilities.get,
                                 {"deep_temporal": .1, "deep_temporal_bio": .1, "full": .1},
                                 reliability=dict(p_cal=.5, prediction_reliable=False, ood=False))
        self.assertEqual(result["state"], "uncertain")
        self.assertEqual(result["consensus"]["status"], "insufficient_evidence")

    def test_missing_scores_keep_denominator_and_are_not_low(self):
        m = metrics([0, 1, 0], [.2, np.nan, .3], [True, False, False])
        self.assertEqual(m["total_paths"], 3)
        self.assertEqual(m["scoreable_paths"], 2)
        self.assertEqual(m["published_paths"], 1)
        self.assertIsNone(m["auroc"])


if __name__ == "__main__":
    unittest.main()
