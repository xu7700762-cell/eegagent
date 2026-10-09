"""Scientific invariants of whole-path MIL and train-only feature normalization."""
import unittest

import numpy as np
import torch

from .model_ablation_v3 import PathMIL, fit_mil, predict_mil, role_indices


class PathMILTests(unittest.TestCase):
    def test_window_order_does_not_change_attention_bag_prediction(self):
        torch.manual_seed(2026)
        head = PathMIL(8).eval()
        values = torch.randn(11, 8)
        with torch.inference_mode():
            original = head(values)
            permuted = head(values[torch.randperm(len(values))])
        self.assertLess(float((original - permuted).abs()), 1e-6)

    def test_duplicate_entire_bag_does_not_give_more_label_weight(self):
        torch.manual_seed(2026)
        head = PathMIL(8).eval()
        values = torch.randn(7, 8)
        with torch.inference_mode():
            original = head(values)
            duplicated = head(torch.cat((values, values), dim=0))
        self.assertLess(float((original - duplicated).abs()), 1e-6)

    def test_base_only_normalization_and_training_ignore_outer_data(self):
        rng = np.random.default_rng(2026)
        features = rng.normal(size=(16, 8)).astype(np.float32)
        paths = [dict(subject_key=i+1, label=i % 2, accepted_windows=2,
                      window_start=2*i, window_end=2*i+2) for i in range(8)]
        base = np.arange(6)
        head_a, audit_a = fit_mil(features, paths, base, torch.device("cpu"), epochs=2)
        changed = features.copy()
        changed[12:] += 100000
        paths_changed = [dict(p) for p in paths]
        for p in paths_changed[6:]:
            p["label"] = 1-p["label"]
        head_b, audit_b = fit_mil(changed, paths_changed, base, torch.device("cpu"), epochs=2)
        self.assertEqual(audit_a["training_indices"], list(range(6)))
        self.assertEqual(audit_b["train_subjects"], list(range(1, 7)))
        self.assertEqual(audit_a["final_state_sha256"], audit_b["final_state_sha256"])
        self.assertTrue(torch.equal(head_a.center, head_b.center))
        self.assertTrue(torch.equal(head_a.scale, head_b.scale))

    def test_empty_path_remains_missing_in_denominator(self):
        paths = [dict(subject_key=1, label=0, accepted_windows=0, window_start=0, window_end=0),
                 dict(subject_key=2, label=1, accepted_windows=2, window_start=0, window_end=2)]
        p = predict_mil(PathMIL(8), np.zeros((2, 8), np.float32), paths, [0, 1], torch.device("cpu"))
        self.assertTrue(np.isnan(p[0]))
        self.assertTrue(np.isfinite(p[1]))

    def test_roles_use_frozen_subject_partition(self):
        paths = [dict(subject_key=s) for s in (1, 1, 2, 3, 3, 4)]
        split = dict(base_train=[1, 2], policy_validation=[3], outer_test=[4])
        self.assertEqual(role_indices(paths, split, "base_train").tolist(), [0, 1, 2])
        self.assertEqual(role_indices(paths, split, "policy_validation").tolist(), [3, 4])
        self.assertEqual(role_indices(paths, split, "outer_test").tolist(), [5])


if __name__ == "__main__":
    unittest.main()
