import unittest
import numpy as np

from .agent_tools_v1 import candidate_oof, case_predictions, indices
from .agent_policy_v1 import apply_rule, audit_rule


class AgentToolContracts(unittest.TestCase):
    def test_oof_does_not_use_heldout_or_outer_labels(self):
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(48, 7))
        groups = np.repeat(np.arange(8), 6)
        y = (x[:, 0] > 0).astype(int)
        base = np.flatnonzero(groups < 3)
        meta = np.flatnonzero((groups >= 3) & (groups < 7))
        p = candidate_oof(x, y, groups, base, meta, "lr_0.1")
        poisoned = y.copy()
        poisoned[groups >= 7] = 1 - poisoned[groups >= 7]
        q = candidate_oof(x, poisoned, groups, base, meta, "lr_0.1")
        np.testing.assert_array_equal(p, q)
        # A meta subject's own labels cannot affect its OOF scores.
        poisoned = y.copy()
        poisoned[groups == 3] = 1 - poisoned[groups == 3]
        q = candidate_oof(x, poisoned, groups, base, meta, "lr_0.1")
        np.testing.assert_array_equal(p[groups[meta] == 3], q[groups[meta] == 3])

    def test_cases_are_subject_diverse_and_not_balanced(self):
        x = np.arange(10, dtype=float)[:, None]
        y = np.asarray([1] * 8 + [0, 0])
        groups = np.repeat(np.arange(5), 2)
        _, retrieved = case_predictions(x, y, groups, np.arange(8), np.array([9]), (3, False))
        chosen = [i for i, _ in retrieved[0]]
        self.assertEqual(len(set(groups[chosen])), 3)
        self.assertEqual(set(y[chosen]), {1})
        self.assertNotIn(9, chosen)

    def test_nested_correction_does_not_approve_harmful_flips(self):
        y = np.tile([0, 1], 12)
        baseline = np.where(y == 1, .9, .1)
        tools = np.tile((1-baseline)[:, None], (1, 6))
        result = audit_rule(y, baseline, tools, np.repeat(np.arange(4), 6))
        self.assertFalse(result["enabled"])
        np.testing.assert_array_equal(apply_rule(baseline, tools, result["rule"]), baseline)


if __name__ == "__main__":
    unittest.main()
