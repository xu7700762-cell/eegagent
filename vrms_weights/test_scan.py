"""Adversarial boundaries and validation-only weight selection."""
from fractions import Fraction as F
import unittest

import numpy as np

from .scan import best_intervals, exact_cells, exact_decisions, select_grid_weight


class WeightScanContracts(unittest.TestCase):
    def test_isolated_boundary_maximum_is_not_lost(self):
        # Both paths are high; at exactly 1/2 both scores equal the threshold.
        # Each neighbouring interval misclassifies one of them.
        cells = exact_cells(np.array([1, 1]), [F("0.2"), F("0.8")], [F("0.8"), F("0.2")])
        best, intervals = best_intervals(cells)
        self.assertEqual(best, 2)
        self.assertEqual(intervals, [(F(1, 2), F(1, 2), True, True, 2)])

    def test_narrow_maximum_missing_from_integer_grid_is_found(self):
        # Path one turns high at .242; path two turns high at .248 and is low.
        n, c = [F("0.4758"), F("0.4752")], [F("0.5758"), F("0.5752")]
        y = np.array([1, 0])
        best, intervals = best_intervals(exact_cells(y, n, c))
        self.assertEqual(best, 2)
        self.assertEqual(intervals, [(F("0.242"), F("0.248"), True, False, 2)])
        grid_counts = [np.count_nonzero(exact_decisions(n, c, F(k, 100)) == y) for k in range(101)]
        self.assertEqual(max(grid_counts), 1)

    def test_decreasing_score_excludes_threshold_for_low_class(self):
        cells = exact_cells(np.array([0]), [F("0.7")], [F("0.3")])
        self.assertEqual(best_intervals(cells), (1, [(F(1, 2), F(1), False, True, 1)]))
        cells = exact_cells(np.array([1]), [F("0.5")], [F("0.5")])
        self.assertEqual(best_intervals(cells), (1, [(F(0), F(1), True, True, 1)]))

    def test_cloud_failure_and_acc_ties_preserve_no_weight(self):
        weight, correct = select_grid_weight([0, 1], [.2, .8], [np.nan, np.nan])
        self.assertEqual((weight, correct), (0, 2))

    def test_heldout_outcomes_cannot_choose_validation_weight(self):
        y = np.array([0, 1, 0, 1])
        n, c = np.array([.6, .4, .1, .9]), np.array([.2, .8, .9, .1])
        validation = np.array([0, 1])
        expected = select_grid_weight(y[validation], n[validation], c[validation])
        y[2:] = 1 - y[2:]
        n[2:], c[2:] = c[2:].copy(), n[2:].copy()
        self.assertEqual(select_grid_weight(y[validation], n[validation], c[validation]), expected)
        self.assertGreater(expected[0], 0)


if __name__ == "__main__":
    unittest.main()
