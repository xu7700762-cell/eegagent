"""Reject clipped endpoints even when the CSV claims completeness."""
import unittest

from .data import boundary_audit
from .physiology import physiological_evidence


class EligibilityAndPhysiology(unittest.TestCase):
    def test_eof_clipping_is_excluded(self):
        row = dict(start_sample_1024="2516402", end_sample_1024="2626240")
        result = boundary_audit(row, [(2516402, 20), (2645592, 22)], 2626240)
        self.assertFalse(result["raw_complete"])
        self.assertIn("end_event_after_eof", result["exclusion_reasons"])
        self.assertEqual(result["actual_end_event_sample"], 2645592)

    def test_exact_complete_event_pair_is_eligible(self):
        self.assertTrue(boundary_audit(dict(start_sample_1024="20", end_sample_1024="40"),
                                      [(20, 20), (40, 22)], 100)["raw_complete"])

    def test_intervening_start_is_not_a_complete_pair(self):
        self.assertFalse(boundary_audit(dict(start_sample_1024="20", end_sample_1024="50"),
                                       [(20, 20), (30, 20), (50, 22)], 100)["raw_complete"])

    def test_unsigned_covariance_and_spectrum_never_vote(self):
        for changes in ([.24, 0, -.12, 0], [-.24, 0, .12, 0]):
            result = physiological_evidence(dict(reference_log_change=changes),
                                            dict(distance=2.15), "valid", .1)
            self.assertFalse(result["verification"]["direction_validated"])
            self.assertIsNone(result["verification"]["conflict_with_deep_model"])
            self.assertEqual(result["verification"]["status"], "inconclusive")

    def test_missing_reference_does_not_create_low_evidence(self):
        result = physiological_evidence(dict(reference_log_change=[1, 2, 3, 4]), dict(distance=5))
        self.assertIsNone(result["biomarker"]["delta_change"])
        self.assertIsNone(result["covariance"]["distance"])


if __name__ == "__main__":
    unittest.main()
