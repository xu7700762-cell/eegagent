"""V3 support, OOF lineage, independent audit and risk-target regressions."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from .reliability_v3 import fit_reliability_v3


class ReliabilityV3Contracts(unittest.TestCase):
    def inputs(self):
        groups = np.repeat(np.arange(1, 13), 8)
        y = np.tile([0, 1], 48)
        raw = np.where(y, .95, .05)
        descriptors = np.repeat(np.tile(np.linspace(-1., 1., 8), 12)[:, None], 60, axis=1)
        fractions = np.ones(len(y))
        roles = {name: np.flatnonzero(np.isin(groups, subjects)) for name, subjects in
                 [('base_indices', [1, 2]), ('meta_indices', [3, 4, 5, 6, 7]),
                  ('policy_indices', [8, 9, 10, 11])]}
        oof = np.full(len(y), np.nan)
        oof[roles['meta_indices']] = raw[roles['meta_indices']]
        lineage = [dict(heldout_subject=heldout, head_fit_subjects=[s for s in range(3, 8) if s != heldout],
            query_indices=np.flatnonzero(groups == heldout).tolist()) for heldout in range(3, 8)]
        return raw, y, groups, descriptors, fractions, dict(**roles,
            meta_oof_probability=oof, meta_oof_provenance=lineage)

    def fit(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        return fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)

    def test_valid_model_releases_only_the_independently_audited_candidate(self):
        model = self.fit()
        self.assertIsNotNone(model.thresholds)
        self.assertEqual(model.thresholds, model.candidate_thresholds)
        self.assertEqual(model.validation['status'], 'validated')
        self.assertEqual(model.validation['primary_selection_subjects'], [8, 9])
        self.assertEqual(model.validation['primary_audit_subjects'], [10, 11])
        self.assertLessEqual(model.validation['primary_audit']['mean_implied_risk'], .2)
        self.assertEqual(model.validation['primary_audit']['accepted_subjects'], 2)
        self.assertIsNotNone(model.validation['primary_audit']['subject_bootstrap_risk_95ci'])
        for row in model.validation['nested_folds']:
            self.assertNotIn(row['heldout_subject'], row['selection_subjects'])
            self.assertFalse(set(row['selection_indices']) & set(row['audit_indices']))
        result = model.assess(.95, model.center, 1., reference_available=False)
        self.assertTrue(result['prediction_reliable'])
        self.assertFalse(result['rejection_reasons'])
        self.assertEqual(result['signal_quality']['reference_quality'], 'unavailable')

    def test_head_cannot_self_validate_calibration(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        roles['meta_oof_provenance'][0]['head_fit_subjects'].append(3)
        with self.assertRaisesRegex(ValueError, 'heldout subject'):
            fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)

    def test_final_head_in_sample_meta_scores_cannot_change_calibration_fit(self):
        original = self.fit()
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        raw[roles['meta_indices']] = 1 - raw[roles['meta_indices']]
        changed = fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        self.assertEqual(original.calibrate(.95), changed.calibrate(.95))
        self.assertEqual(original.validation, changed.validation)

    def test_duplicate_or_incomplete_oof_provenance_is_rejected(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        roles['meta_oof_provenance'][0]['query_indices'].append(roles['meta_oof_provenance'][1]['query_indices'][0])
        with self.assertRaisesRegex(ValueError, 'complete subject'):
            fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)

    def test_policy_role_overlap_or_manual_partition_override_is_rejected(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        roles['policy_indices'] = np.r_[roles['policy_indices'], roles['meta_indices'][:1]]
        with self.assertRaisesRegex(ValueError, 'subject-disjoint'):
            fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles,
                               plan=dict(audit_subjects=[8, 9]))

    def test_audit_failure_cannot_reselect_or_publish_a_training_success(self):
        original = self.fit()
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        y[np.isin(groups, [10, 11])] ^= 1
        model = fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        self.assertEqual(model.candidate_thresholds, original.candidate_thresholds)
        self.assertIsNone(model.thresholds)
        self.assertIn('policy_independent_audit_failed', model.validation['failed_checks'])
        result = model.assess(.95, model.center, 1.)
        self.assertFalse(result['prediction_reliable'])
        self.assertIn('policy_unvalidated', result['rejection_reasons'])

    def test_near_half_group_cannot_use_perfect_labels_to_self_certify_tiny_margin(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        policy = roles['policy_indices']
        raw[policy] = np.where(y[policy], .500001, .499999)
        model = fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        self.assertIsNotNone(model.calibrator)
        self.assertIsNone(model.thresholds)
        self.assertIn('estimated_risk_exceeds_target', model.validation['failed_checks'])
        self.assertIn('policy_nested_validation_failed', model.validation['failed_checks'])
        result = model.assess(.500001, model.center, 1.)
        self.assertFalse(result['prediction_reliable'])
        self.assertGreater(result['individual_implied_risk'], .49)

    def test_coverage_warning_does_not_become_a_signal_quality_failure(self):
        model = self.fit()
        warning = model.assess(.95, model.center, .25)
        self.assertTrue(warning['coverage_warning'])
        self.assertFalse(warning['signal_quality_bad'])
        self.assertIn('QC_coverage_warning', warning['rejection_reasons'])
        self.assertFalse(warning['prediction_reliable'])
        invalid = model.assess(.95, model.center, 1., qc_passed=False)
        self.assertTrue(invalid['signal_quality_bad'])
        self.assertIn('QC_failed', invalid['rejection_reasons'])
        self.assertTrue(model.assess(None, None, 0.)['signal_quality_bad'])

    def test_qc_threshold_never_depends_on_policy_distribution_or_labels(self):
        original = self.fit()
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        fractions[roles['policy_indices']] = .1
        model = fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        self.assertEqual(original.qc_threshold, model.qc_threshold)
        self.assertEqual(model.qc_threshold, 1.)
        self.assertIsNone(model.thresholds)

    def test_outer_scores_labels_and_descriptors_cannot_change_any_parameter(self):
        original = self.fit()
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        outer = groups == 12
        raw[outer], y[outer], descriptors[outer], fractions[outer] = .999, 9, 1e6, .01
        roles['meta_oof_probability'][outer] = .001
        changed = fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        self.assertEqual(original.validation, changed.validation)
        self.assertEqual(original.thresholds, changed.thresholds)
        self.assertEqual(original.calibrate(.95), changed.calibrate(.95))
        np.testing.assert_array_equal(original.center, changed.center)

    def test_two_classes_require_independent_subject_support(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        for subject in (8, 9, 10, 11):
            y[groups == subject] = subject % 2
            raw[groups == subject] = .95 if subject % 2 else .05
        model = fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        self.assertIsNone(model.thresholds)
        self.assertIn('true_class_support_subjects_insufficient', model.validation['failed_checks'])
        self.assertIn('predicted_class_support_subjects_insufficient', model.validation['failed_checks'])

    def test_missing_predictions_return_all_reasons_without_inventing_pcal(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        roles['meta_oof_probability'][roles['meta_indices']] = np.nan
        model = fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles)
        result = model.assess(float('nan'), None, 1.)
        self.assertIsNone(result['p_cal'])
        self.assertFalse(result['signal_quality_bad'])
        self.assertTrue({'calibration_unavailable', 'prediction_unavailable', 'policy_unvalidated',
                         'OOD_reference_unavailable'}.issubset(result['rejection_reasons']))
        self.assertEqual(model.validation['calibration_status'], 'unavailable')

    def test_predeclared_accuracy_or_risk_goals_cannot_be_loosened(self):
        raw, y, groups, descriptors, fractions, roles = self.inputs()
        for override in (dict(maximum_estimated_risk=.5), dict(target_accuracy=.5), dict(minimum_cases=1)):
            with self.assertRaisesRegex(ValueError, 'cannot be relaxed'):
                fit_reliability_v3(raw, y, groups, descriptors, fractions, **roles, plan=override)

    def test_ood_and_margin_reasons_coexist_and_serialization_preserves_gate(self):
        model = self.fit()
        value = model.assess(.5, model.center + 100., 1.)
        self.assertFalse(value['prediction_reliable'])
        self.assertTrue({'OOD', 'margin_below_threshold'}.issubset(value['rejection_reasons']))
        self.assertFalse(value['signal_quality_bad'])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'reliability.joblib'
            joblib.dump(model, filename)
            reloaded = joblib.load(filename)
            self.assertEqual(value, reloaded.assess(.5, model.center + 100., 1.))


if __name__ == '__main__':
    unittest.main()
