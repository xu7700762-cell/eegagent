"""Leakage, abstention, calibrated-score preservation and actual lazy replay."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from .engine import adaptive_policy
from .experiment import ToolReplay
from .reliability import ReliabilityModel, fit_reliability


class ReliabilityContracts(unittest.TestCase):
    def inputs(self):
        groups = np.repeat(np.arange(1, 10), 8)
        y = np.tile([0, 1], 36)
        p = np.where(y, .85, .15)
        descriptor = np.tile(np.linspace(-1., 1., 8), 9)[:, None]
        descriptor = np.repeat(descriptor, 60, axis=1)
        fractions = np.ones(len(y))
        roles = {name: np.flatnonzero(np.isin(groups, subjects)) for name, subjects in
                 [('base_indices', [1, 2]), ('head_indices', [3, 4]),
                  ('calibration_indices', [5, 6]), ('policy_indices', [7, 8])]}
        return p, y, groups, descriptor, fractions, roles

    def fit(self):
        p, y, groups, descriptor, fraction, roles = self.inputs()
        return fit_reliability(p, y, groups, descriptor, fraction, **roles)

    def test_calibration_cannot_use_head_fit_subjects(self):
        p, y, groups, descriptor, fraction, roles = self.inputs()
        roles['calibration_indices'] = roles['head_indices']
        with self.assertRaisesRegex(ValueError, 'subject-disjoint'):
            fit_reliability(p, y, groups, descriptor, fraction, **roles)

    def test_outer_predictions_labels_and_signals_cannot_change_fitted_gates(self):
        p, y, groups, descriptor, fraction, roles = self.inputs()
        original = fit_reliability(p, y, groups, descriptor, fraction, **roles)
        outer = groups == 9
        p[outer], y[outer], descriptor[outer], fraction[outer] = .99, 1, 1e6, .01
        changed = fit_reliability(p, y, groups, descriptor, fraction, **roles)
        self.assertIsNotNone(original.thresholds)
        self.assertEqual(original.thresholds, changed.thresholds)
        self.assertEqual(original.calibrate(.85), changed.calibrate(.85))
        np.testing.assert_array_equal(original.center, changed.center)

    def test_missing_fit_is_uncertain_even_for_extreme_raw_probability(self):
        reliability = ReliabilityModel().assess(.99, np.zeros(60), 1.)
        result = adaptive_policy(lambda name: {} if name == 'case_retrieval' else .99,
                                 reliability=reliability)
        self.assertIsNone(result['probability'])
        self.assertEqual(result['state'], 'uncertain')

    def test_quality_and_ood_reject_with_empirical_thresholds(self):
        model = self.fit()
        descriptor = model.center
        good = model.assess(.85, descriptor, 1.)
        self.assertTrue(good['prediction_reliable'])
        self.assertTrue(model.assess(.85, descriptor, .1)['signal_quality_bad'])
        shifted = model.assess(.85, descriptor + 100., 1.)
        self.assertTrue(shifted['ood'])
        self.assertFalse(shifted['prediction_reliable'])
        self.assertNotIn('head_fit_subjects', good['provenance'])

    def test_pcal_unchanged_when_cases_and_physiology_disagree(self):
        calls = []
        def score(name):
            calls.append(name)
            return dict(retrieval_quality=dict(reliable=False), evidence_conflict=True) if name == 'case_retrieval' else .01
        result = adaptive_policy(score, reliability=dict(p_cal=.55, prediction_reliable=False, ood=False))
        self.assertEqual(result['probability'], .55)
        self.assertEqual(result['state'], 'uncertain')
        self.assertEqual(calls, ['deep', 'case_retrieval', 'biomarker', 'covariance'])

    def test_reliable_cases_do_not_eagerly_run_physiology(self):
        calls = []
        def score(name):
            calls.append(name)
            return dict(retrieval_quality=dict(reliable=True), evidence_conflict=False) if name == 'case_retrieval' else .55
        result = adaptive_policy(score, reliability=dict(p_cal=.55, prediction_reliable=False, ood=False))
        self.assertEqual(result['state'], 'uncertain')
        self.assertEqual(calls, ['deep', 'case_retrieval'])

    def test_bad_quality_stops_before_model_and_tools(self):
        result = adaptive_policy(lambda name: self.fail('tool executed'),
                                 reliability=dict(p_cal=None, signal_quality_bad=True))
        self.assertEqual(result['state'], 'insufficient_data')

    def test_calibration_and_gates_survive_joblib_reload(self):
        model = self.fit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bundle.joblib'
            joblib.dump(model, path)
            loaded = joblib.load(path)
        self.assertEqual(model.assess(.85, model.center, 1.), loaded.assess(.85, loaded.center, 1.))

    def test_tool_replay_fast_exit_only_runs_quality_deep_and_reliability(self):
        class FixedHead:
            def predict_proba(self, x):
                return np.tile([.15, .85], (len(x), 1))
        class FixedCNN(torch.nn.Module):
            def forward(self, x):
                return torch.full((len(x),), 2.)
        model = self.fit()
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(2, 30, 1280)).astype(np.float32)
        from .reliability import signal_descriptor
        model.center = signal_descriptor(x)
        model.scale = np.ones(60)
        bundle = dict(cnn=FixedCNN(), mean=torch.zeros(1, 30, 1), scale=torch.ones(1, 30, 1),
                      meta={'deep': FixedHead()}, reliability=model)
        path = dict(episode_handle='synthetic', path_end_sec=10., total_possible_windows=2)
        replay = ToolReplay(path, x, [0., 5.], np.eye(30), np.ones((30, 4)), bundle, torch.device('cpu'))
        result = replay.assess('adaptive')
        self.assertEqual(result['result']['state'], 'high')
        self.assertEqual(set(result['tool_calls']), {'SignalQualityChecker', 'DeepVRMSDetector', 'UncertaintyEvaluator'})
        self.assertNotIn('CaseRetriever', replay.memo)
        self.assertNotIn('BiomarkerCalculator', replay.memo)

    def test_tool_replay_quality_gate_skips_cnn_and_expensive_tools(self):
        class ForbiddenCNN(torch.nn.Module):
            def forward(self, x):
                raise AssertionError('low quality input reached CNN')
        x = np.random.default_rng(2026).normal(size=(1, 30, 1280)).astype(np.float32)
        bundle = dict(cnn=ForbiddenCNN(), reliability=self.fit())
        path = dict(episode_handle='synthetic', path_end_sec=20., total_possible_windows=4)
        replay = ToolReplay(path, x, [0.], np.eye(30), np.ones((30, 4)), bundle, torch.device('cpu'))
        pack = replay.assess('adaptive')
        self.assertEqual(pack['result']['state'], 'insufficient_data')
        self.assertIsNone(pack['result']['probability'])
        self.assertEqual(set(pack['tool_calls']), {'SignalQualityChecker', 'UncertaintyEvaluator'})
        snapshot = replay.evidence_snapshot()
        self.assertEqual(snapshot['qc_accepted_fraction'], .25)
        self.assertNotIn('DeepVRMSDetector', replay.calls)

    def test_empty_windows_are_insufficient_and_invalid_reference_is_unavailable(self):
        x = np.empty((0, 30, 1280), np.float32)
        path = dict(episode_handle='synthetic', path_end_sec=10., total_possible_windows=2)
        replay = ToolReplay(path, x, [], None, None, {}, torch.device('cpu'))
        pack = replay.assess('adaptive')
        self.assertFalse(pack['qc_passed'])
        self.assertFalse(pack['reference_available'])
        self.assertEqual(pack['result']['state'], 'insufficient_data')
        self.assertIsNone(pack['reliability']['p_cal'])
        self.assertNotIn('DeepVRMSDetector', replay.calls)
        invalid = ToolReplay(path, x, [], -np.eye(30), np.zeros((30, 4)), {}, torch.device('cpu'))
        self.assertFalse(invalid.reference_available)
        self.assertFalse(invalid.reference_power_available)

    def test_unsigned_physiology_exists_without_auxiliary_classifier(self):
        class FixedCNN(torch.nn.Module):
            def forward(self, x):
                return torch.full((len(x),), 2.)
        x = np.random.default_rng(2026).normal(size=(2, 30, 1280)).astype(np.float32)
        bundle = dict(cnn=FixedCNN(), mean=torch.zeros(1, 30, 1), scale=torch.ones(1, 30, 1), meta={},
                      classifiers=dict(bio_absolute=None, bio_reference=None, covariance=None))
        path = dict(episode_handle='synthetic', path_end_sec=10., total_possible_windows=2)
        replay = ToolReplay(path, x, [0., 5.], np.eye(30), np.ones((30, 4)), bundle, torch.device('cpu'))
        pack = replay.assess('adaptive')
        physiology = pack['physiological_evidence']
        self.assertIsNotNone(physiology['biomarker']['delta_change'])
        self.assertIsNotNone(physiology['covariance']['distance'])
        self.assertEqual(physiology['verification']['status'], 'inconclusive')
        self.assertFalse(physiology['verification']['direction_validated'])
        self.assertIsNone(pack['result']['probability'])
        self.assertEqual(pack['result']['state'], 'uncertain')


if __name__ == '__main__':
    unittest.main()
