"""Guard reference leakage, retrieval inputs and single-query contracts."""
import copy
import json
import unittest

import numpy as np

from .common import api_job, message_for, select_source_indices
from .prepare import head_crossfit, retrieve_examples
from .evaluate import fit_predict, cv_predict, feature_matrix, selective_threshold


def evidence(value):
    return dict(temporal=dict(raw_mean_probability=value, raw_probability_std=.1, raw_recent_mean=value,
                raw_recent_slope_per_second=.001, raw_last_probability=value),
                relative_band_power=[.25, .25, .25, .25], reference_log_change=None,
                covariance_distance=None, qc_accepted_fraction=1., reference_available=False,
                probabilities=dict(deep_probability=value))


class ReferenceContracts(unittest.TestCase):
    def test_heldout_meta_labels_cannot_change_its_calibration_prediction(self):
        x = np.arange(20, dtype=float).reshape(-1, 1) / 20
        y = np.tile([0, 1], 10)
        groups = np.repeat(np.arange(5), 4)
        p, _ = head_crossfit(x, y, groups, np.arange(20), .1)
        changed = y.copy()
        changed[groups == 2] = 1 - changed[groups == 2]
        q, audit = head_crossfit(x, changed, groups, np.arange(20), .1)
        np.testing.assert_array_equal(p[groups == 2], q[groups == 2])
        for fold in audit["folds"]:
            self.assertNotIn(fold["heldout_subject"], fold["training_subjects"])

    def test_retrieval_ignores_label_fitted_model_probabilities(self):
        bank = [dict(path_index=i, role="meta", evidence=evidence(i / 10)) for i in range(10)]
        evaluation = {i: dict(subject_key=i // 2, label=i % 2) for i in range(10)}
        _, before = retrieve_examples(evidence(.45), bank, evaluation)
        modified = copy.deepcopy(bank)
        for r in modified:
            r["evidence"]["probabilities"]["deep_probability"] = 1 - r["evidence"]["probabilities"]["deep_probability"]
        _, after = retrieve_examples(evidence(.45), modified, evaluation)
        self.assertEqual(before, after)

    def test_unauthorized_example_subject_is_rejected(self):
        records = [dict(path_index=0, role="meta", evidence=evidence(.2))]
        with self.assertRaises(ValueError):
            select_source_indices(records, {1, 2}, {0: dict(subject_key=99)}, "meta")

    def test_ground_truth_and_identity_are_rejected(self):
        for key in ("label", "subject_key", "path_score", "path_index"):
            corrupted = evidence(.2)
            corrupted[key] = 1
            with self.assertRaises(ValueError):
                message_for(corrupted, [])

    def test_api_job_has_one_anonymous_query_and_no_source_identity(self):
        message = message_for(evidence(.2), [dict(evidence=evidence(.3), observed_class="low")])
        j = api_job(dict(message=message, outer_subject=99, path_index=123, role="outer_test"))
        self.assertIsNone(j["outer_subject"])
        self.assertEqual(j["example_indices"], [])
        self.assertEqual(j["mapping"], {"qsingle": {"role": "single_case"}})
        self.assertEqual(len(json.loads(j["user_message"])["queries"]), 1)

    def test_target_features_do_not_fit_the_fusion_scaler(self):
        train = dict(numeric=np.array([.1,.2,.3,.4,.6,.7,.8,.9]), enhanced=np.array([.2,.3,.2,.4,.7,.6,.8,.7]))
        target = dict(numeric=np.array([.4,.8]), enhanced=np.array([.6,.7]))
        spec = dict(name="fusion", kind="logistic", features="two_probabilities", C=.1)
        y = np.array([0,0,0,0,1,1,1,1])
        _, first = fit_predict(spec,train,y,target,[0,.25,1])
        altered = dict(numeric=np.array([.001,.999]), enhanced=np.array([.999,.001]))
        _, second = fit_predict(spec,train,y,altered,[0,.25,1])
        np.testing.assert_array_equal(first["estimator"][0].mean_,second["estimator"][0].mean_)
        np.testing.assert_array_equal(first["estimator"][1].coef_,second["estimator"][1].coef_)

    def test_group_cv_prediction_excludes_the_target_subject_labels(self):
        data = dict(numeric=np.linspace(.1,.9,16),enhanced=np.linspace(.2,.8,16))
        y = np.tile([0,1],8)
        groups = np.repeat(np.arange(4),4)
        spec = dict(name="fusion",kind="logistic",features="two_probabilities",C=.1)
        first = cv_predict(spec,data,y,groups,[0,.25,1])
        modified=y.copy();modified[groups==1]=1-modified[groups==1]
        second=cv_predict(spec,data,modified,groups,[0,.25,1])
        np.testing.assert_array_equal(first[groups==1],second[groups==1])

    def test_no_selective_threshold_is_invented_when_target_accuracy_fails(self):
        y=np.tile([0,1],10);p=np.full(20,.8)
        plan=dict(selective_confidence_thresholds=[.5,.6,.7,.8,.9],selective_minimum_validation_cases=8,
                  selective_minimum_validation_coverage=.7,selective_target_validation_accuracy=.8)
        self.assertIsNone(selective_threshold(y,p,plan))


if __name__ == "__main__":
    unittest.main()
