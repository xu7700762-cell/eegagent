"""Synthetic retrieval/privacy tests; no EEG files, model training or APIs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .common import message_for, read_json, sha, write_json
from .evaluate import evaluate_retrieval_v2
from .prepare import prepare_v2, retrieve_balanced_examples
from .retrieval import CaseRetriever
from .validate import validate_v2


def evidence(value):
    return dict(temporal=dict(raw_mean_probability=value, raw_probability_std=.1,
        raw_recent_mean=value, raw_recent_slope_per_second=.001, raw_last_probability=value),
        qc_accepted_fraction=1., reliability=dict(p_cal=value))


def bank_fixture():
    bank, evaluation = [], {}
    for subject in range(5):
        for value, label in ((.1, 0), (.2, 0), (.8, 1), (.9, 1)):
            index = len(bank)
            bank.append(dict(path_index=index, role="meta", evidence=evidence(value)))
            evaluation[index] = dict(path_index=index, subject_key=subject, label=label)
    return bank, evaluation


class RetrievalContracts(unittest.TestCase):
    def test_true_top_k_preserves_unbalanced_neighborhood_and_subject_cap(self):
        bank, evaluation = bank_fixture()
        result = CaseRetriever(bank, evaluation)(evidence(.85))
        quality = result["retrieval_quality"]
        self.assertEqual((quality["high_count"], quality["low_count"]), (5, 0))
        self.assertEqual(quality["distinct_subject_count"], 5)
        self.assertEqual(len({evaluation[i]["subject_key"] for i in result["selected_indices"]}), 5)
        self.assertEqual(quality["distances"], sorted(quality["distances"]))
        self.assertNotIn("high_probability", quality)
        control, _ = retrieve_balanced_examples(evidence(.85), bank, evaluation)
        self.assertEqual(sum(example["observed_class"] == "high" for example in control), 4)
        self.assertEqual(sum(example["observed_class"] == "low" for example in control), 4)

    def test_query_subject_excluded_from_cases_scaling_and_threshold_labels(self):
        bank, evaluation = bank_fixture()
        first = CaseRetriever(bank, evaluation)(evidence(.85), query_subject=2)
        changed_bank, changed_evaluation = copy.deepcopy(bank), copy.deepcopy(evaluation)
        for record in changed_bank:
            if changed_evaluation[record["path_index"]]["subject_key"] == 2:
                record["evidence"] = evidence(1000.)
                changed_evaluation[record["path_index"]]["label"] ^= 1
        second = CaseRetriever(changed_bank, changed_evaluation)(evidence(.85), query_subject=2)
        self.assertEqual(first, second)
        self.assertTrue(all(evaluation[i]["subject_key"] != 2 for i in first["selected_indices"]))
        audit = CaseRetriever(bank, evaluation).calibration_audit()
        for fold in audit["folds"]:
            self.assertNotIn(fold["heldout_subject"], fold["training_subjects"])

    def test_selection_ignores_class_labels_and_fitted_scores_and_physiology(self):
        bank, evaluation = bank_fixture()
        before = CaseRetriever(bank, evaluation)(evidence(.85))["selected_indices"]
        altered, modified = copy.deepcopy(bank), copy.deepcopy(evaluation)
        for record in altered:
            record["evidence"].update(probabilities=dict(deep_probability=.01),
                                     relative_band_power=[1000.] * 4, covariance_distance=10000.)
            modified[record["path_index"]]["label"] ^= 1
        after = CaseRetriever(altered, modified)(evidence(.85))["selected_indices"]
        self.assertEqual(before, after)

    def test_internal_range_is_learned_and_far_query_is_rejected(self):
        bank, evaluation = bank_fixture()
        retriever = CaseRetriever(bank, evaluation)
        self.assertEqual(retriever(evidence(.8))["retrieval_quality"]["status"], "reliable")
        far = retriever(evidence(100.))["retrieval_quality"]
        self.assertFalse(far["reliable"])
        self.assertTrue(far["outside_reliable_range"])
        unvalidated = CaseRetriever(bank, evaluation, minimum_cases=100)(evidence(.8))["retrieval_quality"]
        self.assertEqual(unvalidated["status"], "unvalidated")
        self.assertIsNone(unvalidated["distance_threshold"])

    def test_missing_temporal_features_cannot_retrieve_using_qc_alone(self):
        bank, evaluation = bank_fixture()
        result = CaseRetriever(bank, evaluation)(dict(qc_accepted_fraction=1.))
        self.assertEqual(result["examples"], [])
        self.assertFalse(result["retrieval_quality"]["reliable"])
        control, indices = retrieve_balanced_examples(dict(qc_accepted_fraction=1.), bank, evaluation)
        self.assertEqual(control, [])
        self.assertEqual(indices, [])
        json.dumps(result, allow_nan=False)

    def test_mixed_evidence_and_model_disagreement_are_descriptive(self):
        bank, evaluation = bank_fixture()
        for index, source in evaluation.items():
            if source["subject_key"] in (0, 1):
                source["label"] ^= 1
        result = CaseRetriever(bank, evaluation)(evidence(.85))
        self.assertTrue(result["evidence_conflict"])
        self.assertEqual(result["retrieval_quality"]["high_count"], 3)
        self.assertEqual(result["retrieval_quality"]["low_count"], 2)
        message_for(evidence(.85), result["examples"])
        unanimous_bank, unanimous_eval = bank_fixture()
        disagreement = CaseRetriever(unanimous_bank, unanimous_eval)(evidence(.85), p_cal=.1)
        self.assertTrue(disagreement["conflict_with_deep_model"])

    def test_outer_record_cannot_enter_case_bank(self):
        bank, evaluation = bank_fixture()
        bank[0]["role"] = "outer_test"
        with self.assertRaises(ValueError):
            CaseRetriever(bank, evaluation)

    def test_v2_offline_prepare_validate_evaluate_and_outer_label_invariance(self):
        bank, evaluation = bank_fixture()
        records = [dict(path_index=record["path_index"], role="meta", evidence=record["evidence"]) for record in bank]
        for subject, role, value in ((5, "policy_validation", .85), (6, "outer_test", .15)):
            index = len(records)
            evaluation[index] = dict(path_index=index, subject_key=subject, label=1)
            records.append(dict(path_index=index, role=role, evidence=evidence(value)))
        fold = dict(outer_subject=6, split=dict(base_train=[], meta_calibration=list(range(5)),
                    policy_validation=[5], outer_test=[6]), records=records)
        with tempfile.TemporaryDirectory() as directory:
            cloud, out = Path(directory) / "cloud", Path(directory) / "refine"
            cloud.mkdir()
            write_json(cloud / "protocol.json", dict(schema_version="uncertainty_agent_v2", total_paths=22, subjects=7))
            write_json(cloud / "fold_evidence.json", [fold])
            write_json(cloud / "evaluation.json", list(evaluation.values()))
            prepare_v2(cloud, out)
            self.assertEqual(validate_v2(out)["single_query_jobs_checked"], 4)
            summary = evaluate_retrieval_v2(out)
            self.assertFalse(summary["llm_probability_modification"])
            self.assertEqual(summary["explanations"]["gpt"]["status"], "not_run")
            jobs = read_json(out / "jobs.json")
            evaluation[21]["label"] ^= 1
            write_json(cloud / "evaluation.json", list(evaluation.values()))
            second = Path(directory) / "second"
            prepare_v2(cloud, second)
            self.assertEqual(jobs, read_json(second / "jobs.json"))
            with self.assertRaises(FileExistsError):
                prepare_v2(cloud, second)
            implementation = Path(directory) / "frozen_implementation.py"
            implementation.write_text("original", encoding="utf-8")
            protocol = read_json(second / "protocol.json")
            protocol["implementation_sha256"][str(implementation)] = sha(implementation)
            write_json(second / "protocol.json", protocol)
            validate_v2(second)
            implementation.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_retrieval_v2(second)


if __name__ == "__main__":
    unittest.main()
