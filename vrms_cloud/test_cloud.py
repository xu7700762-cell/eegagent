"""Test blinded prompts and failure-sensitive parsing, not implementation mirrors."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vrms_cloud.cloud import SCHEMA, build_job, call_job, check_blind, parse_predictions
from vrms_cloud.supervisor import assess_evidence, summarize_assessment


class CloudContracts(unittest.TestCase):
    def test_query_label_changes_cannot_change_prompt(self):
        fold = dict(outer_subject=9, records=[dict(path_index=0, role="meta", evidence={"p": .4}),
                                            dict(path_index=1, role="outer_test", evidence={"p": .6}),
                                            dict(path_index=2, role="policy_validation", evidence={"p": .5})])
        evaluation = {i: dict(label=i % 2, subject_key=i + 1) for i in range(3)}
        first = build_job(fold, evaluation, "few_shot")
        evaluation[1]["label"] ^= 1
        evaluation[2]["label"] ^= 1
        second = build_job(fold, evaluation, "few_shot")
        self.assertEqual(first["user_message"], second["user_message"])
        self.assertEqual(first["example_indices"], [0])

    def test_private_fields_cannot_enter_nested_evidence(self):
        for key in ("label", "subject_key", "path_start_sec", "ssq"):
            with self.assertRaises(ValueError):
                check_blind({"nested": [{key: 0}]})

    def response(self, predictions, status="completed"):
        return dict(status=status, output=[dict(content=[dict(type="output_text", text=json.dumps(dict(predictions=predictions)))])])

    def prediction(self, ident="a"):
        return dict(id=ident, supporting_evidence=["deep calibrated probability is high"],
                    conflicting_evidence=["retrieved neighbours contain both classes"],
                    missing_evidence=["initial reference unavailable"], explanation="Evidence remains mixed")

    def test_partial_duplicate_or_invalid_response_is_failure(self):
        cases = [[self.prediction()], [self.prediction(), self.prediction()],
                 [{**self.prediction("a"), "high_probability": .99}, self.prediction("b")],
                 [{**self.prediction("a"), "supporting_evidence": "high"}, self.prediction("b")]]
        for case in cases:
            with self.assertRaises(ValueError):
                parse_predictions(self.response(case), ["a", "b"])

    def test_incomplete_response_cannot_count_as_cloud_result(self):
        with self.assertRaises(ValueError):
            parse_predictions(self.response([self.prediction()], "incomplete"), ["a"])
        self.assertEqual(len(parse_predictions(self.response([self.prediction()]), ["a"])), 1)

    def reliability(self, **overrides):
        result = dict(p_raw=.55, p_cal=.57, calibration_status="fitted", policy_status="validated",
                    signal_quality_bad=False, prediction_reliable=True, ood=False,
                    evidence_conflict=False)
        return {**result, **overrides}

    def test_schema_has_no_llm_probability_or_class(self):
        fields = SCHEMA["properties"]["predictions"]["items"]["properties"]
        self.assertEqual(set(fields), {"id", "supporting_evidence", "conflicting_evidence",
                                      "missing_evidence", "explanation"})

    def test_llm_cannot_change_probability_or_resolve_machine_uncertainty(self):
        reliability = self.reliability()
        reliability.update(prediction_reliable=False)
        response = dict(success=True, seconds=1., predictions=[dict(
            **self.prediction("qsingle"), high_probability=.99, state="high")])
        with patch("vrms_cloud.supervisor.existing_provider", return_value={}), \
             patch("vrms_cloud.supervisor.call_job", return_value=response):
            result = assess_evidence({"reliability": reliability}, [], .55, "unused", llm_weight=1.)
        self.assertEqual(result["high_probability"], .57)
        self.assertEqual(result["state"], "uncertain")
        self.assertEqual(result["effective_llm_weight"], 0)

    def test_reliable_and_bad_quality_paths_do_not_call_cloud(self):
        reliability = self.reliability()
        with patch("vrms_cloud.supervisor.call_job") as call:
            result = assess_evidence({"reliability": reliability}, [], .55, "unused")
            self.assertEqual(result["state"], "high")
            reliability.update(p_cal=.42)
            result = assess_evidence({"reliability": reliability}, [], .55, "unused")
            self.assertEqual(result["state"], "low")
            reliability.update(signal_quality_bad=True)
            result = assess_evidence({"reliability": reliability}, [], .55, "unused")
            self.assertEqual(result["state"], "insufficient_data")
        call.assert_not_called()

    def test_uncalibrated_score_is_not_published_as_calibrated_probability(self):
        result = summarize_assessment({"p": .95}, .95, dict(success=False, seconds=1.))
        self.assertEqual(result["state"], "uncertain")
        self.assertIsNone(result["high_probability"])
        self.assertEqual(result["p_raw"], .95)
        self.assertTrue(result["fallback_used"])

    def test_incomplete_quality_provenance_cannot_release_a_class(self):
        reliability = self.reliability()
        reliability.pop("signal_quality_bad")
        result = summarize_assessment({"reliability": reliability}, .55)
        self.assertEqual(result["state"], "uncertain")

    def test_model_probability_survives_api_failure_without_class_override(self):
        reliability = self.reliability()
        reliability.update(ood=True, prediction_reliable=False)
        result = summarize_assessment({"reliability": reliability}, .55, dict(success=False, seconds=2.))
        self.assertEqual(result["state"], "uncertain")
        self.assertEqual(result["high_probability"], .57)
        self.assertTrue(result["fallback_used"])

    def test_old_or_forged_probability_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "call.json"
            path.write_text(json.dumps(dict(request_sha256="same", success=True, predictions=[
                dict(id="a", high_probability=.9, state="high", uncertain=False, reason="old score")
            ])), encoding="utf-8")
            with self.assertRaises(ValueError):
                call_job({}, dict(request_sha256="same", mapping={"a": {}}), path)

    def test_cache_is_not_reused_after_model_or_endpoint_changes(self):
        provider = dict(model="model-one", provider="configured_responses", base_url="https://one.test/v1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "call.json"
            path.write_text(json.dumps(dict(request_sha256="same", success=True,
                predictions=[self.prediction()], **provider)), encoding="utf-8")
            job = dict(request_sha256="same", mapping={"a": {}})
            self.assertTrue(call_job(provider, job, path)["success"])
            for field, value in (("model", "model-two"), ("base_url", "https://two.test/v1"),
                                 ("provider", "different_responses")):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provider or model changed"):
                    call_job({**provider, field: value}, job, path)


if __name__ == "__main__":
    unittest.main()
