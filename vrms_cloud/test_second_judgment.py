"""Blinding, forced-class parsing and safe API-failure tests; no live API."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from .cloud import check_blind
from .second_judgment import call_job, parse_response, request_hash


def prediction():
    return dict(id="qsingle", decision="low", confidence="low", supporting_evidence=[],
                conflicting_evidence=[], missing_evidence=["校准缺失"], explanation="仅作强制分类实验。")


def response(p=None):
    return dict(status="completed", output=[dict(type="message", content=[dict(
                type="output_text", text=json.dumps(prediction() if p is None else p))])])


class SecondJudgmentTests(unittest.TestCase):
    def test_forced_decision_and_no_extra_probability(self):
        self.assertEqual(parse_response(response())["decision"], "low")
        for decision in ("uncertain", "insufficient_data"):
            with self.assertRaises(ValueError):
                parse_response(response(dict(prediction(), decision=decision)))
        with self.assertRaises(ValueError):
            parse_response(response(dict(prediction(), probability=.9)))
        with self.assertRaises(ValueError):
            parse_response(response(dict(prediction(), id="qother")))

    def test_incomplete_and_refusal_rejected(self):
        with self.assertRaises(ValueError):
            parse_response(dict(response(), status="incomplete"))
        with self.assertRaises(ValueError):
            parse_response(dict(status="completed", output=[dict(content=[dict(type="refusal")])]))

    def test_nested_query_blinding(self):
        for field in ("label", "score", "subject_key", "path_index", "episode_handle"):
            with self.assertRaises(ValueError):
                check_blind(dict(query=dict(evidence={field: 1})))
        check_blind(dict(query=dict(id="qsingle"), examples=[dict(observed_class="high")]))

    def test_failure_is_recorded_without_secret(self):
        provider = dict(model="test_model", base_url="http://localhost/v1", key="DO_NOT_SAVE_THIS")
        message = json.dumps(dict(query=dict(id="qsingle", evidence={})))
        job = dict(path_index=0, user_message=message, request_sha256=request_hash(message))
        with tempfile.TemporaryDirectory() as folder, patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")), patch("time.sleep"):
            destination = Path(folder) / "failure.json"
            result = call_job(provider, job, destination)
            self.assertFalse(result["success"])
            self.assertEqual(len(result["attempts"]), 2)
            self.assertNotIn("prediction", result)
            self.assertNotIn(provider["key"], destination.read_text(encoding="utf-8"))
            call_job(provider, job, destination)
            with self.assertRaises(ValueError):
                call_job(dict(provider, model="other"), job, destination)


if __name__ == "__main__":
    unittest.main()
