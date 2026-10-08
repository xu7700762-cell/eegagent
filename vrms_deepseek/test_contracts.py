"""Check native DeepSeek response handling, label isolation, and failure fallback."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vrms_deepseek.supervisor import assess_evidence, messages_for, normalize_response


class DeepSeekContracts(unittest.TestCase):
    def test_nested_test_label_is_rejected(self):
        with self.assertRaises(ValueError):
            messages_for({"nested": {"label": 1}}, [])
        content = json.loads(messages_for({"probability": .6}, [
            {"evidence": {"probability": .7}, "observed_class": "high"}])[1]["content"])
        self.assertEqual(len(content["queries"]), 1)
        self.assertEqual(content["queries"][0]["id"], "qsingle")

    def test_truncation_is_a_failure_even_if_json_parses(self):
        reply = {"predictions": [{"id": "qsingle", "high_probability": .7,
                                 "state": "high", "uncertain": True, "reason": "weak evidence"}]}
        with self.assertRaises(ValueError):
            normalize_response(reply, {"choices": [{"finish_reason": "length"}]})
        with self.assertRaises(ValueError):
            normalize_response(reply, {"choices": [{"finish_reason": "stop"}]})
        reply = {"predictions": [{"id": "qsingle", "supporting_evidence": [],
                                 "conflicting_evidence": [], "missing_evidence": ["reference"],
                                 "explanation": "Insufficient independently validated evidence"}]}
        self.assertEqual(normalize_response(reply, {"choices": [{"finish_reason": "stop"}]}), reply["predictions"][0])

    def test_api_failure_keeps_numerical_result_without_provider_fallback(self):
        provider = SimpleNamespace(model="deepseek-flash", thinking_disabled=True)
        def fail(*args, **kwargs):
            raise RuntimeError("Sensitive provider exception text must not be logged")
        provider.call = fail
        with tempfile.TemporaryDirectory() as tmp, patch("vrms_deepseek.supervisor.time.sleep"):
            path = Path(tmp) / "call.json"
            reliability = dict(p_raw=.62, p_cal=.61, calibration_status="fitted", policy_status="validated",
                               signal_quality_bad=False, prediction_reliable=False, ood=False,
                               evidence_conflict=False)
            pack = assess_evidence(provider, SimpleNamespace(response=None),
                {"base_url": "https://api.deepseek.com"}, {"p": .62}, [], .62, path,
                reliability=reliability)
            self.assertEqual(pack["high_probability"], .61)
            self.assertEqual(pack["state"], "uncertain")
            self.assertTrue(pack["fallback_used"])
            self.assertEqual(pack["effective_llm_weight"], 0)
            log = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(log["attempts"]), 2)
            self.assertEqual(log["model"], "deepseek-flash")
            self.assertNotIn("Sensitive", path.read_text(encoding="utf-8"))

    def test_quality_rejection_skips_provider(self):
        provider = SimpleNamespace(model="deepseek-flash")
        with patch("vrms_deepseek.supervisor.call_evidence") as call:
            result = assess_evidence(provider, None, {}, {}, [], None, "unused",
                reliability=dict(p_cal=None, signal_quality_bad=True))
        call.assert_not_called()
        self.assertEqual(result["state"], "insufficient_data")
        self.assertFalse(result["cloud_called"])


if __name__ == "__main__":
    unittest.main()
