import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from eegagent_config import api_settings
from vrms_cloud.cloud import existing_provider
from vrms_deepseek.supervisor import make_provider, call_evidence


class PortableProviders(unittest.TestCase):
    def test_missing_project_settings_do_not_read_application_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "EEG_GPT_KEY"):
                api_settings("EEG_GPT", config_file=None)

    def test_vendor_settings_are_separate_and_environment_overrides_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp)/"project.env"
            path.write_text('EEG_API_BASE_URL="https://api.deepseek.com"\nEEG_API_MODEL=deepseek-flash\nEEG_API_KEY=file-placeholder\n', encoding="utf-8")
            os.environ["EEG_API_KEY"] = "environment-placeholder"
            settings = api_settings("EEG_API", path)
            self.assertEqual(settings["key"], "environment-placeholder")
            with self.assertRaises(ValueError):
                api_settings("EEG_GPT", path)

    def test_responses_provider_uses_explicit_configuration(self):
        env = dict(EEG_CONFIG_FILE="missing-file.env", EEG_GPT_BASE_URL="https://example.test/v1/",
                   EEG_GPT_MODEL="configured-model", EEG_GPT_KEY="test-placeholder")
        with patch.dict(os.environ, env, clear=True):
            settings = existing_provider()
        self.assertEqual(settings["base_url"], "https://example.test/v1")
        self.assertEqual(settings["model"], "configured-model")
        self.assertEqual(settings["wire_api"], "responses")

    def test_deepseek_wire_settings_budget_and_credential_free_log(self):
        prediction = dict(id="qsingle", supporting_evidence=["synthetic deep score"],
                          conflicting_evidence=[], missing_evidence=["reference"],
                          explanation="Synthetic evidence remains incomplete")
        raw = dict(model="deepseek-flash", choices=[dict(finish_reason="stop",
                   message=dict(content=json.dumps(dict(predictions=[prediction]))))])
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(dict(predictions=[prediction]))))],
            model_dump=lambda **kwargs: raw)
        client = MagicMock()
        client.chat.completions.create.return_value = response
        create = client.chat.completions.create
        env = dict(EEG_API_BASE_URL="https://api.deepseek.com", EEG_API_MODEL="deepseek-flash", EEG_API_KEY="test-placeholder")
        with patch.dict(os.environ, env, clear=True), patch("openai.OpenAI", return_value=client) as constructor:
            provider, captured, public = make_provider(config_file=None, max_calls=1)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"call.json"
                result = call_evidence(provider, captured, public, {"probability": .6}, [], path)
                self.assertTrue(result["success"])
                self.assertNotIn("test-placeholder", path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(RuntimeError, "budget"):
                provider.call("CloudSupervisor", {}, messages=[])
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 60)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["max_tokens"], 700)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(provider.summary()["budget_rejections"], 1)
