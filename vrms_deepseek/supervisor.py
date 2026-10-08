"""Standalone DeepSeek Supervisor using explicit project API settings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import threading
import time

from vrms_cloud.cloud import SCHEMA, SYSTEM, check_blind, parse_predictions, write_json
from vrms_cloud.supervisor import summarize_assessment, validate_inputs

CONFIG_FILE = Path(os.environ.get("EEG_CONFIG_FILE", ".env"))
OUTPUT_FORMAT = "\nReturn a JSON object satisfying this schema: " + json.dumps(SCHEMA, separators=(",", ":"))


def make_provider(config_file=CONFIG_FILE, max_calls=300):
    from eegagent_config import api_settings
    from .provider import CloudProvider
    settings = api_settings("EEG_API", config_file)
    if "deepseek" not in settings["base_url"].lower():
        raise ValueError("Expected the project DeepSeek endpoint; no Codex fallback")
    public = dict(configured=True, base_url=settings["base_url"], model_id=settings["model"])
    provider = CloudProvider(settings, max_calls=max_calls)
    captured = threading.local()
    create = provider.client.chat.completions.create
    def capture(**kwargs):
        response = create(**kwargs)
        captured.response = response.model_dump(exclude_none=True)
        return response
    provider.client.chat.completions.create = capture
    return provider, captured, public


def messages_for(evidence, examples):
    check_blind(evidence)
    for example in examples:
        check_blind(example["evidence"])
        if example["observed_class"] not in ("high", "low"):
            raise ValueError("Invalid internal example class")
    message = json.dumps(dict(task="Explain this anonymous path's support, conflict and missing evidence",
                              examples=examples, queries=[dict(id="qsingle", evidence=evidence)]),
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return [dict(role="system", content=SYSTEM + OUTPUT_FORMAT), dict(role="user", content=message)]


def normalize_response(reply, raw):
    if raw.get("choices", [{}])[0].get("finish_reason") != "stop":
        raise ValueError("Incomplete DeepSeek response")
    response = dict(status="completed", output=[dict(content=[dict(type="output_text", text=json.dumps(reply))])])
    return parse_predictions(response, ["qsingle"])[0]


def call_evidence(provider, captured, public, evidence, examples, destination):
    messages = messages_for(evidence, examples)
    request = dict(model=provider.model, messages=messages, temperature=0, max_tokens=700,
                   response_format={"type": "json_object"})
    if provider.thinking_disabled:
        request["thinking"] = {"type": "disabled"}
    sha = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    destination = Path(destination)
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous["request_sha256"] != sha or previous["base_url"] != public["base_url"]:
            raise ValueError("Cached DeepSeek request differs; use a new output directory")
        if previous.get("success"):
            normalize_response(dict(predictions=[previous["prediction"]]),
                               dict(choices=[dict(finish_reason="stop")]))
        return previous
    log = dict(provider="project_deepseek", model=provider.model, base_url=public["base_url"],
               request=request, request_sha256=sha, success=False, attempts=[])
    start = time.perf_counter()
    for attempt in range(2):
        captured.response = None
        begin = time.perf_counter()
        try:
            reply = provider.call("CloudSupervisor", {}, messages=messages)
            raw = captured.response
            prediction = normalize_response(reply, raw)
            log["attempts"].append(dict(attempt=attempt + 1, status="success",
                                       seconds=time.perf_counter() - begin, response=raw))
            log.update(success=True, prediction=prediction, response_model=raw.get("model"))
            break
        except Exception as exc:
            # Provider exception strings may include credentials; store type only.
            record = dict(attempt=attempt + 1, status="failed", error_type=type(exc).__name__,
                          seconds=time.perf_counter() - begin)
            if captured.response is not None:
                record["response"] = captured.response
            log["attempts"].append(record)
            if attempt == 0:
                time.sleep(1)
    log["seconds"] = time.perf_counter() - start
    write_json(destination, log)
    return log


def assess_evidence(provider, captured, public, evidence, examples, numeric_probability,
                    log_path, llm_weight=None, *, reliability=None, decision=None):
    context = validate_inputs(evidence, examples, numeric_probability, llm_weight, reliability, decision)
    if context["state"] != "uncertain":
        result = summarize_assessment(evidence, numeric_probability, reliability=reliability, decision=decision)
        return {**result, "configured_model": provider.model, "response_model": None}
    response = call_evidence(provider, captured, public, evidence, examples, log_path)
    result = summarize_assessment(evidence, numeric_probability, response,
                                  reliability=reliability, decision=decision)
    return {**result, "configured_model": response["model"], "response_model": response.get("response_model")}
