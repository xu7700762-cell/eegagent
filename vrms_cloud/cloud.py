"""Explicitly configured Responses API client for bounded EEG evidence calls."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request


DEFAULT_OUT = Path("outputs/vrms_cloud/v2_seed2026")
MODES = ("zero_shot", "few_shot", "improved_few_shot")
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"predictions": {
        "type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                "conflicting_evidence": {"type": "array", "items": {"type": "string"}},
                "missing_evidence": {"type": "array", "items": {"type": "string"}},
                "explanation": {"type": "string"}},
            "required": ["id", "supporting_evidence", "conflicting_evidence",
                         "missing_evidence", "explanation"]}}},
    "required": ["predictions"]}

SYSTEM = """You analyze evidence for a deterministic uncertainty-aware EEG Supervisor.
The endpoint is a WHOLE PATH end-of-path questionnaire score: low <30, high >=30.
It is not a clinical diagnosis or an instantaneous symptom label. Assess each
anonymous query independently. No class quotas, guessing path order, or treating
queries as a known same-subject sequence. Probabilities from small internally
trained EEG models are noisy evidence, not true labels. Deep and Temporal share
one model and are correlated; full_probability is already their numerical
fusion with available Bio/Cov and must not count as another independent vote.
Raw window probabilities inherit weak path supervision. Increasing scores do
not prove increasing symptoms. Spectral changes can also reflect fatigue,
arousal and artifacts. Covariance distance is unsigned change, not high-class
evidence by itself. Missing reference/evidence is unknown, never evidence for
low. Relative band powers use delta/theta/alpha/beta; reference_log_change uses
natural logarithms versus the initial pre-task rest when available. Do not
invent a universally valid direction for those features.
The deep model alone produces the classification probability. Its calibrated
probability and internally validated reliability gates are supplied separately.
Never create, change or blend a classification probability, choose a final
class, or treat a neighbour class fraction as a calibrated probability. The
local Supervisor determines high/low/uncertain/insufficient_data. Retrieved
cases can support or conflict with the deep-model class; they do not prove the
current path label. An unreliable or distant retrieval is missing evidence.
Describe physiological changes objectively. Only explicitly validated internal
directional evidence can support high/low; unsigned covariance distance alone
cannot do so. Do not resolve uncertainty by inventing an unavailable measure.
Use supplied evidence and internally labelled examples to return lists of
supporting_evidence, conflicting_evidence and missing_evidence plus a concise
explanation per query. Every statement must name an available measure or an
explicitly missing one. Include each supplied query id exactly once. Return
schema JSON with no probability, class, confidence or uncertainty fields.
"""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def existing_provider():
    from eegagent_config import api_settings
    settings = api_settings("EEG_GPT", os.environ.get("EEG_CONFIG_FILE", ".env"))
    return dict(**settings, provider="configured_responses", wire_api="responses")


FORBIDDEN = {"label", "score", "path_score", "ssq", "subject_key", "subject_id",
             "path_index", "path_start_sec", "path_end_sec", "start_sample",
             "end_sample", "true_label", "outer_subject", "role", "episode_handle"}


def check_blind(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("Ground truth or source identity entered query evidence")
        for v in value.values():
            check_blind(v)
    elif isinstance(value, list):
        for v in value:
            check_blind(v)


def choose_examples(records, evaluation, seed, per_class=4):
    """Only this fold's independent meta/calibration subjects supply examples."""
    candidates = [r for r in records if r["role"] == "meta"]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = []
    for label in (0, 1):
        supported = [r for r in candidates if evaluation[r["path_index"]]["label"] == label]
        seen = set()
        diverse, rest = [], []
        for record in supported:
            subject = evaluation[record["path_index"]]["subject_key"]
            if subject in seen:
                rest.append(record)
            else:
                seen.add(subject)
                diverse.append(record)
        chosen.extend((diverse + rest)[:per_class])
    rng.shuffle(chosen)
    return chosen


def build_job(fold, evaluation, mode):
    rng = random.Random(2026 + fold["outer_subject"])
    records = [r for r in fold["records"] if r["role"] in ("policy_validation", "outer_test")]
    rng.shuffle(records)
    queries, mapping = [], {}
    for record in records:
        handle = "q" + f"{rng.getrandbits(48):012x}"
        evidence = dict(record["evidence"])
        if mode != "improved_few_shot":
            evidence.pop("improved_probabilities", None)
        check_blind(evidence)
        queries.append(dict(id=handle, evidence=evidence))
        mapping[handle] = dict(path_index=record["path_index"], role=record["role"])
    examples, example_indices = [], []
    if mode != "zero_shot":
        for record in choose_examples(fold["records"], evaluation, 2026 + fold["outer_subject"]):
            evidence = dict(record["evidence"])
            if mode != "improved_few_shot":
                evidence.pop("improved_probabilities", None)
            check_blind(evidence)
            examples.append(dict(evidence=evidence, observed_class="high" if
                                 evaluation[record["path_index"]]["label"] else "low"))
            example_indices.append(record["path_index"])
    prompt = dict(task="Explain support, conflict and missing EEG evidence for the local Supervisor",
                  examples=examples, queries=queries)
    message = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return dict(outer_subject=fold["outer_subject"], mode=mode, user_message=message,
                mapping=mapping, example_indices=example_indices,
                request_sha256=hashlib.sha256((SYSTEM + message).encode()).hexdigest())


def parse_predictions(response, expected_ids):
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise ValueError("Cloud response did not complete")
    output = response.get("output", [])
    if not isinstance(output, list):
        raise ValueError("Invalid response output")
    texts = []
    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get("content", []), list):
            raise ValueError("Invalid response content")
        for part in item.get("content", []):
            if not isinstance(part, dict):
                raise ValueError("Invalid response part")
            if part.get("type") == "output_text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("Invalid response text")
                texts.append(part["text"])
    output_text = "".join(texts)
    result = json.loads(output_text)
    if not isinstance(result, dict) or set(result) != {"predictions"}:
        raise ValueError("Invalid evidence response object")
    predictions = result["predictions"]
    required = {"id", "supporting_evidence", "conflicting_evidence", "missing_evidence", "explanation"}
    if not isinstance(predictions, list) or any(not isinstance(p, dict) or set(p) != required for p in predictions):
        raise ValueError("Response must contain evidence analysis only")
    if any(not isinstance(p["id"], str) for p in predictions):
        raise ValueError("Invalid response id")
    if len(predictions) != len(expected_ids) or {p["id"] for p in predictions} != set(expected_ids):
        raise ValueError("Missing, duplicate, or unknown response ids")
    for prediction in predictions:
        for name in ("supporting_evidence", "conflicting_evidence", "missing_evidence"):
            if not isinstance(prediction[name], list) or any(not isinstance(v, str) for v in prediction[name]):
                raise ValueError("Evidence fields must be string lists")
        if not isinstance(prediction["explanation"], str):
            raise ValueError("Invalid explanation")
    return predictions


def call_job(provider, job, destination):
    destination = Path(destination)
    if destination.exists():
        old = json.loads(destination.read_text(encoding="utf-8"))
        if old["request_sha256"] != job["request_sha256"]:
            raise ValueError("Stored request hash differs; use a new run directory")
        if old.get("success"):
            # Cached success is audited too; a forged score cannot bypass parsing.
            parse_predictions(dict(status="completed", output=[dict(content=[dict(
                type="output_text", text=json.dumps(dict(predictions=old["predictions"])))])]), job["mapping"])
        if any(old.get(name) != provider.get(name) for name in ("model", "provider", "base_url")):
            raise ValueError("Cached cloud provider or model changed; use a new run directory")
        return old
    body = dict(model=provider["model"],
                input=[dict(role="system", content=SYSTEM), dict(role="user", content=job["user_message"])],
                reasoning=dict(effort="low"), text=dict(format=dict(type="json_schema",
                    name="vrms_predictions", schema=SCHEMA, strict=True)),
                max_output_tokens=8192, store=False)
    # The stored request is credential-free and contains no query labels or source identity.
    log = {**job, "model": provider["model"], "provider": provider["provider"],
           "base_url": provider["base_url"], "body": body, "attempts": [], "success": False}
    started = time.perf_counter()
    for attempt in range(2):
        begin = time.perf_counter()
        try:
            request = urllib.request.Request(provider["base_url"] + "/responses",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": "Bearer " + provider["key"], "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.load(response)
            predictions = parse_predictions(raw, job["mapping"])
            log["attempts"].append(dict(attempt=attempt + 1, seconds=time.perf_counter() - begin,
                                        status="success", response=raw))
            log.update(success=True, predictions=predictions)
            break
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, KeyError) as exc:
            error = dict(attempt=attempt + 1, seconds=time.perf_counter() - begin,
                         error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                error["http_status"] = exc.code
            # Never log headers, auth files, exception request objects, or provider secrets.
            log["attempts"].append(error)
            if attempt == 0:
                time.sleep(1)
    log["seconds"] = time.perf_counter() - started
    write_json(destination, log)
    return log


def main():
    # V2 acquisition is per-path and lazy. Batch helpers remain for offline
    # blinded-prompt diagnostics, but are no longer a production request path.
    from .independent import main as independent_main
    independent_main()


if __name__ == "__main__":
    main()
