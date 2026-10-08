"""Explicitly configured Responses API client for bounded EEG evidence calls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request


DEFAULT_OUT = Path("outputs/vrms_cloud/20261008_seed2026")
MODES = ("zero_shot", "few_shot", "improved_few_shot")
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"predictions": {
        "type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "high_probability": {"type": "number"},
                "state": {"type": "string", "enum": ["high", "low"]},
                "uncertain": {"type": "boolean"},
                "reason": {"type": "string"}},
            "required": ["id", "high_probability", "state", "uncertain", "reason"]}}},
    "required": ["predictions"]}

SYSTEM = """You are the cloud Supervisor in an exploratory EEG VR discomfort experiment.
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
Use all available evidence, and any supplied internally labelled examples,
to produce a measured high-class probability, binary state, uncertainty flag
and a short reason (one clause, <=100 characters) per query. State must be high
iff probability >=0.5. Uncertainty is an unvalidated flag, not calibrated
confidence. Include each supplied query id exactly once. Return schema JSON.
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
    prompt = dict(task="Infer whole-path high/low from EEG evidence only",
                  examples=examples, queries=queries)
    message = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return dict(outer_subject=fold["outer_subject"], mode=mode, user_message=message,
                mapping=mapping, example_indices=example_indices,
                request_sha256=hashlib.sha256((SYSTEM + message).encode()).hexdigest())


def parse_predictions(response, expected_ids):
    if response.get("status") != "completed":
        raise ValueError("Cloud response did not complete")
    output_text = "".join(part.get("text", "") for item in response.get("output", [])
                          for part in item.get("content", []) if part.get("type") == "output_text")
    result = json.loads(output_text)
    predictions = result["predictions"]
    if len(predictions) != len(expected_ids) or {p["id"] for p in predictions} != set(expected_ids):
        raise ValueError("Missing, duplicate, or unknown response ids")
    for prediction in predictions:
        p = prediction["high_probability"]
        if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid probability")
        if prediction["state"] != ("high" if p >= .5 else "low"):
            raise ValueError("Class and probability disagree")
        if not isinstance(prediction["uncertain"], bool) or not isinstance(prediction["reason"], str):
            raise ValueError("Invalid uncertainty or explanation")
    return predictions


def call_job(provider, job, destination):
    destination = Path(destination)
    if destination.exists():
        old = json.loads(destination.read_text(encoding="utf-8"))
        if old["request_sha256"] != job["request_sha256"]:
            raise ValueError("Stored request hash differs; use a new run directory")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES[:2]))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-folds", type=int)
    args = parser.parse_args()
    provider = existing_provider()
    folds = json.loads((args.out / "fold_evidence.json").read_text(encoding="utf-8"))
    evaluation = {int(r["path_index"]): r for r in json.loads((args.out / "evaluation.json").read_text(encoding="utf-8"))}
    if args.limit_folds:
        folds = folds[:args.limit_folds]
    jobs = [build_job(f, evaluation, mode) for mode in args.modes for f in folds]
    directory = args.out / "cloud_calls"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "cloud_provider.json", {k: v for k, v in provider.items() if k != "key"})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(call_job, provider, job,
                              directory / f"{job['mode']}_fold_{job['outer_subject']:02d}.json"): job for job in jobs}
        for done in as_completed(pending):
            result = done.result()
            print(f"cloud {result['mode']} fold {result['outer_subject']:02d}: "
                  f"success={result['success']} cases={len(result['mapping'])} "
                  f"seconds={result['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
