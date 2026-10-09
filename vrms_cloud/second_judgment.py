"""Exploratory forced-binary GPT judgment of frozen FEMBA+MIL LOSO evidence.

This module does not participate in the production explanation-only Supervisor.
Preparation uses frozen checkpoints; API requests contain one anonymous query.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from .cloud import check_blind, existing_provider, write_json

DEFAULT_OUT = Path("outputs/vrms_second_judgment/femba_gpt_20261008")
DEFAULT_MODELS = Path("outputs/vrms_model_ablation/v3_20261008")
METHOD = "femba_frozen_mil"
SYSTEM = """You perform an exploratory SECOND binary classification of one anonymous
EEG path using frozen FEMBA+MIL evidence and five internally labelled examples.
The target is the WHOLE PATH end-of-path questionnaire category: low <30,
high >=30. It is not an instantaneous symptom or clinical diagnosis.
Return exactly one decision, high or low, even when evidence is weak; express
weak evidence using confidence=low and missing_evidence, never an uncertain
decision. Your decision is the experimental final class, not merely prose.
The model's raw_high_probability is UNCALIBRATED and may be overconfident.
The labelled examples were not used to fit this fold's MIL head, and each
comes from a different internal subject. Examples are selected by EEG feature
distance WITHOUT class balancing. Do not assume any class quota for queries,
infer identity or sequence, or use information absent from this request.
Distances are RMS differences of mean frozen FEMBA window embeddings after
base-training-only feature standardization. Their reliability is unvalidated;
neighbour fractions are descriptive and are NOT a calibrated query probability.
Consider the raw model score and its errors in the supplied labelled examples,
the closeness and agreement of examples, QC, and physiological descriptions.
Retain or revise the model class based on the supplied evidence; do not flip
merely to disagree, and do not invent an unavailable measurement.
Band order is delta/theta/alpha/beta. Reference changes are natural-log power
ratios versus initial pre-task rest. Delta increase, alpha decrease, or a larger
unsigned covariance distance do not universally imply high. Physiological
direction has NOT been independently validated. Small-sample associations in
examples are tentative, not physiological proof. Missing reference is unknown,
not low-class evidence. Quality loss is uncertainty, not low-class evidence.
Use concise Chinese evidence and explanation, naming actual measurements.
Keep each evidence list at most three short items and explanation under 180
Chinese characters. Return strict schema JSON, id=qsingle, with no numerical
LLM probability. Confidence is qualitative and is not a calibrated measure.
"""
SCHEMA = dict(type="object", additionalProperties=False, properties={
    "id": dict(type="string", enum=["qsingle"]),
    "decision": dict(type="string", enum=["high", "low"]),
    "confidence": dict(type="string", enum=["low", "medium", "high"]),
    "supporting_evidence": dict(type="array", items=dict(type="string")),
    "conflicting_evidence": dict(type="array", items=dict(type="string")),
    "missing_evidence": dict(type="array", items=dict(type="string")),
    "explanation": dict(type="string")},
    required=["id", "decision", "confidence", "supporting_evidence",
              "conflicting_evidence", "missing_evidence", "explanation"])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def request_body(message, model):
    return dict(model=model, input=[dict(role="system", content=SYSTEM),
                dict(role="user", content=message)], reasoning=dict(effort="low"),
                text=dict(format=dict(type="json_schema", name="eeg_second_judgment",
                                      schema=SCHEMA, strict=True)),
                max_output_tokens=4096, store=False)


def request_hash(message):
    body = request_body(message, "MODEL_SELECTED_BEFORE_CALLS")
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()


def prepare(out=DEFAULT_OUT, models=DEFAULT_MODELS):
    """Re-predict outer/meta with the existing head; no fitting or label tuning."""
    import numpy as np
    import torch
    from vrms_pilot.model_ablation_v3 import (PathMIL, load_dataset, load_features,
        predict_mil, read_json, role_indices, state_digest, verify_frozen)
    from vrms_pilot.data import read_csv
    from vrms_pilot.physiology import physiological_evidence

    out, models = Path(out), Path(models)
    if out.exists():
        raise FileExistsError("Preserve previous experiment; use a new output directory")
    frozen = verify_frozen(models)
    pilot = Path(frozen["pilot_out"])
    paths, splits, _, _ = load_dataset(pilot)
    source_rows = {int(r["path_index"]): r for r in read_csv(models / "model_results/path_oof.csv")}
    extraction = read_json(models / "feature_cache/extraction_audit.json")
    cache = pilot / "cache"
    arrays = {n: np.load(cache / (n + ".npy"), mmap_mode="r") for n in
              ("bio_absolute", "bio_reference", "covariance")}
    descriptors = []
    for path in paths:
        sl = slice(path["window_start"], path["window_end"])
        if path["accepted_windows"] < 1:
            raise ValueError("This forced-class experiment requires nonempty paths")
        powers = arrays["bio_absolute"][sl, 120:].mean(axis=0).reshape(30, 4).mean(axis=0)
        valid = path["reference_available"]
        changes = arrays["bio_reference"][sl, 240:].mean(axis=0).reshape(30, 4).mean(axis=0) if valid else None
        distance = float(arrays["covariance"][sl, -1].mean()) if valid else None
        descriptors.append(dict(signal_quality=dict(accepted_windows=path["accepted_windows"],
            total_complete_windows=path["total_possible_windows"],
            accepted_window_fraction=path["accepted_windows"] / path["total_possible_windows"],
            reference_quality="valid" if valid else "unavailable"),
            physiological_evidence=physiological_evidence(
                biomarker=dict(relative_band_power=powers.tolist(),
                               reference_log_change=None if changes is None else changes.tolist()),
                covariance=dict(distance=distance), reference_quality="valid" if valid else "unavailable")))

    def evidence(i, p):
        # Construct a whitelist; never copy a labelled path record into a prompt.
        return dict(raw_high_probability=float(p), calibrated=False, **descriptors[i])

    out.mkdir(parents=True)
    jobs, folds = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split, record in zip(splits, extraction["records"]):
        subject = split["outer_subject"]
        if record["outer_subject"] != subject:
            raise ValueError("Changed feature fold order")
        features = load_features(models, record, METHOD)
        checkpoint = models / "model_results/checkpoints" / f"outer_subject_{subject:02d}_{METHOD}.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["split"] != split or state["feature_sha256"] != record["feature_files"][METHOD]["sha256"]:
            raise ValueError("Checkpoint split/features differ")
        if state["training"]["train_subjects"] != sorted(split["base_train"]):
            raise ValueError("MIL was not fitted on base14 only")
        head = PathMIL(state["input_dim"]).to(device)
        head.load_state_dict(state["state_dict"])
        if state_digest(head) != state["training"]["final_state_sha256"]:
            raise ValueError("MIL checkpoint weights changed")
        meta, outer = [role_indices(paths, split, role) for role in ("meta_calibration", "outer_test")]
        p = predict_mil(head, features, paths, np.r_[meta, outer], device)
        # Distances use frozen base14 statistics; no test or meta fit occurs.
        center = head.center.detach().cpu().numpy()
        scale = head.scale.detach().cpu().numpy()
        z = np.clip((features - center) / scale, -20, 20)
        bags = np.asarray([z[path["window_start"]:path["window_end"]].mean(axis=0) for path in paths])
        error = max(abs(p[i] - float(source_rows[i][METHOD + "_probability"])) for i in outer)
        if error > 1e-6:
            raise ValueError("Frozen baseline probabilities changed")
        for i in outer:
            i = int(i)
            distances = np.sqrt(np.mean((bags[meta] - bags[i]) ** 2, axis=1))
            ordered = sorted(zip(meta.tolist(), distances.tolist()), key=lambda t: (t[1], t[0]))
            selected, seen = [], set()
            for j, distance in ordered:
                s = paths[j]["subject_key"]
                if s not in seen:
                    selected.append((j, distance))
                    seen.add(s)
                if len(selected) == 5:
                    break
            if len(selected) != 5 or seen != set(split["meta_calibration"]) or subject in seen:
                raise ValueError("Invalid internal subject-diverse examples")
            examples = [dict(distance=distance, observed_class="high" if paths[j]["label"] else "low",
                             evidence=evidence(j, p[j])) for j, distance in selected]
            prompt = dict(task="Second binary judgment of this one anonymous EEG path",
                query=dict(id="qsingle", evidence=evidence(i, p[i])), examples=examples,
                retrieval=dict(k=5, different_subjects=5, high_count=sum(e["observed_class"] == "high" for e in examples),
                    low_count=sum(e["observed_class"] == "low" for e in examples),
                    distance_definition="RMS base14-standardized mean FEMBA embedding difference",
                    reliability="unvalidated", selection="nearest per subject; no class quotas"))
            check_blind(prompt)
            message = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            jobs.append(dict(path_index=i, outer_subject=subject,
                example_indices=[j for j, _ in selected], user_message=message,
                request_sha256=request_hash(message)))
        folds.append(dict(outer_subject=subject, checkpoint_sha256=sha256(checkpoint),
            feature_sha256=record["feature_files"][METHOD]["sha256"], baseline_max_abs_error=error,
            meta_subjects=split["meta_calibration"], training_subjects=split["base_train"]))
        print(f"prepare fold {len(folds)}/24", flush=True)
    jobs.sort(key=lambda j: j["path_index"])
    if [j["path_index"] for j in jobs] != list(range(146)):
        raise ValueError("Expected exactly the same 146 outer queries")
    write_json(out / "jobs.json", jobs)
    write_json(out / "protocol.json", dict(schema_version="femba_gpt_second_binary_v1",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        frozen_before_calls=True, paths=146, folds=24, input_method=METHOD,
        decision="GPT high/low replaces model class directly; no numeric fusion",
        failure_rule="retain original p_raw>=0.5 class only after all recorded API attempts fail",
        evaluation="all146 and successful-call-only; accuracy, balanced accuracy, macroF1, paired subject bootstrap",
        prompt_tuning_on_outer_labels=False, examples="meta5 only, nearest per subject, no class balancing",
        physiology="read existing objective caches; no Bio/Cov class votes, no new fitting",
        tool_execution="offline second-classification experiment; not a lazy deployment benchmark",
        system=SYSTEM, schema=SCHEMA, client=dict(reasoning_effort="low", max_output_tokens=4096, store=False),
        source_hashes={"baseline_csv": sha256(models / "model_results/path_oof.csv"),
            "split_manifest": sha256(pilot / "split_manifest.json"),
            "evaluation_manifest": sha256(cache / "evaluation_manifest.json"),
            **{n + ".npy": sha256(cache / (n + ".npy")) for n in arrays}},
        folds_audit=folds, code_sha256=sha256(__file__), jobs_sha256=sha256(out / "jobs.json"),
        pretraining_overlap_excluded=False, exploratory=True))


def validate_prediction(prediction):
    if not isinstance(prediction, dict) or set(prediction) != set(SCHEMA["required"]):
        raise ValueError("Invalid second-judgment fields")
    if prediction["id"] != "qsingle" or prediction["decision"] not in ("high", "low"):
        raise ValueError("One known query and a forced binary decision are required")
    if prediction["confidence"] not in ("low", "medium", "high"):
        raise ValueError("Invalid qualitative confidence")
    for name in ("supporting_evidence", "conflicting_evidence", "missing_evidence"):
        if not isinstance(prediction[name], list) or any(not isinstance(s, str) for s in prediction[name]):
            raise ValueError("Evidence must be string arrays")
    if not isinstance(prediction["explanation"], str) or not prediction["explanation"].strip():
        raise ValueError("Missing explanation")
    return prediction


def parse_response(raw):
    if not isinstance(raw, dict) or raw.get("status") != "completed":
        raise ValueError("Response incomplete")
    texts = []
    for item in raw.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "refusal":
                raise ValueError("Response refusal")
            if part.get("type") == "output_text":
                texts.append(part["text"])
    return validate_prediction(json.loads("".join(texts)))


def load_jobs(out):
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol["code_sha256"] != sha256(__file__) or protocol["jobs_sha256"] != sha256(out / "jobs.json"):
        raise ValueError("Frozen experiment changed; use another directory")
    jobs = json.loads((out / "jobs.json").read_text(encoding="utf-8"))
    for job in jobs:
        check_blind(json.loads(job["user_message"]))
        if request_hash(job["user_message"]) != job["request_sha256"]:
            raise ValueError("Prompt hash differs")
    return jobs


def call_job(provider, job, destination):
    destination = Path(destination)
    if destination.exists():
        log = json.loads(destination.read_text(encoding="utf-8"))
        if (log["request_sha256"] != job["request_sha256"] or log["model"] != provider["model"]
                or log["base_url"] != provider["base_url"]):
            raise ValueError("Cached request/provider differs")
        if log["success"]:
            validate_prediction(log["prediction"])
        return log
    body = request_body(job["user_message"], provider["model"])
    check_blind(json.loads(job["user_message"]))
    if request_hash(job["user_message"]) != job["request_sha256"]:
        raise ValueError("Call prompt differs")
    log = dict(path_index=job["path_index"], request_sha256=job["request_sha256"],
               model=provider["model"], base_url=provider["base_url"], body=body,
               success=False, attempts=[])
    begin = time.perf_counter()
    for attempt in range(2):
        started = time.perf_counter()
        try:
            request = urllib.request.Request(provider["base_url"] + "/responses",
                data=json.dumps(body, ensure_ascii=False).encode(),
                headers={"Authorization": "Bearer " + provider["key"], "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(request, timeout=180) as r:
                raw = json.load(r)
            prediction = parse_response(raw)
            log["attempts"].append(dict(number=attempt + 1, seconds=time.perf_counter() - started,
                                        status="success", response=raw))
            log.update(success=True, prediction=prediction)
            break
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, KeyError, TypeError) as exc:
            error = dict(number=attempt + 1, seconds=time.perf_counter() - started,
                         status="failure", error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                error["http_status"] = exc.code
            # No credentials, request headers or exception request objects in logs.
            log["attempts"].append(error)
            if attempt == 0:
                time.sleep(1)
    log["seconds"] = time.perf_counter() - begin
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, log)
    return log


def run(out=DEFAULT_OUT, workers=4, probe=False, recover=False):
    out = Path(out)
    jobs = load_jobs(out)
    provider = existing_provider()
    run_info = dict(model=provider["model"], base_url=provider["base_url"], wire_api="responses",
                    protocol_sha256=sha256(out / "protocol.json"))
    info_path = out / "provider.json"
    if info_path.exists() and json.loads(info_path.read_text(encoding="utf-8")) != run_info:
        raise ValueError("Experiment model/provider changed")
    write_json(info_path, run_info)
    directory = out / ("recovery" if recover else "calls")
    if recover:
        jobs = [j for j in jobs if (out / "calls" / f'{j["path_index"]:03d}.json').exists()
                and not json.loads((out / "calls" / f'{j["path_index"]:03d}.json').read_text(encoding="utf-8"))["success"]]
    elif probe:
        jobs = jobs[:1]
    begin = time.perf_counter()
    successes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_job, provider, j, directory / f'{j["path_index"]:03d}.json'): j for j in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            log = future.result()
            successes += int(log["success"])
            print(json.dumps(dict(completed=number, total=len(jobs), success=successes,
                path_index=log["path_index"], call_success=log["success"], seconds=round(log["seconds"], 1)),
                ensure_ascii=False), flush=True)
    write_json(out / ("recovery_run.json" if recover else "probe_run.json" if probe else "main_run.json"),
               dict(requested_paths=len(jobs), successful_paths=successes, workers=workers,
                    elapsed_seconds=time.perf_counter() - begin, recovery=recover))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out, args.models)
    else:
        run(args.out, args.workers, args.probe, args.recover)


if __name__ == "__main__":
    main()
