"""Read-only scoring/audit of the independently frozen GPT classification run."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from .second_judgment import (DEFAULT_MODELS, DEFAULT_OUT, SCHEMA, SYSTEM, load_jobs,
    parse_response, request_body, sha256, validate_prediction)
from .cloud import write_json

DEFAULT_PILOT = Path("outputs/vrms_pilot/v2_gpt_flow_20261008")


def classification_metrics(y, predicted):
    y, predicted = np.asarray(y, int), np.asarray(predicted, int)
    if not len(y):
        return None
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    dual_class = len(np.unique(y)) == 2
    return dict(n=len(y), correct=int((y == predicted).sum()), accuracy=float((y == predicted).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)) if dual_class else None,
        macro_f1=float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        sensitivity=None if not matrix[1].sum() else float(matrix[1, 1] / matrix[1].sum()),
        specificity=None if not matrix[0].sum() else float(matrix[0, 0] / matrix[0].sum()),
        confusion_matrix=matrix.tolist())


def paired_subject_ci(y, baseline, second, subjects, repetitions=5000):
    rng = np.random.default_rng(2026)
    keys = np.unique(subjects)
    differences = []
    balanced_differences = []
    for _ in range(repetitions):
        draw = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([np.flatnonzero(subjects == key) for key in draw])
        yy, a, b = y[indices], baseline[indices], second[indices]
        differences.append(float(np.mean(b == yy) - np.mean(a == yy)))
        if len(np.unique(yy)) == 2:
            balanced_differences.append(float(balanced_accuracy_score(yy, b) - balanced_accuracy_score(yy, a)))
    return dict(unit="subject; paired predictions; repeated subjects retain all their paths",
        repetitions=repetitions, seed=2026,
        accuracy_difference=float(np.mean(y == second) - np.mean(y == baseline)),
        accuracy_difference_95ci=np.quantile(differences, [.025, .975]).tolist(),
        bacc_difference_95ci=np.quantile(balanced_differences, [.025, .975]).tolist(),
        exploratory=True)


def evaluate(out=DEFAULT_OUT, models=DEFAULT_MODELS, pilot=DEFAULT_PILOT):
    out, models, pilot = Path(out), Path(models), Path(pilot)
    jobs = load_jobs(out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol["system"] != SYSTEM or protocol["schema"] != SCHEMA:
        raise ValueError("Frozen protocol differs")
    paths = json.loads((pilot / "cache/evaluation_manifest.json").read_text(encoding="utf-8"))
    splits = {s["outer_subject"]: s for s in json.loads((pilot / "split_manifest.json").read_text(encoding="utf-8"))}
    with (models / "model_results/path_oof.csv").open(encoding="utf-8-sig", newline="") as f:
        original = {int(r["path_index"]): r for r in csv.DictReader(f)}
    sources = {"baseline_csv": models / "model_results/path_oof.csv",
               "split_manifest": pilot / "split_manifest.json",
               "evaluation_manifest": pilot / "cache/evaluation_manifest.json",
               **{n + ".npy": pilot / "cache" / (n + ".npy") for n in ("bio_absolute", "bio_reference", "covariance")}}
    if any(sha256(p) != protocol["source_hashes"][name] for name, p in sources.items()):
        raise ValueError("Experiment inputs changed")
    for fold in protocol["folds_audit"]:
        stem = f'outer_subject_{fold["outer_subject"]:02d}_femba_frozen_mil'
        if (sha256(models / "model_results/checkpoints" / (stem + ".pt")) != fold["checkpoint_sha256"]
                or sha256(models / "feature_cache" / (stem + ".npy")) != fold["feature_sha256"]):
            raise ValueError("Frozen model/features changed after requests")
    provider = json.loads((out / "provider.json").read_text(encoding="utf-8"))
    rows, predictions, adopted, all_logs, response_models = [], [], [], [], Counter()
    for job in jobs:
        i = job["path_index"]
        path, source = paths[i], original[i]
        prompt = json.loads(job["user_message"])
        if int(source["subject_key"]) != path["subject_key"] or int(source["label"]) != path["label"]:
            raise ValueError("Incorrect evaluation pairing")
        if prompt["query"]["evidence"]["raw_high_probability"] != float(source["femba_frozen_mil_probability"]):
            raise ValueError("Wrong model score in query")
        meta_subjects = {paths[j]["subject_key"] for j in job["example_indices"]}
        if meta_subjects != set(splits[job["outer_subject"]]["meta_calibration"]) or path["subject_key"] in meta_subjects:
            raise ValueError("Query subject entered labelled examples")
        if len(prompt["examples"]) != 5 or len(job["example_indices"]) != 5:
            raise ValueError("Wrong number of examples")
        for j, example in zip(job["example_indices"], prompt["examples"]):
            if example["observed_class"] != ("high" if paths[j]["label"] else "low"):
                raise ValueError("Incorrect historical class")
        logs = []
        for directory in ("calls", "recovery"):
            destination = out / directory / f"{i:03d}.json"
            if not destination.exists():
                if directory == "calls":
                    raise ValueError("Incomplete experiment; all 146 main calls are required")
                continue
            log = json.loads(destination.read_text(encoding="utf-8"))
            if (log["request_sha256"] != job["request_sha256"] or log["path_index"] != i
                    or log["body"] != request_body(job["user_message"], provider["model"])
                    or log["model"] != provider["model"] or log["base_url"] != provider["base_url"]):
                raise ValueError("Stored call differs from frozen anonymous request")
            if log["success"]:
                success_attempts = [a for a in log["attempts"] if a["status"] == "success"]
                if len(success_attempts) != 1 or parse_response(success_attempts[0]["response"]) != validate_prediction(log["prediction"]):
                    raise ValueError("Prediction is not the saved successful API response")
            logs.append((directory, log))
            all_logs.append(log)
        chosen = next(((directory, log) for directory, log in logs if log["success"]), None)
        probability = float(source["femba_frozen_mil_probability"])
        baseline = int(probability >= .5)
        prediction = None if chosen is None else chosen[1]["prediction"]
        final = baseline if prediction is None else int(prediction["decision"] == "high")
        adopted.append(prediction is not None)
        if prediction is not None:
            predictions.append(dict(path_index=i, source=chosen[0], **prediction))
        # This deterministic extra-evidence control is descriptive; no label fit.
        high_count = prompt["retrieval"]["high_count"]
        row = dict(path_index=i, subject_key=path["subject_key"], true_class=path["label"],
            femba_probability=probability, femba_class=baseline, llm_success=prediction is not None,
            final_class=final, llm_class=None if prediction is None else final,
            result_source="baseline_fallback" if chosen is None else chosen[0],
            confidence=None if prediction is None else prediction["confidence"],
            baseline_correct=baseline == path["label"], final_correct=final == path["label"],
            changed=baseline != final, neighbor_high_count=high_count, neighbor_majority_class=int(high_count >= 3),
            nearest_distance=prompt["examples"][0]["distance"],
            explanation="API failed; original model class retained" if prediction is None else prediction["explanation"])
        rows.append(row)
    y = np.asarray([r["true_class"] for r in rows])
    a = np.asarray([r["femba_class"] for r in rows])
    b = np.asarray([r["final_class"] for r in rows])
    neighbor = np.asarray([r["neighbor_majority_class"] for r in rows])
    groups = np.asarray([r["subject_key"] for r in rows])
    success = np.asarray(adopted)
    if len(rows) != 146 or int((y == a).sum()) != 89:
        raise ValueError("The expected 89/146 baseline is not reproduced")
    attempts = [attempt for log in all_logs for attempt in log["attempts"]]
    usage = Counter()
    for attempt in attempts:
        if "response" not in attempt:
            continue
        raw = attempt["response"]
        response_models[raw.get("model", "missing")] += 1
        tokens = raw.get("usage", {})
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage[key] += tokens.get(key, 0)
        usage["cached_input_tokens"] += tokens.get("input_tokens_details", {}).get("cached_tokens", 0)
        usage["reasoning_tokens"] += tokens.get("output_tokens_details", {}).get("reasoning_tokens", 0)
    flip = dict(changed=int((a != b).sum()), wrong_to_correct=int(((a != y) & (b == y)).sum()),
                correct_to_wrong=int(((a == y) & (b != y)).sum()), unchanged_correct=int(((a == y) & (a == b)).sum()),
                unchanged_wrong=int(((a != y) & (a == b)).sum()))
    per_subject = [dict(subject_key=int(key), baseline=classification_metrics(y[groups == key], a[groups == key]),
                        second=classification_metrics(y[groups == key], b[groups == key])) for key in np.unique(groups)]
    summary = dict(status="completed_exploratory_second_classification", paths=146, subjects=24,
        model=provider["model"], response_model_counts=dict(response_models),
        baseline=classification_metrics(y, a), gpt_second_with_fallback=classification_metrics(y, b),
        successful_calls=int(success.sum()), fallback_paths=int((~success).sum()),
        successful_subset=dict(baseline=classification_metrics(y[success], a[success]),
                               gpt=classification_metrics(y[success], b[success])),
        neighbor_majority_control=classification_metrics(y, neighbor), flips=flip,
        paired_subject_bootstrap=paired_subject_ci(y, a, b, groups),
        confidence_counts=dict(Counter(r["confidence"] for r in rows if r["llm_success"])),
        per_subject=per_subject,
        api=dict(total_attempts=len(attempts), failures=dict(Counter(
                f'{a["error_type"]}:{a.get("http_status", "")}' for a in attempts if a["status"] != "success")),
            usage=dict(usage), summed_call_seconds=sum(log["seconds"] for log in all_logs)),
        audit=dict(anonymous_single_query=True, meta_only_examples=True, subject_diverse=True,
                   original_sources_unchanged=True, raw_responses_reparsed=True, calibrated_llm_probability=False,
                   tuning_on_outer_labels=False, prompt_hashes_checked=True,
                   code_sha256=sha256(__file__), protocol_sha256=sha256(out / "protocol.json")))
    summary["api"]["main_successes"] = sum(json.loads(p.read_text(encoding="utf-8"))["success"] for p in (out / "calls").glob("*.json"))
    write_json(out / "summary.json", summary)
    write_json(out / "predictions.json", predictions)
    with (out / "path_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: summary[k] for k in ("baseline", "gpt_second_with_fallback", "successful_calls",
          "fallback_paths", "flips", "paired_subject_bootstrap", "neighbor_majority_control", "api")}, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    args = parser.parse_args()
    evaluate(args.out, args.models, args.pilot)


if __name__ == "__main__":
    main()
