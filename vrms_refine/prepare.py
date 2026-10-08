"""Build baseline/enhanced single-path requests; no outer-label selection.

Enhanced examples and reliability use leave-one-meta-subject-out calibration
heads, conditional on frozen encoders. MIL epoch selection previously used all
meta subjects; that remaining dependence is disclosed, not called fully
cross-fitted encoder evaluation.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

from vrms_cloud.cloud import choose_examples
from vrms_cloud.improve import PathMILCNN, mil_scores, path_features, tangent_features
from vrms_pilot.experiment import base_scores, fit_logistic, logit, pool_features, probabilities
from vrms_pilot.model import CompactEEGCNN
from vrms_weights.scan import metrics
from .common import CLOUD, DEEPSEEK, PILOT, DEFAULT_OUT, job_hash, message_for, read_json, sha, write_json, select_source_indices


def fraction(value):
    return None if not np.isfinite(value) else round(float(value), 4)


def head_crossfit(x, y, groups, meta, c):
    p = np.full(len(y), np.nan)
    provenance, prior_fallbacks = [], 0
    for subject in np.unique(groups[meta]):
        training = meta[groups[meta] != subject]
        target = meta[groups[meta] == subject]
        model = fit_logistic(x, y, training, c=c, balanced=False)
        if model is None:
            good = target[np.isfinite(x[target]).all(axis=1)]
            p[good] = (float(y[training].sum()) + .5) / (len(training) + 1)
            prior_fallbacks += len(good)
        else:
            p[target] = probabilities(model, x)[target]
        provenance.append(dict(heldout_subject=int(subject), training_subjects=np.unique(groups[training]).tolist(),
                               target_indices=target.tolist(), training_indices=training.tolist()))
    return p, dict(folds=provenance, prior_fallbacks=prior_fallbacks)


def raw_vector(evidence):
    temporal = evidence["temporal"]
    values = [temporal[k] for k in ("raw_mean_probability", "raw_probability_std", "raw_recent_mean",
                                   "raw_recent_slope_per_second", "raw_last_probability")]
    values += evidence["relative_band_power"]
    values += evidence["reference_log_change"] or [None] * 4
    values += [evidence["covariance_distance"], evidence["qc_accepted_fraction"], float(evidence["reference_available"])]
    return np.asarray([np.nan if v is None else v for v in values], float)


def retrieve_examples(query, bank, evaluation, per_class=4):
    """Scale on the internal bank only; raw EEG features exclude calibrated scores."""
    x = np.stack([raw_vector(r["evidence"]) for r in bank])
    q = raw_vector(query)
    scale = np.asarray([max(float(np.std(x[np.isfinite(x[:, j]), j])), 1e-4)
                        if np.isfinite(x[:, j]).any() else 1. for j in range(x.shape[1])])
    distance = []
    for row in x:
        common = np.isfinite(row) & np.isfinite(q)
        distance.append(float(np.mean(np.minimum(((row[common] - q[common]) / scale[common]) ** 2, 25))))
    chosen = []
    for label in (0, 1):
        ordered = sorted([i for i, r in enumerate(bank) if evaluation[r["path_index"]]["label"] == label],
                         key=lambda i: (distance[i], bank[i]["path_index"]))
        diverse, remaining, seen = [], [], set()
        for i in ordered:
            subject = evaluation[bank[i]["path_index"]]["subject_key"]
            if subject in seen:
                remaining.append(i)
            else:
                diverse.append(i)
                seen.add(subject)
        chosen += (diverse + remaining)[:per_class]
    chosen.sort(key=lambda i: (distance[i], bank[i]["path_index"]))
    examples = [dict(evidence=bank[i]["evidence"], observed_class="high" if evaluation[bank[i]["path_index"]]["label"] else "low")
                for i in chosen]
    return examples, [bank[i]["path_index"] for i in chosen]


def reliability_pack(scores, y, meta):
    result = {}
    for name, p in scores.items():
        good = meta[np.isfinite(p[meta])]
        if not len(good):
            result[name] = dict(cases=0, accuracy=None, balanced_accuracy=None, brier=None)
            continue
        m = metrics(y[good], p[good])
        result[name] = dict(cases=len(good), accuracy=fraction(m["acc"]),
                            balanced_accuracy=fraction(m["bacc"]) if m["bacc"] is not None else None,
                            brier=fraction(m["brier"]))
    return result


def local_path(value):
    value = str(value).replace("\\", "/")
    if sys.platform != "win32" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    return Path(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Keep completed runs; choose a new output directory")
    old_checks = read_json(Path("outputs/vrms_weights/20261008_seed2026/validation.json"))["original_sha256"]
    protected = {str(local_path(p)): expected for p, expected in old_checks.items()}
    for p, expected in protected.items():
        if sha(p) != expected:
            raise ValueError(f"An original source changed: {p}")
    checkpoints = list((PILOT / "results/checkpoints").glob("*")) + list((CLOUD / "improvements/checkpoints").glob("*"))
    protected.update({str(p): sha(p) for p in checkpoints if p.is_file()})
    args.out.mkdir(parents=True)
    write_json(args.out / "analysis_plan.json", read_json(Path(__file__).with_name("analysis_plan.json")))
    write_json(args.out / "protocol.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        stage="frozen_before_new_cloud_outcomes", seed=2026, subjects=24, paths=147,
        numerical_models_retrained=False, splits_changed=False, single_path_requests=True,
        models=dict(gpt="gpt-6.1-sol", deepseek="deepseek-flash"),
        interventions=["raw-feature similar meta examples", "leave-one-meta-subject-out calibration scores and reliability",
                       "validation-selected simple fusion", "validation-selected calibration and selective publication"],
        remaining_dependency="Frozen MIL epoch previously selected using all 5 meta subjects; calibration heads alone cross-fitted",
        outer_labels_used_for_prompts_or_selection=False, prior_outer_results_seen=True, exploratory=True,
        new_requests_per_vendor=1309, max_attempts_per_request=2,
        acceptance=dict(accuracy_minimum_gain_paths=5, paired_subject_bootstrap_95ci_lower_gt0=True,
                        bacc_must_not_decrease=True, brier_max_increase=.005,
                        calibration_minimum_relative_brier_improvement=.10, calibration_max_lost_correct_paths=1,
                        calibration_brier_gain_ci_lower_gt0=True,
                        selective_minimum_coverage=.70, selective_target_accuracy=.80,
                        api_minimum_success_rate=.99),
        original_sha256=protected))
    cache = PILOT / "cache"
    paths = read_json(cache / "evaluation_manifest.json")
    audit = read_json(cache / "dataset_audit.json")
    folds = read_json(CLOUD / "fold_evidence.json")
    evaluation = {r["path_index"]: r for r in read_json(CLOUD / "evaluation.json")}
    y = np.asarray([p["label"] for p in paths])
    groups = np.asarray([p["subject_key"] for p in paths])
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    window_records = [__import__("json").loads(s) for s in (cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = np.asarray([r["start_sample"] / 1024 for r in window_records])
    pooled = {key: pool_features(np.load(cache / filename, mmap_mode="r"), paths) for key, filename in
              (("bio_absolute", "bio_absolute.npy"), ("bio_reference", "bio_reference.npy"), ("covariance", "covariance.npy"))}
    features, covariances = path_features(cache, paths, windows, audit["channel_order"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    jobs, diagnostics = [], []
    for fold in folds:
        s, split = fold["outer_subject"], fold["split"]
        base = np.flatnonzero(np.isin(groups, split["base_train"]))
        meta = np.flatnonzero(np.isin(groups, split["meta_calibration"]))
        meta_records = select_source_indices(fold["records"], set(split["meta_calibration"]), evaluation, "meta")
        if s in split["meta_calibration"] or s in split["policy_validation"]:
            raise ValueError("Outer subject leaked into internal data")
        state = torch.load(PILOT / f"results/checkpoints/outer_subject_{s:02d}.pt", map_location=device, weights_only=True)
        original = joblib.load(PILOT / f"results/checkpoints/outer_subject_{s:02d}.joblib")
        cnn = CompactEEGCNN().to(device)
        cnn.load_state_dict(state["model"])
        cnn.eval()
        raw, _ = base_scores(paths, windows, starts, cnn, state["mean"].to(device), state["scale"].to(device),
                             device, original["classifiers"], pooled, meta)
        old_scores, crossfit = {}, {}
        for name, x in raw.items():
            old_scores[name], crossfit[name] = head_crossfit(x, y, groups, meta, .3)
        old_scores["full"] = np.where(np.isfinite(old_scores["full"]), old_scores["full"], old_scores["deep_temporal_bio"])
        improved = joblib.load(CLOUD / f"improvements/checkpoints/outer_subject_{s:02d}.joblib")
        new_scores = {}
        for name in ("regional_spectrum", "spatial_spectrum", "tangent_covariance"):
            x = tangent_features(covariances, base)[0] if name == "tangent_covariance" else features[name]
            raw_probability = probabilities(improved["models"][name]["model"], x)
            xcal = logit(raw_probability)[:, None]
            new_scores[name], crossfit[name] = head_crossfit(xcal, y, groups, meta, .1)
        mil_state = torch.load(CLOUD / f"improvements/checkpoints/outer_subject_{s:02d}.pt", map_location=device, weights_only=True)
        mil = PathMILCNN().to(device)
        mil.load_state_dict(mil_state["model"])
        dt = mil_scores(mil, windows, paths, starts, meta, device)
        new_scores["path_mil"], crossfit["path_mil"] = head_crossfit(dt, y, groups, meta, .1)
        new_scores["original_full"] = old_scores["full"]
        new_scores["candidate_mean"] = np.mean([new_scores[k] for k in
                         ("original_full", "regional_spectrum", "spatial_spectrum", "tangent_covariance", "path_mil")], axis=0)
        new_scores["validated_numeric"] = new_scores[improved["selected"]]
        reliability = reliability_pack({**{k: p for k, p in old_scores.items() if k != "deep_temporal_bio"},
                                        **{k: p for k, p in new_scores.items() if k != "original_full"}}, y, meta)
        bank = copy.deepcopy(meta_records)
        for record in bank:
            i = record["path_index"]
            record["evidence"]["probabilities"] = {k + "_probability": fraction(old_scores[k][i])
                        for k in ("deep", "deep_temporal", "biomarker", "covariance", "full")}
            record["evidence"]["improved_probabilities"] = {k + "_probability": fraction(p[i])
                        for k, p in new_scores.items() if k != "original_full"}
        baseline_examples = [dict(evidence=r["evidence"], observed_class="high" if evaluation[r["path_index"]]["label"] else "low")
                             for r in choose_examples(fold["records"], evaluation, 2026 + s)]
        contexts = []
        for record in fold["records"]:
            if record["role"] not in ("policy_validation", "outer_test"):
                continue
            i = record["path_index"]
            if record["role"] == "policy_validation":
                message = message_for(record["evidence"], baseline_examples)
                jobs.append(dict(mode="baseline", outer_subject=s, path_index=i, role=record["role"],
                                 subject_key=int(groups[i]), message=message, message_sha256=job_hash(message)))
            examples, indices = retrieve_examples(record["evidence"], bank, evaluation)
            evidence = copy.deepcopy(record["evidence"])
            evidence["analysis_context"] = dict(
                reliability_source="leave-one-meta-subject-out calibration, conditional on frozen encoders",
                reference_case_source="5 internal meta subjects; query subject excluded", meta_subject_count=5,
                model_reliability=reliability,
                dependency_notes=["Deep and Temporal share the same original CNN", "Full is a numerical fusion, not an independent vote",
                                  "Candidate mean and validated numeric reuse component models"],
                limitations="Small reference set; MIL epoch used meta subjects, so encoder selection is not fully cross-fitted",
                guidance="Compare similar labelled cases and model reliability. Cite concrete supporting and conflicting evidence in reason. Do not inflate confidence or infer missing measurements. No universal spectral or covariance direction.")
            message = message_for(evidence, examples)
            jobs.append(dict(mode="enhanced", outer_subject=s, path_index=i, role=record["role"], subject_key=int(groups[i]),
                             message=message, message_sha256=job_hash(message)))
            contexts.append(dict(path_index=i, role=record["role"], selected_example_indices=indices))
        diagnostics.append(dict(outer_subject=s, selected_numeric=improved["selected"], meta_subjects=split["meta_calibration"],
                                crossfit=crossfit, model_reliability=reliability, contexts=contexts))
        print(f"prepared fold {s:02d}; requests={len(jobs)}; calibration-crossfit examples={len(bank)}", flush=True)
        del cnn, mil
    if len(jobs) != 1309 or sum(j["mode"] == "baseline" for j in jobs) != 581:
        raise ValueError("Unexpected request counts")
    for p, expected in protected.items():
        if sha(p) != expected:
            raise ValueError("A protected input changed")
    write_json(args.out / "jobs.json", jobs)
    write_json(args.out / "preparation_audit.json", dict(status="passed", requests_per_vendor=len(jobs),
        baseline_policy_requests=581, enhanced_policy_requests=581, enhanced_outer_requests=147,
        source_hashes_unchanged=True, query_labels_absent=True, other_test_paths_absent=True,
        reliability_uses_meta_only=True, retrieval_scaling_uses_meta_only=True, diagnostics=diagnostics))
    print("Preparation complete; no API calls made.", flush=True)


if __name__ == "__main__":
    main()
