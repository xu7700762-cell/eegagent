"""Reconstruct structured evidence from frozen checkpoints, without refitting."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from vrms_pilot.data import digest, read_csv, write_json
from vrms_pilot.engine import temporal_features
from vrms_pilot.experiment import DEFAULT_OUT as PILOT_OUT, pool_features
from vrms_pilot.model import CompactEEGCNN, predict_windows

DEFAULT_OUT = Path("outputs/vrms_cloud/20261008_seed2026")


def rounded(value):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache, result = PILOT_OUT / "cache", PILOT_OUT / "results"
    paths = json.loads((cache / "evaluation_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((cache / "dataset_audit.json").read_text(encoding="utf-8"))
    splits = json.loads((PILOT_OUT / "split_manifest.json").read_text(encoding="utf-8"))
    inner = read_csv(result / "inner_predictions.csv")
    outer = {int(r["path_index"]): r for r in read_csv(result / "path_oof.csv")}
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    source = [json.loads(s) for s in (cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    times = np.asarray([r["start_sample"] / 1024 for r in source])
    pooled = {key: pool_features(np.load(cache / name, mmap_mode="r"), paths) for key, name in
              (("bio_absolute", "bio_absolute.npy"), ("bio_reference", "bio_reference.npy"), ("covariance", "covariance.npy"))}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    evidence_folds = []
    for split in splits:
        subject = split["outer_subject"]
        state = torch.load(result / "checkpoints" / f"outer_subject_{subject:02d}.pt",
                           map_location=device, weights_only=True)
        components = joblib.load(result / "checkpoints" / f"outer_subject_{subject:02d}.joblib")
        assert components["split"] == split
        model = CompactEEGCNN().to(device)
        model.load_state_dict(state["model"])
        model.eval()
        records = []
        inner_fold = {int(r["path_index"]): r for r in inner if int(r["outer_subject"]) == subject}
        for i, path in enumerate(paths):
            role = ("outer_test" if path["subject_key"] == subject else "meta" if
                    path["subject_key"] in split["meta_calibration"] else "policy_validation" if
                    path["subject_key"] in split["policy_validation"] else None)
            if role is None:
                continue
            scores = outer[i] if role == "outer_test" else inner_fold[i]
            keys = ("deep", "deep_temporal", "biomarker", "covariance", "full")
            probability = {}
            for key in keys:
                raw = scores[key + "_probability"] if role == "outer_test" else scores[key]
                if key == "full" and not raw and role != "outer_test":
                    raw = scores["deep_temporal_bio"]
                probability[key + "_probability"] = rounded(float(raw)) if raw else None
            sl = slice(path["window_start"], path["window_end"])
            z = predict_windows(model, state["mean"].to(device), state["scale"].to(device), windows[sl], device)
            dt = temporal_features(z, times[sl])
            # PSD feature layout: 120 log-powers, 120 relative powers; then 120 reference log-ratios.
            absolute = pooled["bio_absolute"][i]
            ref = pooled["bio_reference"][i]
            relative = absolute[120:].reshape(30, 4).mean(axis=0)
            changes = ref[240:].reshape(30, 4).mean(axis=0) if np.isfinite(ref).all() else None
            temporal = dict(raw_mean_probability=rounded(np.mean(1 / (1 + np.exp(-z)))),
                            raw_probability_std=rounded(dt[1]), raw_recent_mean=rounded(dt[3]),
                            raw_recent_slope_per_second=rounded(dt[4]), raw_last_probability=rounded(dt[5]))
            evidence = dict(probabilities=probability, temporal=temporal,
                            relative_band_power=[rounded(v) for v in relative],
                            reference_log_change=None if changes is None else [rounded(v) for v in changes],
                            covariance_distance=rounded(pooled["covariance"][i, -1]),
                            reference_available=path["reference_available"],
                            qc_accepted_fraction=rounded(path["accepted_windows"] / path["total_possible_windows"]))
            records.append(dict(path_index=i, role=role, evidence=evidence))
        evidence_folds.append(dict(outer_subject=subject, split=split, records=records))
        print(f"evidence fold {subject:02d}: {len(records)} meta/validation/test paths", flush=True)
    write_json(args.out / "fold_evidence.json", evidence_folds)
    write_json(args.out / "evaluation.json", [dict(path_index=i, subject_key=p["subject_key"], label=p["label"])
                                              for i, p in enumerate(paths)])
    protocol = dict(status="frozen_before_cloud_evaluation", seed=2026, total_paths=147, subjects=24,
                    endpoint="whole_path_score_ge30", outer="same 24 LOSO folds as frozen pilot",
                    query_labels_sent=False, raw_eeg_sent=False, source_identifiers_sent=False,
                    example_source="fold-specific 5 meta subjects; 4 examples per class when available",
                    example_calibration_caveat="Base predictions are held out; meta heads have seen meta labels",
                    numerical_candidates="original full, improved path models selected within the fold",
                    llm_modes=["zero_shot", "few_shot", "improved_few_shot"],
                    blend_weights=[0, .25, .5, .75, 1], fixed_blend_weight=.25,
                    selection="independent 4-subject policy-validation only; ties prefer less LLM weight",
                    threshold=.5, calls_per_mode=24, query_batch="one randomly shuffled fold batch; classify independently",
                    failure_policy="retain all cases, report coverage, numeric fallback for combined systems",
                    exploratory=True, prior_outer_results_previously_examined=True, pristine_external_validation=False,
                    source_artifact_sha256={name: digest(result / name) for name in
                                            ("path_oof.csv", "inner_predictions.csv", "protocol.json")},
                    split_manifest_sha256=digest(PILOT_OUT / "split_manifest.json"),
                    code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                    improvement_plan="path-level MIL loss, dropout/window normalization; low-dimensional spectral and train-reference covariance classifiers; grouped internal regularization search")
    write_json(args.out / "protocol.json", protocol)


if __name__ == "__main__":
    main()
