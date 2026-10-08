"""Verify actual requests, frozen sources and all improved outer checkpoints."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from vrms_pilot.data import digest, read_csv, write_json
from vrms_pilot.experiment import DEFAULT_DATA, DEFAULT_OUT as PILOT_OUT, probabilities
from .cloud import SYSTEM, build_job, check_blind, choose_examples, parse_predictions
from .improve import PathMILCNN, path_features, tangent_features, mil_scores
from .prepare import DEFAULT_OUT
from .evaluate import metric


def main():
    out = DEFAULT_OUT
    cache = PILOT_OUT / "cache"
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    for name, sha in protocol["source_artifact_sha256"].items():
        assert digest(PILOT_OUT / "results" / name) == sha
    assert digest(PILOT_OUT / "split_manifest.json") == protocol["split_manifest_sha256"]
    old_protocol = json.loads((PILOT_OUT / "results/protocol.json").read_text(encoding="utf-8"))
    for name, sha in old_protocol["code_sha256"].items():
        assert digest(Path("vrms_pilot") / name) == sha, "Frozen pilot source was changed"
    improved_protocol = json.loads((out / "improvements/protocol.json").read_text(encoding="utf-8"))
    assert digest(Path("vrms_cloud/improve.py")) == improved_protocol["source_sha256"]
    evaluation = {r["path_index"]: r for r in json.loads((out / "evaluation.json").read_text(encoding="utf-8"))}
    folds = json.loads((out / "fold_evidence.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    clouds = []
    for mode in summary["cloud"]["modes"]:
        for fold in folds:
            job = build_job(fold, evaluation, mode)
            log = json.loads((out / "cloud_calls" / f"{mode}_fold_{fold['outer_subject']:02d}.json").read_text(encoding="utf-8"))
            assert job["request_sha256"] == log["request_sha256"]
            assert job["user_message"] == log["body"]["input"][1]["content"]
            message = json.loads(job["user_message"])
            for query in message["queries"]:
                check_blind(query)
            for example in message["examples"]:
                check_blind(example["evidence"])
            assert set(job["example_indices"]).isdisjoint(r["path_index"] for r in job["mapping"].values())
            if log["success"]:
                assert parse_predictions(log["attempts"][-1]["response"], job["mapping"]) == log["predictions"]
            clouds.append(log)
    paths = json.loads((cache / "evaluation_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((cache / "dataset_audit.json").read_text(encoding="utf-8"))
    assert digest(DEFAULT_DATA / "labels/task_segments_with_path_scores.csv") == audit["label_sha256"]
    assert digest(DEFAULT_DATA / "labels/state_segments_by_mark.csv") == audit["state_sha256"]
    for source in audit["sources"]:
        raw = Path(source["path"])
        assert raw.stat().st_size == source["size_bytes"]
        assert raw.stat().st_mtime_ns == source["mtime_ns"]
        assert digest(str(raw) + ".dpo") == source["dpo_sha256"]
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    window_rows = [json.loads(s) for s in (cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = np.asarray([r["start_sample"] / 1024 for r in window_rows])
    features, cov = path_features(cache, paths, windows, audit["channel_order"])
    predictions = read_csv(out / "improvements/predictions.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    errors = []
    for fold in folds:
        split = fold["split"]
        s = split["outer_subject"]
        groups = [set(split[k]) for k in ("base_train", "meta_calibration", "policy_validation", "outer_test")]
        assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
        base = [i for i, p in enumerate(paths) if p["subject_key"] in split["base_train"]]
        test = [i for i, p in enumerate(paths) if p["subject_key"] == s]
        filename = out / "improvements/checkpoints" / f"outer_subject_{s:02d}"
        bundle = joblib.load(str(filename) + ".joblib")
        state = torch.load(str(filename) + ".pt", map_location=device, weights_only=False)
        assert bundle["split"] == state["split"] == split
        model = PathMILCNN().to(device)
        model.load_state_dict(state["model"])
        dt = mil_scores(model, windows, paths, starts, test, device)
        scores = {"path_mil": probabilities(bundle["calibrators"]["path_mil"], dt)}
        for name in ("regional_spectrum", "spatial_spectrum", "tangent_covariance"):
            if name == "tangent_covariance":
                x, reference = tangent_features(cov, base)
                np.testing.assert_array_equal(reference, bundle["models"][name]["reference"])
            else:
                x = features[name]
            raw = probabilities(bundle["models"][name]["model"], x)
            z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))[:, None]
            scores[name] = probabilities(bundle["calibrators"][name], z)
        rows = {int(r["path_index"]): r for r in predictions if int(r["outer_subject"]) == s and r["role"] == "outer_test"}
        for i in test:
            for name, p in scores.items():
                errors.append(abs(p[i] - float(rows[i][name])))
    assert max(errors) < 2e-5
    oof = read_csv(out / "path_oof.csv")
    assert len(oof) == len({int(r["path_index"]) for r in oof}) == 147
    yy = np.asarray([int(r["label"]) for r in oof])
    for name, expected in summary["methods"].items():
        pp = np.asarray([float(r[name]) if r[name] else np.nan for r in oof])
        assert metric(yy, pp) == expected, f"Metric mismatch: {name}"
    independent_checked = 0
    if "single_path_cloud" in summary:
        for fold in folds:
            examples = [dict(evidence=r["evidence"], observed_class="high" if evaluation[r["path_index"]]["label"] else "low")
                        for r in choose_examples(fold["records"], evaluation, 2026 + fold["outer_subject"])]
            for record in fold["records"]:
                if record["role"] != "outer_test":
                    continue
                i = record["path_index"]
                log = json.loads((out / "independent_calls" / f"path_{i:03d}_cloud_call.json").read_text(encoding="utf-8"))
                body = log["body"]
                message = body["input"][1]["content"]
                prompt = json.loads(message)
                assert prompt["examples"] == examples
                assert prompt["queries"] == [dict(id="qsingle", evidence=record["evidence"])]
                assert "previous_response_id" not in body
                check_blind(prompt["queries"])
                assert log["request_sha256"] == hashlib.sha256((SYSTEM + message).encode()).hexdigest()
                pack = json.loads((out / "independent_calls" / f"path_{i:03d}_evidence_pack.json").read_text(encoding="utf-8"))
                row = next(r for r in predictions if int(r["path_index"]) == i and r["role"] == "outer_test")
                numeric = float(row["validated_numeric"])
                if log["success"]:
                    response = parse_predictions(log["attempts"][-1]["response"], ["qsingle"])[0]
                    assert pack["high_probability"] == .75 * numeric + .25 * response["high_probability"]
                else:
                    assert pack["high_probability"] == numeric and pack["fallback_used"]
                assert not pack["confidence_calibrated"]
                independent_checked += 1
        assert independent_checked == 147
    result = dict(status="passed", original_source_and_results_unchanged=True,
                  raw_stat_header_and_label_hashes_unchanged=True,
                  outer_labels_absent_from_requests=True, example_subjects_internal_only=True,
                  cloud_request_hashes_checked=len(clouds), cloud_responses_checked=sum(c["success"] for c in clouds),
                  independent_single_path_requests_checked=independent_checked,
                  complete_outer_paths=147, subjects=24, improvement_checkpoint_comparisons=len(errors),
                  max_reload_probability_error=float(max(errors)), metrics_recomputed=True,
                  llm_participates_in_classification=True, clinical_or_instantaneous_validation=False,
                  implementation_sha256={p.name: digest(p) for p in Path("vrms_cloud").glob("*.py")})
    write_json(out / "validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
