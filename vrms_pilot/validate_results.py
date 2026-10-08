"""Validate finished real-data artifacts without retraining or retuning."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .data import digest, write_json
from .experiment import DEFAULT_DATA, DEFAULT_OUT, ToolReplay, metrics
from .model import CompactEEGCNN


def main():
    out = DEFAULT_OUT
    result = out / "results"
    import csv
    with (result / "path_oof.csv").open(encoding="utf-8-sig") as f:
        oof = list(csv.DictReader(f))
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "cache/dataset_audit.json").read_text(encoding="utf-8"))
    paths = json.loads((out / "cache/evaluation_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((result / "protocol.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed_single_seed_exploratory_loso"
    assert summary["outer_folds"] == len(splits) == 24
    assert len(oof) == 147 == len(paths)
    assert len({int(r["path_index"]) for r in oof}) == 147
    assert len({int(r["subject_key"]) for r in oof}) == 24
    assert digest(DEFAULT_DATA / "labels/task_segments_with_path_scores.csv") == audit["label_sha256"]
    assert digest(DEFAULT_DATA / "labels/state_segments_by_mark.csv") == audit["state_sha256"]
    for source in audit["sources"]:
        raw = Path(source["path"])
        assert raw.stat().st_size == source["size_bytes"]
        assert raw.stat().st_mtime_ns == source["mtime_ns"]
        assert digest(str(raw) + ".dpo") == source["dpo_sha256"]
    for filename, expected_hash in protocol["code_sha256"].items():
        assert digest(Path(__file__).parent / filename) == expected_hash, f"Training code changed: {filename}"
    forbidden = {"label", "score", "path_score", "ssq", "SSQ", "subject_key", "true_label"}
    def check_blind(obj):
        if isinstance(obj, dict):
            assert not forbidden.intersection(obj), "Ground truth leaked into online evidence"
            for v in obj.values():
                check_blind(v)
        elif isinstance(obj, list):
            for v in obj:
                check_blind(v)
    packs = [json.loads(s) for s in (result / "assessment_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(packs) == 4 * 147
    order = ["SignalQualityChecker", "DeepVRMSDetector", "TemporalAnalyzer", "BiomarkerCalculator", "CovarianceAnalyzer"]
    for pack in packs:
        check_blind(pack)
        assert len(pack["tool_calls"]) == len(set(pack["tool_calls"]))
        calls = pack["tool_calls"]
        assert calls == order[:len(calls)]
        assert pack["ensemble_std"] is None
    lookup = {int(r["path_index"]): r for r in oof}
    window_records = [json.loads(s) for s in (out / "cache/window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    for record in window_records:
        check_blind(record)
        assert record["end_sample"] <= paths[record["path_index"]]["path_end_sec"] * 1024 + 1
    windows = np.load(out / "cache/windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    references = np.load(out / "cache/path_references.npy")
    powers = np.load(out / "cache/path_reference_power.npy")
    starts = np.asarray([r["start_sample"] / 1024 for r in window_records])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reload_errors = []
    for split in splits:
        group_names = ("base_train", "meta_calibration", "policy_validation", "outer_test")
        groups = [set(split[name]) for name in group_names]
        assert [len(s) for s in groups] == [14, 5, 4, 1]
        assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
        checkpoint = result / "checkpoints" / f"outer_subject_{split['outer_subject']:02d}"
        state = torch.load(str(checkpoint) + ".pt", map_location=device, weights_only=True)
        components = joblib.load(str(checkpoint) + ".joblib")
        assert state["training_subjects"] == split["base_train"]
        assert state["heldout_subject"] == split["outer_subject"]
        assert components["split"] == split
        model = CompactEEGCNN().to(device)
        model.load_state_dict(state["model"])
        model.eval()
        bundle = dict(cnn=model, mean=state["mean"].to(device), scale=state["scale"].to(device),
                      **{key: components[key] for key in ("classifiers", "meta", "margins")})
        i = next(i for i, p in enumerate(paths) if p["subject_key"] == split["outer_subject"])
        p = paths[i]
        sl = slice(p["window_start"], p["window_end"])
        for method in ("deep", "full", "adaptive"):
            replay = ToolReplay(p, windows[sl], starts[sl], references[i], powers[i], bundle, device)
            actual = replay.assess(method)["result"]["probability"]
            expected = float(lookup[i][f"{method}_probability"])
            error = abs(actual - expected)
            assert error < 2e-5, (split["outer_subject"], method, error)
            reload_errors.append(error)
    for name in ("deep", "deep_temporal", "biomarker", "covariance", "full", "adaptive"):
        yy = [int(r["label"]) for r in oof]
        pp = [float(r[f"{name}_probability"]) if r[f"{name}_probability"] else np.nan for r in oof]
        published = [r[f"{name}_state"] in ("high", "low") for r in oof]
        recomputed = metrics(yy, pp, published)
        for key in ("numeric_threshold_diagnostic", "published_conditional", "auroc", "ece", "brier",
                    "scoreable_paths", "published_paths"):
            assert recomputed[key] == summary["methods"][name][key], (name, key)
    validation = dict(status="passed", folds=24, unique_paths=147, online_evidence_records=len(packs),
                      split_disjointness=True, raw_files_stat_and_dpo_hashes_unchanged=True,
                      label_hashes_unchanged=True, online_ground_truth_absent=True,
                      training_code_hashes_match=True, metrics_recomputed=True,
                      checkpoint_reload_comparisons=len(reload_errors),
                      max_probability_reload_error=max(reload_errors), pi_tested=False, llm_tested=False)
    write_json(result / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
