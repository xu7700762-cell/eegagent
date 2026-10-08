"""Validate V2 subject isolation, frozen sources, scores and lazy checkpoint replay."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .data import digest, read_csv, write_json
from .experiment import DEFAULT_DATA, DEFAULT_OUT, ToolReplay, metrics
from .model import CompactEEGCNN


def validate(out=DEFAULT_OUT, data_root=DEFAULT_DATA):
    out = Path(out)
    result, cache = out / "results", out / "cache"
    oof = read_csv(result / "path_oof.csv")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((cache / "dataset_audit.json").read_text(encoding="utf-8"))
    paths = json.loads((cache / "evaluation_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((result / "protocol.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == "raw_event_eligibility_v2"
    assert protocol["schema_version"] == "pilot_v2" and not protocol["smoke"]
    assert summary["outer_folds"] == len(splits) == audit["subjects"]
    assert len(oof) == len({int(r["path_index"]) for r in oof}) == len(paths) == audit["paths"]
    assert digest(cache / "eligibility_audit.json") == audit["eligibility_sha256"]
    eligibility = json.loads((cache / "eligibility_audit.json").read_text(encoding="utf-8"))
    assert eligibility["paths"] == len(paths)
    assert len(paths) + eligibility["excluded_paths"] == eligibility["candidate_paths"]
    for source in audit["sources"]:
        raw = Path(source["path"])
        assert raw.stat().st_size == source["size_bytes"] and raw.stat().st_mtime_ns == source["mtime_ns"]
        assert digest(str(raw) + ".dpo") == source["dpo_sha256"]
        if source.get("ceo_sha256"):
            assert digest(str(raw) + ".ceo") == source["ceo_sha256"]
    assert digest(Path(data_root) / "labels/task_segments_with_path_scores.csv") == audit["label_sha256"]
    assert digest(Path(data_root) / "labels/state_segments_by_mark.csv") == audit["state_sha256"]
    for filename, expected in protocol["code_sha256"].items():
        assert digest(Path(__file__).parent / filename) == expected, f"Frozen source changed: {filename}"
    for filename, expected in protocol["external_code_sha256"].items():
        assert digest(Path(filename)) == expected, f"Frozen external source changed: {filename}"
    packs = [json.loads(s) for s in (result / "assessment_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(packs) == 4 * len(paths)
    for pack in packs:
        assert len(pack["tool_calls"]) == len(set(pack["tool_calls"]))
        if pack["result"]["stop_reason"] == "validated_calibration_quality_ood_exit":
            assert not set(pack["tool_calls"]) & {"CaseRetriever", "BiomarkerCalculator", "CovarianceAnalyzer"}
    lookup = {int(r["path_index"]): r for r in oof}
    rows = [json.loads(s) for s in (cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    references = np.load(cache / "path_references.npy")
    powers = np.load(cache / "path_reference_power.npy")
    starts = np.asarray([r["start_sample"] / 1024 for r in rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    errors = []
    for split in splits:
        keys = ("base_train", "head_fit", "probability_calibration", "policy_validation", "outer_test")
        groups = [set(split[k]) for k in keys]
        assert [len(g) for g in groups] == [14, 3, 2, 4, 1]
        assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
        assert groups[1] | groups[2] == set(split["meta_calibration"])
        stem = result / "checkpoints" / f"outer_subject_{split['outer_subject']:02d}"
        state = torch.load(str(stem) + ".pt", map_location=device, weights_only=True)
        components = joblib.load(str(stem) + ".joblib")
        assert components["split"] == split and state["training_subjects"] == split["base_train"]
        model = CompactEEGCNN().to(device)
        model.load_state_dict(state["model"])
        model.eval()
        bundle = dict(cnn=model, mean=state["mean"].to(device), scale=state["scale"].to(device),
                      **{k: v for k, v in components.items() if k != "split"})
        for i, p in enumerate(paths):
            if p["subject_key"] != split["outer_subject"]:
                continue
            sl = slice(p["window_start"], p["window_end"])
            for method in ("deep", "full", "adaptive"):
                pack = ToolReplay(p, windows[sl], starts[sl], references[i], powers[i], bundle, device).assess(method)
                actual = pack["result"]["probability"]
                expected = float(lookup[i][f"{method}_probability"]) if lookup[i][f"{method}_probability"] else None
                assert (actual is None) == (expected is None)
                assert pack["result"]["state"] == lookup[i][f"{method}_state"]
                if actual is not None:
                    error = abs(actual - expected)
                    assert error < 2e-5
                    errors.append(error)
                if method == "adaptive" and "reliability" in pack:
                    assert actual == pack["reliability"]["p_cal"]
    for name in ("deep", "deep_temporal", "biomarker", "covariance", "full", "adaptive"):
        yy = [int(r["label"]) for r in oof]
        pp = [float(r[f"{name}_probability"]) if r[f"{name}_probability"] else np.nan for r in oof]
        published = [r[f"{name}_state"] in ("high", "low") for r in oof]
        recomputed = metrics(yy, pp, published)
        for key, value in recomputed.items():
            assert value == summary["methods"][name][key], (name, key)
    validation = dict(status="passed", schema_version="pilot_v2", unique_paths=len(paths),
                      folds=len(splits), raw_event_complete=True, split_disjointness=True,
                      calibration_excludes_head_fit_subjects=True, frozen_sources_match=True,
                      metrics_recomputed=True, checkpoint_comparisons=len(errors),
                      max_probability_reload_error=max(errors, default=0), llm_tested=False, pi_tested=False)
    write_json(result / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    validate(args.out, args.data_root)


if __name__ == "__main__":
    main()
