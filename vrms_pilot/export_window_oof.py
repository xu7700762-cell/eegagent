"""Export window scores from frozen outer checkpoints; no fitting or selection."""
import csv
import json

import numpy as np
import torch

from .data import write_json
from .experiment import DEFAULT_OUT
from .model import CompactEEGCNN, predict_windows


def main():
    out = DEFAULT_OUT
    cache, result = out / "cache", out / "results"
    paths = json.loads((cache / "evaluation_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((cache / "dataset_audit.json").read_text(encoding="utf-8"))
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    records = [json.loads(s) for s in (cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    logits = np.full(len(windows), np.nan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for subject in sorted({p["subject_key"] for p in paths}):
        state = torch.load(result / "checkpoints" / f"outer_subject_{subject:02d}.pt", map_location=device, weights_only=True)
        model = CompactEEGCNN().to(device)
        model.load_state_dict(state["model"])
        model.eval()
        ids = np.concatenate([np.arange(p["window_start"], p["window_end"])
                              for p in paths if p["subject_key"] == subject])
        logits[ids] = predict_windows(model, state["mean"].to(device), state["scale"].to(device), windows[ids], device)
    assert np.isfinite(logits).all()
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    evaluation = []
    with (result / "window_oof_scores.jsonl").open("w", encoding="utf-8") as f:
        for i, source in enumerate(records):
            path = paths[source["path_index"]]
            row = dict(evidence_id=f"window-{i:05d}", episode_handle=source["episode_handle"],
                       start_sample=source["start_sample"], end_sample=source["end_sample"],
                       data_cutoff_sec=source["cutoff_sec"], deep_window_logit=float(logits[i]),
                       deep_window_score=float(probabilities[i]),
                       score_scope="weak_path_model_score_not_instantaneous_symptom_probability")
            f.write(json.dumps(row, allow_nan=False) + "\n")
            evaluation.append(dict(evidence_id=row["evidence_id"], subject_key=path["subject_key"],
                                   path_index=path["path_index"], score=path["score"], label=path["label"],
                                   label_scope="whole_path_weak_label", outer_subject=path["subject_key"]))
    with (result / "window_oof_evaluation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(evaluation[0]))
        writer.writeheader()
        writer.writerows(evaluation)
    note = dict(status="completed", window_scores=len(records), paths=len(paths),
                heldout_subjects=24, refit=False, thresholds_retuned=False,
                live_scores_label_free=True, evaluation_labels_separate=True,
                labels_inherited_from_whole_path=True)
    write_json(result / "window_oof_export.json", note)
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
