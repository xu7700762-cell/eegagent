"""Fold-internal improvements aligned with the whole-path endpoint.

Keep the original 14/5/4/test subjects. Select regularization on grouped base
training folds, epochs/calibration on meta subjects, and final numerical
candidate on independent policy subjects. Outer labels are never used here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from vrms_pilot.data import covariance, digest, read_csv, write_json
from vrms_pilot.engine import temporal_features
from vrms_pilot.experiment import DEFAULT_OUT as PILOT_OUT, fit_logistic, probabilities, save_csv
from vrms_pilot.model import CompactEEGCNN, predict_windows, seed_everything
from .prepare import DEFAULT_OUT

C_VALUES = (.001, .01, .1, 1.)
EPOCH_CANDIDATES = (8, 16, 32)
REGIONS = (("Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8", "F11", "F12"),
           ("FC3", "FCz", "FC4", "FT11", "FT12"),
           ("C3", "Cz", "C4"), ("CP3", "CPz", "CP4", "P3", "Pz", "P4"),
           ("O1", "Oz", "O2"), ("T7", "T8", "P7", "P8"))


def path_features(cache, paths, windows, channel_names):
    bio = np.load(cache / "bio_absolute.npy", mmap_mode="r")
    region_ids = [[channel_names.index(c) for c in region] for region in REGIONS]
    spectral, regional, covariances = [], [], []
    for path in paths:
        sl = slice(path["window_start"], path["window_end"])
        power = np.asarray(bio[sl, :120], dtype=float).reshape(-1, 30, 4)
        # Ratios remove broadband scale without using other test paths/windows.
        ratios = power[:, :, [0, 1, 3]] - power[:, :, 2:3]
        third = max(1, len(power) // 3)
        change = ratios[-third:].mean(axis=0) - ratios[:third].mean(axis=0)
        spectral.append(np.r_[ratios.mean(axis=0).ravel(), change.ravel()])
        regions = np.stack([ratios[:, ids].mean(axis=1) for ids in region_ids], axis=1)
        regional.append(np.r_[regions.mean(axis=0).ravel(), regions.std(axis=0).ravel(),
                              regions[-third:].mean(axis=0).ravel(),
                              (regions[-third:].mean(axis=0) - regions[:third].mean(axis=0)).ravel()])
        matrices = [covariance(x) for x in windows[sl]]
        matrices = [c / np.trace(c) for c in matrices]
        covariances.append(np.mean(matrices, axis=0))
    return dict(spatial_spectrum=np.asarray(spectral), regional_spectrum=np.asarray(regional)), np.asarray(covariances)


def tangent_features(covariances, train_indices):
    reference = covariances[train_indices].mean(axis=0)
    e, u = np.linalg.eigh(reference)
    invroot = (u * (1 / np.sqrt(np.maximum(e, 1e-12)))) @ u.T
    ii, jj = np.triu_indices(30)
    features = []
    for covariance_matrix in covariances:
        c = invroot @ covariance_matrix @ invroot
        e, u = np.linalg.eigh((c + c.T) / 2)
        logc = (u * np.log(np.maximum(e, 1e-12))) @ u.T
        features.append(logc[ii, jj] * np.where(ii == jj, 1, np.sqrt(2)))
    return np.asarray(features), reference


def select_logistic(name, features, covariances, y, subjects, base):
    folds = list(GroupKFold(n_splits=3).split(base, y[base], subjects[base]))
    scores = []
    for c in C_VALUES:
        observed, predicted = [], []
        for training_positions, validation_positions in folds:
            train, val = base[training_positions], base[validation_positions]
            x = tangent_features(covariances, train)[0] if name == "tangent_covariance" else features[name]
            model = fit_logistic(x, y, train, c=c)
            p = probabilities(model, x)[val]
            if not np.isfinite(p).all():
                raise ValueError("Internal grouped model cannot provide all probabilities")
            observed.extend(y[val])
            predicted.extend(p >= .5)
        scores.append(float(balanced_accuracy_score(observed, predicted)))
    selected = int(np.argmax(scores))  # smaller C first in ties
    x, reference = tangent_features(covariances, base) if name == "tangent_covariance" else (features[name], None)
    model = fit_logistic(x, y, base, c=C_VALUES[selected])
    return model, x, reference, dict(c=C_VALUES[selected], grouped_bacc=scores,
                                     c_candidates=list(C_VALUES))


class PathMILCNN(CompactEEGCNN):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.35)

    def forward(self, x):
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1), z.std(dim=-1, correction=0)), dim=1)
        return self.head(self.dropout(pooled)).squeeze(-1)


def normalize_windows(x):
    x = torch.as_tensor(np.array(x, dtype=np.float32))
    return ((x - x.mean(dim=-1, keepdim=True)) /
            x.std(dim=-1, correction=0, keepdim=True).clamp_min(.01)).clamp(-20, 20)


def mil_scores(model, windows, paths, starts, indices, device):
    features = np.full((len(paths), 7), np.nan)
    model.eval()
    with torch.inference_mode():
        for i in indices:
            path = paths[i]
            sl = slice(path["window_start"], path["window_end"])
            x = normalize_windows(windows[sl]).to(device)
            logits = model(x).cpu().numpy().astype(float)
            features[i] = temporal_features(logits, starts[sl])
    return features


def train_mil(windows, paths, starts, base, meta, device, y):
    seed_everything(2026)
    model = PathMILCNN().to(device)
    # Each path contributes ONE BCE term; its windows are a bag of weak evidence.
    bags = {i: normalize_windows(windows[paths[i]["window_start"]:paths[i]["window_end"]]).to(device) for i in base}
    counts = np.bincount(y[base], minlength=2)
    weights = len(base) / (2 * counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    rng = np.random.default_rng(2026)
    losses, candidates = [], []
    best, best_loss, best_epoch = None, float("inf"), None
    for epoch in range(1, max(EPOCH_CANDIDATES) + 1):
        model.train()
        order = rng.permutation(base)
        total = 0.
        for offset in range(0, len(order), 8):
            indices = order[offset:offset + 8]
            sampled = [bags[i][rng.choice(len(bags[i]), 16, replace=len(bags[i]) < 16)] for i in indices]
            x = torch.stack(sampled)
            logits = model(x.flatten(0, 1)).reshape(len(indices), 16).mean(dim=1)
            targets = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            w = torch.as_tensor(weights[y[indices]], dtype=torch.float32, device=device)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(base))
        if epoch in EPOCH_CANDIDATES:
            meta_features = mil_scores(model, windows, paths, starts, meta, device)
            z = np.clip(meta_features[meta, 0], -30, 30)
            validation_loss = float(np.mean(np.logaddexp(0, z) - y[meta] * z))
            candidates.append(dict(epoch=epoch, meta_path_bce=validation_loss))
            if validation_loss < best_loss:
                best_loss, best_epoch, best = validation_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    return model, dict(selected_epoch=best_epoch, epoch_selection=candidates, training_path_losses=losses,
                       objective="BCE of mean sampled-window logits per path", dropout=.35,
                       normalization="within each arrived window/channel", train_paths=len(base))


def calibrate(raw_features, y, meta):
    model = fit_logistic(raw_features, y, meta, c=.1, balanced=False)
    return model, probabilities(model, raw_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")).get("schema_version") == "uncertainty_agent_v2":
        raise ValueError("Historical candidate training is excluded from V2; use the frozen V2 pilot probability")
    cache = PILOT_OUT / "cache"
    paths = json.loads((cache / "evaluation_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((cache / "dataset_audit.json").read_text(encoding="utf-8"))
    splits = json.loads((PILOT_OUT / "split_manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((args.out / "fold_evidence.json").read_text(encoding="utf-8"))
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    window_rows = [json.loads(s) for s in (cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = np.asarray([r["start_sample"] / 1024 for r in window_rows])
    features, covariances = path_features(cache, paths, windows, audit["channel_order"])
    y = np.asarray([p["label"] for p in paths])
    subjects = np.asarray([p["subject_key"] for p in paths])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = args.out / "improvements"
    directory.mkdir(exist_ok=True)
    (directory / "checkpoints").mkdir(exist_ok=True)
    # This plan is saved before candidate OOF outcomes are available.
    write_json(directory / "protocol.json", dict(seed=2026, base_subjects=14, meta_subjects=5,
               policy_subjects=4, outer_test_subjects=1, c_candidates=list(C_VALUES),
               mil_epochs=list(EPOCH_CANDIDATES), outer_labels_used_for_selection=False,
               calibration_c=.1, selector="max policy-validation BACC; ties prefer original full",
               feature_timing="path end only; never another test path for normalization/reference",
               source_sha256=digest(Path(__file__))))
    rows, diagnostics = [], []
    began = time.perf_counter()
    for fold, split in zip(evidence, splits):
        assert fold["outer_subject"] == split["outer_subject"]
        base = np.flatnonzero(np.isin(subjects, split["base_train"]))
        meta = np.flatnonzero(np.isin(subjects, split["meta_calibration"]))
        val = np.flatnonzero(np.isin(subjects, split["policy_validation"]))
        test = np.flatnonzero(subjects == split["outer_subject"])
        models, calibrators, scores, details = {}, {}, {}, {}
        for name in ("regional_spectrum", "spatial_spectrum", "tangent_covariance"):
            model, x, reference, detail = select_logistic(name, features, covariances, y, subjects, base)
            raw = probabilities(model, x)
            z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))[:, None]
            calibrators[name], scores[name] = calibrate(z, y, meta)
            models[name] = dict(model=model, reference=reference)
            details[name] = detail
        mil, training = train_mil(windows, paths, starts, base, meta, device, y)
        dt = mil_scores(mil, windows, paths, starts, np.r_[meta, val, test], device)
        calibrators["path_mil"], scores["path_mil"] = calibrate(dt, y, meta)
        original = np.full(len(paths), np.nan)
        for record in fold["records"]:
            original[record["path_index"]] = record["evidence"]["probabilities"]["full_probability"]
        scores = dict(original_full=original, **scores)
        scores["candidate_mean"] = np.mean([scores[k] for k in
                                            ("original_full", "regional_spectrum", "spatial_spectrum", "tangent_covariance", "path_mil")], axis=0)
        # Only independent policy labels choose the deployable candidate.
        validation = {name: float(balanced_accuracy_score(y[val], p[val] >= .5)) for name, p in scores.items()}
        selected = max(validation, key=validation.get)
        scores["validated_numeric"] = scores[selected]
        for record in fold["records"]:
            i = record["path_index"]
            improved = {k + "_probability": round(float(v[i]), 4) for k, v in scores.items() if k != "original_full"}
            record["evidence"]["improved_probabilities"] = improved
            rows.append(dict(outer_subject=split["outer_subject"], path_index=int(i),
                             subject_key=int(subjects[i]), role=record["role"], label=int(y[i]),
                             selected_numeric=selected, **{k: float(v[i]) for k, v in scores.items()}))
        filename = directory / "checkpoints" / f"outer_subject_{split['outer_subject']:02d}"
        joblib.dump(dict(models=models, calibrators=calibrators, split=split, selected=selected), str(filename) + ".joblib")
        torch.save(dict(model=mil.state_dict(), split=split, training=training), str(filename) + ".pt")
        diagnostics.append(dict(outer_subject=split["outer_subject"], selection=selected,
                                policy_bacc=validation, model_selection=details, mil_training=training))
        save_csv(directory / "predictions.csv", rows)
        write_json(directory / "fold_diagnostics.json", diagnostics)
        print(f"improvement fold {split['outer_subject']:02d}: selected={selected} "
              f"mil_epoch={training['selected_epoch']} elapsed={time.perf_counter()-began:.1f}s", flush=True)
        del mil
        torch.cuda.empty_cache()
    # The original LLM modes ignore the new evidence block; their request hashes stay identical.
    write_json(args.out / "fold_evidence.json", evidence)
    write_json(directory / "run_status.json", dict(status="completed_24_fold_exploratory", folds=24,
                                                  elapsed_seconds=time.perf_counter() - began,
                                                  source_sha256=digest(Path(__file__))))


if __name__ == "__main__":
    main()
