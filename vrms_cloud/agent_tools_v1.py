"""Independent, subject-validated numeric tools for the forced-binary Agent.

The locked FEMBA+MIL baseline is never fitted or changed. Tool hyperparameters
use subject OOF meta predictions, and policy subjects evaluate corrections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .cloud import write_json
from .second_judgment import DEFAULT_MODELS, sha256

DEFAULT_OUT = Path("outputs/vrms_agent_loso/v4_tools_r1_20261008")
FAMILIES = ("femba_features", "spectral", "spatial_covariance")
CONFIGS = ("lr_0.01", "lr_0.1", "lr_1", "rbf_1", "rbf_10", "pca16_rbf_1", "pca32_lr_0.1", "trees")
CASE_CONFIGS = tuple(product((3, 5, 9), (False, True)))
SEED = 2026


def indices(paths, subjects):
    return np.asarray([i for i, p in enumerate(paths) if p["subject_key"] in set(subjects)], int)


def binary_metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    selected = np.isfinite(p)
    if not selected.any():
        return dict(paths=len(y), available=0, acc=None, bacc=None, auroc=None)
    yy, pp = y[selected], p[selected]
    return dict(paths=len(y), available=int(selected.sum()), acc=float(accuracy_score(yy, pp >= .5)),
        bacc=float(balanced_accuracy_score(yy, pp >= .5)) if len(np.unique(yy)) == 2 else None,
        auroc=float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None)


def subject_weights(groups):
    counts = Counter(groups)
    return np.asarray([len(groups) / (len(counts) * counts[g]) for g in groups], float)


def classifier(config):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if config.startswith("pca"):
        size = int(config.split("_")[0][3:])
        steps.append(PCA(n_components=size, svd_solver="full"))
        config = config.split("_", 1)[1]
    if config.startswith("lr_"):
        model = LogisticRegression(C=float(config.split("_")[1]), class_weight="balanced",
                                   solver="liblinear", max_iter=2000, random_state=SEED)
    elif config.startswith("rbf_"):
        model = SVC(C=float(config.split("_")[1]), kernel="rbf", class_weight="balanced")
    elif config == "trees":
        model = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=3, max_depth=5,
            class_weight="balanced", max_features="sqrt", n_jobs=1, random_state=SEED)
    else:
        raise ValueError("Unknown frozen tool configuration")
    return make_pipeline(*steps, model)


def predict_score(model, values):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), float)
    return logit(np.clip(model.predict_proba(values)[:, 1], 1e-6, 1 - 1e-6))


def fit_calibrator(scores, y, groups):
    if len(np.unique(y)) != 2 or len(np.unique(groups)) < 3:
        return None
    model = LogisticRegression(C=1, solver="liblinear", max_iter=1000)
    model.fit(np.asarray(scores)[:, None], y, sample_weight=subject_weights(groups))
    return model if model.coef_[0, 0] > 0 else None


def calibrated(calibrator, scores):
    return expit(scores) if calibrator is None else calibrator.predict_proba(np.asarray(scores)[:, None])[:, 1]


def case_predictions(x, y, groups, train, query, config):
    k, weighted = config
    scaler = StandardScaler().fit(x[train])
    reference, queries = scaler.transform(x[train]), scaler.transform(x[query])
    predictions, records = [], []
    for q in queries:
        distances = np.sqrt(np.mean((reference - q) ** 2, axis=1))
        ordered = sorted(zip(train.tolist(), distances.tolist()), key=lambda v: (v[1], v[0]))
        seen, chosen = set(), []
        for i, distance in ordered:
            if groups[i] not in seen:
                chosen.append((i, distance))
                seen.add(groups[i])
            if len(chosen) == k:
                break
        weights = np.asarray([1 / max(d, 1e-6) if weighted else 1 for _, d in chosen])
        value = np.average([y[i] for i, _ in chosen], weights=weights)
        predictions.append(float(value))
        records.append(chosen)
    return np.asarray(predictions), records


def prepare(out=DEFAULT_OUT, models=DEFAULT_MODELS):
    import torch
    from vrms_pilot.data import covariance, read_csv
    from vrms_pilot.model_ablation_v3 import (PathMIL, load_dataset, load_features,
        predict_mil, read_json, state_digest, verify_frozen)

    out, models = Path(out), Path(models)
    if out.exists():
        raise FileExistsError("Preserve existing tool preparation; choose a new run directory")
    frozen = verify_frozen(models)
    paths, splits, windows, audit = load_dataset(frozen["pilot_out"])
    pilot = Path(frozen["pilot_out"])
    absolute = np.load(pilot / "cache/bio_absolute.npy", mmap_mode="r")
    ref = np.load(pilot / "cache/bio_reference.npy", mmap_mode="r")
    spatial, spectrum, descriptions = [], [], []
    for p in paths:
        start, stop = p["window_start"], p["window_end"]
        a = np.asarray(absolute[start:stop], float)
        reference = np.asarray(ref[start:stop, 240:], float)
        early = a[:max(1, len(a) // 3), :120].mean(axis=0)
        late = a[-max(1, len(a) // 3):, :120].mean(axis=0)
        # All statistics are within the current path and do not consume labels.
        spectrum.append(np.r_[a.mean(axis=0), a.std(axis=0), late - early,
            np.nanmean(reference, axis=0) if p["reference_available"] else np.zeros(120),
            float(p["reference_available"])])
        covariances = np.asarray([covariance(w) for w in windows[start:stop]])
        scales = np.trace(covariances, axis1=1, axis2=2) / 30
        normalized = covariances / scales[:, None, None]
        mean_cov = normalized.mean(axis=0)
        eigenvalues, vectors = np.linalg.eigh(mean_cov)
        log_cov = (vectors * np.log(np.maximum(eigenvalues, 1e-10))) @ vectors.T
        ii, jj = np.triu_indices(30)
        spatial.append(np.r_[log_cov[ii, jj] * np.where(ii == jj, 1., np.sqrt(2.)),
                             np.log(scales).mean(), np.log(scales).std(),
                             np.log(np.diagonal(covariances, axis1=1, axis2=2)).mean(axis=0)])
        descriptions.append(dict(relative_band_power=a[:, 120:].mean(axis=0).reshape(30, 4).mean(axis=0).tolist(),
            reference_log_change=reference.mean(axis=0).reshape(30, 4).mean(axis=0).tolist()
                if p["reference_available"] else None,
            reference_quality="valid" if p["reference_available"] else "unavailable",
            valid_window_fraction=p["accepted_windows"] / p["total_possible_windows"],
            accepted_windows=p["accepted_windows"], direction_validated=False))
    out.mkdir(parents=True)
    (out / "features").mkdir()
    np.savez_compressed(out / "features/common.npz", spectral=np.asarray(spectrum), spatial_covariance=np.asarray(spatial))
    write_json(out / "descriptions.json", descriptions)
    write_json(out / "split_manifest.json", splits)
    # Query truth is local and used only by the separate evaluator, never API prompts.
    write_json(out / "local_paths.json", paths)
    extraction = read_json(models / "feature_cache/extraction_audit.json")
    original = {int(r["path_index"]): r for r in read_csv(models / "model_results/path_oof.csv")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for split, feature_record in zip(splits, extraction["records"]):
        subject = split["outer_subject"]
        if feature_record["outer_subject"] != subject:
            raise ValueError("Wrong cached feature fold")
        features = load_features(models, feature_record, "femba_frozen_mil")
        stem = models / "model_results/checkpoints" / f"outer_subject_{subject:02d}_femba_frozen_mil.pt"
        state = torch.load(stem, map_location=device, weights_only=True)
        head = PathMIL(state["input_dim"]).to(device)
        head.load_state_dict(state["state_dict"])
        if state["split"] != split or state_digest(head) != state["training"]["final_state_sha256"]:
            raise ValueError("The original MIL head changed")
        probability = predict_mil(head, features, paths, np.arange(len(paths)), device)
        values = [features[p["window_start"]:p["window_end"]] for p in paths]
        means, deviations = np.asarray([v.mean(axis=0) for v in values]), np.asarray([v.std(axis=0) for v in values])
        path = out / "features" / f"fold_{subject:02d}.npz"
        np.savez_compressed(path, femba_features=np.c_[means, deviations], femba_mean=means, baseline_probability=probability)
        outer = indices(paths, [subject])
        error = max(abs(probability[i] - float(original[i]["femba_frozen_mil_probability"])) for i in outer)
        if error != 0:
            raise ValueError("The locked baseline probability changed")
        records.append(dict(outer_subject=subject, original_head_sha256=sha256(stem),
                            feature_sha256=sha256(path), baseline_max_abs_error=error))
        print(f"tool features fold {len(records)}/24", flush=True)
    write_json(out / "protocol.json", dict(schema_version="agent_tool_selection_v1", frozen_before_training=True,
        acceptance=dict(accuracy=.70, minimum_correction_to_damage_ratio=2., final_classes=["High", "Low"]),
        baseline=dict(locked_correct=89, paths=146, threshold=.5, accuracy=89/146),
        candidates=dict(families=FAMILIES, classifiers=CONFIGS, cases=CASE_CONFIGS),
        fitting="meta5 subject OOF: fit base14 plus other4 meta; final tool fit base14+meta5 (19), policy4 held out",
        calibration="Platt on meta subject OOF scores; fitted head predictions never calibrate themselves",
        correction_validation="policy4 only; leave-one-policy-subject-out gate selection, heldout veto",
        outer_label_selection=False, prior_outer_results_exploratory=True, baseline_train_subjects=14,
        tool_train_subjects=19, model_provenance_limit="same local FEMBA checkpoint, pretraining overlap not excluded",
        source_hashes=dict(baseline=sha256(models / "model_results/path_oof.csv"),
                          common_features=sha256(out / "features/common.npz"),
                          original_split=sha256(pilot / "split_manifest.json")),
        feature_folds=records, code_sha256=sha256(__file__), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def candidate_oof(x, y, groups, base, meta, config):
    result = np.full(len(meta), np.nan)
    for subject in np.unique(groups[meta]):
        train = np.r_[base, meta[groups[meta] != subject]]
        query = meta[groups[meta] == subject]
        model = classifier(config).fit(x[train], y[train])
        result[np.isin(meta, query)] = predict_score(model, x[query])
    return result


def selection_key(y, p, groups):
    correct = (p >= .5) == y
    subject_mean = np.mean([correct[groups == g].mean() for g in np.unique(groups)])
    return float(subject_mean), float(correct.mean()), float(balanced_accuracy_score(y, p >= .5))


def correction_counts(y, baseline, candidate):
    before, after = baseline >= .5, candidate >= .5
    corrected = int(((before != y) & (after == y)).sum())
    damaged = int(((before == y) & (after != y)).sum())
    return dict(corrected=corrected, damaged=damaged, net=corrected-damaged,
                changes=int((before != after).sum()))


def fit_fold(out, split):
    out = Path(out)
    paths = json.loads((out / "local_paths.json").read_text(encoding="utf-8"))
    y = np.asarray([p["label"] for p in paths])
    groups = np.asarray([p["subject_key"] for p in paths])
    base, meta, policy = [indices(paths, split[key]) for key in ("base_train", "meta_calibration", "policy_validation")]
    # No estimator or selector receives this fold's outer labels.
    if split["outer_subject"] in set(groups[np.r_[base, meta, policy]]):
        raise ValueError("Outer subject entered tool fitting")
    common = dict(np.load(out / "features/common.npz"))
    features = dict(np.load(out / "features" / f"fold_{split['outer_subject']:02d}.npz"))
    matrices = {**{k: common[k] for k in FAMILIES if k != "femba_features"},
                "femba_features": features["femba_features"]}
    selected, reports, meta_columns, policy_columns = {}, {}, [], []
    train = np.r_[base, meta]
    for family in FAMILIES:
        x = matrices[family]
        results, candidates = [], []
        for config in CONFIGS:
            score = candidate_oof(x, y, groups, base, meta, config)
            p = expit(score)
            results.append(score)
            candidates.append(dict(config=config, meta_oof=binary_metrics(y[meta], p),
                                   selection_key=selection_key(y[meta], p, groups[meta])))
        best = max(range(len(CONFIGS)), key=lambda i: (candidates[i]["selection_key"], -i))
        score = results[best]
        calibrator = fit_calibrator(score, y[meta], groups[meta])
        model = classifier(CONFIGS[best]).fit(x[train], y[train])
        meta_p, policy_p = calibrated(calibrator, score), calibrated(calibrator, predict_score(model, x[policy]))
        selected[family] = dict(model=model, calibrator=calibrator, configuration=CONFIGS[best])
        meta_columns.append(meta_p)
        policy_columns.append(policy_p)
        reports[family] = dict(selected_configuration=CONFIGS[best], candidates=candidates,
            meta_oof_uncalibrated=candidates[best]["meta_oof"], policy=binary_metrics(y[policy], policy_p),
            policy_correction=correction_counts(y[policy], features["baseline_probability"][policy], policy_p),
            calibration_usable=calibrator is not None)
    cases, case_candidates = [], []
    for config in CASE_CONFIGS:
        p = np.full(len(meta), np.nan)
        for subject in np.unique(groups[meta]):
            support = np.r_[base, meta[groups[meta] != subject]]
            query = meta[groups[meta] == subject]
            p[np.isin(meta, query)], _ = case_predictions(features["femba_mean"], y, groups, support, query, config)
        cases.append(p)
        case_candidates.append(dict(config=list(config), meta_oof=binary_metrics(y[meta], p),
                                    selection_key=selection_key(y[meta], p, groups[meta])))
    best = max(range(len(CASE_CONFIGS)), key=lambda i: (case_candidates[i]["selection_key"], -i))
    case_raw = cases[best]
    case_cal = fit_calibrator(logit(np.clip(case_raw, 1e-5, 1-1e-5)), y[meta], groups[meta])
    policy_raw, _ = case_predictions(features["femba_mean"], y, groups, train, policy, CASE_CONFIGS[best])
    meta_case = calibrated(case_cal, logit(np.clip(case_raw, 1e-5, 1-1e-5)))
    policy_case = calibrated(case_cal, logit(np.clip(policy_raw, 1e-5, 1-1e-5)))
    meta_columns.append(meta_case)
    policy_columns.append(policy_case)
    selected["case_retrieval"] = dict(config=CASE_CONFIGS[best], calibrator=case_cal,
        scaler=StandardScaler().fit(features["femba_mean"][train]),
        reference_features=features["femba_mean"][train], reference_labels=y[train],
        reference_subjects=groups[train], reference_indices=train)
    reports["case_retrieval"] = dict(selected_configuration=list(CASE_CONFIGS[best]), candidates=case_candidates,
        meta_oof_uncalibrated=case_candidates[best]["meta_oof"], policy=binary_metrics(y[policy], policy_case),
        policy_correction=correction_counts(y[policy], features["baseline_probability"][policy], policy_case))
    raw_meta = features["baseline_probability"][meta]
    raw_policy = features["baseline_probability"][policy]
    deep_cal = fit_calibrator(logit(np.clip(raw_meta, 1e-6, 1-1e-6)), y[meta], groups[meta])
    selected["deep_calibrator"] = deep_cal
    meta_x = np.c_[raw_meta, np.asarray(meta_columns).T]
    policy_x = np.c_[raw_policy, np.asarray(policy_columns).T]
    fusion = make_pipeline(StandardScaler(), LogisticRegression(C=.1, solver="liblinear", max_iter=1000))
    fusion.fit(logit(np.clip(meta_x, .001, .999)), y[meta])
    selected["fusion"] = fusion
    fusion_policy = fusion.predict_proba(logit(np.clip(policy_x, .001, .999)))[:, 1]
    reports["fusion"] = dict(policy=binary_metrics(y[policy], fusion_policy),
        policy_correction=correction_counts(y[policy], raw_policy, fusion_policy))
    numeric = np.c_[policy_x[:, 1:], policy_x[:, 1:].mean(axis=1), fusion_policy]
    reports["equal_average"] = dict(policy=binary_metrics(y[policy], numeric[:, -2]),
        policy_correction=correction_counts(y[policy], raw_policy, numeric[:, -2]))
    write_json(out / "internal_validation" / f"fold_{split['outer_subject']:02d}.json",
        dict(outer_subject=split["outer_subject"], tool_train_subjects=sorted(map(int, set(groups[train]))),
             meta_oof_subjects=sorted(map(int, set(groups[meta]))), independent_policy_subjects=sorted(map(int, set(groups[policy]))),
             outer_labels_used=False, reports=reports))
    joblib.dump(dict(split=split, tools=selected, reports=reports, family_order=list(FAMILIES) + ["case_retrieval"],
                     meta_indices=meta, policy_indices=policy, train_indices=train),
                out / "checkpoints" / f"fold_{split['outer_subject']:02d}.joblib")
    np.savez_compressed(out / "internal_validation" / f"fold_{split['outer_subject']:02d}.npz",
        meta_indices=meta, meta_probability=meta_x, policy_indices=policy, policy_probability=policy_x,
        policy_fusion=fusion_policy, policy_labels=y[policy], policy_subjects=groups[policy])
    return reports


def train(out=DEFAULT_OUT):
    from threadpoolctl import threadpool_limits
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol["code_sha256"] != sha256(__file__):
        raise ValueError("Tool preparation source changed")
    if (out / "checkpoints").exists():
        raise FileExistsError("Preserve partial/finished tool training")
    (out / "checkpoints").mkdir()
    (out / "internal_validation").mkdir()
    splits = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    start, records = time.perf_counter(), []
    with threadpool_limits(limits=2):
        for n, split in enumerate(splits, 1):
            begin = time.perf_counter()
            reports = fit_fold(out, split)
            records.append(reports)
            print(json.dumps(dict(fold=n, total=24, seconds=round(time.perf_counter()-begin, 1),
                policy_acc={name: value["policy"]["acc"] for name, value in reports.items()})), flush=True)
    write_json(out / "training_summary.json", dict(status="completed_internal_tools", outer_evaluated=False,
        folds=24, elapsed_seconds=time.perf_counter()-start,
        mean_internal_policy_acc={name:float(np.mean([r[name]["policy"]["acc"] for r in records])) for name in records[0]},
        checkpoint_sha256={p.name:sha256(p) for p in (out / "checkpoints").glob("*.joblib")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare(args.out, args.models) if args.prepare else train(args.out)


if __name__ == "__main__":
    main()
