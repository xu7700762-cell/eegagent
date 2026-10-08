"""24-subject LOSO exploratory experiment and real lazy tool replay."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (audit_eligibility, covariance_features, digest, prepare, qc, reference_invroot,
                   spectral_features, write_json)
from .engine import adaptive_policy, temporal_features
from .model import CompactEEGCNN, predict_windows, train_cnn
from .reliability import ReliabilityModel, fit_reliability, signal_descriptor

SEED = 2026
DEFAULT_DATA = Path(os.environ.get("EEG_DATA_ROOT", "data"))
DEFAULT_OUT = Path("outputs/vrms_pilot/v2_seed2026")


def make_splits(paths):
    subjects = sorted({p["subject_key"] for p in paths})
    result = []
    for subject in subjects:
        remaining = [s for s in subjects if s != subject]
        rng = np.random.default_rng(SEED + subject)
        for attempt in range(1000):
            order = rng.permutation(remaining).tolist()
            parts = dict(base_train=order[:14], meta_calibration=order[14:19], policy_validation=order[19:])
            # Internal stratification only: the held-out subject's labels are never read.
            adequate = all(min(np.bincount([p["label"] for p in paths if p["subject_key"] in group], minlength=2)) >= 3
                           for group in parts.values())
            head_fit, probability_calibration = order[14:17], order[17:19]
            adequate = adequate and all(min(np.bincount([p['label'] for p in paths if p['subject_key'] in group], minlength=2)) >= 2
                                        for group in (head_fit, probability_calibration))
            if adequate:
                break
        else:
            raise ValueError("Cannot build internally class-supported split")
        parts["outer_test"] = [subject]
        groups = list(parts.values())
        if any(set(a) & set(b) for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise AssertionError("Subject leakage")
        result.append(dict(outer_subject=subject, selection_attempt=attempt,
                           head_fit=head_fit, probability_calibration=probability_calibration, **parts))
    return result


def pool_features(values, paths):
    result = np.full((len(paths), values.shape[1]), np.nan)
    for i, p in enumerate(paths):
        if p["window_end"] > p["window_start"]:
            part = values[p["window_start"]:p["window_end"]]
            if np.isfinite(part).all():
                result[i] = part.mean(axis=0)
    return result


def fit_logistic(x, y, indices, c=0.1, balanced=True):
    indices = np.asarray(indices, dtype=int)
    indices = indices[np.isfinite(x[indices]).all(axis=1)]
    if len(indices) < 6 or len(np.unique(y[indices])) != 2:
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear", max_iter=1000,
                                                               class_weight="balanced" if balanced else None, random_state=SEED))
    return model.fit(x[indices], y[indices])


def probabilities(model, x):
    p = np.full(len(x), np.nan)
    good = np.isfinite(x).all(axis=1)
    if model is not None and good.any():
        p[good] = model.predict_proba(x[good])[:, 1]
    return p


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def base_scores(paths, windows, times, model, mean, scale, device, classifiers, pooled, indices):
    dt = np.full((len(paths), 7), np.nan)
    window_predictions = {}
    for i in indices:
        path = paths[i]
        sl = slice(path["window_start"], path["window_end"])
        if path["accepted_windows"]:
            z = predict_windows(model, mean, scale, windows[sl], device)
            window_predictions[i] = z
            dt[i] = temporal_features(z, times[sl])
    abs_prob = probabilities(classifiers["bio_absolute"], pooled["bio_absolute"])
    ref_prob = probabilities(classifiers["bio_reference"], pooled["bio_reference"])
    bio = np.where(np.isfinite(ref_prob), ref_prob, abs_prob)
    cov = probabilities(classifiers["covariance"], pooled["covariance"])
    features = dict(deep=dt[:, :1], deep_temporal=dt,
                    biomarker=logit(bio)[:, None], covariance=logit(cov)[:, None],
                    deep_temporal_bio=np.c_[dt, logit(bio)],
                    full=np.c_[dt, logit(bio), logit(cov)])
    return features, window_predictions


class ToolReplay:
    """Single VRMSAgent execution context; tool results are actually computed lazily."""
    def __init__(self, path, x, starts, reference, reference_power, bundle, device):
        self.path, self.x, self.starts = path, x, starts
        self.reference = np.full((30, 30), np.nan) if reference is None else np.asarray(reference)
        self.reference_power = np.full((30, 4), np.nan) if reference_power is None else np.asarray(reference_power)
        self.reference_available = (self.reference.shape == (30, 30) and np.isfinite(self.reference).all()
                                    and np.allclose(self.reference, self.reference.T)
                                    and np.linalg.eigvalsh(self.reference).min() > 0)
        self.reference_power_available = (self.reference_power.shape == (30, 4)
                                          and np.isfinite(self.reference_power).all() and (self.reference_power > 0).all())
        self.bundle, self.device = bundle, device
        self.memo, self.timings, self.calls = {}, {}, []
        self.dt = None
        self.bio_logit, self.cov_logit = None, None

    def tool(self, name, operation):
        if name not in self.memo:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            self.memo[name] = operation()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.timings[name] = (time.perf_counter() - begin) * 1000
            self.calls.append(name)
        return self.memo[name]

    def head(self, name, features):
        estimator = self.bundle["meta"].get(name)
        if estimator is None or features is None or not np.isfinite(features).all():
            return None
        return float(estimator.predict_proba(np.asarray(features)[None])[:, 1][0])

    def quality(self):
        def operation():
            measurements = [qc(x) for x in self.x]
            amplitudes = [m['max_ptp_uv'] for _, m in measurements if m['max_ptp_uv'] is not None]
            self.memo['quality_summary'] = dict(checked_windows=len(measurements),
                finite_windows=sum(m['finite'] for _, m in measurements),
                max_ptp_uv=max(amplitudes, default=None),
                minimum_nonflat_channels=min((m['nonflat_channels'] for _, m in measurements), default=0),
                scope='engineering_sanity_checks_not_validated_artifact_detection')
            return bool(measurements) and all(good for good, _ in measurements)
        return self.tool("SignalQualityChecker", operation)

    def reliability(self):
        if 'UncertaintyEvaluator' in self.memo:
            return self.memo['UncertaintyEvaluator']
        fraction = len(self.x) / max(self.path['total_possible_windows'], 1)
        model = self.bundle.get('reliability') or ReliabilityModel()
        good = self.quality()
        preliminary = model.assess(None, None, fraction, qc_passed=good,
                                   reference_available=bool(self.reference_available))
        raw = None if preliminary['signal_quality_bad'] else self.score('deep')
        def operation():
            if preliminary['signal_quality_bad']:
                result = preliminary
            else:
                result = model.assess(raw, signal_descriptor(self.x), fraction, qc_passed=good,
                                      reference_available=bool(self.reference_available))
            result['signal_quality']['artifact_summary'] = self.memo['quality_summary']
            return result
        return self.tool('UncertaintyEvaluator', operation)

    def evidence_snapshot(self):
        """Label-blind deep/temporal/QC evidence; no eager RAG, Bio or Cov."""
        gate = self.reliability()
        if gate['signal_quality_bad']:
            return dict(probabilities={}, temporal={}, reliability=gate,
                        qc_accepted_fraction=len(self.x) / max(self.path['total_possible_windows'], 1),
                        reference_available=bool(self.reference_available))
        raw = self.score('deep')
        temporal_score = self.score('deep_temporal')
        dt = self.dt
        temporal = dict(raw_mean_probability=float(np.mean(1 / (1 + np.exp(-np.clip(self.memo['window_logits'], -30, 30))))),
                        raw_probability_std=float(dt[1]), raw_recent_mean=float(dt[3]),
                        raw_recent_slope_per_second=float(dt[4]), raw_last_probability=float(dt[5]))
        evidence = dict(probabilities=dict(deep_probability=raw, deep_temporal_probability=temporal_score,
                                          calibrated_deep_probability=gate['p_cal']),
                        temporal=temporal, reliability=gate,
                        qc_accepted_fraction=len(self.x) / max(self.path['total_possible_windows'], 1),
                        reference_available=bool(self.reference_available))
        if 'physiological_evidence' in self.memo:
            evidence['physiological_evidence'] = self.memo['physiological_evidence']
        if 'CaseRetriever' in self.memo:
            retrieval = self.memo['CaseRetriever']
            evidence['case_retrieval'] = {key: value for key, value in retrieval.items() if key != 'selected_indices'}
        return evidence

    def _physiology(self):
        from .physiology import physiological_evidence
        self.memo['physiological_evidence'] = physiological_evidence(
            biomarker=self.memo.get('biomarker_description'), covariance=self.memo.get('covariance_description'),
            reference_quality='valid' if self.reference_power_available and self.reference_available else 'unavailable',
            p_cal=self.reliability()['p_cal'])

    def score(self, name):
        if name == 'case_retrieval':
            def retrieve():
                retriever = self.bundle.get('case_retriever')
                if retriever is None:
                    return dict(examples=[], retrieval_quality=dict(status='unvalidated', reliable=False,
                                reason='case_bank_unavailable'), evidence_conflict=False)
                return retriever(self.evidence_snapshot(), query_subject=self.path.get('subject_key'),
                                 p_cal=self.reliability()['p_cal'])
            return self.tool('CaseRetriever', retrieve)
        if name == "deep":
            z = self.tool("DeepVRMSDetector", lambda: predict_windows(self.bundle["cnn"], self.bundle["mean"],
                          self.bundle["scale"], self.x, self.device))
            self.memo["window_logits"] = z
            return self.head("deep", [z.mean()]) if len(z) else None
        if name == "deep_temporal":
            self.score("deep")
            self.dt = self.tool("TemporalAnalyzer", lambda: temporal_features(self.memo["window_logits"], self.starts))
            return self.head(name, self.dt)
        if name == "biomarker":
            def bio_operation():
                absolute, relative = [], []
                ref_valid = self.reference_power_available and self.reference_available
                for x in self.x:
                    f, power = spectral_features(x)
                    absolute.append(f)
                    if ref_valid:
                        relative.append(np.r_[f, np.log((power + 1e-10) / (self.reference_power + 1e-10)).ravel()])
                self.memo['biomarker_description'] = dict(
                    relative_band_power=np.mean(absolute, axis=0)[120:].reshape(30, 4).mean(axis=0).tolist(),
                    reference_log_change=(np.mean(relative, axis=0)[240:].reshape(30, 4).mean(axis=0).tolist()
                                          if relative else None))
                classifier = self.bundle["classifiers"]["bio_reference"] if relative else None
                features = np.mean(relative, axis=0) if classifier is not None else np.mean(absolute, axis=0)
                if classifier is None:
                    classifier = self.bundle["classifiers"]["bio_absolute"]
                if classifier is None:
                    return None
                return float(classifier.predict_proba(features[None])[:, 1][0])
            p = self.tool("BiomarkerCalculator", bio_operation)
            self._physiology()
            self.bio_logit = None if p is None else float(logit(p))
            return self.head(name, None if p is None else [self.bio_logit])
        if name == "covariance":
            def cov_operation():
                classifier = self.bundle["classifiers"]["covariance"]
                if not self.reference_available:
                    return None
                invroot = reference_invroot(self.reference)
                f = np.mean([covariance_features(x, self.reference, invroot) for x in self.x], axis=0)
                self.memo['covariance_description'] = dict(distance=float(f[-1]))
                if classifier is None:
                    return None
                return float(classifier.predict_proba(f[None])[:, 1][0])
            p = self.tool("CovarianceAnalyzer", cov_operation)
            self._physiology()
            self.cov_logit = None if p is None else float(logit(p))
            return self.head(name, None if p is None else [self.cov_logit])
        self.score("deep_temporal")
        self.score("biomarker")
        if self.dt is None or self.bio_logit is None:
            return None
        features = np.r_[self.dt, self.bio_logit]
        if name == "full":
            self.score("covariance")
            if self.cov_logit is None:
                return None
            features = np.r_[features, self.cov_logit]
        return self.head(name, features)

    def assess(self, method):
        begin = time.perf_counter()
        good = self.quality()
        reference_ok = bool(self.reference_available)
        if method == 'adaptive':
            reliability = self.reliability()
        if not good or not len(self.x):
            result = dict(probability=None, state="insufficient_data", stage="qc", stop_reason="invalid_input")
        elif method == "adaptive":
            result = adaptive_policy(self.score, self.bundle.get("margins"), reference_ok, reliability=reliability)
            reliability['evidence_conflict'] = result.get('evidence_conflict', False)
        else:
            stage = "full" if method == "full" and reference_ok else "deep_temporal_bio" if method == "full" else method
            p = self.score(stage)
            result = dict(probability=p, state=("high" if p >= .5 else "low") if p is not None else "uncertain",
                          stage=stage, stop_reason="fixed_baseline_numeric_output")
        elapsed = (time.perf_counter() - begin) * 1000
        pack = dict(schema_version="pilot_v2", evidence_id=f"{self.path['episode_handle']}-{method}",
                    episode_handle=self.path["episode_handle"], assessment_scope="path_end",
                    data_cutoff_sec=self.path["path_end_sec"], qc_passed=good,
                    valid_windows=len(self.x), possible_windows=self.path["total_possible_windows"],
                    reference_available=bool(self.reference_available),
                    versions=dict(preprocessing="causal_p0_0.5_45_v1", model="compact_cnn_seed2026",
                                  probability_calibration="2_independent_meta_subjects_excluding_3_head_fit",
                                  policy="independent_validation_subjects_v2"),
                    tool_calls=self.calls, tool_latency_ms=self.timings, assessment_latency_ms=elapsed,
                    ensemble_std=None, result=result,
                    verification=dict(verified=True, label_source="numerical_engine",
                                      confidence_scope="exploratory_development_gate_not_clinical"))
        if method == 'adaptive':
            pack['reliability'] = self.reliability()
        if 'physiological_evidence' in self.memo:
            pack['physiological_evidence'] = self.memo['physiological_evidence']
        if 'CaseRetriever' in self.memo:
            pack['case_retrieval'] = {key: value for key, value in self.memo['CaseRetriever'].items() if key != 'selected_indices'}
        return pack


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for i in range(bins):
        selected = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if selected.any():
            value += selected.mean() * abs(y[selected].mean() - p[selected].mean())
    return float(value)


def metrics(y, p, published):
    y, p, published = np.asarray(y), np.asarray(p, float), np.asarray(published, bool)
    valid = np.isfinite(p)
    score = dict(total_paths=len(y), scoreable_paths=int(valid.sum()), published_paths=int((valid & published).sum()),
                 score_coverage=float(valid.mean()), publication_coverage=float((valid & published).mean()))
    def classify(mask):
        if not mask.any() or len(np.unique(y[mask])) < 2:
            return dict(bacc=None, macro_f1=None)
        predicted = p[mask] >= .5
        return dict(bacc=float(balanced_accuracy_score(y[mask], predicted)),
                    macro_f1=float(f1_score(y[mask], predicted, average="macro", zero_division=0)))
    score["numeric_threshold_diagnostic"] = classify(valid)
    score["published_conditional"] = classify(valid & published)
    if valid.any() and len(np.unique(y[valid])) == 2:
        score.update(auroc=float(roc_auc_score(y[valid], p[valid])),
                     ece=expected_calibration_error(y[valid], p[valid]),
                     brier=float(brier_score_loss(y[valid], p[valid])),
                     confusion_matrix=confusion_matrix(y[valid], p[valid] >= .5).tolist())
    else:
        score.update(auroc=None, ece=None, brier=None, confusion_matrix=None)
    return score


def save_csv(path, records):
    if not records:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(out, epochs=12, smoke=False):
    out, cache = Path(out), Path(out) / "cache"
    paths = json.loads((cache / "evaluation_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((cache / "dataset_audit.json").read_text(encoding="utf-8"))
    if audit.get('schema_version') != 'raw_event_eligibility_v2':
        raise ValueError('V2 training requires freshly audited raw-event-complete caches; historical caches are not eligible')
    rows = [json.loads(s) for s in (cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    starts = np.asarray([r["start_sample"] / 1024 for r in rows])
    reference = np.load(cache / "path_references.npy")
    reference_power = np.load(cache / "path_reference_power.npy")
    pooled = {key: pool_features(np.load(cache / name, mmap_mode="r"), paths) for key, name in
              (("bio_absolute", "bio_absolute.npy"), ("bio_reference", "bio_reference.npy"), ("covariance", "covariance.npy"))}
    y = np.asarray([p["label"] for p in paths])
    subject = np.asarray([p["subject_key"] for p in paths])
    split_path = out / "split_manifest.json"
    splits = make_splits(paths)
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != splits:
        raise ValueError("Frozen split manifest mismatch")
    write_json(split_path, splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = out / ("smoke" if smoke else "results")
    if run_dir.exists():
        raise FileExistsError('Preserve completed or partial runs; choose a fresh V2 output directory')
    run_dir.mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    protocol = dict(schema_version='pilot_v2', seed=SEED, epochs=epochs, smoke=smoke, outer="LOSO",
                    inner="14 base / 3 head-fit + 2 probability-calibration meta / 4 policy subjects",
                    data_label_sha256=audit["label_sha256"], window_label_scope="weak_whole_path",
                    threshold=.5, clinical_symptom_truth=False, validation_target_accuracy=.80,
                    validation_minimum_coverage=.35, thresholds='empirical independent policy-subject values',
                    python=platform.python_version(), torch=torch.__version__, device=str(device),
                    gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                    external_code_sha256={"vrms_refine/retrieval.py": digest(Path("vrms_refine/retrieval.py"))})
    write_json(run_dir / "protocol.json", protocol)
    all_records, all_packs, fold_audit, inner_records = [], [], [], []
    selected_splits = splits[:1] if smoke else splits
    overall_start = time.perf_counter()
    for fold_number, split in enumerate(selected_splits, 1):
        fold_start = time.perf_counter()
        base = np.flatnonzero(np.isin(subject, split["base_train"]))
        meta = np.flatnonzero(np.isin(subject, split["meta_calibration"]))
        head_fit = np.flatnonzero(np.isin(subject, split['head_fit']))
        calibration = np.flatnonzero(np.isin(subject, split['probability_calibration']))
        val = np.flatnonzero(np.isin(subject, split["policy_validation"]))
        test = np.flatnonzero(subject == split["outer_subject"])
        cnn, mean, scale, training = train_cnn(windows, paths, base, device, epochs=epochs, seed=SEED)
        classifiers = {k: fit_logistic(v, y, base, c=.01 if k == "covariance" else .1) for k, v in pooled.items()}
        indices = np.r_[meta, val, test]
        features, win_scores = base_scores(paths, windows, starts, cnn, mean, scale, device, classifiers, pooled, indices)
        heads = {k: fit_logistic(v, y, head_fit, c=.3, balanced=False) for k, v in features.items()}
        head_scores = {k: probabilities(heads[k], v) for k, v in features.items()}
        descriptors = np.full((len(paths), 60), np.nan)
        for i, path in enumerate(paths):
            descriptor = signal_descriptor(windows[path['window_start']:path['window_end']])
            if descriptor is not None:
                descriptors[i] = descriptor
        fractions = [p['accepted_windows'] / max(p['total_possible_windows'], 1) for p in paths]
        reliability = fit_reliability(head_scores['deep'], y, subject, descriptors, fractions,
                                      base_indices=base, head_indices=head_fit,
                                      calibration_indices=calibration, policy_indices=val)
        from vrms_refine.retrieval import CaseRetriever
        bank = []
        for i in meta:
            if i not in win_scores:
                continue
            dt, z = features['deep_temporal'][i], win_scores[i]
            bank.append(dict(path_index=int(i), role='meta', evidence=dict(
                temporal=dict(raw_mean_probability=float(np.mean(1 / (1 + np.exp(-np.clip(z, -30, 30))))),
                              raw_probability_std=float(dt[1]), raw_recent_mean=float(dt[3]),
                              raw_recent_slope_per_second=float(dt[4]), raw_last_probability=float(dt[5])),
                qc_accepted_fraction=float(fractions[i]))))
        evaluation = {i: dict(subject_key=int(subject[i]), label=int(y[i])) for i in meta}
        retriever = CaseRetriever(bank, evaluation, k=5, allowed_subjects=split['meta_calibration']) if bank else None
        bundle = dict(cnn=cnn, mean=mean, scale=scale, classifiers=classifiers, meta=heads,
                      reliability=reliability, case_retriever=retriever)
        checkpoint = run_dir / "checkpoints" / f"outer_subject_{split['outer_subject']:02d}"
        torch.save(dict(model=cnn.state_dict(), mean=mean.cpu(), scale=scale.cpu(), seed=SEED,
                        training_subjects=split["base_train"], heldout_subject=split["outer_subject"]), str(checkpoint) + ".pt")
        joblib.dump(dict(classifiers=classifiers, meta=heads, split=split,
                         reliability=reliability, case_retriever=retriever), str(checkpoint) + ".joblib")
        for i in np.r_[meta, val]:
            inner_records.append(dict(outer_subject=split["outer_subject"], path_index=int(i),
                                      subject_key=int(subject[i]), role="meta" if i in meta else "policy_validation",
                                      reliability_role='head_fit' if i in head_fit else 'probability_calibration' if i in calibration else 'policy_validation',
                                      label=int(y[i]), **{k: None if not np.isfinite(v[i]) else float(v[i])
                                                         for k, v in head_scores.items()},
                                      calibrated_deep_probability=reliability.calibrate(head_scores['deep'][i])))
        for i in test:
            p = paths[i]
            sl = slice(p["window_start"], p["window_end"])
            row = dict(path_index=int(i), subject_key=int(subject[i]), label=int(y[i]), score=p["score"],
                       valid_windows=p["accepted_windows"], reference_available=p["reference_available"])
            for method in ("deep", "deep_temporal", "full", "adaptive"):
                replay = ToolReplay(p, windows[sl], starts[sl], reference[i], reference_power[i], bundle, device)
                pack = replay.assess(method)
                probability = pack["result"]["probability"]
                row[f"{method}_probability"] = probability
                row[f"{method}_state"] = pack["result"]["state"]
                row[f"{method}_tool_calls"] = len(pack["tool_calls"])
                row[f"{method}_latency_ms"] = pack["assessment_latency_ms"]
                all_packs.append(pack)
                expected_key = pack["result"]["stage"]
                expected = (reliability.calibrate(head_scores['deep'][i]) if method == 'adaptive'
                            else head_scores.get(expected_key, np.full(len(paths), np.nan))[i])
                if probability is not None and expected is not None and np.isfinite(expected):
                    if abs(probability - expected) > 2e-5:
                        raise AssertionError(f"Cache versus real tool replay mismatch: {method}, path {i}")
            for method in ("biomarker", "covariance"):
                value = head_scores[method][i]
                row[f"{method}_probability"] = None if not np.isfinite(value) else float(value)
                row[f"{method}_state"] = "insufficient_data" if not np.isfinite(value) else "high" if value >= .5 else "low"
            all_records.append(row)
        fold_audit.append(dict(**split, **training, reliability_provenance=reliability.provenance,
                               reliability_validation=reliability.validation,
                               seconds=time.perf_counter() - fold_start,
                               memory_rss_mb=psutil.Process().memory_info().rss / 2 ** 20))
        save_csv(run_dir / "path_oof.csv", all_records)
        save_csv(run_dir / "inner_predictions.csv", inner_records)
        write_json(run_dir / "fold_audit.json", fold_audit)
        print(f"fold {fold_number}/{len(selected_splits)} subject {split['outer_subject']:02d}: "
              f"paths={len(test)} loss={training['losses'][0]:.3f}->{training['losses'][-1]:.3f} "
              f"reliability={reliability.validation['status']} time={time.perf_counter()-fold_start:.1f}s", flush=True)
        del cnn, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (run_dir / "assessment_evidence.jsonl").open("w", encoding="utf-8") as f:
        for pack in all_packs:
            f.write(json.dumps(pack, ensure_ascii=False, allow_nan=False) + "\n")
    summary = summarize(all_records)
    summary.update(status="smoke_only" if smoke else "completed_single_seed_exploratory_loso",
                   elapsed_seconds=time.perf_counter() - overall_start, outer_folds=len(selected_splits),
                   trained_from_scratch=True, pi_measured=False, llm_tested=False,
                   raw_causal_preprocessing=True, primary_endpoint="whole_path_score_ge30")
    write_json(run_dir / "summary.json", summary)
    write_report(run_dir, summary, audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def summarize(rows):
    result = dict(total_paths=len(rows), methods={})
    yy = np.asarray([r["label"] for r in rows])
    for name in ("deep", "deep_temporal", "biomarker", "covariance", "full", "adaptive"):
        p = np.asarray([np.nan if r[f"{name}_probability"] is None else r[f"{name}_probability"] for r in rows])
        pub = [r[f"{name}_state"] in ("high", "low") for r in rows]
        m = metrics(yy, p, pub)
        if name in ("deep", "deep_temporal", "full", "adaptive"):
            m.update(tool_calls_mean=float(np.mean([r[f"{name}_tool_calls"] for r in rows])),
                     desktop_latency_median_ms=float(np.median([r[f"{name}_latency_ms"] for r in rows])),
                     desktop_latency_p95_ms=float(np.quantile([r[f"{name}_latency_ms"] for r in rows], .95)))
        result["methods"][name] = m
    # Common-score paired subject-cluster bootstrap; no windows are resampled as independent subjects.
    pfull = np.asarray([np.nan if r["full_probability"] is None else r["full_probability"] for r in rows])
    padapt = np.asarray([np.nan if r["adaptive_probability"] is None else r["adaptive_probability"] for r in rows])
    pdeep = np.asarray([np.nan if r["deep_probability"] is None else r["deep_probability"] for r in rows])
    group = np.asarray([r["subject_key"] for r in rows])
    common = np.isfinite(pfull) & np.isfinite(padapt) & np.isfinite(pdeep)
    comparisons = {}
    for label, a, b in (("full_minus_deep", pfull, pdeep), ("adaptive_minus_full", padapt, pfull)):
        differences = []
        rng = np.random.default_rng(SEED)
        subjects = np.unique(group[common])
        if len(subjects) > 1 and len(np.unique(yy[common])) == 2:
            for _ in range(1000):
                chosen = rng.choice(subjects, len(subjects), replace=True)
                ix = np.concatenate([np.flatnonzero(common & (group == s)) for s in chosen])
                if len(np.unique(yy[ix])) == 2:
                    differences.append(balanced_accuracy_score(yy[ix], a[ix] >= .5) - balanced_accuracy_score(yy[ix], b[ix] >= .5))
        delta = (balanced_accuracy_score(yy[common], a[common] >= .5) - balanced_accuracy_score(yy[common], b[common] >= .5)) if len(np.unique(yy[common])) == 2 else None
        comparisons[label] = dict(common_paths=int(common.sum()), bacc_difference=delta,
                                  subject_bootstrap_95ci=np.quantile(differences, [.025, .975]).tolist() if differences else None)
    result["paired_numeric_diagnostic"] = comparisons
    return result


def write_report(out, summary, audit):
    def number(x, percent=False):
        return "—" if x is None else f"{100*x:.2f}%" if percent else f"{x:.3f}"
    lines = ["# VRMS真实数据单种子试验", "", f"状态：{summary['status']}；外层{summary['outer_folds']}折，共{summary['total_paths']}路径。",
             "", "## 评价口径", "", "从原始CDT重建因果P0并排除EOF截断路径，5秒窗仅继承路径弱标签，主要评价整条路径评分高/低。外层LOSO；其余被试14基础训练、3分类头拟合、2独立概率校准、4策略验证，五类被试不重叠。单种子2026，固定epochs；本阶段未调用LLM。",
             "", "V2 Adaptive的连续分数始终是独立校准的深度模型p_cal。Case RAG和生理工具提供解释证据，不修改p_cal；没有通过内部可靠性门槛时保持uncertain，质量不足时输出insufficient_data。Full等固定数值方法保留为诊断对照。",
             "", "下表BACC是所有合法连续分数按固定0.5阈值得到的数值诊断，包含尚未通过发布门槛的uncertain记录；正式发布类别的条件BACC和覆盖另列。它不等同真实即时症状检测。",
             "", "|方法|可评分路径|数值BACC|AUROC|Macro-F1|ECE|发布覆盖|发布条件BACC|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in summary["methods"].items():
        lines.append(f"|{name}|{m['scoreable_paths']}|{number(m['numeric_threshold_diagnostic']['bacc'],True)}|{number(m['auroc'])}|{number(m['numeric_threshold_diagnostic']['macro_f1'])}|{number(m['ece'])}|{number(m['publication_coverage'],True)}|{number(m['published_conditional']['bacc'],True)}|")
    lines += ["", "## 实际工具回放", "", "每条外层测试路径重新执行数值工具，Adaptive通过校准、信号质量与OOD门槛后可直接结束；困难样本先懒执行Case RAG，检索不可靠或冲突时再实际计算Bio/Cov。缓存与实时计算概率一致检查已完成。计时是这台电脑对已预处理有效窗口的数值评估耗时，含QC、模型和工具；不含原始流积累/预处理、首任务校准等待、Pi或云API/报告，所以不能解释为完整部署延迟或节能。",
              "", "|方法|平均专业工具调用/路径|电脑评估中位数ms|P95 ms|", "|---|---:|---:|---:|"]
    for name in ("deep", "deep_temporal", "full", "adaptive"):
        m = summary["methods"][name]
        lines.append(f"|{name}|{m['tool_calls_mean']:.2f}|{m['desktop_latency_median_ms']:.2f}|{m['desktop_latency_p95_ms']:.2f}|")
    lines += ["", "## 配对比较", ""]
    for name, comparison in summary["paired_numeric_diagnostic"].items():
        lines.append(f"- {name}：共同可评分{comparison['common_paths']}路径，数值BACC差={number(comparison['bacc_difference'],True)}；按被试聚类95%区间={comparison['subject_bootstrap_95ci']}。")
    references = sum(s["reference_available"] for s in audit["subjects_audit"])
    mismatches = [Path(s["path"]).name for s in audit["sources"] if s["metadata_count_mismatch"]]
    lines += ["", "## 限制与产物", "", f"- 有效初始参考：{references}/{audit['subjects']}人；有效5秒窗口{audit['accepted_windows']}。参考不足时仅描述绝对谱，Cov不可用；Adaptive不以生理描述修改深度概率。",
              "- Delta/Alpha等谱变化和协方差距离仅作客观描述；尚无独立方向验证时，生理verification保持inconclusive。邻域类别计数不是校准症状概率。",
              f"- DPO样本计数与实际完整数据不符的记录：{mismatches}；使用字节导出的完整样本数并核查任务边界，未修改原文件。",
              "- 单种子、小型CNN、较小内部校准/策略集、固定试验参数；不是充分优化模型、未经查看的外部验证、正式概率可靠性证明、即时预警或Pi实测。",
              "- 技术QC只做预声明sanity检查，没有验证对眼电等全部伪迹的敏感性。",
              "- 全部测试路径保留状态；不能拿Adaptive已发布子集与Full全部路径直接比准确率。数值诊断配对比较不代表拒判策略已达到临床使用要求。",
              "- path_oof.csv保存完整路径结果；inner_predictions.csv保存内部来源；split_manifest.json在上级目录；assessment_evidence.jsonl不含真实评分/标签；checkpoints保存每个外层模型；protocol.json记录代码哈希和环境。",
              "", "当前结论必须以以上实际结果为准；三类证据不一定有增益，无法发布的结果应保持不确定。"]
    (out / "试验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "prepare", "smoke", "run"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'audit':
        _, eligibility = audit_eligibility(args.data_root)
        target = args.out / 'eligibility_audit.json'
        if target.exists():
            raise FileExistsError('Preserve the existing audit; choose a fresh output directory')
        write_json(target, eligibility)
        print(json.dumps({k: v for k, v in eligibility.items() if k != 'candidates'}, indent=2), flush=True)
    elif args.stage == "prepare":
        prepare(args.data_root, args.out / "cache")
    else:
        run(args.out, epochs=2 if args.stage == "smoke" else args.epochs, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
