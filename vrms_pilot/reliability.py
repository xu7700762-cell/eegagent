"""Subject-disjoint probability calibration and internally validated exit gates.

The EEG reference uses base-training signals only. Calibration labels and policy
labels belong to two separate held-out subject sets, neither used by the head.
These small development-set gates are exploratory, not clinical quality limits.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def signal_descriptor(windows):
    """Cheap channel scale descriptor; requires no spectral/covariance tool."""
    x = np.asarray(windows)
    if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
        return None
    scale = np.std(x, axis=2).mean(axis=0)
    amplitude = np.ptp(x, axis=2).mean(axis=0)
    return np.log(np.maximum(np.r_[scale, amplitude], 1e-10))


class ReliabilityModel:
    def __init__(self):
        self.calibrator = None
        self.center, self.scale = None, None
        self.thresholds = None
        self.provenance = {}
        self.validation = {}

    def ood_score(self, descriptor):
        if descriptor is None or self.center is None:
            return None
        x = np.asarray(descriptor, float)
        if x.shape != self.center.shape or not np.isfinite(x).all():
            return None
        return float(np.sqrt(np.mean(((x - self.center) / self.scale) ** 2)))

    def calibrate(self, probability):
        if self.calibrator is None or probability is None or not np.isfinite(probability):
            return None
        p = np.clip(float(probability), 1e-6, 1 - 1e-6)
        return float(self.calibrator.predict_proba([[np.log(p / (1 - p))]])[0, 1])

    def assess(self, probability, descriptor, valid_fraction, *, qc_passed=True, reference_available=False):
        p_cal, distance = self.calibrate(probability), self.ood_score(descriptor)
        thresholds = self.thresholds
        quality_bad = (not qc_passed or not np.isfinite(valid_fraction) or valid_fraction <= 0
                       or (thresholds is not None and valid_fraction < thresholds['min_valid_fraction']))
        ood = None if thresholds is None or distance is None else distance > thresholds['ood_max']
        reliable = (p_cal is not None and thresholds is not None and distance is not None
                    and not quality_bad and not ood and abs(p_cal - .5) >= thresholds['margin'])
        return dict(p_raw=None if probability is None or not np.isfinite(probability) else float(probability),
                    p_cal=p_cal, calibration_status='fitted' if self.calibrator is not None else 'unavailable',
                    policy_status='validated' if thresholds is not None else 'unvalidated',
                    prediction_reliable=bool(reliable), signal_quality_bad=bool(quality_bad),
                    ood_score=distance, ood=ood, evidence_conflict=False,
                    signal_quality=dict(valid_window_fraction=float(valid_fraction) if np.isfinite(valid_fraction) else None,
                                        qc_passed=bool(qc_passed), reference_quality='valid' if reference_available else 'unavailable',
                                        threshold=None if thresholds is None else thresholds['min_valid_fraction']),
                    thresholds=thresholds,
                    provenance={key.replace('_subjects', '_subject_count'): len(value)
                                if key.endswith('_subjects') else value for key, value in self.provenance.items()},
                    confidence_scope='internal_subject_holdout_exploratory_not_clinical')


def fit_reliability(raw_probability, y, groups, descriptors, valid_fractions, *,
                    base_indices, head_indices, calibration_indices, policy_indices,
                    target_accuracy=.80, minimum_coverage=.35, minimum_cases=8):
    """Fit exclusively on the declared roles; overlap is an error, never fallback."""
    p, y, groups = np.asarray(raw_probability, float), np.asarray(y), np.asarray(groups)
    descriptors, fractions = np.asarray(descriptors, float), np.asarray(valid_fractions, float)
    role_indices = [np.asarray(v, int) for v in (base_indices, head_indices, calibration_indices, policy_indices)]
    role_names = ('base_reference', 'head_fit', 'probability_calibration', 'policy_validation')
    subject_sets = [set(groups[i].tolist()) for i in role_indices]
    if any(a & b for i, a in enumerate(subject_sets) for b in subject_sets[i + 1:]):
        raise ValueError('Reliability roles must be subject-disjoint; head-fit predictions cannot calibrate themselves')
    model = ReliabilityModel()
    model.provenance = {name + '_subjects': sorted(s) for name, s in zip(role_names, subject_sets)}
    model.provenance.update(calibration_prediction_source='frozen_head_on_independent_subjects',
                            threshold_source='independent_policy_subjects_only',
                            ood_reference_source='base_training_signals_only')
    base, _, calibration, policy = role_indices
    ref = descriptors[base]
    ref = ref[np.isfinite(ref).all(axis=1)]
    if len(ref):
        model.center = np.median(ref, axis=0)
        mad = 1.4826 * np.median(np.abs(ref - model.center), axis=0)
        model.scale = np.where(mad > 1e-8, mad, np.maximum(np.std(ref, axis=0), 1e-8))
    calibration = calibration[np.isfinite(p[calibration])]
    if len(calibration) >= 6 and len(np.unique(y[calibration])) == 2 and len(subject_sets[2]) >= 2:
        z = np.log(np.clip(p[calibration], 1e-6, 1 - 1e-6) / (1 - np.clip(p[calibration], 1e-6, 1 - 1e-6)))
        calibrator = LogisticRegression(C=1., solver='liblinear', max_iter=1000, random_state=2026)
        calibrator.fit(z[:, None], y[calibration])
        if calibrator.coef_[0, 0] > 0:
            model.calibrator = calibrator
    if model.calibrator is None or model.center is None or len(subject_sets[3]) < 2:
        model.validation = dict(status='unvalidated', reason='insufficient_independent_calibration_or_reference')
        return model
    calibrated = np.asarray([model.calibrate(v) if np.isfinite(v) else np.nan for v in p[policy]])
    distances = np.asarray([model.ood_score(v) for v in descriptors[policy]], float)
    quality = fractions[policy]
    valid = np.isfinite(calibrated) & np.isfinite(distances) & np.isfinite(quality) & (quality > 0)
    if valid.sum() < minimum_cases or len(np.unique(y[policy][valid])) != 2:
        model.validation = dict(status='unvalidated', reason='insufficient_policy_cases')
        return model
    # All score cutoffs are empirical policy-set values. The accuracy/coverage
    # targets are declared operating objectives, not assumed reliability limits.
    margins = np.unique(np.quantile(np.abs(calibrated[valid] - .5), np.linspace(0, 1, 11)))
    ood_limits = np.unique(np.quantile(distances[valid], [.5, .75, .9, 1.]))
    quality_limits = np.unique(np.quantile(quality[valid], [0., .25, .5, .75]))
    candidates = []
    for margin in margins:
        for distance in ood_limits:
            for fraction in quality_limits:
                accepted = valid & (np.abs(calibrated - .5) >= margin) & (distances <= distance) & (quality >= fraction)
                count, coverage = int(accepted.sum()), float(accepted.mean())
                if count < minimum_cases or coverage < minimum_coverage:
                    continue
                accuracy = float(np.mean((calibrated[accepted] >= .5) == y[policy][accepted]))
                if accuracy >= target_accuracy:
                    candidates.append((coverage, accuracy, float(fraction), -float(distance), -float(margin), count))
    model.validation = dict(status='unvalidated', policy_cases=len(policy), target_accuracy=target_accuracy,
                            minimum_coverage=minimum_coverage, calibration_cases=len(calibration),
                            policy_brier=float(np.mean((calibrated[valid] - y[policy][valid]) ** 2)),
                            policy_raw_brier=float(np.mean((p[policy][valid] - y[policy][valid]) ** 2)))
    if candidates:
        coverage, accuracy, fraction, distance, margin, count = max(candidates)
        model.thresholds = dict(margin=-margin, ood_max=-distance, min_valid_fraction=fraction)
        model.validation.update(status='validated', selected_coverage=coverage, selected_accuracy=accuracy,
                                selected_cases=count, thresholds=model.thresholds)
    else:
        model.validation['reason'] = 'no_internal_gate_met_declared_targets'
    return model
