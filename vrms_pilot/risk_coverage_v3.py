"""Descriptive held-out risk/coverage diagnostics, never threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from .experiment import expected_calibration_error


def score_metrics(y, probability):
    y, p = np.asarray(y, int), np.asarray(probability, float)
    valid = np.isfinite(p)
    result = dict(total_paths=len(y), scoreable_paths=int(valid.sum()), score_coverage=float(valid.mean()) if len(y) else None,
                  acc=None, bacc=None, auroc=None, brier=None, ece=None)
    if valid.any():
        yy, pp = y[valid], p[valid]
        result.update(acc=float(accuracy_score(yy, pp >= .5)),
                      brier=float(brier_score_loss(yy, pp)), ece=expected_calibration_error(yy, pp))
        if len(np.unique(yy)) == 2:
            result.update(bacc=float(balanced_accuracy_score(yy, pp >= .5)), auroc=float(roc_auc_score(yy, pp)))
    return result


def risk_curve(y, probability, subjects):
    """Sort by absolute margin with whole tied groups; missing scores abstain.

    Coverage always uses all eligible paths as its denominator. Labels are
    read only after ranking. No oracle ranking, smoothing or arbitrary tie
    breaking is used, and the curve cannot validate an operating gate.
    """
    y, p, groups = np.asarray(y, int), np.asarray(probability, float), np.asarray(subjects)
    finite = np.isfinite(p)
    confidence = np.abs(p - .5)
    thresholds = np.unique(confidence[finite])[::-1]
    rows = []
    for threshold in thresholds:
        accepted = finite & (confidence >= threshold)
        correct = (p[accepted] >= .5) == y[accepted]
        group_accuracy = [float(np.mean((p[accepted & (groups == g)] >= .5) == y[accepted & (groups == g)]))
                          for g in np.unique(groups[accepted])]
        rows.append(dict(margin=float(threshold), accepted_paths=int(accepted.sum()),
                         coverage=float(accepted.mean()), errors=int((~correct).sum()),
                         risk=float(np.mean(~correct)), subject_macro_risk=1-float(np.mean(group_accuracy)),
                         accepted_subjects=len(group_accuracy),
                         mean_implied_risk=float(np.mean(np.minimum(p[accepted], 1-p[accepted])))))
    points = []
    for target in (.2, .4, .6):
        point = next((row for row in rows if row['coverage'] >= target), None)
        if point is not None:
            mask=finite & (confidence >= point['margin'])
            unique=np.unique(groups)
            rng=np.random.default_rng(2026)
            risks=[]
            for _ in range(1000):
                drawn=rng.choice(unique,len(unique),replace=True)
                selected=np.concatenate([np.flatnonzero(mask & (groups==g)) for g in drawn])
                if len(selected):
                    risks.append(float(np.mean((p[selected]>=.5)!=y[selected])))
            point=dict(point,subject_bootstrap_95ci=np.quantile(risks,[.025,.975]).tolist() if risks else None)
        points.append(dict(target_coverage=target, achieved=point is not None,
                           actual_point=point, note='First whole-tie coverage >= target; descriptive only'))
    prediction_correct = ((p[finite] >= .5) == y[finite]).astype(int)
    discrimination = (float(roc_auc_score(prediction_correct, confidence[finite]))
                      if len(np.unique(prediction_correct)) == 2 else None)
    return dict(rows=rows, target_points=points, denominator=len(y), rank_by='absolute_probability_margin',
                correct_vs_error_margin_auroc=discrimination,
                no_outer_label_threshold_selection=True, ties_kept_together=True)


def paired_subject_bootstrap(y, a, b, subjects, seed=2026, iterations=1000):
    """Paired subject resampling on common paths; exploratory percentile CI."""
    y, a, b, groups = np.asarray(y, int), np.asarray(a, float), np.asarray(b, float), np.asarray(subjects)
    common = np.isfinite(a) & np.isfinite(b)
    unique = np.unique(groups[common])
    values = dict(bacc=[], auroc=[], brier=[])
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selected = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([np.flatnonzero(common & (groups == g)) for g in selected]) if len(selected) else np.array([], int)
        if not len(ix) or len(np.unique(y[ix])) != 2:
            continue
        ma, mb = score_metrics(y[ix], a[ix]), score_metrics(y[ix], b[ix])
        for key in values:
            values[key].append(mb[key]-ma[key])
    original_a, original_b = score_metrics(y[common], a[common]), score_metrics(y[common], b[common])
    return dict(common_paths=int(common.sum()), common_subjects=len(unique), seed=seed, iterations=iterations,
                comparison='b_minus_a', results={key:dict(difference=original_b[key]-original_a[key]
                     if original_b[key] is not None and original_a[key] is not None else None,
                     subject_bootstrap_95ci=np.quantile(value,[.025,.975]).tolist() if value else None,
                     valid_replicates=len(value)) for key,value in values.items()},
                limitation='Previously inspected exploratory dataset; CI does not correct selection or pretraining overlap')
