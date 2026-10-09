"""Subject OOF calibration and independently audited numeric reliability V3.

The CNN is frozen. Calibration consumes meta scores from heads that excluded
each score's subject, never predictions from a head fitted on those same rows.
A predeclared two-policy-subject selector freezes one candidate; the other two
subjects can only audit or reject it. Additional 3/1 policy LOSO audits can
veto deployment, but cannot change or replace the primary candidate. Accepted
mean implied risk is an operating objective, not a statistical risk guarantee.
"""
from __future__ import annotations

from itertools import product
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .reliability import ReliabilityModel


DEFAULT_PLAN = dict(target_accuracy=.80, maximum_estimated_risk=.20,
    minimum_coverage=.35, minimum_cases=8, minimum_subjects=2,
    minimum_class_cases=2, minimum_class_subjects=2,
    qc_fraction_quantile=.05, margin_quantiles=(0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.),
    ood_quantiles=(.5, .75, .9, 1.))


def _plan(overrides):
    value = {**DEFAULT_PLAN, **(overrides or {})}
    if set(value) != set(DEFAULT_PLAN):
        raise ValueError("Unknown V3 reliability plan field")
    # Existing goals may be made stricter, never relaxed to rescue coverage.
    if (not .80 <= value['target_accuracy'] <= 1.
            or not 0 <= value['maximum_estimated_risk'] <= .20
            or not .35 <= value['minimum_coverage'] <= 1.
            or value['minimum_cases'] < 8 or value['minimum_subjects'] < 2
            or value['minimum_class_cases'] < 2 or value['minimum_class_subjects'] < 2
            or not 0 <= value['qc_fraction_quantile'] <= 1.):
        raise ValueError("V3 operating goals cannot be relaxed")
    for name in ('margin_quantiles', 'ood_quantiles'):
        if not value[name] or any(not 0 <= q <= 1 for q in value[name]):
            raise ValueError("Candidate quantiles must lie in [0,1]")
    return value


def _probability(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def _support(y, p, groups):
    true, predicted = {}, {}
    for label in (0, 1):
        actual = y == label
        forecast = (p >= .5) == label
        true[str(label)] = dict(cases=int(actual.sum()), subjects=int(len(np.unique(groups[actual]))))
        predicted[str(label)] = dict(cases=int(forecast.sum()), subjects=int(len(np.unique(groups[forecast]))))
    return dict(true_class_support=true, predicted_class_support=predicted)


def _risk_summary(y, p, groups, accepted):
    """Audit denominators include unavailable scores and refused signals."""
    selected = np.asarray(accepted, bool)
    yy, pp, gg = y[selected], p[selected], groups[selected]
    correct = (pp >= .5) == yy
    per_subject = [dict(subject=int(subject), cases=int(sum(gg == subject)),
        error_rate=float(np.mean(~correct[gg == subject]))) for subject in np.unique(gg)]
    risk = float(np.mean(~correct)) if len(correct) else None
    interval = None
    if len(per_subject) >= 2:
        # Exact small-n subject bootstrap: subjects, not correlated paths, are
        # the resampling units. Two audit subjects produce only four draws.
        draws = []
        for chosen in product(range(len(per_subject)), repeat=len(per_subject)):
            cases = sum(per_subject[i]['cases'] for i in chosen)
            draws.append(sum(per_subject[i]['cases'] * per_subject[i]['error_rate'] for i in chosen) / cases)
        interval = np.quantile(draws, [.025, .975]).tolist()
    return dict(total_cases=len(y), accepted_cases=int(selected.sum()),
        coverage=float(selected.mean()) if len(selected) else 0.,
        accepted_subjects=len(per_subject), accepted_subject_ids=[r['subject'] for r in per_subject],
        errors=int(sum(~correct)), empirical_risk=risk,
        accuracy=None if risk is None else 1 - risk,
        subject_macro_risk=float(np.mean([r['error_rate'] for r in per_subject])) if per_subject else None,
        mean_implied_risk=float(np.mean(np.minimum(pp, 1 - pp))) if len(pp) else None,
        probability_range=[float(pp.min()), float(pp.max())] if len(pp) else None,
        auroc=float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None,
        per_subject=per_subject, subject_bootstrap_risk_95ci=interval,
        ci_scope='exploratory_subject_bootstrap; two_audit_subjects_do_not_establish_a_risk_bound',
        **_support(yy, pp, gg))


def _failed_checks(summary, plan, required_subjects=None):
    failures = []
    if summary['accepted_cases'] < plan['minimum_cases']:
        failures.append('insufficient_accepted_cases')
    if summary['coverage'] < plan['minimum_coverage']:
        failures.append('insufficient_coverage')
    if summary['accepted_subjects'] < plan['minimum_subjects']:
        failures.append('insufficient_independent_subject_support')
    if required_subjects is not None and set(summary['accepted_subject_ids']) != set(required_subjects):
        failures.append('audit_subject_without_accepted_cases')
    for name in ('true_class_support', 'predicted_class_support'):
        if any(v['cases'] < plan['minimum_class_cases'] for v in summary[name].values()):
            failures.append(name + '_insufficient')
        if any(v['subjects'] < plan['minimum_class_subjects'] for v in summary[name].values()):
            failures.append(name + '_subjects_insufficient')
    if summary['empirical_risk'] is None or summary['empirical_risk'] > 1 - plan['target_accuracy'] + 1e-12:
        failures.append('empirical_risk_exceeds_target')
    if summary['subject_macro_risk'] is None or summary['subject_macro_risk'] > 1 - plan['target_accuracy'] + 1e-12:
        failures.append('subject_macro_risk_exceeds_target')
    if summary['mean_implied_risk'] is None or summary['mean_implied_risk'] > plan['maximum_estimated_risk'] + 1e-12:
        failures.append('estimated_risk_exceeds_target')
    if summary['auroc'] is None or summary['auroc'] <= .5:
        failures.append('probability_discrimination_failed')
    return failures


def _accept(p, distance, fractions, qc_passed, qc_threshold, thresholds):
    valid = np.isfinite(p) & np.isfinite(distance) & np.isfinite(fractions) & qc_passed
    valid &= (fractions > 0) & (fractions <= 1)
    if qc_threshold is not None:
        valid &= fractions >= qc_threshold
    if thresholds is None:
        return np.zeros(len(p), dtype=bool)
    return valid & (np.abs(p - .5) >= thresholds['margin']) & (distance <= thresholds['ood_max'])


def _select(y, p, groups, distance, fractions, qc_passed, qc_threshold, plan):
    valid = np.isfinite(p) & np.isfinite(distance) & np.isfinite(fractions) & qc_passed & (fractions > 0) & (fractions <= 1)
    if qc_threshold is not None:
        valid &= fractions >= qc_threshold
    summaries, best = [], None
    if valid.any():
        margins = np.unique(np.quantile(np.abs(p[valid] - .5), plan['margin_quantiles']))
        limits = np.unique(np.quantile(distance[valid], plan['ood_quantiles']))
        for margin in margins:
            for limit in limits:
                threshold = dict(margin=float(margin), ood_max=float(limit))
                accepted = _accept(p, distance, fractions, qc_passed, qc_threshold, threshold)
                summary = _risk_summary(y, p, groups, accepted)
                failures = _failed_checks(summary, plan)
                summaries.append(dict(thresholds=threshold, failed_checks=failures, summary=summary))
                if not failures:
                    # Coverage first, then lower actual risk, lower implied risk,
                    # stricter margin and smaller OOD range. Audit is not read.
                    rank = (summary['coverage'], -summary['empirical_risk'], -summary['mean_implied_risk'], margin, -limit)
                    if best is None or rank > best[0]:
                        best = (rank, threshold, summary)
    return dict(status='selected' if best else 'unvalidated',
        candidate_thresholds=None if best is None else best[1],
        selected_summary=None if best is None else best[2],
        candidate_count=len(summaries), candidates=summaries,
        failed_checks=[] if best else sorted({reason for row in summaries for reason in row['failed_checks']})
            or ['no_scoreable_selection_cases'])


def _validate_lineage(meta, groups, lineage):
    subjects = set(groups[meta].tolist())
    if len(lineage) != len(subjects):
        raise ValueError("Need one OOF head provenance record per meta subject")
    covered, seen = set(), set()
    for fold in lineage:
        heldout = fold['heldout_subject']
        fitting = fold['head_fit_subjects']
        query = fold['query_indices']
        if heldout in seen or heldout not in subjects or heldout in fitting:
            raise ValueError("OOF head cannot see its heldout subject")
        if set(fitting) != subjects - {heldout} or len(fitting) != 4:
            raise ValueError("OOF head must fit exactly the other four meta subjects")
        expected = set(meta[groups[meta] == heldout].tolist())
        if set(query) != expected or len(query) != len(expected) or covered & set(query):
            raise ValueError("OOF query provenance does not match its complete subject")
        if 'head_fit_indices' in fold:
            expected_fit = set(meta[np.isin(groups[meta], fitting)].tolist())
            if set(fold['head_fit_indices']) != expected_fit:
                raise ValueError("OOF head fit indices violate subject separation")
        covered.update(query)
        seen.add(heldout)
    if covered != set(meta.tolist()):
        raise ValueError("OOF provenance omits meta predictions")


class ReliabilityModelV3(ReliabilityModel):
    def __init__(self):
        super().__init__()
        self.qc_threshold = None
        self.candidate_thresholds = None
        self.plan = dict(DEFAULT_PLAN)

    def calibrate(self, probability):
        probability = _probability(probability)
        return None if probability is None else super().calibrate(probability)

    def assess(self, probability, descriptor, valid_fraction, *, qc_passed=True, reference_available=False):
        raw, p = _probability(probability), self.calibrate(probability)
        distance = self.ood_score(descriptor)
        fraction = float(valid_fraction) if valid_fraction is not None else float('nan')
        hard_bad = not qc_passed or not math.isfinite(fraction) or fraction <= 0 or fraction > 1
        coverage_warning = (not hard_bad and self.qc_threshold is not None and fraction < self.qc_threshold)
        ood = None if self.thresholds is None or distance is None else distance > self.thresholds['ood_max']
        reasons = []
        if hard_bad:
            reasons.append('QC_failed')
        if coverage_warning:
            reasons.append('QC_coverage_warning')
        if self.qc_threshold is None:
            reasons.append('QC_reference_unavailable')
        if raw is None:
            reasons.append('prediction_unavailable')
        if p is None:
            reasons.append('calibration_unavailable')
        if distance is None:
            reasons.append('OOD_reference_unavailable')
        if self.thresholds is None:
            reasons.append('policy_unvalidated')
            reasons.extend(self.validation.get('failed_checks', []))
        elif p is not None and abs(p - .5) < self.thresholds['margin']:
            reasons.append('margin_below_threshold')
        if ood is True:
            reasons.append('OOD')
        reasons = list(dict.fromkeys(reasons))
        return dict(schema_version='reliability_v3', p_raw=raw, p_cal=p,
            calibration_status='fitted' if self.calibrator is not None else 'unavailable',
            policy_status='validated' if self.thresholds is not None else 'unvalidated',
            signal_quality_bad=bool(hard_bad), coverage_warning=bool(coverage_warning),
            prediction_reliable=not reasons, rejection_reasons=reasons,
            ood_score=distance, ood=ood, evidence_conflict=False,
            individual_implied_risk=None if p is None else min(p, 1 - p),
            individual_risk_scope='descriptive; accepted_set_mean_risk_does_not_guarantee_individual_confidence',
            signal_quality=dict(valid_window_fraction=fraction if math.isfinite(fraction) else None,
                qc_passed=bool(qc_passed), coverage_warning=bool(coverage_warning), threshold=self.qc_threshold,
                reference_quality='valid' if reference_available else 'unavailable',
                threshold_source='base_training_valid_fraction_without_labels'),
            thresholds=self.thresholds, candidate_thresholds=self.candidate_thresholds,
            validation_mean_implied_risk=(self.validation.get('primary_audit') or {}).get('mean_implied_risk'),
            provenance={key.replace('_subjects', '_subject_count'): len(value)
                        if key.endswith('_subjects') else value for key, value in self.provenance.items()
                        if key != 'meta_oof_provenance'},
            confidence_scope='exploratory_independent_subject_audit_not_clinical_or_finite_sample_guarantee')


def fit_reliability_v3(raw_probability, y, groups, descriptors, valid_fractions, *,
        base_indices, meta_indices, policy_indices, meta_oof_probability,
        meta_oof_provenance, qc_passed=None, plan=None):
    """Fit internal roles only; unassigned/outer labels never select a parameter."""
    raw, y, groups = np.asarray(raw_probability, float), np.asarray(y), np.asarray(groups)
    descriptors, fractions = np.asarray(descriptors, float), np.asarray(valid_fractions, float)
    oof = np.asarray(meta_oof_probability, float)
    n = len(raw)
    if (y.shape != (n,) or groups.shape != (n,) or fractions.shape != (n,) or oof.shape != (n,)
            or descriptors.ndim != 2 or len(descriptors) != n):
        raise ValueError("Reliability arrays must share a path axis")
    quality = np.ones(n, dtype=bool) if qc_passed is None else np.asarray(qc_passed, bool)
    if quality.shape != (n,):
        raise ValueError("QC flags must match paths")
    base, meta, policy = [np.asarray(indices, int) for indices in (base_indices, meta_indices, policy_indices)]
    roles = (base, meta, policy)
    if any(len(set(ix.tolist())) != len(ix) or np.any(ix < 0) or np.any(ix >= n) for ix in roles):
        raise ValueError("Invalid or duplicate role indices")
    subjects = [set(groups[ix].tolist()) for ix in roles]
    if any(a & b for i, a in enumerate(subjects) for b in subjects[i + 1:]):
        raise ValueError("Base, meta and policy roles must be subject-disjoint")
    if len(subjects[1]) != 5 or len(subjects[2]) != 4:
        raise ValueError("V3 requires five meta and four policy subjects")
    if any(set(ix.tolist()) != set(np.flatnonzero(np.isin(groups, list(ss))).tolist()) for ix, ss in zip(roles, subjects)):
        raise ValueError("Role indices must contain every path of each role subject")
    if not np.isin(y[np.r_[meta, policy]], [0, 1]).all():
        raise ValueError("Internal meta/policy labels must be binary")
    _validate_lineage(meta, groups, meta_oof_provenance)
    model = ReliabilityModelV3()
    model.plan = _plan(plan)
    policy_subjects = sorted(subjects[2])
    selection_subjects, audit_subjects = policy_subjects[:2], policy_subjects[2:]
    model.provenance = dict(base_reference_subjects=sorted(subjects[0]), meta_calibration_subjects=sorted(subjects[1]),
        policy_selection_subjects=selection_subjects, policy_audit_subjects=audit_subjects,
        meta_oof_provenance=meta_oof_provenance,
        calibration_prediction_source='subject_OOF_head4_scores; production_head5_scores_are_never_calibration_fit_rows',
        calibration_scale_caveat='OOF_heads_fit_four_meta_subjects; production_head_fits_five; policy_audits_test_that_shift',
        policy_partition_rule='sorted_subject_ID_first_two_select_last_two_audit_frozen_before_labels',
        nested_validation_source='three_select_one_audit_subject_LOSO; veto_only_no_reselection')
    checks = []
    reference = base[quality[base] & np.isfinite(descriptors[base]).all(axis=1)
                     & np.isfinite(fractions[base]) & (fractions[base] > 0) & (fractions[base] <= 1)]
    if len(reference) and len(np.unique(groups[reference])) >= 2:
        values = descriptors[reference]
        model.center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - model.center), axis=0)
        model.scale = np.where(mad > 1e-8, mad, np.maximum(np.std(values, axis=0), 1e-8))
    else:
        checks.append('OOD_reference_unavailable')
    quality_reference = base[quality[base] & np.isfinite(fractions[base]) & (fractions[base] > 0) & (fractions[base] <= 1)]
    if len(quality_reference) and len(np.unique(groups[quality_reference])) >= 2:
        model.qc_threshold = float(np.quantile(fractions[quality_reference], model.plan['qc_fraction_quantile']))
    else:
        checks.append('QC_reference_unavailable')
    calibration = meta[quality[meta] & np.isfinite(oof[meta]) & (oof[meta] >= 0) & (oof[meta] <= 1)
                       & np.isfinite(fractions[meta]) & (fractions[meta] > 0) & (fractions[meta] <= 1)]
    calibration_diagnostic = dict(cases=len(calibration), subjects=len(np.unique(groups[calibration])),
        scope='subject_OOF_head_discrimination; recalibrator_fit_metrics_are_not_independent_validation')
    if (len(calibration) < model.plan['minimum_cases'] or len(np.unique(groups[calibration])) != 5
            or len(np.unique(y[calibration])) != 2):
        checks.append('calibration_support_insufficient')
    else:
        pp, yy = oof[calibration], y[calibration]
        calibration_diagnostic.update(auroc=float(roc_auc_score(yy, pp)),
            probability_range=[float(pp.min()), float(pp.max())], **_support(yy, pp, groups[calibration]))
        supported = all(v['subjects'] >= 2 and v['cases'] >= 2 for v in calibration_diagnostic['true_class_support'].values())
        if not supported:
            checks.append('calibration_true_class_subject_support_insufficient')
        if calibration_diagnostic['auroc'] <= .5 or np.ptp(pp) <= 1e-8:
            checks.append('calibration_discrimination_failed')
        if supported and calibration_diagnostic['auroc'] > .5 and np.ptp(pp) > 1e-8:
            logits = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
            calibrator = LogisticRegression(C=1., solver='liblinear', max_iter=1000, random_state=2026)
            calibrator.fit(logits[:, None], yy)
            calibration_diagnostic['slope'] = float(calibrator.coef_[0, 0])
            if calibrator.coef_[0, 0] > 0:
                model.calibrator = calibrator
            else:
                checks.append('calibration_nonmonotone')
    calibrated = [model.calibrate(v) for v in raw]
    p = np.asarray([np.nan if value is None else value for value in calibrated])
    distance = np.asarray([model.ood_score(v) for v in descriptors], float)
    selection = policy[np.isin(groups[policy], selection_subjects)]
    audit = policy[np.isin(groups[policy], audit_subjects)]
    selector = _select(y[selection], p[selection], groups[selection], distance[selection], fractions[selection],
                       quality[selection], model.qc_threshold, model.plan)
    model.candidate_thresholds = selector['candidate_thresholds']
    audit_accept = _accept(p[audit], distance[audit], fractions[audit], quality[audit], model.qc_threshold, model.candidate_thresholds)
    primary_audit = _risk_summary(y[audit], p[audit], groups[audit], audit_accept)
    primary_failures = _failed_checks(primary_audit, model.plan, audit_subjects)
    if model.candidate_thresholds is None:
        checks.append('policy_selection_failed')
        checks.extend(selector['failed_checks'])
    if primary_failures:
        checks.append('policy_independent_audit_failed')
        checks.extend(primary_failures)
    nested_accept = np.zeros(len(policy), dtype=bool)
    nested, nested_failed = [], False
    for heldout in policy_subjects:
        fit_mask, target_mask = groups[policy] != heldout, groups[policy] == heldout
        ix = policy[fit_mask]
        selected = _select(y[ix], p[ix], groups[ix], distance[ix], fractions[ix], quality[ix], model.qc_threshold, model.plan)
        ix = policy[target_mask]
        accepted = _accept(p[ix], distance[ix], fractions[ix], quality[ix], model.qc_threshold, selected['candidate_thresholds'])
        nested_accept[target_mask] = accepted
        nested_failed |= selected['candidate_thresholds'] is None
        nested.append(dict(heldout_subject=int(heldout), selection_subjects=sorted(set(groups[policy[fit_mask]].tolist())),
            selection_indices=policy[fit_mask].tolist(), audit_indices=ix.tolist(),
            candidate_thresholds=selected['candidate_thresholds'], selection_failed_checks=selected['failed_checks'],
            audit=_risk_summary(y[ix], p[ix], groups[ix], accepted)))
    nested_audit = _risk_summary(y[policy], p[policy], groups[policy], nested_accept)
    nested_checks = _failed_checks(nested_audit, model.plan, policy_subjects)
    if nested_failed or nested_checks:
        checks.append('policy_nested_validation_failed')
        checks.extend(nested_checks)
    checks = list(dict.fromkeys(checks))
    if not checks and model.calibrator is not None and model.candidate_thresholds is not None:
        model.thresholds = dict(model.candidate_thresholds)
    model.validation = dict(schema_version='reliability_v3',
        calibration_status='fitted' if model.calibrator is not None else 'unavailable',
        status='validated' if model.thresholds is not None else 'unvalidated', failed_checks=checks,
        plan=model.plan, qc_threshold=model.qc_threshold, qc_source='base_training_without_labels',
        calibration=calibration_diagnostic, primary_selection_subjects=selection_subjects,
        primary_audit_subjects=audit_subjects, primary_selection=selector, primary_audit=primary_audit,
        primary_audit_failed_checks=primary_failures, candidate_thresholds=model.candidate_thresholds,
        deployed_thresholds=model.thresholds, audit_failure_never_reselects=True,
        nested_folds=nested, nested_audit=nested_audit, nested_failed_checks=nested_checks,
        nested_scope='heldout_subject_predictions_from_different_frozen_candidates; selection_procedure_diagnostic_veto_only',
        estimated_risk_is_model_implied_not_true_risk=True, clinical_or_finite_sample_risk_guarantee=False)
    return model
