# -*- coding: utf-8 -*-
"""Apply frozen numerical model parameters to freshly encoded raw EEG.

No query labels, feature/prediction tables or questionnaire files are read. The
case component is a frozen training-only nearest-neighbour estimator; its fit
parameters stay inside this local tool and are never exposed to the agents.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

from .features import covariance, spectral_features, predict_score, compact_spectrum, filterbank_covariance
from .assets import Assets

class FrozenNumeric:
    def __init__(self, fold, manifest):
        self.fold = fold
        self.assets = Assets(manifest)
        self.first, a = self.assets.bundle(fold, "first")
        self.second, b = self.assets.bundle(fold, "second")
        self.selector, c = self.assets.bundle(fold, "selector")
        if self.first['split'] != self.second['split'] or self.first['split'] != self.selector['split']:
            raise ValueError('Numerical model roles differ')
        bank = self.first['tools']['case_retrieval']
        if fold in bank['reference_subjects']:
            raise ValueError('Query subject in frozen estimator support')
        self.hashes = {'numeric_first_sha256': a, 'numeric_second_sha256': b, 'numeric_selector_sha256': c}

    def predict(self, windows, features, mil_probability, baseline=None):
        absolute, powers = zip(*(spectral_features(w) for w in windows))
        a = np.asarray(absolute, float)
        n = max(1, len(a) // 3)
        reference = spectral_features(baseline)[1] if baseline is not None else None
        changes = None if reference is None else np.asarray([np.log((p + 1e-10) / (reference + 1e-10)).ravel() for p in powers])
        spectrum = np.r_[a.mean(axis=0), a.std(axis=0), a[-n:, :120].mean(axis=0) - a[:n, :120].mean(axis=0),
                         np.zeros(120) if changes is None else changes.mean(axis=0), float(changes is not None)]
        covs = np.asarray([covariance(w) for w in windows])
        scales = np.trace(covs, axis1=1, axis2=2) / 30
        mean_cov = (covs / scales[:, None, None]).mean(axis=0)
        eigen, vectors = np.linalg.eigh(mean_cov)
        log_cov = (vectors * np.log(np.maximum(eigen, 1e-10))) @ vectors.T
        ii, jj = np.triu_indices(30)
        spatial = np.r_[log_cov[ii, jj] * np.where(ii == jj, 1., np.sqrt(2.)), np.log(scales).mean(),
                        np.log(scales).std(), np.log(np.diagonal(covs, axis1=1, axis2=2)).mean(axis=0)]
        feature_mean, feature_std = features.mean(axis=0), features.std(axis=0)
        chunks = np.array_split(features, 3)
        matrices = {'vrms_features': np.r_[feature_mean, feature_std], 'spectral': spectrum,
                    'spatial_covariance': spatial, 'vrms_mean': feature_mean,
                    'vrms_temporal': np.r_[feature_mean, feature_std, chunks[-1].mean(axis=0) - chunks[0].mean(axis=0)],
                    'spectral_compact': compact_spectrum(absolute, changes), 'filterbank_csp': filterbank_covariance(windows)}
        scores, evidence = {}, []
        for family, x in matrices.items():
            bank = self.second['tools'] if family in ('vrms_mean', 'vrms_temporal', 'spectral_compact', 'filterbank_csp') else self.first['tools']
            score = float(predict_score(bank[family]['model'], x[None])[0])
            scores[family] = score
            evidence.append({'tool': family, 'raw_decision_score': score, 'predicted_class': 'High' if score >= 0 else 'Low'})
        bank = self.first['tools']['case_retrieval']
        reference_features = bank['scaler'].transform(bank['reference_features'])
        query = bank['scaler'].transform(feature_mean[None])[0]
        distance = np.sqrt(np.mean((reference_features - query) ** 2, axis=1))
        k, weighted = bank['config']
        chosen, seen = [], set()
        for index in np.argsort(distance, kind='stable'):
            subject = int(bank['reference_subjects'][index])
            if subject not in seen:
                chosen.append(int(index)); seen.add(subject)
            if len(chosen) == k:
                break
        weights = 1 / np.maximum(distance[chosen], 1e-6) if weighted else np.ones(len(chosen))
        vote = float(np.average(bank['reference_labels'][chosen], weights=weights))
        scores['case_retrieval'] = float(logit(np.clip(vote, 1e-5, 1 - 1e-5)))
        evidence.append({'tool': 'case_retrieval', 'raw_decision_score': scores['case_retrieval'], 'predicted_class': 'High' if vote >= .5 else 'Low'})
        combined = [float(logit(np.clip(mil_probability, 1e-6, 1 - 1e-6)))]
        for name in self.selector['feature_names'][1:]:
            score = scores[name]
            if name in ('vrms_mean', 'vrms_temporal', 'spectral_compact', 'filterbank_csp'):
                score = float(logit(np.clip(expit(score), 1e-6, 1 - 1e-6)))
            combined.append(score)
        model = self.selector['model']
        x = np.asarray(combined)[None]
        probability = float(expit(model.decision_function(x))[0]) if hasattr(model, 'decision_function') else float(model.predict_proba(x)[0, 1])
        # Only frozen internal-validation summaries are exposed to the language
        # Agent. Query identity, labels, arrays and estimator support stay local.
        profiles = {}
        for item in evidence:
            family = item['tool']
            bundle = self.second if family in ('vrms_mean', 'vrms_temporal', 'spectral_compact', 'filterbank_csp') else self.first
            fitted = bundle['tools'][family]
            calibrator = fitted.get('calibrator')
            item['calibration_usable'] = calibrator is not None
            item['calibrated_high_probability'] = (None if calibrator is None else
                float(calibrator.predict_proba([[item['raw_decision_score']]])[0, 1]))
            report = bundle.get('reports', {}).get(family, {})
            profiles[family] = {key: report[key] for key in
                ('policy', 'policy_correction', 'raw_policy', 'calibrated_policy', 'raw_correction', 'calibrated_correction')
                if key in report}
        case = next(item for item in evidence if item['tool'] == 'case_retrieval')
        case.update(descriptive_vote=vote, high_count=int(np.sum(bank['reference_labels'][chosen] == 1)),
                    low_count=int(np.sum(bank['reference_labels'][chosen] == 0)),
                    different_subjects=len(chosen), closest_distance=float(distance[chosen].min()))
        report = self.selector.get('report', {})
        profiles['corrective_evidence'] = {key: report[key] for key in ('policy_gated', 'correction', 'margin') if key in report}
        audit_path = self.assets.optional(self.fold, 'nested_audit') if hasattr(self, 'assets') else None
        if audit_path is not None and audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            profiles['corrective_evidence']['independent_nested_audit'] = {
                key: audit[key] for key in ('approved', 'nested_validation', 'correction')}
        return {'raw_combined_score': probability, 'predicted_class': 'High' if probability >= .5 else 'Low',
                'probability_calibrated': False, 'actual_tools': evidence, 'reference_available': reference is not None,
                'tool_validation_profiles': profiles}
