"""Numerical evidence definitions and deterministic Supervisor policy."""
from __future__ import annotations

import numpy as np


def temporal_features(logits, starts_sec=None):
    logits = np.asarray(logits, dtype=float)
    if not len(logits):
        return None
    p = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    times = np.arange(len(p)) * 5.0 if starts_sec is None else np.asarray(starts_sec)
    recent = times >= times[-1] - 25.0
    recent_p, recent_time = p[recent], times[recent]
    slope = 0.0
    if len(recent_p) >= 2:
        centered = recent_time - recent_time.mean()
        slope = float(np.sum(centered * (recent_p - recent_p.mean())) / np.sum(centered ** 2))
    consecutive = 0
    for i in range(len(p) - 1, -1, -1):
        if p[i] < 0.5 or (i < len(p) - 1 and times[i + 1] - times[i] > 5.01):
            break
        consecutive += 1
    return np.array([logits.mean(), p.std(), p.max(), recent_p.mean(), slope,
                     p[-1], min(consecutive, 12) / 12.0], dtype=float)


def signed_support(probability, threshold=0.5):
    if probability is None:
        return None
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly inside (0, 1)")
    return ((probability - threshold) / (1 - threshold) if probability >= threshold
            else (probability - threshold) / threshold)


def consensus(probabilities, min_strength=0.05):
    support = [signed_support(p) for p in probabilities if p is not None and np.isfinite(p)]
    if min_strength <= 0:
        raise ValueError("min_strength must be positive")
    strength = sum(abs(s) for s in support)
    if not support or strength < min_strength:
        return dict(support_score=None, conflict_score=None, strength=strength,
                    status="insufficient_evidence")
    return dict(support_score=float(np.mean(support)),
                conflict_score=float(1 - abs(sum(support)) / strength),
                strength=float(strength), status="measured")


def choose_margin(probabilities, y, minimum_coverage=0.35, target_accuracy=0.80):
    """Historical V1 margin diagnostic; unused by V2 adaptive decisions."""
    p, y = np.asarray(probabilities), np.asarray(y)
    finite = np.isfinite(p)
    for margin in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        accept = finite & (np.abs(p - 0.5) >= margin)
        if accept.sum() >= 8 and accept.mean() >= minimum_coverage:
            if np.mean((p[accept] >= 0.5) == y[accept]) >= target_accuracy:
                return margin
    return None


def adaptive_policy(scores, margins=None, reference_available=True, *, reliability=None):
    """Acquire cases, then physiology lazily; preserve the calibrated deep score.

    ``margins`` is retained for callers loading historical bundles but cannot
    authorize a V2 exit. Publication needs independently fitted reliability.
    Case label counts and unsigned physiology never become new probabilities.
    """
    gate = dict(reliability or {})
    trace = []
    p = gate.get("p_cal")
    p = float(p) if p is not None and np.isfinite(p) else None
    if gate.get("signal_quality_bad"):
        return dict(probability=p, state="insufficient_data", stage="deep", requested_tools=trace,
                    stop_reason="signal_quality_rejected", evidence_conflict=False,
                    consensus=consensus([]))
    scores("deep")
    trace.append("DeepVRMSDetector")
    reliable = (p is not None and gate.get("prediction_reliable") is True
                and gate.get("ood") is False and not gate.get("evidence_conflict", False))
    if reliable:
        return dict(probability=p, state="high" if p >= .5 else "low", stage="deep", requested_tools=trace,
                    stop_reason="validated_calibration_quality_ood_exit", evidence_conflict=False,
                    consensus=consensus([p]))
    retrieval = scores("case_retrieval") or {}
    trace.append("CaseRetriever")
    quality = retrieval.get("retrieval_quality", {})
    conflict = bool(gate.get("evidence_conflict", False) or retrieval.get("evidence_conflict", False))
    physiological_scores = []
    if quality.get("reliable") is not True or conflict:
        physiological_scores.append(scores("biomarker"))
        trace.append("BiomarkerCalculator")
        if reference_available:
            physiological_scores.append(scores("covariance"))
            trace.append("CovarianceAnalyzer")
    return dict(probability=p, state="uncertain", stage="deep", requested_tools=trace,
                stop_reason="unresolved_or_unvalidated_exit", evidence_conflict=conflict,
                retrieval_quality=quality, consensus=consensus(physiological_scores))
