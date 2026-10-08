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
    """Only the independent inner policy-validation labels may enter here."""
    p, y = np.asarray(probabilities), np.asarray(y)
    finite = np.isfinite(p)
    for margin in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        accept = finite & (np.abs(p - 0.5) >= margin)
        if accept.sum() >= 8 and accept.mean() >= minimum_coverage:
            if np.mean((p[accept] >= 0.5) == y[accept]) >= target_accuracy:
                return margin
    return None


def adaptive_policy(scores, margins, reference_available=True):
    """Stops current evidence acquisition, not stream collection.

    scores are supplied lazily by callable tool/meta functions. Deep and Temporal
    form one group. Covariance may remain unavailable or fail to resolve conflict.
    """
    trace = []
    qdeep = scores("deep")
    trace.append("DeepVRMSDetector")
    qdt = scores("deep_temporal")
    trace.append("TemporalAnalyzer")
    stage = "deep_temporal"
    q = qdt
    consensus_result = consensus([qdt])
    def acceptable(name, probability, conflict):
        margin = margins.get(name)
        return (probability is not None and np.isfinite(probability) and margin is not None
                and abs(probability - 0.5) >= margin
                and consensus_result["status"] == "measured"
                and (conflict is None or conflict < 0.5))
    if acceptable(stage, q, consensus_result["conflict_score"]):
        return dict(probability=float(q), state="high" if q >= .5 else "low",
                    stage=stage, requested_tools=trace, stop_reason="validated_margin_temporal",
                    consensus=consensus_result)
    qb = scores("biomarker")
    trace.append("BiomarkerCalculator")
    stage = "deep_temporal_bio"
    q = scores(stage)
    consensus_result = consensus([qdt, qb])
    if acceptable(stage, q, consensus_result["conflict_score"]):
        return dict(probability=float(q), state="high" if q >= .5 else "low",
                    stage=stage, requested_tools=trace, stop_reason="validated_margin_bio",
                    consensus=consensus_result)
    if reference_available:
        qc = scores("covariance")
        trace.append("CovarianceAnalyzer")
        stage = "full"
        q = scores(stage)
        consensus_result = consensus([qdt, qb, qc])
    resolved = acceptable(stage, q, consensus_result["conflict_score"])
    return dict(probability=None if q is None or not np.isfinite(q) else float(q),
                state=("high" if q >= .5 else "low") if resolved else "uncertain",
                stage=stage, requested_tools=trace,
                stop_reason="validated_full_evidence" if resolved else "unresolved_or_unvalidated_exit",
                consensus=consensus_result)
