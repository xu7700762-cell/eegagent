"""Numeric-only V3 Supervisor: a validated risk gate releases model scores."""
from __future__ import annotations

import numpy as np


def reliability_decision(reliability):
    """Preserve calibrated probability; expose every blocking condition.

    This experiment acquires no RAG/physiology/cloud evidence. A usable signal
    with an unvalidated gate remains uncertain; coverage warnings are not QC
    failures. The unchanged V2 engine stays available for frozen replay.
    """
    gate = dict(reliability)
    p = gate.get('p_cal')
    p = float(p) if p is not None and np.isfinite(p) else None
    reasons = list(gate.get('rejection_reasons', []))
    if gate.get('signal_quality_bad'):
        state = 'insufficient_data'
    elif (p is not None and gate.get('prediction_reliable') is True
          and gate.get('policy_status') == 'validated'
          and gate.get('ood') is False and not reasons):
        state = 'high' if p >= .5 else 'low'
    else:
        state = 'uncertain'
    return dict(probability=p, state=state, rejection_reasons=reasons,
                stop_reason='validated_risk_release' if state in ('high', 'low')
                else 'technical_qc_failure' if state == 'insufficient_data'
                else 'risk_or_validation_abstention',
                probability_source='subject_oof_calibrated_deep_head',
                tool_calls=['SignalQualityChecker', 'DeepVRMSDetector', 'UncertaintyEvaluator'],
                llm_called=False, case_retrieval_called=False)
