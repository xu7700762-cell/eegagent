"""Deterministic reliability gates, with optional explanation-only cloud analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .cloud import SYSTEM, call_job, check_blind, existing_provider, write_json

DEFAULT_OUT = Path("outputs/vrms_cloud/v2_seed2026")
STATES = ("high", "low", "uncertain", "insufficient_data")


def probability_or_none(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability in [0,1]")
    return float(value)


def assessment_context(evidence, numeric_probability, reliability=None, decision=None):
    """Only machine evidence controls a decision; missing calibration abstains."""
    raw = probability_or_none(numeric_probability, "numeric_probability")
    if reliability is None:
        reliability = (decision or {}).get("reliability", evidence.get("reliability", {}))
    if not isinstance(reliability, dict):
        raise ValueError("Reliability must be a machine-generated object")
    p_cal = probability_or_none(reliability.get("p_cal"), "p_cal")
    if p_cal is not None and reliability.get("calibration_status") != "fitted":
        raise ValueError("Calibrated probability needs fitted calibration provenance")
    if decision is not None and "probability" in decision:
        supplied = probability_or_none(decision["probability"], "decision probability")
        if supplied != p_cal:
            raise ValueError("Decision and calibrated deep probability disagree")
    if reliability.get("signal_quality_bad") is True:
        state, reason = "insufficient_data", "signal_quality_rejected"
    elif (p_cal is not None and reliability.get("prediction_reliable") is True
          and reliability.get("signal_quality_bad") is False
          and reliability.get("policy_status") == "validated"
          and reliability.get("ood") is False
          and reliability.get("evidence_conflict") is False):
        state, reason = ("high" if p_cal >= .5 else "low"), "validated_deep_prediction"
    else:
        state, reason = "uncertain", "unresolved_or_unvalidated_evidence"
    if decision is not None:
        if decision.get("state") not in STATES:
            raise ValueError("Invalid deterministic decision state")
        # A lazy policy may abstain after finding unavailable/unreliable tools.
        # It may never release a class that these reliability gates reject.
        if decision["state"] in ("high", "low") and decision["state"] != state:
            raise ValueError("Decision bypasses calibrated reliability gates")
        if decision["state"] == "uncertain" and state != "insufficient_data":
            state, reason = "uncertain", decision.get("stop_reason", reason)
        elif decision["state"] == "insufficient_data" and state != "insufficient_data":
            raise ValueError("Quality refusal needs a failed signal-quality gate")
    return dict(p_raw=raw, p_cal=p_cal, state=state, reason=reason, reliability=reliability)


def summarize_assessment(evidence, numeric_probability, response=None, *, reliability=None, decision=None):
    """Shared vendor-independent result; the LLM can never modify p_cal/state."""
    context = assessment_context(evidence, numeric_probability, reliability, decision)
    response = response or {}
    called = bool(response) and response.get("skipped_reason") is None
    success = response.get("success") is True
    analysis = response.get("prediction") if success else None
    if success and analysis is None:
        analysis = response["predictions"][0]
    requested = (decision or {}).get("requested_tools", [])
    return dict(schema_version="uncertainty_agent_v2", assessment_scope="whole_path_end",
                state=context["state"], high_probability=context["p_cal"],
                p_cal=context["p_cal"], p_raw=context["p_raw"],
                probability_source="calibrated_deep_model", numerical_probability=context["p_raw"],
                reliability=context["reliability"], decision_reason=context["reason"],
                supporting_evidence=[] if analysis is None else analysis["supporting_evidence"],
                conflicting_evidence=[] if analysis is None else analysis["conflicting_evidence"],
                missing_evidence=[] if analysis is None else analysis["missing_evidence"],
                explanation=None if analysis is None else analysis["explanation"],
                requested_tools=requested, effective_llm_weight=0,
                cloud_called=called, cloud_success=success,
                measured_cloud_seconds=response.get("seconds", 0.),
                cloud_skipped_reason=None if called else response.get("skipped_reason", context["reason"]),
                fallback_used=called and not success,
                endpoint="whole-path score <30 / >=30; provisional research classification")


def validate_inputs(evidence, examples, numeric_probability, llm_weight, reliability, decision):
    # Retain the positional V1 parameter only so old callers cannot accidentally
    # re-enable score fusion. Its effective weight is always zero in V2.
    if llm_weight is not None:
        probability_or_none(llm_weight, "legacy llm_weight")
    check_blind(evidence)
    for example in examples:
        check_blind(example["evidence"])
        if example["observed_class"] not in ("high", "low"):
            raise ValueError("Examples need explicit internal high/low classes")
    return assessment_context(evidence, numeric_probability, reliability, decision)


def assess_evidence(evidence, examples, numeric_probability, log_path, llm_weight=None, *,
                    reliability=None, decision=None):
    context = validate_inputs(evidence, examples, numeric_probability, llm_weight, reliability, decision)
    if context["state"] != "uncertain":
        return summarize_assessment(evidence, numeric_probability, reliability=reliability, decision=decision)
    message = json.dumps(dict(task="Explain this anonymous path's support, conflict and missing evidence",
                              examples=examples, queries=[dict(id="qsingle", evidence=evidence)]),
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    job = dict(mode="single_evidence_assessment_v2", outer_subject=None, user_message=message,
               mapping={"qsingle": {"role": "single_case"}}, example_indices=[],
               request_sha256=hashlib.sha256((SYSTEM + message).encode()).hexdigest())
    response = call_job(existing_provider(), job, log_path)
    return summarize_assessment(evidence, numeric_probability, response,
                                reliability=reliability, decision=decision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--path-index", type=int, default=0)
    args = parser.parse_args()
    # Single-case demonstrations use the same lazy V2 entrypoint as evaluation.
    from .independent import assess_path
    result = assess_path(args.out, args.path_index)
    write_json(args.out / f"single_case_{args.path_index:03d}_evidence_pack.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
