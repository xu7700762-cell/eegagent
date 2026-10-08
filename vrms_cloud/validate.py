"""Validate V2 frozen artifacts and recompute four-state selective metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from vrms_pilot.data import digest, write_json
from .cloud import SYSTEM, check_blind, parse_predictions
from .evaluate import evaluate_v2
from .independent import load_prepared, prepare_assessment
from .prepare import DEFAULT_OUT


def same_evidence(a, b):
    """Allow inference roundoff across devices, but keep roles/fields exact."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_evidence(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_evidence(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5)
    return type(a) is type(b) and a == b


def validate_v2(out=DEFAULT_OUT, offline=False):
    out = Path(out)
    data, folds, protocol = load_prepared(out)
    for filename, expected in protocol["code_sha256"].items():
        if digest(Path(__file__).parent / filename) != expected:
            raise ValueError(f"Frozen cloud source changed: {filename}")
    status = json.loads((out / ("offline_status.json" if offline else "independent_status.json")).read_text(encoding="utf-8"))
    folder = out / ("offline_calls" if offline else "independent_calls")
    calls = 0
    bundles = {}
    records = {r["path_index"]: (fold, r) for fold in folds for r in fold["records"] if r["role"] == "outer_test"}
    for row in status["results"]:
        i = row["path_index"]
        pack = json.loads((folder / f"path_{i:03d}_evidence_pack.json").read_text(encoding="utf-8"))
        fold, record = records[i]
        subject = fold["outer_subject"]
        if subject not in bundles:
            bundles.clear()
            bundles[subject] = data.load_bundle(fold["split"])
        inputs, replay_pack = prepare_assessment(data, record, bundles[subject])
        if (not same_evidence(pack["p_cal"], inputs["reliability"]["p_cal"]) or pack["state"] != inputs["decision"]["state"]
                or pack["tool_calls"] != replay_pack["tool_calls"]):
            raise ValueError("Saved result differs from frozen lazy model replay")
        expected_call = not offline and inputs["decision"]["state"] == "uncertain"
        if pack.get("cloud_called") is not expected_call:
            raise ValueError("Cloud call flag differs from the deterministic policy")
        if pack["high_probability"] != pack["p_cal"] or pack["effective_llm_weight"] != 0:
            raise ValueError("Cloud changed model probability")
        if pack["state"] in ("high", "low"):
            reliability = pack["reliability"]
            if not (reliability["prediction_reliable"] and reliability["policy_status"] == "validated"
                    and reliability["ood"] is False and reliability["evidence_conflict"] is False):
                raise ValueError("Published class bypassed reliability gate")
        if pack["decision_reason"] == "validated_deep_prediction":
            if set(pack["tool_calls"]) & {"CaseRetriever", "BiomarkerCalculator", "CovarianceAnalyzer"}:
                raise ValueError("Easy path acquired optional tools")
        if pack.get("cloud_called"):
            log = json.loads((folder / f"path_{i:03d}_cloud_call.json").read_text(encoding="utf-8"))
            if pack["cloud_success"] != log["success"] or pack["fallback_used"] != (not log["success"]):
                raise ValueError("Saved cloud success/fallback differs from API record")
            is_deepseek = log.get("provider") == "project_deepseek"
            message = log["request"]["messages"][1]["content"] if is_deepseek else log["user_message"]
            expected_hash = (hashlib.sha256(json.dumps(log["request"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                             if is_deepseek else hashlib.sha256((SYSTEM + message).encode()).hexdigest())
            if log["request_sha256"] != expected_hash:
                raise ValueError("Request hash changed")
            request = json.loads(message)
            if len(request["queries"]) != 1:
                raise ValueError("Independent requests require exactly one anonymous query")
            check_blind(request["queries"])
            for example in request["examples"]:
                check_blind(example["evidence"])
            if not same_evidence(request["examples"], inputs["examples"]) or not same_evidence(request["queries"][0]["evidence"], inputs["evidence"]):
                raise ValueError("Cloud request differs from frozen lazy tool evidence")
            if log["success"]:
                if is_deepseek:
                    from vrms_deepseek.supervisor import normalize_response
                    raw = log["attempts"][-1]["response"]
                    parsed = normalize_response(json.loads(raw["choices"][0]["message"]["content"]), raw)
                    stored = log["prediction"]
                else:
                    parsed = parse_predictions(log["attempts"][-1]["response"], log["mapping"])
                    stored = log["predictions"]
                if parsed != stored:
                    raise ValueError("Cached explanation changed")
                analysis = stored if is_deepseek else stored[0]
                if any(pack[name] != analysis[name] for name in
                       ("supporting_evidence", "conflicting_evidence", "missing_evidence", "explanation")):
                    raise ValueError("Saved explanation differs from API record")
            elif (pack["supporting_evidence"] or pack["conflicting_evidence"] or pack["missing_evidence"]
                  or pack["explanation"] is not None):
                raise ValueError("Failed API call cannot invent a cloud explanation")
            calls += 1
        elif (pack["cloud_success"] or pack["fallback_used"] or pack["supporting_evidence"]
              or pack["conflicting_evidence"] or pack["missing_evidence"] or pack["explanation"] is not None):
            raise ValueError("Skipped API call cannot contain a cloud explanation")
    score = evaluate_v2(out, offline)
    validation = dict(schema_version="uncertainty_agent_v2", status="passed", paths=len(data.paths),
                      model_probability_unchanged=True, dynamic_tools_verified=True,
                      independent_requests_checked=calls, frozen_hashes_checked=True,
                      metrics_recomputed=True, offline=offline, partial_run=score["partial_run"])
    write_json(out / ("offline_validation.json" if offline else "validation.json"), validation)
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_v2(args.out, args.offline), indent=2), flush=True)


if __name__ == "__main__":
    main()
