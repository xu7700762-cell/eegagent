"""Recompute final paired ACC and replay every optimized RAG context offline."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from eeg_agent.rag_experiment import RESULTS, load_replay, payload_for, read


def verify_results(directory=RESULTS):
    directory = Path(directory)
    inputs, banks, protocol, manifest = load_replay(directory)
    with (directory / "optimized_rag_paths.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = read(directory / "optimized_rag_summary.json")
    audit = read(directory / "optimized_rag_source_audit.json")
    if len(rows) != 146 or {row["query_id"] for row in rows} != set(inputs):
        raise ValueError("Incomplete or duplicated score rows")
    by_id = {row["query_id"]: row for row in rows}
    pilot = [row for row in rows if row["pilot24"] == "1"]
    if (len(pilot) != 24 or len({row["subject_group"] for row in pilot}) != 24
            or {row["query_id"] for row in pilot} != set(protocol["pilot24_ids"])):
        raise ValueError("Pilot membership changed")
    if not audit["passed"] or not audit["fully_completed_scope"] or audit["errors"]:
        raise ValueError("Original completed-output audit did not pass")
    metrics, confidence, providers = {}, {}, {}
    for arm, prefix, summary_key in (("without_rag", "without_rag", "without_rag"), ("percentile_group", "rag", "optimized_rag")):
        for row in rows:
            if (row["true_class"] not in ("Low", "High") or row[prefix + "_class"] not in ("Low", "High")
                    or row[prefix + "_confidence"] not in ("low", "medium", "high")):
                raise ValueError("Invalid score category")
        correct = sum(row[prefix + "_class"] == row["true_class"] for row in rows)
        recall = {label: sum(row[prefix + "_class"] == label and row["true_class"] == label for row in rows)
                  / sum(row["true_class"] == label for row in rows) for label in ("High", "Low")}
        counts = dict(Counter(row[prefix + "_confidence"] for row in rows))
        expected = {"fixed_total": 146, "valid": 146, "correct": correct, "accuracy": correct / 146,
                    "recall": recall, "balanced_accuracy": sum(recall.values()) / 2, "confidence_counts": counts}
        for key, value in expected.items():
            if summary[summary_key][key] != value:
                raise ValueError("Summary differs from actual row scores: " + arm + "/" + key)
        metrics[arm] = expected
        confidence[arm] = {}
        for level in ("low", "medium", "high"):
            selected = [row for row in rows if row[prefix + "_confidence"] == level]
            n = sum(row[prefix + "_class"] == row["true_class"] for row in selected)
            confidence[arm][level] = {"total": len(selected), "correct": n, "accuracy": n / len(selected) if selected else None}
        providers[arm] = dict(Counter(row[prefix + "_provider"] for row in rows))
        if providers[arm] != summary["declared_execution_route_counts"][arm]:
            raise ValueError("Declared provider counts differ from rows")
        proofs = audit["proofs"][arm]
        if len(proofs) != 146 or {proof["query_id"] for proof in proofs} != set(by_id):
            raise ValueError("Incomplete original output proof index")
        for proof in proofs:
            row = by_id[proof["query_id"]]
            if (proof["final_class"] != row[prefix + "_class"] or proof["confidence"] != row[prefix + "_confidence"]
                    or proof["correct"] != (row[prefix + "_class"] == row["true_class"])
                    or not proof["native_responses"]
                    or any(item["status"] != "completed" or item["model"] != protocol["model"] for item in proof["native_responses"])):
                raise ValueError("Published score differs from original completed-output proof")
    corrected = [row["query_id"] for row in rows if row["rag_class"] == row["true_class"] and row["without_rag_class"] != row["true_class"]]
    damaged = [row["query_id"] for row in rows if row["rag_class"] != row["true_class"] and row["without_rag_class"] == row["true_class"]]
    paired = {"both_valid": 146, "corrected": len(corrected), "damaged": len(damaged), "net": len(corrected) - len(damaged)}
    for key, value in paired.items():
        if summary["optimized_rag"]["paired"][key] != value:
            raise ValueError("Paired score differs from actual rows")
    pilot_net = sum(int(row["rag_class"] == row["true_class"]) - int(row["without_rag_class"] == row["true_class"]) for row in pilot)
    if (summary["net_correct"] != paired["net"] or summary["pilot24_net"] != pilot_net
            or summary["remaining122_net"] != paired["net"] - pilot_net
            or summary["accuracy_gain_percentage_points"] != paired["net"] / 146 * 100
            or summary["target_plus_four_met"] != (paired["net"] >= 4)):
        raise ValueError("Net improvement or target claim differs from paired scores")
    warning_ids = {row["query_id"] for row in rows if row["without_rag_citation_warning"] == "1"}
    warnings = summary["citation_warnings"]["without_rag"]
    if warning_ids != set(warnings["citation_warning_queries"]) or len(warning_ids) != warnings["citation_warning_count"]:
        raise ValueError("Citation warning count differs from rows")
    # Numeric fixtures are from transformed historical requests, not this code's output.
    for q in inputs:
        without = payload_for(q, "without_rag", inputs, banks, manifest)
        with_rag = payload_for(q, "percentile_group", inputs, banks, manifest)
        context = with_rag["initial_evidence"].pop("policy_heldout_case_context")
        if with_rag != without or context["retrieval"]["returned_subjects"] != 4:
            raise ValueError("Arm inputs differ beyond RAG or reference subjects missing")
    return {"status": "passed", "contexts_verified": len(inputs), "reference_folds": len(banks),
            "metrics": metrics, "paired": paired, "confidence": confidence, "declared_providers": providers,
            "pilot24_net": pilot_net, "remaining122_net": paired["net"] - pilot_net,
            "target_plus_four_met": paired["net"] >= 4, "citation_warning_paths": len(warning_ids), "model_API_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", help="Print one anonymous numeric payload, such as q001; no API call")
    parser.add_argument("--arm", choices=("without_rag", "percentile_group"), default="percentile_group")
    args = parser.parse_args()
    if args.payload:
        inputs, banks, protocol, manifest = load_replay()
        result = {"system": protocol["system"], "model": protocol["model"], "reasoning_effort": protocol["reasoning_effort"],
                  "max_output_tokens": protocol["max_output_tokens"], "payload": payload_for(args.payload, args.arm, inputs, banks, manifest)}
    else:
        result = verify_results()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
