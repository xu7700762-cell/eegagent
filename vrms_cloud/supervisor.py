"""Reusable cloud final-synthesis entrypoint, using the existing provider.

This participates in the numerical decision, not only in its explanation.
The returned confidence is an exploratory score, not a reliability guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .cloud import SYSTEM, call_job, check_blind, choose_examples, existing_provider, write_json

DEFAULT_OUT = Path("outputs/vrms_cloud/20261008_seed2026")


def assess_evidence(evidence, examples, numeric_probability, log_path, llm_weight=.25):
    if not 0 <= llm_weight <= 1 or not 0 <= numeric_probability <= 1:
        raise ValueError("Probability and blend weight must be within [0,1]")
    check_blind(evidence)
    for example in examples:
        check_blind(example["evidence"])
        if example["observed_class"] not in ("high", "low"):
            raise ValueError("Examples need explicit internal high/low classes")
    message = json.dumps(dict(task="Assess this one anonymous path independently",
                              examples=examples, queries=[dict(id="qsingle", evidence=evidence)]),
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    job = dict(mode="single_evidence_assessment", outer_subject=None, user_message=message,
               mapping={"qsingle": {"role": "single_case"}}, example_indices=[],
               request_sha256=hashlib.sha256((SYSTEM + message).encode()).hexdigest())
    response = call_job(existing_provider(), job, log_path)
    cloud = response["predictions"][0] if response["success"] else None
    probability = numeric_probability if cloud is None else (
        (1 - llm_weight) * numeric_probability + llm_weight * cloud["high_probability"])
    return dict(schema_version="cloud_supervisor_v1", assessment_scope="whole_path_end",
                state="high" if probability >= .5 else "low", high_probability=probability,
                confidence_score=max(probability, 1 - probability), confidence_calibrated=False,
                numerical_probability=numeric_probability,
                cloud_probability=None if cloud is None else cloud["high_probability"],
                cloud_uncertain=True if cloud is None else cloud["uncertain"],
                effective_llm_weight=llm_weight if cloud is not None else 0,
                cloud_success=response["success"], measured_cloud_seconds=response["seconds"],
                explanation=dict(cloud_reason=None if cloud is None else cloud["reason"],
                                 fusion="(1 - LLM weight) * numerical probability + LLM weight * cloud probability",
                                 endpoint="whole-path score <30 / >=30; provisional research classification"),
                fallback_used=cloud is None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument("--llm-weight", type=float, default=.25)
    args = parser.parse_args()
    evaluation = {r["path_index"]: r for r in json.loads((args.out / "evaluation.json").read_text(encoding="utf-8"))}
    folds = json.loads((args.out / "fold_evidence.json").read_text(encoding="utf-8"))
    subject = evaluation[args.path_index]["subject_key"]
    fold = next(f for f in folds if f["outer_subject"] == subject)
    record = next(r for r in fold["records"] if r["path_index"] == args.path_index and r["role"] == "outer_test")
    examples = [dict(evidence=r["evidence"], observed_class="high" if evaluation[r["path_index"]]["label"] else "low")
                for r in choose_examples(fold["records"], evaluation, 2026 + subject)]
    numeric = record["evidence"]["improved_probabilities"]["validated_numeric_probability"]
    # Separate demonstration files: this extra call is not added to benchmark metrics.
    result = assess_evidence(record["evidence"], examples, numeric,
                             args.out / f"single_case_{args.path_index:03d}_cloud_call.json", args.llm_weight)
    write_json(args.out / f"single_case_{args.path_index:03d}_evidence_pack.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
