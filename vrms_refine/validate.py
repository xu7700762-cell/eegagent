"""Recompute V2 retrieval provenance and anonymous explanation requests."""
import argparse
import copy
import json
from pathlib import Path

from vrms_cloud.cloud import check_blind
from .common import DEFAULT_OUT, check_v2_artifacts, job_hash, message_for, read_json, sha, write_json
from .prepare import retrieve_balanced_examples
from .retrieval import CaseRetriever


def validate_v2(out):
    out = Path(out)
    protocol = check_v2_artifacts(out)
    audit, jobs = (read_json(out / name) for name in ("preparation_audit.json", "jobs.json"))
    if protocol.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("V2 validator cannot validate historical fusion requests")
    for filename, expected in protocol["original_sha256"].items():
        if sha(filename) != expected:
            raise ValueError("Frozen evidence source changed")
    cloud_out = Path(protocol["cloud_out"])
    folds = {fold["outer_subject"]: fold for fold in read_json(cloud_out / "fold_evidence.json")}
    evaluation = {record["path_index"]: record for record in read_json(cloud_out / "evaluation.json")}
    diagnostics = {row["outer_subject"]: row for row in audit["diagnostics"]}
    expected_keys = {(mode, subject, record["path_index"]) for subject, fold in folds.items()
                     for record in fold["records"] if record["role"] in ("policy_validation", "outer_test")
                     for mode in ("case_rag", "balanced_few_shot_control")}
    keys = {(job["mode"], job["outer_subject"], job["path_index"]) for job in jobs}
    if keys != expected_keys or len(keys) != len(jobs) or len(jobs) != protocol["expected_requests"]:
        raise ValueError("Duplicate or missing V2 requests")
    retrievers, banks = {}, {}
    for subject, fold in folds.items():
        split = fold["split"]
        partitions = [set(split[name]) for name in ("base_train", "meta_calibration", "policy_validation", "outer_test")]
        if sum(map(len, partitions)) != len(set.union(*partitions)) or partitions[-1] != {subject}:
            raise ValueError("Subject partitions overlap")
        bank = [dict(path_index=record["path_index"], role="meta", evidence=record["evidence"])
                for record in fold["records"] if record["role"] == "meta"]
        retriever = CaseRetriever(bank, evaluation, allowed_subjects=partitions[1])
        banks[subject], retrievers[subject] = bank, retriever
        diagnostic = diagnostics[subject]
        if diagnostic["retrieval_calibration"] != retriever.calibration_audit() or diagnostic["top_k_loso"] != retriever.loso_evaluation():
            raise ValueError("Retrieval LOSO diagnostics changed")
        for row in diagnostic["top_k_loso"]["rows"]:
            if any(evaluation[index]["subject_key"] == row["heldout_subject"] for index in row["selected_indices"]):
                raise ValueError("Retrieval included its heldout subject")
    for job in jobs:
        subject, index = job["outer_subject"], job["path_index"]
        fold = folds[subject]
        record = next(record for record in fold["records"] if record["path_index"] == index)
        query_subject = evaluation[index]["subject_key"]
        if job["role"] != record["role"] or job["subject_key"] != query_subject or query_subject in fold["split"]["meta_calibration"]:
            raise ValueError("Query role or subject partition changed")
        message = json.loads(job["message"])
        if len(message["queries"]) != 1 or message["queries"][0]["id"] != "qsingle":
            raise ValueError("Request is not one anonymous path")
        check_blind(message["queries"])
        query = copy.deepcopy(message["queries"][0]["evidence"])
        context = query.pop("analysis_context")
        if query != record["evidence"]:
            raise ValueError("Underlying query measurements changed")
        if job["mode"] == "case_rag":
            result = retrievers[subject](query, query_subject=query_subject)
            examples, indices = result["examples"], result["selected_indices"]
            if result["retrieval_quality"] != context["retrieval_quality"]:
                raise ValueError("Retrieval quality changed")
            if len({evaluation[i]["subject_key"] for i in indices}) != len(indices) or len(indices) > 5:
                raise ValueError("Top K subject cap violated")
        else:
            examples, indices = retrieve_balanced_examples(query, banks[subject], evaluation, query_subject=query_subject)
        provenance = next(row for row in diagnostics[subject]["contexts"] if row["mode"] == job["mode"] and row["path_index"] == index)
        if indices != provenance["selected_example_indices"] or context["retrieval_quality"] != provenance["retrieval_quality"]:
            raise ValueError("Reference mapping changed")
        if any(evaluation[i]["subject_key"] not in fold["split"]["meta_calibration"] for i in indices):
            raise ValueError("Reference escaped internal meta subjects")
        if message["examples"] != examples or message_for(message["queries"][0]["evidence"], examples) != job["message"]:
            raise ValueError("Example content changed")
        if job_hash(job["message"]) != job["message_sha256"]:
            raise ValueError("Request hash changed")
    result = dict(status="passed", schema_version="uncertainty_agent_v2", single_query_jobs_checked=len(jobs),
        top_k_subject_diversity_checked=True, nested_subject_loso_checked=True,
        query_labels_absent=True, reference_sources_meta_only=True, physiology_not_required=True,
        jobs_sha256=sha(out / "jobs.json"), plan_sha256=sha(out / "analysis_plan.json"),
        audit_sha256=sha(out / "preparation_audit.json"), protocol_sha256=sha(out / "protocol.json"))
    write_json(out / "structural_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(validate_v2(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
