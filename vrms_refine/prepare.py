"""Freeze read-only V2 retrieval comparisons; no training/current-path tools.

Runtime acquisition uses CaseRetriever through ToolReplay. This offline prompt
comparison reads only deep/temporal/QC snapshots and never prepares physiology.
"""
import argparse
import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import CLOUD, DEFAULT_OUT, job_hash, message_for, read_json, sha, write_json
from .retrieval import TEMPORAL_FIELDS, CaseRetriever, case_vector, _scale
from vrms_cloud.cloud import SYSTEM


def head_crossfit(x, y, groups, meta, c):
    """Retained head-only diagnostic, not a fully cross-fitted encoder."""
    p = np.full(len(y), np.nan)
    provenance, prior_fallbacks = [], 0
    for subject in np.unique(groups[meta]):
        training = meta[groups[meta] != subject]
        target = meta[groups[meta] == subject]
        fitting = training[np.isfinite(x[training]).all(axis=1)]
        good = target[np.isfinite(x[target]).all(axis=1)]
        if len(fitting) < 6 or len(np.unique(y[fitting])) < 2:
            p[good] = (float(y[training].sum()) + .5) / (len(training) + 1)
            prior_fallbacks += len(good)
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear", max_iter=1000,
                                                                       class_weight=None, random_state=2026))
            model.fit(x[fitting], y[fitting])
            p[good] = model.predict_proba(x[good])[:, 1]
        provenance.append(dict(heldout_subject=int(subject), training_subjects=np.unique(groups[training]).tolist(),
                               target_indices=target.tolist(), training_indices=training.tolist()))
    return p, dict(folds=provenance, prior_fallbacks=prior_fallbacks)


def raw_vector(evidence):
    return case_vector(evidence)


def retrieve_examples(query, bank, evaluation, k=5, query_subject=None):
    """True distance Top K, one case per subject, with no class quota."""
    result = CaseRetriever(bank, evaluation, k=k)(query, query_subject=query_subject)
    return result["examples"], result["selected_indices"]


def retrieve_balanced_examples(query, bank, evaluation, per_class=4, query_subject=None):
    """Preserved 4 high + 4 low policy, explicitly a similar few-shot control.

    The common V2 fingerprint isolates label quotas and subject caps. The old
    diversity-first policy allows repeat subjects after the first diversity pass.
    """
    bank = [record for record in bank if evaluation[record["path_index"]]["subject_key"] != query_subject]
    validated = CaseRetriever(bank, evaluation)
    q, scale = case_vector(query), _scale(validated.records)
    candidates = []
    for record in validated.records:
        row = case_vector(record["evidence"])
        common = np.isfinite(q) & np.isfinite(row)
        if common[:len(TEMPORAL_FIELDS)].any():
            candidates.append((float(np.mean(((row[common] - q[common]) / scale[common]) ** 2)), record))
    candidates.sort(key=lambda item: (item[0], item[1]["path_index"]))
    chosen = []
    for label in (0, 1):
        diverse, remaining, seen = [], [], set()
        for item in candidates:
            if item[1]["label"] != label:
                continue
            subject = item[1]["subject"]
            if subject in seen:
                remaining.append(item)
            else:
                diverse.append(item)
                seen.add(subject)
        chosen.extend((diverse + remaining)[:per_class])
    chosen.sort(key=lambda item: (item[0], item[1]["path_index"]))
    return [dict(evidence=copy.deepcopy(record["evidence"]), observed_class="high" if record["label"] else "low",
                 retrieval_distance=distance) for distance, record in chosen], [record["path_index"] for _, record in chosen]


def balanced_loso_evaluation(bank, evaluation):
    rows = []
    for record in bank:
        source = evaluation[record["path_index"]]
        examples, indices = retrieve_balanced_examples(record["evidence"], bank, evaluation, query_subject=source["subject_key"])
        high = sum(example["observed_class"] == "high" for example in examples)
        low = len(examples) - high
        vote = 1 if high > low else 0 if low > high else None
        rows.append(dict(path_index=record["path_index"], heldout_subject=source["subject_key"],
                         selected_indices=indices, high_count=high, low_count=low, vote_label=vote,
                         correct=vote is not None and vote == source["label"]))
    return dict(source="meta-only subject-LOSO balanced few-shot control", cases=len(rows),
                neighbor_vote_accuracy=sum(row["correct"] for row in rows) / len(rows) if rows else None,
                ties_are_unresolved=True, rows=rows,
                limitation="A forced class-balanced prompt is not a neighborhood classifier; judge explanation usefulness separately")


def prepare_v2(cloud_out, out):
    cloud_out, out = Path(cloud_out), Path(out)
    if out.exists():
        raise FileExistsError("Keep completed runs; choose a new V2 output directory")
    source_protocol = read_json(cloud_out / "protocol.json")
    if source_protocol.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("V2 retrieval requires corrected V2 cloud evidence; do not reuse historical caches")
    folds = read_json(cloud_out / "fold_evidence.json")
    evaluation = {record["path_index"]: record for record in read_json(cloud_out / "evaluation.json")}
    jobs, diagnostics = [], []
    for fold in folds:
        split = fold["split"]
        meta_subjects = set(split["meta_calibration"])
        if fold["outer_subject"] in meta_subjects:
            raise ValueError("Outer subject entered reference bank")
        bank = [dict(path_index=record["path_index"], role="meta", evidence=record["evidence"])
                for record in fold["records"] if record["role"] == "meta"]
        retriever = CaseRetriever(bank, evaluation, allowed_subjects=meta_subjects)
        contexts = []
        for record in fold["records"]:
            if record["role"] not in ("policy_validation", "outer_test"):
                continue
            index, query = record["path_index"], record["evidence"]
            subject = evaluation[index]["subject_key"]
            expected = set(split["policy_validation"] if record["role"] == "policy_validation" else split["outer_test"])
            if subject not in expected or subject in meta_subjects:
                raise ValueError("Query subject partition mismatch")
            result = retriever(query, query_subject=subject)
            balanced, balanced_indices = retrieve_balanced_examples(query, bank, evaluation, query_subject=subject)
            control_quality = dict(mode="balanced_few_shot_control", reliable=False, status="control_without_reliability_claim",
                high_count=sum(example["observed_class"] == "high" for example in balanced),
                low_count=sum(example["observed_class"] == "low" for example in balanced),
                distinct_subject_count=len({evaluation[i]["subject_key"] for i in balanced_indices}),
                distances=[example["retrieval_distance"] for example in balanced], class_counts_are_not_calibrated_probabilities=True)
            for mode, examples, indices, quality in (
                ("case_rag", result["examples"], result["selected_indices"], result["retrieval_quality"]),
                ("balanced_few_shot_control", balanced, balanced_indices, control_quality)):
                evidence = copy.deepcopy(query)
                evidence["analysis_context"] = dict(
                    reference_case_source="internal meta subjects only; query subject excluded", retrieval_quality=quality,
                    guidance="Describe support, conflicts and missing measurements. Neighborhood counts are not symptom probabilities. The LLM cannot alter the deep model probability.")
                message = message_for(evidence, examples)
                jobs.append(dict(mode=mode, outer_subject=fold["outer_subject"], path_index=index,
                                 subject_key=subject, role=record["role"], message=message, message_sha256=job_hash(message)))
                contexts.append(dict(mode=mode, path_index=index, selected_example_indices=indices, retrieval_quality=quality))
        diagnostics.append(dict(outer_subject=fold["outer_subject"], meta_subjects=sorted(meta_subjects),
                                retrieval_calibration=retriever.calibration_audit(), top_k_loso=retriever.loso_evaluation(),
                                balanced_control_loso=balanced_loso_evaluation(bank, evaluation), contexts=contexts))
    keys = {(job["mode"], job["outer_subject"], job["path_index"]) for job in jobs}
    if len(keys) != len(jobs) or not jobs:
        raise ValueError("Duplicate or empty V2 retrieval requests")
    sources = {str((cloud_out / name).resolve()): sha(cloud_out / name)
               for name in ("protocol.json", "fold_evidence.json", "evaluation.json")}
    out.mkdir(parents=True)
    write_json(out / "analysis_plan.json", dict(schema_version="uncertainty_agent_v2", frozen_before_api=True,
        llm_probability_modification=False, retrieval=dict(k=5, subject_cap=1, class_quota=None, fingerprint="raw deep-temporal + QC",
            reliability="meta-only subject-LOSO distance cutoff; no cutoff when gates fail",
            control="balanced_few_shot_control: up to 4 cases per class"),
        comparison="Nested meta subject-LOSO coverage and neighbor votes; later compare evidence descriptions",
        no_full_training_or_api_performed_by_preparation=True))
    write_json(out / "protocol.json", dict(schema_version="uncertainty_agent_v2", cloud_out=str(cloud_out.resolve()),
        created_utc=datetime.now(timezone.utc).isoformat(), total_paths=source_protocol["total_paths"], subjects=source_protocol["subjects"],
        models=dict(gpt="gpt-6.1-sol", deepseek="deepseek-flash"), llm_role="explanation_only", numerical_models_retrained=False,
        runtime_tools="CaseRetriever is lazy through ToolReplay; this preparation freezes an offline prompt comparison only",
        original_sha256=sources, expected_requests=len(jobs), source_protocol=source_protocol,
        implementation_sha256={str(path.resolve()): sha(path) for path in Path(__file__).parent.glob("*.py")},
        system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
        outer_labels_used_for_retrieval_or_threshold_selection=False, exploratory=True))
    write_json(out / "jobs.json", jobs)
    write_json(out / "preparation_audit.json", dict(status="passed", requests_per_vendor=len(jobs), query_labels_absent=True,
        retrieval_scaling_and_threshold_selection_meta_only=True, physiology_not_required_for_query_fingerprint=True,
        diagnostics=diagnostics))
    print(f"Prepared {len(jobs)} V2 explanation comparisons; no training or API calls made.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cloud-out", type=Path, default=CLOUD)
    args = parser.parse_args()
    prepare_v2(args.cloud_out, args.out)


if __name__ == "__main__":
    main()
