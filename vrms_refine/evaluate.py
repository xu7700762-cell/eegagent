"""V2 retrieval and explanation audit, with generic numeric CV diagnostics."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import DEFAULT_OUT, check_v2_artifacts, read_json, sha, write_json


def subset(data, indices):
    return {k: np.asarray(v)[indices] for k, v in data.items()}


def feature_matrix(data, name):
    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    if name == "baseline25_only":
        return logit(data["baseline25"])[:, None]
    two = np.c_[logit(data["numeric"]), logit(data["enhanced"])]
    if name == "two_probabilities":
        return two
    if name == "probabilities_quality_conflict":
        return np.c_[two, np.abs(data["numeric"] - data["enhanced"]), data["qc"], data["reference"], data["spread"]]
    raise ValueError("Unknown, non-predeclared fusion features")


def fit_predict(spec, train, y_train, target, weights):
    """No target labels are accepted by this interface."""
    if spec["kind"] == "fixed":
        return target[spec["source"]].copy(), dict(spec=spec)
    if spec["kind"] != "logistic":
        raise ValueError("Unknown candidate")
    if len(np.unique(y_train)) < 2:
        return target["numeric"].copy(), dict(spec=spec, numeric_fallback=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"], solver="liblinear",
                          max_iter=2000, class_weight=None, random_state=2026))
    model.fit(feature_matrix(train, spec["features"]), y_train)
    return model.predict_proba(feature_matrix(target, spec["features"]))[:, 1], dict(spec=spec, estimator=model)


def cv_predict(spec, data, y, groups, weights):
    p = np.full(len(y), np.nan)
    for subject in np.unique(groups):
        training = np.flatnonzero(groups != subject)
        target = np.flatnonzero(groups == subject)
        if set(groups[training]) & set(groups[target]):
            raise ValueError("Fusion CV subject leakage")
        p[target], _ = fit_predict(spec, subset(data, training), y[training], subset(data, target), weights)
    if not np.isfinite(p).all():
        raise ValueError("Fusion CV did not score every validation path")
    return p


def selective_threshold(y, p, plan):
    confidence = np.maximum(p, 1-p)
    for threshold in plan["selective_confidence_thresholds"]:
        accept = confidence >= threshold
        if (accept.sum() >= plan["selective_minimum_validation_cases"] and
            accept.mean() >= plan["selective_minimum_validation_coverage"] and
            np.mean((p[accept] >= .5) == y[accept]) >= plan["selective_target_validation_accuracy"]):
            return float(threshold)
    return None


def evaluate_retrieval_v2(out, vendors=("gpt", "deepseek")):
    """Report actual cached LOSO retrieval diagnostics and explanation coverage.

    Retrieval votes are diagnostic labels, never replacement model probabilities.
    Outer-fold internal banks overlap, so per-fold results are not pooled OOF.
    """
    out = Path(out)
    protocol = check_v2_artifacts(out, require_validation=True)
    if protocol.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("This evaluator requires explanation-only V2 artifacts")
    if read_json(out / "structural_validation.json")["status"] != "passed":
        raise ValueError("Validate retrieval sources before evaluation")
    for filename, expected in protocol["original_sha256"].items():
        if sha(filename) != expected:
            raise ValueError("Frozen evidence changed")
    audit = read_json(out / "preparation_audit.json")
    retrieval = []
    for diagnostic in audit["diagnostics"]:
        top, balanced = diagnostic["top_k_loso"], diagnostic["balanced_control_loso"]
        retrieval.append(dict(outer_subject=diagnostic["outer_subject"], internal_cases=top["cases"],
            top_k_neighbor_vote_accuracy=top["neighbor_vote_accuracy"], top_k_reliable_coverage=top["coverage"],
            top_k_reliable_neighbor_vote_accuracy=top["reliable_neighbor_vote_accuracy"],
            balanced_control_neighbor_vote_accuracy=balanced["neighbor_vote_accuracy"],
            balanced_ties_are_unresolved=True))
    jobs = read_json(out / "jobs.json")
    jobs_by_key = {(job["mode"], job["outer_subject"], job["path_index"]): job for job in jobs}
    explanations = {}
    for vendor in vendors:
        status_path = out / vendor / "run_status.json"
        if not status_path.exists():
            explanations[vendor] = dict(status="not_run", expected_requests=len(jobs), completed_requests=0)
            continue
        status = read_json(status_path)
        rows = status["results"]
        keys = {(row["mode"], row["outer_subject"], row["path_index"]) for row in rows}
        expected_keys = {(job["mode"], job["outer_subject"], job["path_index"]) for job in jobs}
        if len(keys) != len(rows) or not keys <= expected_keys:
            raise ValueError("Duplicate or unknown explanation results")
        if status["status"] == "completed" and keys != expected_keys:
            raise ValueError("Claimed complete explanation run has missing requests")
        successful = 0
        for row in rows:
            log_path = Path(row["selected_log_path"])
            log = read_json(log_path)
            if sha(log_path) != row["log_sha256"] or log["success"] != row["success"]:
                raise ValueError("Explanation log changed")
            job = jobs_by_key[row["mode"], row["outer_subject"], row["path_index"]]
            actual_message = log["user_message"] if vendor == "gpt" else log["request"]["messages"][1]["content"]
            if actual_message != job["message"]:
                raise ValueError("Actual explanation request differs from the frozen comparison")
            prediction = (log["predictions"][0] if vendor == "gpt" else log["prediction"]) if log["success"] else None
            if prediction != row["explanation"]:
                raise ValueError("Stored explanation differs from its API record")
            if prediction is not None:
                if "high_probability" in prediction or "state" in prediction:
                    raise ValueError("V2 LLM response contains a prohibited prediction")
                for name in ("supporting_evidence", "conflicting_evidence", "missing_evidence"):
                    if not isinstance(prediction[name], list):
                        raise ValueError("Invalid evidence explanation")
                if not isinstance(prediction["explanation"], str):
                    raise ValueError("Missing explanation text")
                successful += 1
        explanations[vendor] = dict(status=status["status"], expected_requests=len(jobs), completed_requests=len(rows),
                                    successful_requests=successful, coverage=successful / len(jobs))
    summary = dict(schema_version="uncertainty_agent_v2", status="retrieval_diagnostics_only", folds=retrieval,
        explanations=explanations, llm_probability_modification=False,
        limitation="Internal banks overlap across outer folds; per-fold LOSO results are diagnostics, not independent outer OOF or proof of LLM benefit",
        top_k_class_counts_are_not_probabilities=True, no_new_training_or_api_calls=True)
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vendors", nargs="+", choices=("gpt", "deepseek"), default=["gpt", "deepseek"])
    args = parser.parse_args()
    protocol = read_json(args.out / "protocol.json")
    if protocol.get("schema_version") == "uncertainty_agent_v2":
        summary = evaluate_retrieval_v2(args.out, args.vendors)
        print(__import__("json").dumps(summary, ensure_ascii=False, indent=2))
        return
    raise ValueError("Use the historical Git version to evaluate V1 probability fusion; V2 never blends LLM scores")


if __name__ == "__main__":
    main()
