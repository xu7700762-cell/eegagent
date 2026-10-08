"""Paired path ACC evaluation with subject-cluster uncertainty and API accounting."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, brier_score_loss

from vrms_pilot.data import digest, read_csv, write_json
from vrms_pilot.experiment import DEFAULT_OUT as PILOT_OUT, expected_calibration_error, save_csv
from .prepare import DEFAULT_OUT


def metric(y, p):
    p = np.asarray(p, float)
    mask = np.isfinite(p)
    result = dict(total_paths=len(y), scored_paths=int(mask.sum()), coverage=float(mask.mean()),
                  correct_paths=int(np.count_nonzero((p[mask] >= .5) == y[mask])))
    if not mask.any():
        return {**result, "acc": None, "bacc": None, "auroc": None}
    result.update(acc=float(accuracy_score(y[mask], p[mask] >= .5)),
                  bacc=float(balanced_accuracy_score(y[mask], p[mask] >= .5)) if len(np.unique(y[mask])) == 2 else None,
                  auroc=float(roc_auc_score(y[mask], p[mask])) if len(np.unique(y[mask])) == 2 else None,
                  brier=float(brier_score_loss(y[mask], p[mask])),
                  ece=expected_calibration_error(y[mask], p[mask]),
                  confusion=confusion_matrix(y[mask], p[mask] >= .5, labels=[0, 1]).tolist())
    return result


def paired(y, subjects, a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return dict(common_paths=0)
    yy, aa, bb, group = y[mask], a[mask] >= .5, b[mask] >= .5, subjects[mask]
    rng = np.random.default_rng(2026)
    groups = np.unique(group)
    acc, bacc = [], []
    for _ in range(1000):
        selected = rng.choice(groups, len(groups), replace=True)
        ix = np.concatenate([np.flatnonzero(group == s) for s in selected])
        acc.append(np.mean(aa[ix] == yy[ix]) - np.mean(bb[ix] == yy[ix]))
        if len(np.unique(yy[ix])) == 2:
            bacc.append(balanced_accuracy_score(yy[ix], aa[ix]) - balanced_accuracy_score(yy[ix], bb[ix]))
    return dict(common_paths=int(mask.sum()), acc_difference=float(np.mean(aa == yy) - np.mean(bb == yy)),
                acc_subject_bootstrap_95ci=np.quantile(acc, [.025, .975]).tolist(),
                bacc_difference=float(balanced_accuracy_score(yy, aa) - balanced_accuracy_score(yy, bb)),
                bacc_subject_bootstrap_95ci=np.quantile(bacc, [.025, .975]).tolist(),
                changed_decisions=int(np.count_nonzero(aa != bb)),
                corrected=int(np.count_nonzero((aa == yy) & (bb != yy))),
                harmed=int(np.count_nonzero((aa != yy) & (bb == yy))))


def evaluate_v2(out=DEFAULT_OUT, offline=False):
    """Score frozen model probabilities and published states on separate denominators."""
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("Use a V2 directory; historical fusion outputs remain separate")
    evaluation = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    status = json.loads((out / ("offline_status.json" if offline else "independent_status.json")).read_text(encoding="utf-8"))
    if status["paths"] != len(evaluation) or status.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("Incomplete V2 run")
    by_id = {r["path_index"]: r for r in status["results"]}
    if len(by_id) != len(evaluation) or set(by_id) != {r["path_index"] for r in evaluation}:
        raise ValueError("Duplicate or missing outer results")
    packs_dir = out / ("offline_calls" if offline else "independent_calls")
    rows, labels, probabilities, published = [], [], [], []
    for expected in evaluation:
        i = expected["path_index"]
        pack = json.loads((packs_dir / f"path_{i:03d}_evidence_pack.json").read_text(encoding="utf-8"))
        p = pack["high_probability"]
        if pack.get("effective_llm_weight") != 0 or p != pack.get("p_cal"):
            raise ValueError("An explanation modified the calibrated model probability")
        if pack["state"] not in ("high", "low", "uncertain", "insufficient_data"):
            raise ValueError("Invalid state")
        accepted = pack["state"] in ("high", "low")
        if accepted and (p is None or pack["state"] != ("high" if p >= .5 else "low")):
            raise ValueError("Published state contradicts model probability")
        labels.append(expected["label"])
        probabilities.append(np.nan if p is None else p)
        published.append(accepted)
        rows.append(dict(**expected, probability=p, state=pack["state"], cloud_called=pack.get("cloud_called", False),
                         cloud_success=pack["cloud_success"], tool_calls=len(pack["tool_calls"])))
    from vrms_pilot.experiment import metrics
    scored = metrics(labels, probabilities, published)
    score = dict(schema_version="uncertainty_agent_v2", status="offline_dry_run" if offline else "completed_exploratory_v2",
                 paths=len(rows), partial_run=protocol["partial_run"], model=scored,
                 state_counts=dict(Counter(r["state"] for r in rows)),
                 cloud_calls=sum(r["cloud_called"] for r in rows), cloud_success=sum(r["cloud_success"] for r in rows),
                 tools_mean=float(np.mean([r["tool_calls"] for r in rows])),
                 llm_probability_changes=0, probability_source="calibrated deep model only",
                 comparison_note="Coverage and conditional accuracy use distinct denominators; no fresh historical comparison")
    write_json(out / ("offline_summary.json" if offline else "summary.json"), score)
    save_csv(out / ("offline_oof.csv" if offline else "path_oof.csv"), rows)
    print(json.dumps(score, indent=2), flush=True)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    evaluate_v2(args.out, args.offline)


if __name__ == "__main__":
    main()