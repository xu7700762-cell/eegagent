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


def choose_blend(y, numeric, cloud):
    """Call only with this outer fold's independent policy-validation cases."""
    best = dict(mode="none", weight=0., bacc=float(balanced_accuracy_score(y, numeric >= .5)))
    for weight in (.25, .5, .75, 1.):
        for mode, probability in cloud.items():
            probability = np.where(np.isfinite(probability), probability, numeric)
            p = (1 - weight) * numeric + weight * probability
            score = float(balanced_accuracy_score(y, p >= .5))
            if score > best["bacc"] + 1e-12:
                best = dict(mode=mode, weight=weight, bacc=score)
    return best


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", default=["zero_shot", "few_shot", "improved_few_shot"])
    args = parser.parse_args()
    evaluation = json.loads((args.out / "evaluation.json").read_text(encoding="utf-8"))
    lookup = {r["path_index"]: r for r in evaluation}
    y = np.asarray([lookup[i]["label"] for i in range(147)])
    subjects = np.asarray([lookup[i]["subject_key"] for i in range(147)])
    folds = json.loads((args.out / "fold_evidence.json").read_text(encoding="utf-8"))
    oof = read_csv(PILOT_OUT / "results/path_oof.csv")
    inner = read_csv(PILOT_OUT / "results/inner_predictions.csv")
    original_fold = {f["outer_subject"]: {} for f in folds}
    methods = {name: np.full(147, np.nan) for name in ("deep", "original_full", "original_adaptive")}
    for r in oof:
        i = int(r["path_index"])
        methods["deep"][i] = float(r["deep_probability"])
        methods["original_full"][i] = float(r["full_probability"])
        methods["original_adaptive"][i] = float(r["adaptive_probability"])
        original_fold[int(r["subject_key"])][i] = float(r["full_probability"])
    for r in inner:
        original_fold[int(r["outer_subject"])][int(r["path_index"])] = float(r["full"] or r["deep_temporal_bio"])
    cloud_fold, all_calls, explanations = {}, [], []
    for mode in args.modes:
        methods[mode] = np.full(147, np.nan)
        cloud_fold[mode] = {}
        for fold in folds:
            s = fold["outer_subject"]
            filename = args.out / "cloud_calls" / f"{mode}_fold_{s:02d}.json"
            if not filename.exists():
                raise ValueError(f"Incomplete mode: {mode}, missing fold {s}")
            log = json.loads(filename.read_text(encoding="utf-8"))
            assert log["mode"] == mode and log["outer_subject"] == s
            example_indices = log["example_indices"]
            assert all(lookup[i]["subject_key"] in fold["split"]["meta_calibration"] for i in example_indices)
            assert all(lookup[i]["subject_key"] != s for i in example_indices)
            values = {r["path_index"]: np.nan for r in log["mapping"].values()}
            all_calls.append(log)
            if log["success"]:
                for prediction in log["predictions"]:
                    mapping = log["mapping"][prediction["id"]]
                    i = mapping["path_index"]
                    values[i] = prediction["high_probability"]
                    if mapping["role"] == "outer_test":
                        assert lookup[i]["subject_key"] == s
                        methods[mode][i] = prediction["high_probability"]
                        explanations.append(dict(mode=mode, path_index=i, probability=prediction["high_probability"],
                                                 uncertain=prediction["uncertain"], reason=prediction["reason"]))
            cloud_fold[mode][s] = values
    improvements = read_csv(args.out / "improvements/predictions.csv")
    improved_fold = {}
    improvement_names = ("regional_spectrum", "spatial_spectrum", "tangent_covariance", "path_mil", "candidate_mean", "validated_numeric")
    for name in improvement_names:
        methods[name] = np.full(147, np.nan)
    for r in improvements:
        s, i = int(r["outer_subject"]), int(r["path_index"])
        improved_fold.setdefault(s, {})[i] = float(r["validated_numeric"])
        if r["role"] == "outer_test":
            assert lookup[i]["subject_key"] == s
            for name in improvement_names:
                methods[name][i] = float(r[name])
    selectors = []
    for numeric_name, source, available_modes in (
            ("original", original_fold, [m for m in ("zero_shot", "few_shot") if m in args.modes]),
            ("improved", improved_fold, [m for m in ("improved_few_shot",) if m in args.modes])):
        if not available_modes:
            continue
        fixed_mode = available_modes[-1]
        methods[numeric_name + "_llm25"] = np.full(147, np.nan)
        methods[numeric_name + "_validated_llm"] = np.full(147, np.nan)
        for fold in folds:
            s = fold["outer_subject"]
            val = [r["path_index"] for r in fold["records"] if r["role"] == "policy_validation"]
            test = [r["path_index"] for r in fold["records"] if r["role"] == "outer_test"]
            assert all(lookup[i]["subject_key"] in fold["split"]["policy_validation"] for i in val)
            selected = choose_blend(y[val], np.asarray([source[s][i] for i in val]),
                                    {m: np.asarray([cloud_fold[m][s][i] for i in val]) for m in available_modes})
            selectors.append(dict(outer_subject=s, numeric=numeric_name, **selected))
            for i in test:
                numeric = source[s][i]
                p = cloud_fold[fixed_mode][s][i]
                methods[numeric_name + "_llm25"][i] = .75 * numeric + .25 * (p if np.isfinite(p) else numeric)
                p = numeric if selected["mode"] == "none" else cloud_fold[selected["mode"]][s][i]
                methods[numeric_name + "_validated_llm"][i] = ((1 - selected["weight"]) * numeric + selected["weight"] *
                                                              (p if np.isfinite(p) else numeric))
    independent = None
    if (args.out / "independent_status.json").exists():
        independent = json.loads((args.out / "independent_status.json").read_text(encoding="utf-8"))
        assert independent["status"] == "completed" and independent["paths"] == 147
        methods["independent_cloud"] = np.full(147, np.nan)
        methods["independent_llm25"] = np.full(147, np.nan)
        for record in independent["results"]:
            i = record["path_index"]
            methods["independent_cloud"][i] = np.nan if record["cloud_probability"] is None else record["cloud_probability"]
            methods["independent_llm25"][i] = record["probability"]
    summary = dict(status="completed_exploratory_cloud_loso", paths=147, subjects=24,
                   methods={k: metric(y, p) for k, p in methods.items()}, comparisons={})
    comparisons = [(m + "_minus_full", m, "original_full") for m in
                   ("zero_shot", "few_shot", "original_llm25", "original_validated_llm", "validated_numeric") if m in methods]
    comparisons.extend((m + "_minus_improved", m, "validated_numeric") for m in
                       ("improved_few_shot", "improved_llm25", "improved_validated_llm") if m in methods)
    comparisons.extend((m + "_minus_improved", m, "validated_numeric") for m in
                       ("independent_cloud", "independent_llm25") if m in methods)
    if "independent_llm25" in methods:
        comparisons.append(("independent_llm25_minus_full", "independent_llm25", "original_full"))
    for name, a, b in comparisons:
        summary["comparisons"][name] = paired(y, subjects, methods[a], methods[b])
    successful = [log for log in all_calls if log["success"]]
    attempts = [a for log in all_calls for a in log["attempts"]]
    usages = [a["response"].get("usage", {}) for a in attempts if a.get("response")]
    summary["cloud"] = dict(model=all_calls[0]["model"], provider=all_calls[0]["provider"],
                            gateway=all_calls[0]["base_url"], modes=args.modes, logical_batches=len(all_calls),
                            successful_batches=len(successful), failed_batches=len(all_calls) - len(successful),
                            attempted_requests=len(attempts), failed_attempts=sum(a.get("status") != "success" for a in attempts),
                            input_tokens=sum(u.get("input_tokens", 0) for u in usages),
                            output_tokens=sum(u.get("output_tokens", 0) for u in usages),
                            total_tokens=sum(u.get("total_tokens", 0) for u in usages),
                            median_batch_seconds=float(np.median([r["seconds"] for r in successful])),
                            p95_batch_seconds=float(np.quantile([r["seconds"] for r in successful], .95)),
                            note="Fold batches mix shuffled inner-validation and outer cases; excludes synthetic probes; no serial per-person latency claim")
    summary["selectors"] = selectors
    if independent:
        independent_logs = [json.loads((args.out / "independent_calls" / f"path_{i:03d}_cloud_call.json").read_text(encoding="utf-8")) for i in range(147)]
        attempts_single = [a for log in independent_logs for a in log["attempts"]]
        usage_single = [a["response"].get("usage", {}) for a in attempts_single if a.get("response")]
        summary["single_path_cloud"] = dict(paths=147, successful=independent["cloud_success"],
                 attempted_requests=len(attempts_single), failed_attempts=sum(a.get("status") != "success" for a in attempts_single),
                 input_tokens=sum(u.get("input_tokens", 0) for u in usage_single),
                 output_tokens=sum(u.get("output_tokens", 0) for u in usage_single),
                 total_tokens=sum(u.get("total_tokens", 0) for u in usage_single),
                 median_seconds=float(np.median([r["seconds"] for r in independent["results"]])),
                 p95_seconds=float(np.quantile([r["seconds"] for r in independent["results"]], .95)),
                 other_test_queries_in_context=False, previous_response_id_used=False,
                 primary_final_method="independent_llm25", fixed_llm_weight=.25)
    summary["selection_counts"] = {name: dict(Counter(r["selection"] for r in json.loads(
        (args.out / "improvements/fold_diagnostics.json").read_text(encoding="utf-8")))) for name in ["numeric"]}
    write_json(args.out / "summary.json", summary)
    save_csv(args.out / "path_oof.csv", [dict(path_index=i, subject_key=int(subjects[i]), label=int(y[i]),
              **{k: None if not np.isfinite(p[i]) else float(p[i]) for k, p in methods.items()}) for i in range(147)])
    save_csv(args.out / "llm_explanations.csv", explanations)
    save_csv(args.out / "subject_metrics.csv", [dict(subject_key=int(s), paths=int(sum(subjects == s)),
        high_paths=int(sum(y[subjects == s])), **{k + "_acc": metric(y[subjects == s], p[subjects == s])["acc"]
                                                 for k, p in methods.items()}) for s in np.unique(subjects)])
    print(json.dumps({"methods": {k: {field: v.get(field) for field in ("acc", "bacc", "auroc", "correct_paths", "scored_paths")}
                      for k, v in summary["methods"].items()}, "cloud": summary["cloud"], "comparisons": summary["comparisons"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
