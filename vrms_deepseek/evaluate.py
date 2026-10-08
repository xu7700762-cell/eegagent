"""Audit and compare DeepSeek with frozen numerical/GPT predictions, no refit."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from vrms_cloud.cloud import choose_examples, check_blind, write_json
from vrms_cloud.evaluate import metric, paired
from vrms_pilot.experiment import save_csv
from .experiment import DEFAULT_OUT, FROZEN, sha
from .supervisor import messages_for, normalize_response


def local_path(value):
    value = value.replace("\\", "/")
    if sys.platform != "win32" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    return Path(value)


def main():
    out = DEFAULT_OUT
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["source_artifact_sha256"].items():
        assert sha(local_path(name)) == expected, f"Frozen input changed: {name}"
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed" and status["paths"] == 147
    evaluation = {r["path_index"]: r for r in json.loads((FROZEN / "evaluation.json").read_text(encoding="utf-8"))}
    folds = json.loads((FROZEN / "fold_evidence.json").read_text(encoding="utf-8"))
    with (FROZEN / "path_oof.csv").open(encoding="utf-8-sig") as f:
        import csv
        previous = {int(r["path_index"]): r for r in csv.DictReader(f)}
    rows = {r["path_index"]: r for r in status["results"]}
    assert len(rows) == 147
    logs = []
    for fold in folds:
        examples = [dict(evidence=r["evidence"], observed_class="high" if evaluation[r["path_index"]]["label"] else "low")
                    for r in choose_examples(fold["records"], evaluation, 2026 + fold["outer_subject"])]
        for record in fold["records"]:
            if record["role"] != "outer_test":
                continue
            i = record["path_index"]
            log = json.loads((out / "independent_calls" / f"path_{i:03d}_cloud_call.json").read_text(encoding="utf-8"))
            assert log["model"] == protocol["model"]
            assert log["base_url"] == protocol["base_url"]
            request = log["request"]
            assert request["messages"] == messages_for(record["evidence"], examples)
            assert log["request_sha256"] == hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            assert "previous_response_id" not in request
            message = json.loads(request["messages"][1]["content"])
            assert len(message["queries"]) == 1
            check_blind(message["queries"])
            numeric = float(previous[i]["validated_numeric"])
            pack = rows[i]
            if log["success"]:
                raw = log["attempts"][-1]["response"]
                reply = json.loads(raw["choices"][0]["message"]["content"])
                prediction = normalize_response(reply, raw)
                assert prediction == log["prediction"]
                assert "deepseek" in raw["model"].lower()
                assert pack["cloud_probability"] == prediction["high_probability"]
                assert pack["high_probability"] == .75 * numeric + .25 * prediction["high_probability"]
                assert not pack["fallback_used"]
            else:
                assert pack["high_probability"] == numeric and pack["fallback_used"]
            assert pack["state"] == ("high" if pack["high_probability"] >= .5 else "low")
            assert pack["numerical_probability"] == numeric
            logs.append(log)
    y = np.asarray([evaluation[i]["label"] for i in range(147)])
    groups = np.asarray([evaluation[i]["subject_key"] for i in range(147)])
    methods = {name: np.asarray([float(previous[i][key]) for i in range(147)]) for name, key in
               (("original_numeric", "original_full"), ("improved_numeric", "validated_numeric"),
                ("gpt_cloud", "independent_cloud"), ("gpt_fusion25", "independent_llm25"))}
    methods["deepseek_cloud"] = np.asarray([np.nan if rows[i]["cloud_probability"] is None else rows[i]["cloud_probability"] for i in range(147)])
    methods["deepseek_fusion25"] = np.asarray([rows[i]["high_probability"] for i in range(147)])
    summary = dict(status="completed_exploratory_deepseek_147_paths", model=protocol["model"],
                   response_models=dict(Counter(log.get("response_model") for log in logs if log["success"])),
                   paths=147, subjects=24, llm_weight=.25, numerical_training_reused=True,
                   methods={name: metric(y, p) for name, p in methods.items()}, comparisons={})
    for name, a, b in (("deepseek_cloud_minus_numeric", "deepseek_cloud", "improved_numeric"),
                       ("deepseek_fusion_minus_numeric", "deepseek_fusion25", "improved_numeric"),
                       ("deepseek_fusion_minus_gpt", "deepseek_fusion25", "gpt_fusion25")):
        summary["comparisons"][name] = paired(y, groups, methods[a], methods[b])
    attempts = [a for log in logs for a in log["attempts"]]
    usages = [a["response"].get("usage", {}) for a in attempts if a.get("response")]
    summary["api"] = dict(successful_paths=sum(log["success"] for log in logs),
                          logical_requests=147, attempted_requests=len(attempts),
                          failed_attempts=sum(a["status"] != "success" for a in attempts),
                          prompt_tokens=sum(u.get("prompt_tokens", 0) for u in usages),
                          completion_tokens=sum(u.get("completion_tokens", 0) for u in usages),
                          total_tokens=sum(u.get("total_tokens", 0) for u in usages),
                          median_seconds=float(np.median([log["seconds"] for log in logs])),
                          p95_seconds=float(np.quantile([log["seconds"] for log in logs], .95)),
                          monetary_cost_verified=False, config_source=protocol["config_source"],
                          standalone_api_client=True, codex_default_used=False)
    write_json(out / "summary.json", summary)
    save_csv(out / "path_oof.csv", [dict(path_index=i, subject_key=int(groups[i]), label=int(y[i]),
                **{name: None if not np.isfinite(p[i]) else float(p[i]) for name, p in methods.items()}) for i in range(147)])
    validation = dict(status="passed", paths=147, source_input_hashes_unchanged=True,
                      standalone_api_client_source_hashes_checked=True,
                      codex_default_not_used=True, query_labels_and_identity_absent=True,
                      other_test_paths_and_history_absent=True, fold_internal_examples_checked=True,
                      probabilities_and_fusion_recomputed=True, weights_and_models_not_retuned=True,
                      request_hashes_checked=len(logs), original_gpt_results_preserved=True,
                      implementation_sha256={p.name: sha(p) for p in Path("vrms_deepseek").glob("*.py")})
    write_json(out / "validation.json", validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
