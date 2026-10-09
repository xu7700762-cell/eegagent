"""Audit one complete seed repeat and compare paired whole-path decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from .audit_tool_agent_v4 import assert_same
from .cloud import check_blind, write_json
from .evaluate_second_judgment import classification_metrics, paired_subject_ci
from .second_judgment import sha256
from .seed_retest import DEFAULT_OUT, OLD_GPT, first_tools, mil, read, verify_protocol
from .tool_agent_v1 import validate_final
from .tool_agent_v3 import CorrectiveRuntime, SCHEMA, TOOLS
from .tool_agent_v4 import SYSTEM


def corrections(y, baseline, prediction):
    y, baseline, prediction = map(np.asarray, (y, baseline, prediction))
    corrected = int(((baseline != y) & (prediction == y)).sum())
    damaged = int(((baseline == y) & (prediction != y)).sum())
    return dict(corrected=corrected,damaged=damaged,net=corrected-damaged,
                ratio=None if not damaged else corrected/damaged)


def successful_log(out, i):
    original = read(out / "calls" / f"{i:03d}.json")
    if original["success"]:
        return original
    recovery = out / "recovery" / f"{i:03d}.json"
    if recovery.exists():
        result = read(recovery)
        assert result["initial"] == original["initial"]
        if result["success"]:
            return result
    raise ValueError(f"Query {i} needs API/format recovery; keep all main logs")


def evaluate(out=DEFAULT_OUT):
    out = Path(out)
    payload = verify_protocol(out)
    first, second, selector, calls = [out / name for name in ("tools1","tools2","selector","gpt")]
    gpt_protocol = read(calls / "protocol.json")
    assert gpt_protocol["system"] == SYSTEM and gpt_protocol["schema"] == SCHEMA and gpt_protocol["tools"] == TOOLS
    assert gpt_protocol["model"] == payload["model"] and gpt_protocol["base_url"] == payload["base_url"]
    hashes = {}

    def frozen(file, expected):
        actual = sha256(file)
        assert actual == expected, f"Changed seed artifact: {file}"
        hashes[str(file)] = actual

    original_models = Path("outputs/vrms_model_ablation/v3_20261008")
    original_preflight = read(original_models / "preflight.json")
    for filename, expected in original_preflight["implementation_sha256"].items():
        frozen(Path(mil.__file__).parent / filename,expected)
    pilot = Path("outputs/vrms_pilot/v2_gpt_flow_20261008")
    for filename, expected in original_preflight["cache_sha256"].items():
        frozen(pilot / "cache" / filename,expected)
    for record in read(first / "protocol.json")["feature_folds"]:
        frozen(first / "features" / f"fold_{record['outer_subject']:02d}.npz",record["feature_sha256"])
    frozen(first / "features/common.npz",read(first / "protocol.json")["source_hashes"]["common_features"])

    for directory, key in ((first,"tool_training_sha256"),(second,"extended_training_sha256"),(selector,"selector_training_sha256")):
        frozen(directory / "training_summary.json", gpt_protocol[key])
        for filename, expected in read(directory / "training_summary.json")["checkpoint_sha256"].items():
            frozen(directory / "checkpoints" / filename, expected)
    frozen(selector / "nested_audit/summary.json", gpt_protocol["selector_nested_audit_sha256"])
    frozen(out / "models/path_oof.csv", read(first / "protocol.json")["source_hashes"]["baseline"])
    paths, splits = read(first / "local_paths.json"), read(first / "split_manifest.json")
    assert paths == read(first_tools.DEFAULT_OUT / "local_paths.json") == read(second / "local_paths.json")
    assert splits == read(first_tools.DEFAULT_OUT / "split_manifest.json") == read(second / "split_manifest.json")
    assert splits == read(pilot / "split_manifest.json")
    frozen(pilot / "split_manifest.json",payload["original_split_sha256"])
    model_audits = read(out / "models/fold_audit.json")
    assert len(model_audits) == len(splits) == 24
    for split, audit in zip(splits,model_audits):
        subject = split["outer_subject"]
        assert audit["outer_subject"] == subject
        assert audit["training"]["train_subjects"] == sorted(split["base_train"])
        assert audit["training"]["training_indices"] == first_tools.indices(paths,split["base_train"]).tolist()
        assert audit["reload_max_abs_error"] == 0 and audit["training"]["epochs"] == 12
        if payload["scope"] == "full":
            assert audit["training"]["seed"] == payload["seed"]
        frozen(out / "models/checkpoints" / f"fold_{subject:02d}.pt", audit["checkpoint_sha256"])
        assert subject not in split["base_train"] + split["meta_calibration"] + split["policy_validation"]
        for directory in (first,second,selector):
            assert joblib.load(directory / "checkpoints" / f"fold_{subject:02d}.joblib")["split"] == split
        for inner in read(selector / "nested_audit" / f"fold_{subject:02d}.json")["splits"]:
            assert set(inner["selection_subjects"]) == set(split["policy_validation"]) - {inner["heldout_subject"]}
    with (OLD_GPT / "path_comparison.csv").open(encoding="utf-8-sig",newline="") as f:
        reference = list(csv.DictReader(f))
    with (out / "models/path_oof.csv").open(encoding="utf-8-sig",newline="") as f:
        source_rows = list(csv.DictReader(f))
    assert len(reference) == len(source_rows) == len(paths) == 146
    rows, numeric_rows = [], []
    tool_counts, usage, response_models = Counter(), Counter(), Counter()
    rounds = attempts = failures = 0
    with threadpool_limits(limits=2):
        for i, path in enumerate(paths):
            saved = successful_log(calls,i)
            assert saved["query_index"] == i and saved["model"] == payload["model"]
            runtime = CorrectiveRuntime(first,second,selector,path["subject_key"],i)
            assert_same(runtime.initial_evidence(),saved["initial"])
            requested, consumed, final = {}, set(), None
            for step in saved["rounds"]:
                body = step["body"]
                assert body["store"] is False and "previous_response_id" not in body
                assert body["model"] == payload["model"]
                assert body["input"][0] == dict(role="system",content=SYSTEM)
                assert json.loads(body["input"][1]["content"]) == saved["initial"]
                assert body["text"]["format"]["schema"] == SCHEMA
                for item in body["input"][2:]:
                    if item.get("type") == "function_call_output":
                        returned = json.loads(item["output"])
                        check_blind(returned)
                        assert_same(runtime.execute(requested[item["call_id"]]),returned)
                        consumed.add(item["call_id"])
                responses = [a["response"] for a in step["attempts"] if a["status"] == "success"]
                assert len(responses) == 1
                response = responses[0]
                assert response["status"] == "completed" and response["model"] == payload["model"]
                functions = [r for r in response["output"] if r["type"] == "function_call"]
                for function in functions:
                    assert json.loads(function["arguments"]) == dict(query="qsingle")
                    requested[function["call_id"]] = function["name"]
                if not functions:
                    text = "".join(p["text"] for item in response["output"] for p in item.get("content",[]) if p["type"] == "output_text")
                    final = json.loads(text)
            assert requested and consumed == set(requested)
            assert runtime.calls == saved["actual_tool_calls"]
            assert validate_final(final,runtime.calls) == saved["final"]
            # These comparisons are local evaluation; query truth never enters GPT.
            ref, source = reference[i], source_rows[i]
            assert int(ref["path_index"]) == int(source["path_index"]) == i
            assert int(ref["subject_key"]) == int(source["subject_key"]) == path["subject_key"]
            assert int(ref["true_class"]) == int(source["label"]) == path["label"]
            raw = saved["initial"]["baseline"]["raw_high_probability"]
            assert raw == float(source["femba_frozen_mil_probability"])
            adviser = runtime.memo["corrective_evidence"]
            individual = adviser["actual_individual_evidence"]
            tool_counts.update(runtime.calls)
            numeric = dict(path_index=i,subject_key=path["subject_key"],baseline=raw,
                selector_raw=adviser["uncalibrated_combined_score"],
                selector_guarded=int(adviser["selected_margin_suggestion"] == "High"),
                raw_tools={e["tool"]:int(e["raw_learned_class"] == "High") if "raw_learned_class" in e else e["descriptive_vote"] for e in individual})
            numeric_rows.append(numeric)
            rows.append(dict(path_index=i,subject_key=path["subject_key"],true_class=path["label"],
                baseline_probability=raw,baseline_class=int(raw >= .5),agent_class=int(final["final_class"] == "High"),
                raw_combined_probability=numeric["selector_raw"],raw_combined_class=int(numeric["selector_raw"] >= .5),
                original_baseline_class=int(ref["baseline_class"]),original_agent_class=int(ref["agent_class"]),
                confidence=final["confidence"],actual_tools=";".join(runtime.calls),explanation=final["explanation"],
                result_source="main" if read(calls / "calls" / f"{i:03d}.json")["success"] else "recovery"))
            if (i+1)%24 == 0:
                print(json.dumps(dict(audited=i+1,total=146)),flush=True)
    # Count all real attempts, including failed main calls and separate recovery.
    for folder in ("calls","recovery"):
        for file in (calls / folder).glob("*.json"):
            for step in read(file)["rounds"]:
                rounds += 1
                for attempt in step["attempts"]:
                    attempts += 1
                    if attempt["status"] != "success":
                        failures += 1
                        continue
                    response = attempt["response"]
                    response_models[response.get("model","missing")] += 1
                    usage.update({key:response.get("usage",{}).get(key,0) for key in ("input_tokens","output_tokens","total_tokens")})
    y = np.asarray([r["true_class"] for r in rows])
    baseline = np.asarray([r["baseline_class"] for r in rows])
    agent = np.asarray([r["agent_class"] for r in rows])
    combined = np.asarray([r["raw_combined_class"] for r in rows])
    old_baseline = np.asarray([r["original_baseline_class"] for r in rows])
    old_agent = np.asarray([r["original_agent_class"] for r in rows])
    groups = np.asarray([r["subject_key"] for r in rows])
    assert sum(old_baseline == y) == 89 and sum(old_agent == y) == 105
    changes = corrections(y,baseline,agent)
    controls = dict(selector_raw=dict(metrics=classification_metrics(y,combined),**corrections(y,baseline,combined)))
    for family in numeric_rows[0]["raw_tools"]:
        predicted = [r["raw_tools"][family] >= .5 for r in numeric_rows]
        controls[family] = dict(metrics=classification_metrics(y,predicted),**corrections(y,baseline,predicted))
    metric = classification_metrics(y,agent)
    result = dict(status="completed_and_audited",seed=payload["seed"],scope=payload["scope"],paths=146,folds=24,
        baseline=classification_metrics(y,baseline),agent=metric,**changes,numeric_controls=controls,
        original_reference=dict(seed=2026,baseline=classification_metrics(y,old_baseline),agent=classification_metrics(y,old_agent)),
        vs_original_fixed_baseline=corrections(y,old_baseline,agent),vs_same_seed_numeric_fusion=corrections(y,combined,agent),
        paired_subject_bootstrap=paired_subject_ci(y,baseline,agent,groups),
        goal_metric_pass=metric["accuracy"] >= .70 and changes["corrected"] > 0 and changes["corrected"] >= 2*changes["damaged"],
        successful_gpt_paths=146,fallback_paths=0,uncertain_count=0,tool_counts=dict(tool_counts),
        api=dict(rounds=rounds,attempts=attempts,failed_attempts=failures,models=dict(response_models),usage=dict(usage)),
        baseline_score_diagnostics=dict(auroc=float(roc_auc_score(y,[r["baseline_probability"] for r in rows])),brier=float(brier_score_loss(y,[r["baseline_probability"] for r in rows]))),
        audit=dict(status="passed",all146_tool_outputs_replayed=True,raw_final_replies_reparsed=True,model_reload_error_zero=True,
                   frozen_source_hashes_verified=True,outer_subjects_excluded=True,old_results_preserved=True,
                   root_protocol_sha256=sha256(out / "protocol.json"),scorer_sha256=sha256(__file__),artifact_sha256=hashes),
        caveats=["One additional predeclared training seed; not an estimate from many seeds", "GPT sampling not controlled by training seed",
                 "Repeated development on the same dataset", "More internal labelled subjects used by auxiliary tools than MIL baseline", "Pretraining overlap not excluded"])
    with (out / "path_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out / "numeric_predictions.json", numeric_rows)
    write_json(out / "summary.json", result)
    print(json.dumps({key:result[key] for key in ("status","seed","baseline","agent","corrected","damaged","ratio","goal_metric_pass","vs_same_seed_numeric_fusion")},ensure_ascii=False,indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    evaluate(parser.parse_args().out)
