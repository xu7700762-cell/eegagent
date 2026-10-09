"""Recount and replay the fourth Agent experiment without calling an API."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .agent_selector_v3 import DEFAULT_OUT as SELECTOR
from .agent_tools_v1 import DEFAULT_OUT as FIRST
from .cloud import check_blind, write_json
from .second_judgment import DEFAULT_MODELS, sha256
from .tool_agent_v1 import PILOT, validate_final
from .tool_agent_v2 import EXTENDED_TOOLS as SECOND
from .tool_agent_v3 import CorrectiveRuntime
from .tool_agent_v4 import DEFAULT_OUT, SYSTEM


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_same(left, right):
    """Compare replayed evidence, allowing only floating-point roundoff."""
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        assert isinstance(right, (int, float)) and not isinstance(right, bool)
        assert math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10)
    else:
        assert left == right


def audit(out=DEFAULT_OUT):
    out = Path(out)
    protocol, summary = read(out / "protocol.json"), read(out / "summary.json")
    assert protocol["system"] == SYSTEM
    assert summary["goal"]["metrics_pass"]
    assert read(out / "run.json")["success"] == 146
    hashes = {}

    def frozen(path, expected):
        path = Path(path)
        actual = sha256(path)
        assert actual == expected, f"Changed artifact: {path}"
        hashes[str(path)] = actual

    sources = Path(__file__).parent
    frozen(sources / "tool_agent_v3.py", protocol["code_sha256"])
    frozen(sources / "tool_agent_v1.py", protocol["client_code_sha256"])
    for directory, key, filename in [
        (FIRST, "tool_training_sha256", "agent_tools_v1.py"),
        (SECOND, "extended_training_sha256", "agent_tools_v2.py"),
        (SELECTOR, "selector_training_sha256", "agent_selector_v3.py"),
    ]:
        frozen(directory / "training_summary.json", protocol[key])
        frozen(sources / filename, read(directory / "protocol.json")["code_sha256"])
        for name, expected in read(directory / "training_summary.json")["checkpoint_sha256"].items():
            frozen(directory / "checkpoints" / name, expected)
    frozen(SELECTOR / "nested_audit/summary.json", protocol["selector_nested_audit_sha256"])
    nested_summary = read(SELECTOR / "nested_audit/summary.json")
    frozen(sources / "agent_selector_audit_v3.py", nested_summary["code_sha256"])

    first_protocol = read(FIRST / "protocol.json")
    frozen(DEFAULT_MODELS / "model_results/path_oof.csv", first_protocol["source_hashes"]["baseline"])
    frozen(FIRST / "features/common.npz", first_protocol["source_hashes"]["common_features"])
    frozen(PILOT / "split_manifest.json", first_protocol["source_hashes"]["original_split"])
    preflight = read(DEFAULT_MODELS / "preflight.json")
    for name, expected in preflight["cache_sha256"].items():
        frozen(PILOT / "cache" / name, expected)
    for filename, expected in preflight["implementation_sha256"].items():
        frozen(sources.parent / "vrms_pilot" / filename, expected)

    def local_path(value):
        path = Path(value)
        if path.exists() or os.name != "nt":
            return path
        parts = value.split("/")
        if value.startswith("/mnt/") and len(parts) > 3 and len(parts[2]) == 1 and parts[2].isalpha():
            return Path(parts[2].upper() + ":/" + "/".join(parts[3:]))
        return path

    frozen(local_path(preflight["encoder_source"]), preflight["encoder_source_sha256"])
    frozen(local_path(preflight["pretrained_checkpoint"]), preflight["pretrained_checkpoint_sha256"])
    for record in first_protocol["feature_folds"]:
        subject = record["outer_subject"]
        frozen(FIRST / "features" / f"fold_{subject:02d}.npz", record["feature_sha256"])
        frozen(DEFAULT_MODELS / "model_results/checkpoints" / f"outer_subject_{subject:02d}_femba_frozen_mil.pt", record["original_head_sha256"])
        assert record["baseline_max_abs_error"] == 0

    paths, splits = read(FIRST / "local_paths.json"), read(FIRST / "split_manifest.json")
    assert splits == read(PILOT / "split_manifest.json") == read(SECOND / "split_manifest.json")
    assert paths == read(SECOND / "local_paths.json")
    groups = np.asarray([p["subject_key"] for p in paths])
    assert len(paths) == 146 and len(set(groups)) == len(splits) == 24
    for split in splits:
        subject = split["outer_subject"]
        roles = [set(split[key]) for key in ("base_train", "meta_calibration", "policy_validation")]
        assert [len(role) for role in roles] == [14, 5, 4]
        assert subject not in set.union(*roles)
        assert len(set.union(*roles)) == 23
        for directory in (FIRST, SECOND, SELECTOR):
            assert joblib.load(directory / "checkpoints" / f"fold_{subject:02d}.joblib")["split"] == split
        for directory in (FIRST, SECOND):
            with np.load(directory / "internal_validation" / f"fold_{subject:02d}.npz") as values:
                assert set(groups[values["meta_indices"]]) == roles[1]
                assert set(groups[values["policy_indices"]]) == roles[2]
        nested = read(SELECTOR / "nested_audit" / f"fold_{subject:02d}.json")
        assert not nested["outer_labels_used"]
        for internal in nested["splits"]:
            heldout = internal["heldout_subject"]
            assert heldout in roles[2]
            assert set(internal["selection_subjects"]) == roles[2] - {heldout}

    with (out / "path_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    baseline, correct, corrected, damaged, overrides = 0, 0, 0, 0, 0
    replay_records = []
    with threadpool_limits(limits=2):
        for i, path in enumerate(paths):
            saved = read(out / "calls" / f"{i:03d}.json")
            assert saved["success"] and saved["query_index"] == i
            assert saved["model"] == protocol["model"]
            assert saved["base_url"] == protocol["base_url"]
            runtime = CorrectiveRuntime(FIRST, SECOND, SELECTOR, path["subject_key"], i)
            assert_same(runtime.initial_evidence(), saved["initial"])
            requests, outputs, final = {}, set(), None
            for step in saved["rounds"]:
                body = step["body"]
                assert body["store"] is False and "previous_response_id" not in body
                assert body["model"] == protocol["model"]
                assert body["input"][0] == {"role": "system", "content": SYSTEM}
                assert json.loads(body["input"][1]["content"]) == saved["initial"]
                assert body["text"]["format"]["schema"] == protocol["schema"]
                for item in body["input"][2:]:
                    if item.get("type") != "function_call_output":
                        continue
                    name = requests[item["call_id"]]
                    actual = runtime.execute(name)
                    returned = json.loads(item["output"])
                    check_blind(returned)
                    assert_same(actual, returned)
                    outputs.add(item["call_id"])
                responses = [a["response"] for a in step["attempts"] if a["status"] == "success"]
                assert len(responses) == 1
                response = responses[0]
                assert response["model"] == protocol["model"] and response["status"] == "completed"
                functions = [item for item in response["output"] if item["type"] == "function_call"]
                for function in functions:
                    assert json.loads(function["arguments"]) == {"query": "qsingle"}
                    requests[function["call_id"]] = function["name"]
                if not functions:
                    text = "".join(part["text"] for item in response["output"] for part in item.get("content", []) if part["type"] == "output_text")
                    final = json.loads(text)
            assert requests and outputs == set(requests)
            assert runtime.calls == saved["actual_tool_calls"]
            assert set(saved["tool_timings_ms"]) == set(runtime.calls)
            assert all(value >= 0 for value in saved["tool_timings_ms"].values())
            assert validate_final(final, runtime.calls) == saved["final"]
            y = path["label"]
            b = int(saved["initial"]["baseline"]["raw_high_probability"] >= .5)
            a = int(final["final_class"] == "High")
            row = rows[i]
            assert [int(row[key]) for key in ("path_index", "subject_key", "true_class", "baseline_class", "agent_class")] == [i, path["subject_key"], y, b, a]
            baseline += b == y
            correct += a == y
            corrected += b != y and a == y
            damaged += b == y and a != y
            overrides += final["final_class"] != runtime.memo["corrective_evidence"]["raw_combined_class"]
            replay_records.append(dict(query_index=i,outer_subject=path["subject_key"],initial_equal=True,tool_outputs_equal=True,raw_reply_equal=True))
            if (i + 1) % 24 == 0:
                print(json.dumps(dict(replayed=i + 1,total=146)), flush=True)
    assert baseline == 89 and correct == summary["agent"]["correct"]
    assert corrected == summary["corrected"] and damaged == summary["damaged"]
    assert correct / 146 >= .70 and corrected >= 2 * damaged
    poison_file = Path("outputs/vrms_agent_loso/v6_gpt_r1_20261008/allfold_query_label_poison_audit.json")
    poison = read(poison_file)
    assert poison["status"] == "passed" and poison["folds"] == 24
    assert all(r["initial_equal"] and r["all_tool_evidence_equal"] for r in poison["records"])

    # Snapshots preserve experiment sources before any subsequent development.
    snapshot = out / "sources"
    snapshot.mkdir(exist_ok=True)
    for name in ("tool_agent_v1.py", "tool_agent_v2.py", "tool_agent_v3.py", "tool_agent_v4.py", "agent_tools_v1.py", "agent_tools_v2.py", "agent_policy_v1.py", "agent_selector_v3.py", "agent_selector_audit_v3.py", "evaluate_tool_agent.py", "audit_tool_agent_v4.py"):
        destination = snapshot / name
        if destination.exists():
            assert sha256(destination) == sha256(sources / name)
        else:
            shutil.copy2(sources / name, destination)
        hashes[str(destination)] = sha256(destination)
    result = dict(status="passed",paths=146,folds=24,baseline_correct=int(baseline),agent_correct=int(correct),corrected=int(corrected),damaged=int(damaged),numerical_goal_pass=True,
        api_called_during_audit=False,all_saved_tool_outputs_replayed=True,actual_final_replies_reparsed=True,outer_subjects_excluded=True,
        source_and_checkpoint_hashes_verified=True,poison_audit_reference=str(poison_file),poison_audit_sha256=sha256(poison_file),
        gpt_overrides_of_raw_combination=int(overrides),sha256=hashes,replays=replay_records,
        scope="Exploratory repeated LOSO development; metric acceptance does not establish independent generalization or a GPT-specific gain.")
    write_json(out / "completion_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("sha256", "replays")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    audit(parser.parse_args().out)
