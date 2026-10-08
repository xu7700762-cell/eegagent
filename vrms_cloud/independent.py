"""Independent V2 execution: lazy EEG tools followed by optional cloud explanation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from vrms_pilot.data import write_json, digest
from vrms_pilot.experiment import DEFAULT_OUT as PILOT_OUT
from .prepare import DEFAULT_OUT, ReplayData
from .supervisor import assess_evidence, summarize_assessment


def load_prepared(out, pilot_out=None):
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("Expected independently prepared V2 evidence")
    data = ReplayData(pilot_out or protocol["pilot_out"])
    for filename, expected in protocol["code_sha256"].items():
        if digest(Path(__file__).parent / filename) != expected:
            raise ValueError(f"Frozen cloud source changed: {filename}")
    for filename, expected in protocol["implementation_sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"Frozen external implementation changed: {filename}")
    if digest(data.pilot_out / "split_manifest.json") != protocol["split_manifest_sha256"]:
        raise ValueError("Frozen split changed")
    for name, expected in protocol["checkpoint_sha256"].items():
        if digest(data.result / "checkpoints" / name) != expected:
            raise ValueError(f"Frozen checkpoint changed: {name}")
    for name, expected in protocol["source_artifact_sha256"].items():
        if digest(data.result / name) != expected:
            raise ValueError(f"Frozen source artifact changed: {name}")
    for name, expected in protocol["cache_artifact_sha256"].items():
        if digest(data.cache / name) != expected:
            raise ValueError(f"Frozen cache artifact changed: {name}")
    for filename, key in (("fold_evidence.json", "evidence_sha256"), ("evaluation.json", "evaluation_sha256")):
        if digest(out / filename) != protocol[key]:
            raise ValueError(f"Prepared artifact changed: {filename}")
    folds = json.loads((out / "fold_evidence.json").read_text(encoding="utf-8"))
    if [f["split"] for f in folds] != data.splits:
        raise ValueError("Prepared folds differ from checkpoint folds")
    return data, folds, protocol


def prepare_assessment(data, record, bundle):
    replay = data.replay(record["path_index"], bundle)
    pack = replay.assess("adaptive")
    if pack["result"]["state"] == "uncertain":
        evidence = replay.evidence_snapshot()
    else:
        # A successful early exit must not acquire temporal or other evidence
        # merely to build an explanation that the Supervisor will skip.
        evidence = dict(probabilities={}, reliability=replay.reliability(),
                        qc_accepted_fraction=len(replay.x) / max(replay.path["total_possible_windows"], 1),
                        reference_available=pack["reference_available"])
    reliability = pack.get("reliability", evidence.get("reliability", {}))
    retrieval = pack.get("case_retrieval", evidence.get("case_retrieval", {})) or {}
    if not retrieval:
        retrieval = replay.memo.get("CaseRetriever", {}) or {}
    examples = retrieval.get("examples", [])
    if "case_retrieval" in evidence:
        # Cases are sent once at the top level; this block contains retrieval
        # quality only so the same five neighbors are not counted twice.
        evidence["case_retrieval"] = {k: v for k, v in evidence["case_retrieval"].items() if k != "examples"}
    return dict(evidence=evidence, examples=examples, numeric_probability=reliability.get("p_raw"),
                reliability=reliability, decision=pack["result"]), pack


def execute_assessment(inputs, pack, log_path, assessor, dry_run=False):
    if dry_run:
        result = summarize_assessment(inputs["evidence"], inputs["numeric_probability"],
            dict(success=False, seconds=0, skipped_reason="offline_dry_run"),
            reliability=inputs["reliability"], decision=inputs["decision"])
    else:
        result = assessor(**inputs, log_path=log_path)
    result["tool_calls"] = pack["tool_calls"]
    result["tool_latency_ms"] = pack["tool_latency_ms"]
    result["assessment_latency_ms"] = pack["assessment_latency_ms"]
    result["physiological_evidence"] = pack.get("physiological_evidence", inputs["evidence"].get("physiological_evidence"))
    result["case_evidence"] = pack.get("case_retrieval", inputs["evidence"].get("case_retrieval"))
    return result


def run_independent(out=DEFAULT_OUT, pilot_out=None, assessor=assess_evidence, workers=4, dry_run=False):
    out = Path(out)
    data, folds, protocol = load_prepared(out, pilot_out)
    directory = out / ("offline_calls" if dry_run else "independent_calls")
    directory.mkdir(exist_ok=True)
    frozen = dict(schema_version="uncertainty_agent_v2", paths=len(data.paths), subjects=data.audit["subjects"],
                  independent_single_query=True, llm_changes_probability=False,
                  dry_run=dry_run, previous_response_id_used=False,
                  retrieval="subject-diverse Top K=5; no class quota", exploratory=True)
    protocol_path = out / ("offline_protocol.json" if dry_run else "independent_protocol.json")
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Independent protocol changed; use a fresh directory")
    write_json(protocol_path, frozen)
    completed = []
    # GPU numerical work is sequential. Only optional explanation calls run in parallel.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for fold in folds:
            bundle = data.load_bundle(fold["split"])
            for record in fold["records"]:
                if record["role"] != "outer_test":
                    continue
                i = record["path_index"]
                inputs, pack = prepare_assessment(data, record, bundle)
                pending[pool.submit(execute_assessment, inputs, pack,
                    directory / f"path_{i:03d}_cloud_call.json", assessor, dry_run)] = i
        for future in as_completed(pending):
            i = pending[future]
            pack = future.result()
            write_json(directory / f"path_{i:03d}_evidence_pack.json", pack)
            completed.append(dict(path_index=i, state=pack["state"], probability=pack["high_probability"],
                                  cloud_called=pack.get("cloud_called", False), cloud_success=pack["cloud_success"],
                                  seconds=pack["measured_cloud_seconds"], tools=pack["tool_calls"]))
    if len(completed) != len(data.paths) or len({r["path_index"] for r in completed}) != len(data.paths):
        raise ValueError("Missing/duplicate outer test paths")
    status = dict(schema_version="uncertainty_agent_v2", status="offline_dry_run" if dry_run else "completed",
                  paths=len(completed), partial_run=protocol["partial_run"],
                  cloud_success=sum(r["cloud_success"] for r in completed),
                  cloud_calls=sum(r["cloud_called"] for r in completed),
                  results=sorted(completed, key=lambda r: r["path_index"]))
    write_json(out / ("offline_status.json" if dry_run else "independent_status.json"), status)
    print(json.dumps({k: v for k, v in status.items() if k != "results"}, indent=2), flush=True)
    return status


def assess_path(out, path_index, pilot_out=None, dry_run=False):
    data, folds, _ = load_prepared(out, pilot_out)
    if not 0 <= path_index < len(data.paths):
        raise ValueError("Path index outside eligible manifest")
    fold = next(f for f in folds if f["outer_subject"] == data.paths[path_index]["subject_key"])
    record = next(r for r in fold["records"] if r["path_index"] == path_index and r["role"] == "outer_test")
    inputs, pack = prepare_assessment(data, record, data.load_bundle(fold["split"]))
    return execute_assessment(inputs, pack, Path(out) / f"single_case_{path_index:03d}_cloud_call.json", assess_evidence, dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_independent(args.out, args.pilot_out, workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
