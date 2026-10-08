"""Compare the existing DeepSeek API on the exact frozen 147-path EEG evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random

from vrms_cloud.cloud import choose_examples, write_json
from .supervisor import CONFIG_FILE, assess_evidence, make_provider

FROZEN = Path("outputs/vrms_cloud/20261008_seed2026")
DEFAULT_OUT = Path("outputs/vrms_deepseek/20261008_seed2026")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config-file", type=Path, default=CONFIG_FILE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    provider, captured, public = make_provider(args.config_file)
    evaluation = {r["path_index"]: r for r in json.loads((FROZEN / "evaluation.json").read_text(encoding="utf-8"))}
    folds = json.loads((FROZEN / "fold_evidence.json").read_text(encoding="utf-8"))
    with (FROZEN / "improvements/predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        numeric = {int(r["path_index"]): float(r["validated_numeric"]) for r in csv.DictReader(f) if r["role"] == "outer_test"}
    protocol = dict(model=public["model_id"], base_url=public["base_url"], config_source=str(args.config_file),
                    codex_configuration_used=False, actual_api_client="vrms_deepseek/provider.py",
                    paths=147, subjects=24, seed=2026, endpoint="whole_path_score_ge30", llm_weight=.25,
                    model_retrained=False, split_changed=False, evidence_changed=False,
                    requests="one heldout path independently; fold-meta examples only", outer_labels_in_prompt=False,
                    selection="no new prompt, weight or numerical tuning", temperature=0, max_tokens=700,
                    native_thinking_disabled=provider.thinking_disabled, timeout_seconds=60, max_attempts_per_path=2,
                    adapter_call_budget=300, all_failures_retained=True, failure_fallback="same improved numerical probability",
                    source_artifact_sha256={str(p): sha(p) for p in
                        [FROZEN / "fold_evidence.json", FROZEN / "evaluation.json", FROZEN / "improvements/predictions.csv",
                         FROZEN / "summary.json", Path("vrms_deepseek/provider.py"),
                         Path("vrms_deepseek/supervisor.py"), Path("eegagent_config.py")]},
                    exploratory=True, earlier_gpt_outer_results_seen=True)
    path = args.out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Frozen DeepSeek protocol changed")
    else:
        write_json(path, protocol)
    directory = args.out / "independent_calls"
    directory.mkdir(exist_ok=True)
    jobs = []
    for fold in folds:
        examples = [dict(evidence=r["evidence"], observed_class="high" if evaluation[r["path_index"]]["label"] else "low")
                    for r in choose_examples(fold["records"], evaluation, 2026 + fold["outer_subject"])]
        for record in fold["records"]:
            if record["role"] == "outer_test":
                jobs.append(dict(record=record, examples=examples))
    random.Random(2026).shuffle(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(assess_evidence, provider, captured, public, j["record"]["evidence"], j["examples"],
                              numeric[j["record"]["path_index"]],
                              directory / f"path_{j['record']['path_index']:03d}_cloud_call.json", .25): j for j in jobs}
        for future in as_completed(pending):
            job = pending[future]
            i = job["record"]["path_index"]
            pack = future.result()
            write_json(directory / f"path_{i:03d}_evidence_pack.json", pack)
            completed.append(dict(path_index=i, **pack))
            if len(completed) % 15 == 0 or args.limit or len(completed) == len(jobs):
                print(f"DeepSeek {len(completed)}/{len(jobs)}; success={sum(r['cloud_success'] for r in completed)}; "
                      f"configured={public['model_id']}; response={pack['response_model']}", flush=True)
    write_json(args.out / ("probe_status.json" if args.limit else "run_status.json"),
               dict(status="probe_only" if args.limit else "completed", paths=len(completed),
                    successful=sum(r["cloud_success"] for r in completed), results=sorted(completed, key=lambda r: r["path_index"])))
    write_json(args.out / ("probe_client_summary.json" if args.limit else "client_summary.json"), provider.summary())


if __name__ == "__main__":
    main()
