"""Run the actual single-evidence Supervisor separately for every heldout path.

No other heldout path, future query, conversation history, or query label is
sent in a request. The predeclared 25% weight is fixed, not tuned again.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import random

from .cloud import choose_examples, write_json
from .supervisor import assess_evidence

DEFAULT_OUT = Path("outputs/vrms_cloud/20261008_seed2026")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    evaluation = {r["path_index"]: r for r in json.loads((args.out / "evaluation.json").read_text(encoding="utf-8"))}
    folds = json.loads((args.out / "fold_evidence.json").read_text(encoding="utf-8"))
    with (args.out / "improvements/predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        numeric = {int(r["path_index"]): float(r["validated_numeric"]) for r in csv.DictReader(f) if r["role"] == "outer_test"}
    directory = args.out / "independent_calls"
    directory.mkdir(exist_ok=True)
    protocol_path = args.out / "independent_protocol.json"
    protocol = dict(paths=147, subjects=24, requests="one heldout path per independent Responses request",
                    previous_response_id_used=False, llm_weight=.25, weight_source="predeclared fixed blend in initial protocol",
                    source_models="same frozen improved models; no retraining after batch results",
                    examples="same 8-or-fewer fold-internal meta examples as batch mode",
                    test_paths_sent_as_examples=False, outer_labels_used_for_tuning=False,
                    baseline="same unrounded improved numerical probabilities", seed_for_job_order=2026,
                    all_failures_retained=True, failure_fallback="improved numerical probability",
                    exploratory=True)
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Independent protocol changed")
    else:
        write_json(protocol_path, protocol)
    jobs = []
    for fold in folds:
        examples = [dict(evidence=r["evidence"], observed_class="high" if evaluation[r["path_index"]]["label"] else "low")
                    for r in choose_examples(fold["records"], evaluation, 2026 + fold["outer_subject"])]
        for record in fold["records"]:
            if record["role"] == "outer_test":
                jobs.append(dict(record=record, examples=examples))
    random.Random(2026).shuffle(jobs)
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(assess_evidence, j["record"]["evidence"], j["examples"],
                              numeric[j["record"]["path_index"]],
                              directory / f"path_{j['record']['path_index']:03d}_cloud_call.json", .25): j for j in jobs}
        for future in as_completed(pending):
            job = pending[future]
            index = job["record"]["path_index"]
            pack = future.result()
            write_json(directory / f"path_{index:03d}_evidence_pack.json", pack)
            completed.append(dict(path_index=index, cloud_success=pack["cloud_success"],
                                  cloud_probability=pack["cloud_probability"],
                                  probability=pack["high_probability"], seconds=pack["measured_cloud_seconds"]))
            if len(completed) % 15 == 0 or len(completed) == len(jobs):
                print(f"single-path cloud {len(completed)}/{len(jobs)}; "
                      f"success={sum(r['cloud_success'] for r in completed)}", flush=True)
    write_json(args.out / "independent_status.json", dict(status="completed", paths=len(completed),
               cloud_success=sum(r["cloud_success"] for r in completed),
               results=sorted(completed, key=lambda r: r["path_index"])))


if __name__ == "__main__":
    main()
