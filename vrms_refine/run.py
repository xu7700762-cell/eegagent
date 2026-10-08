"""Run real baseline-policy and enhanced single-path cloud requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import threading
import time

from vrms_cloud.cloud import call_job, existing_provider
from vrms_deepseek.supervisor import CONFIG_FILE, call_evidence, make_provider
from .common import DEFAULT_OUT, api_job, read_json, request_path, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vendor", choices=("gpt", "deepseek"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recover-http402", action="store_true")
    args = parser.parse_args()
    jobs = read_json(args.out / "jobs.json")
    protocol = read_json(args.out / "protocol.json")
    if not args.limit and not (args.out / "analysis_plan.json").exists():
        raise ValueError("Freeze the analysis plan before the full API run")
    random.Random(2026).shuffle(jobs)
    if args.limit:
        # Probe both contexts, using the same deterministic jobs in the full run.
        first = [next(j for j in jobs if j["mode"] == mode) for mode in ("baseline", "enhanced")]
        jobs = (first + [j for j in jobs if j not in first])[:args.limit]
    out = args.out / args.vendor
    out.mkdir(exist_ok=True)
    if args.vendor == "gpt":
        provider = existing_provider()
        if provider["model"] != protocol["models"]["gpt"]:
            raise ValueError("Configured GPT model changed")
        public = {k: v for k, v in provider.items() if k != "key"}
    else:
        provider, captured, public = make_provider(CONFIG_FILE, max_calls=2 * len(jobs) + 2)
        if public["model_id"] != protocol["models"]["deepseek"]:
            raise ValueError("Configured DeepSeek model changed")
        original_create = provider.client.chat.completions.create
        def capture_http_status(**kwargs):
            captured.last_http_status = None
            try:
                return original_create(**kwargs)
            except Exception as exc:
                captured.last_http_status = getattr(exc, "status_code", None)
                raise
        provider.client.chat.completions.create = capture_http_status
    write_json(out / "provider.json", public)
    status_path = out / ("probe_status.json" if args.limit else "run_status.json")
    completed, started = [], time.perf_counter()
    blocked = threading.Event()

    def run_job(job):
        path = request_path(args.out, args.vendor, job)
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.recover_http402 and path.exists():
            prior = read_json(path)
            infrastructure_errors = {"APIStatusError", "RateLimitError"}
            if not prior["success"] and all(a.get("error_type") in infrastructure_errors for a in prior["attempts"]):
                path = path.with_name(path.stem + "_recovery.json")
        if blocked.is_set() and not path.exists():
            return None
        if args.vendor == "gpt":
            log = call_job(provider, api_job(job), path)
            prediction = log["predictions"][0] if log["success"] else None
        else:
            message = __import__("json").loads(job["message"])
            log = call_evidence(provider, captured, public, message["queries"][0]["evidence"], message["examples"], path)
            prediction = log["prediction"] if log["success"] else None
            http_status = getattr(captured, "last_http_status", None)
            if not log["success"] and http_status == 402:
                log["infrastructure_http_status"] = 402
                write_json(path, log)
                blocked.set()
        return dict(mode=job["mode"], outer_subject=job["outer_subject"], path_index=job["path_index"], role=job["role"],
                    success=log["success"], probability=None if prediction is None else prediction["high_probability"],
                    uncertain=True if prediction is None else prediction["uncertain"], seconds=log["seconds"],
                    log_sha256=sha(path), selected_log_path=str(path))

    def save_status(status):
        write_json(status_path, dict(status=status, vendor=args.vendor, expected_requests=len(jobs),
            completed_requests=len(completed), successes=sum(r["success"] for r in completed),
            elapsed_seconds=time.perf_counter() - started, results=completed))

    save_status("running_probe" if args.limit else "running")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            completed.append(result)
            if len(completed) % 5 == 0 or len(completed) == len(jobs):
                save_status("running_probe" if args.limit else "running")
            if len(completed) % 25 == 0 or args.limit or len(completed) == len(jobs):
                print(f"{args.vendor}: {len(completed)}/{len(jobs)}; success={sum(r['success'] for r in completed)}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    save_status("interrupted_http402" if blocked.is_set() else "probe_only" if args.limit else "completed")
    if args.vendor == "deepseek":
        write_json(out / ("probe_client_summary.json" if args.limit else "client_summary.json"), provider.summary())


if __name__ == "__main__":
    main()
