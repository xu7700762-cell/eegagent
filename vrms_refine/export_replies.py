"""Export cached V2 public explanations without training or API calls."""
import argparse
import json
from pathlib import Path

from .common import DEFAULT_OUT, check_v2_artifacts, read_json, sha, write_json


def public_reply(log, vendor, expected_model):
    if not log["success"]:
        raise ValueError("A failed call cannot be displayed as an agent reply")
    prediction = log["predictions"][0] if vendor == "gpt" else log["prediction"]
    if "high_probability" in prediction or "state" in prediction:
        raise ValueError("V2 reply must contain an explanation only")
    response = next(attempt["response"] for attempt in reversed(log["attempts"]) if attempt.get("status") == "success")
    if response["model"] != expected_model:
        raise ValueError("Unexpected returned model")
    if vendor == "gpt":
        reply = "".join(part["text"] for item in response.get("output", [])
                        for part in item.get("content", []) if part.get("type") == "output_text")
    else:
        reply = response["choices"][0]["message"]["content"]
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Public final assistant message is missing")
    return prediction, reply


def export_v2(out, vendors):
    out = Path(out).resolve()
    protocol, jobs = check_v2_artifacts(out, require_validation=True), read_json(out / "jobs.json")
    if protocol.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("Do not export historical probability replies as V2 explanations")
    expected = {(job["mode"], job["outer_subject"], job["path_index"]) for job in jobs}
    records = []
    for vendor in vendors:
        status = read_json(out / vendor / "run_status.json")
        keys = {(row["mode"], row["outer_subject"], row["path_index"]) for row in status["results"]}
        if status["status"] != "completed" or keys != expected or len(keys) != len(status["results"]):
            raise ValueError("Explanation run is incomplete or has duplicate cases")
        for row in status["results"]:
            if not row["success"]:
                continue
            path = Path(row["selected_log_path"]).resolve()
            if not path.is_relative_to(out / vendor) or sha(path) != row["log_sha256"]:
                raise ValueError("Reply source changed or is outside the run")
            prediction, message = public_reply(read_json(path), vendor, protocol["models"][vendor])
            if prediction != row["explanation"]:
                raise ValueError("Displayed reply differs from the audited explanation")
            records.append(dict(path_index=row["path_index"], outer_subject=row["outer_subject"], role=row["role"],
                vendor=vendor, mode=row["mode"], model=protocol["models"][vendor], explanation=prediction,
                assistant_message=message, source_path=str(path), source_sha256=row["log_sha256"]))
    write_json(out / "agent_replies.json", dict(schema_version="uncertainty_agent_v2", records=records,
        reply_count=len(records), new_model_api_calls=0, llm_probability_modification=False,
        failed_calls_remain_in_run_status=True))
    return dict(status="exported", replies=len(records), new_model_api_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vendors", nargs="+", choices=("gpt", "deepseek"), default=["gpt", "deepseek"])
    args = parser.parse_args()
    print(json.dumps(export_v2(args.out, args.vendors), ensure_ascii=False))


if __name__ == "__main__":
    main()
