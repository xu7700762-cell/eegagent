"""Credential-free shared contracts, usable by Windows and WSL runners."""
import hashlib
import json
from pathlib import Path

from vrms_cloud.cloud import SYSTEM, check_blind

DEFAULT_OUT = Path("outputs/vrms_refine/v2_seed2026")
CLOUD = Path("outputs/vrms_cloud/v2_seed2026")
DEEPSEEK = Path("outputs/vrms_deepseek/v2_seed2026")
PILOT = Path("outputs/vrms_pilot/v2_seed2026")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def message_for(evidence, examples):
    check_blind(evidence)
    for example in examples:
        check_blind(example["evidence"])
        if example["observed_class"] not in ("high", "low"):
            raise ValueError("Internal examples require high/low classes")
    return json.dumps(dict(task="Explain this anonymous path's support, conflict and missing evidence", examples=examples,
                           queries=[dict(id="qsingle", evidence=evidence)]),
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def job_hash(message):
    return hashlib.sha256((SYSTEM + message).encode()).hexdigest()


def api_job(job):
    return dict(mode="single_evidence_assessment", outer_subject=None, user_message=job["message"],
                mapping={"qsingle": {"role": "single_case"}}, example_indices=[],
                request_sha256=job_hash(job["message"]))


def request_path(out, vendor, job):
    return Path(out) / vendor / "calls" / job["mode"] / f"fold_{job['outer_subject']:02d}_path_{job['path_index']:03d}.json"


def select_source_indices(records, allowed_subjects, evaluation, role):
    selected = [r for r in records if r["role"] == role]
    if any(evaluation[r["path_index"]]["subject_key"] not in allowed_subjects for r in selected):
        raise ValueError("Example source subject is outside the allowed internal partition")
    return selected


def check_v2_artifacts(out, require_validation=False):
    """Bind requests, algorithms and optional validation to one frozen V2 run."""
    out = Path(out)
    protocol = read_json(out / "protocol.json")
    if protocol.get("schema_version") != "uncertainty_agent_v2":
        raise ValueError("Expected V2 explanation-only artifacts")
    for filename, expected in {**protocol["original_sha256"], **protocol["implementation_sha256"]}.items():
        if sha(filename) != expected:
            raise ValueError(f"A frozen evidence source or implementation changed: {filename}")
    if protocol["system_sha256"] != hashlib.sha256(SYSTEM.encode()).hexdigest():
        raise ValueError("Frozen system prompt changed")
    if require_validation:
        validation = read_json(out / "structural_validation.json")
        if validation["status"] != "passed":
            raise ValueError("Validate frozen V2 requests before use")
        for filename, field in (("jobs.json", "jobs_sha256"), ("analysis_plan.json", "plan_sha256"),
                                ("preparation_audit.json", "audit_sha256"), ("protocol.json", "protocol_sha256")):
            if sha(out / filename) != validation[field]:
                raise ValueError(f"Validated artifact changed: {filename}")
    return protocol
