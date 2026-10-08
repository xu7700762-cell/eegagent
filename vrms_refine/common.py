"""Credential-free shared contracts, usable by Windows and WSL runners."""
import hashlib
import json
from pathlib import Path

from vrms_cloud.cloud import SYSTEM, check_blind

DEFAULT_OUT = Path("outputs/vrms_refine/20261008_seed2026")
CLOUD = Path("outputs/vrms_cloud/20261008_seed2026")
DEEPSEEK = Path("outputs/vrms_deepseek/20261008_seed2026")
PILOT = Path("outputs/vrms_pilot/20261007_seed2026")


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
    return json.dumps(dict(task="Assess this one anonymous path independently", examples=examples,
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
