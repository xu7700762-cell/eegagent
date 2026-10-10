# -*- coding: utf-8 -*-
"""Run the registered local EEG preparation tools for one recording.

This script is deliberately process-scoped: a single CDT can occupy several
GB while filtering, so the caller can run one subject per child process and
collect one compact JSON result.  The four preparation tools are invoked via
``execute_tool`` rather than bypassing the tool registry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_agent.domain_tools import execute_tool  # noqa: E402
from eeg_agent.recordings import CHANNEL_NEIGHBORS, RecordingRepository  # noqa: E402


def compact_quality(value: dict | None) -> dict:
    value = value or {}
    window_qc = value.get("window_qc") or {}
    return {
        "status": value.get("status"),
        "score": value.get("score"),
        "warning_score": value.get("warning_score"),
        "bad_channels": list(value.get("bad_channels") or []),
        "blocking_bad_channels": list(value.get("blocking_bad_channels") or []),
        "warning_channels": list(value.get("warning_channels") or []),
        "persistent_bad_channels": list(value.get("persistent_bad_channels") or []),
        "transient_bad_channels": list(value.get("transient_bad_channels") or []),
        "shared_artifact_blocking": bool(value.get("shared_artifact_blocking")),
        "window_qc": {
            "window_samples": window_qc.get("window_samples"),
            "window_seconds": window_qc.get("window_seconds"),
            "total_windows": window_qc.get("total_windows"),
            "usable_windows": window_qc.get("usable_windows"),
            "ignored_windows": window_qc.get("ignored_windows"),
            "shared_artifact_windows": window_qc.get("shared_artifact_windows"),
            "shared_artifact_fraction": window_qc.get("shared_artifact_fraction"),
            "persistence_threshold": window_qc.get("persistence_threshold"),
        },
    }


def donor_map(bad: list[str], names: list[str]) -> tuple[dict, dict]:
    bad_set = set(bad)
    enough, insufficient = {}, {}
    for channel in bad:
        donors = [
            donor for donor in CHANNEL_NEIGHBORS.get(channel, ())
            if donor in names and donor not in bad_set
        ]
        (enough if len(donors) >= 2 else insufficient)[channel] = donors
    return enough, insufficient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="subject-01 ... subject-26")
    parser.add_argument("--data-root", default=None,
                        help="local dataset root; defaults to EEG_DATA_ROOT")
    parser.add_argument("--output-root", default="./outputs/registered_tool_runs")
    parser.add_argument("--repair", action="store_true", help="try auto-detected interpolation")
    args = parser.parse_args()

    data_root = args.data_root or __import__("os").environ.get("EEG_DATA_ROOT")
    if not data_root:
        parser.error("provide --data-root or set EEG_DATA_ROOT")
    repo = RecordingRepository({"data_root": data_root, "output_root": args.output_root})
    evidence = repo.analyze(args.subject)
    record = evidence["recording"]
    loader = execute_tool("EEGFileLoader", evidence)
    preprocessor = execute_tool(
        "EEGPreprocessor", evidence,
        args={"profile": "raw-vrms-causal-v1"},
    )
    quality = execute_tool("EEGQualityAssessor", evidence)
    before = quality.get("quality") or {}
    names = list(record.get("eeg_channels") or [])
    # Only persistent hard faults are interpolation candidates.  Soft
    # correlation warnings remain visible in ``quality_before`` but must not
    # trigger donor selection or an automatic repair request.
    bad = list(before.get("blocking_bad_channels") or [])
    enough, insufficient = donor_map(bad, names)
    repair = execute_tool(
        "EEGChannelRepair", evidence,
        repository=repo,
        recording_id=args.subject,
        args={
            "strategy": "interpolate" if args.repair else "none",
            "auto_detect": bool(args.repair),
            "min_neighbors": 2,
        },
    )
    result = {
        "subject": args.subject,
        "tools_called": [
            "EEGFileLoader", "EEGPreprocessor", "EEGQualityAssessor", "EEGChannelRepair"
        ],
        "loader": {
            "available": loader.get("available"),
            "format": loader.get("format"),
            "sampling_hz": loader.get("sampling_hz"),
            "channel_count": loader.get("channel_count"),
            "duration_seconds": loader.get("duration_seconds"),
        },
        "preprocessor": {
            "available": preprocessor.get("available"),
            "profile": (preprocessor.get("preprocessing") or {}).get("profile"),
            "accepted_windows": preprocessor.get("accepted_windows"),
            "total_windows": preprocessor.get("total_windows"),
            "window_coverage": preprocessor.get("window_coverage"),
        },
        "quality_before": compact_quality(before),
        "accepted_windows": record.get("overall", {}).get("accepted_windows"),
        "total_windows": record.get("overall", {}).get("total_windows"),
        "window_coverage": record.get("overall", {}).get("window_coverage"),
        "model_compatible_before": record.get("model_compatible"),
        "interpolation_candidates": enough,
        "insufficient_donors": insufficient,
        "repair_requested": bool(args.repair),
        "repair": {
            "available": repair.get("available"),
            "applied": repair.get("applied"),
            "reason": repair.get("reason"),
            "repair": repair.get("repair"),
            "quality_after": compact_quality(repair.get("quality_after")),
            "model_compatible": repair.get("model_compatible"),
            "accepted_windows": repair.get("accepted_windows"),
            "window_coverage": repair.get("window_coverage"),
        },
    }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
