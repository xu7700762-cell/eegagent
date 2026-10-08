"""Run explanation-only DeepSeek calls after the shared lazy V2 EEG policy."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from vrms_cloud.cloud import write_json
from vrms_cloud.independent import run_independent
from vrms_cloud.prepare import prepare_v2
from vrms_pilot.experiment import DEFAULT_OUT as PILOT_OUT
from .supervisor import CONFIG_FILE, assess_evidence, make_provider

DEFAULT_OUT = Path("outputs/vrms_deepseek/v2_seed2026")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-out", type=Path, default=PILOT_OUT)
    parser.add_argument("--config-file", type=Path, default=CONFIG_FILE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "protocol.json").exists():
        prepare_v2(args.out, args.pilot_out)
    if args.dry_run:
        return run_independent(args.out, pilot_out=args.pilot_out,
                               workers=args.workers, dry_run=True)
    provider, captured, public = make_provider(args.config_file, max_calls=300)
    write_json(args.out / "provider.json", public)
    result = run_independent(args.out, pilot_out=args.pilot_out,
                            assessor=partial(assess_evidence, provider, captured, public),
                            workers=args.workers)
    write_json(args.out / "client_summary.json", provider.summary())
    return result


if __name__ == "__main__":
    main()
