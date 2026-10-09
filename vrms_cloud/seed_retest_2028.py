"""Predeclared seed2028 repeat using the unchanged seed2027 training recipe."""
import argparse
from pathlib import Path
import shutil

from . import seed_retest as previous
from .cloud import write_json
from .second_judgment import sha256

DEFAULT_OUT = Path("outputs/vrms_agent_seed_retest/seed2028_20261008")
PREVIOUS_OUT = previous.DEFAULT_OUT


def freeze(out=DEFAULT_OUT):
    out = Path(out)
    # Correct the selection description before any training or new prediction.
    payload = previous.freeze_protocol(out,2028,"full")
    payload["seed_selection"] = "2028 selected before any new result; one additional repeat requested"
    payload["previous_seed2027_summary_sha256"] = sha256(PREVIOUS_OUT / "summary.json")
    payload["previous_seed2027_protocol_sha256"] = sha256(PREVIOUS_OUT / "protocol.json")
    payload["source_sha256"][Path(__file__).name] = sha256(__file__)
    write_json(out / "protocol.json",payload)
    shutil.copy2(__file__,out / "sources" / Path(__file__).name)


def verify_previous():
    previous.verify_protocol(PREVIOUS_OUT)
    assert previous.read(PREVIOUS_OUT / "final_verification.json")["status"] == "completed_and_verified"


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=["freeze","models","numeric","gpt"])
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--probe",action="store_true")
    args=parser.parse_args()
    verify_previous()
    if args.stage == "freeze":
        freeze(args.out)
    elif args.stage == "models":
        previous.models(args.out)
    elif args.stage == "numeric":
        previous.numeric(args.out)
    else:
        previous.gpt(args.out,args.probe)
