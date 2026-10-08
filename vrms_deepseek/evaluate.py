"""Evaluate deterministic V2 deep predictions, coverage and explanation accounting."""
import argparse
from pathlib import Path

from vrms_cloud.evaluate import evaluate_v2
from .experiment import DEFAULT_OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    return evaluate_v2(args.out, args.offline)


if __name__ == "__main__":
    main()
