"""Demonstrate the API contract using synthetic evidence; dry-run is offline."""
import argparse
import json
from pathlib import Path

from vrms_cloud.cloud import SCHEMA, check_blind
from vrms_cloud.supervisor import summarize_assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).with_name("evidence.json"))
    parser.add_argument("--vendor", choices=("gpt", "deepseek"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path, default=Path("outputs/example/v2_cloud_call.json"))
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    check_blind(evidence)
    numeric = evidence["reliability"]["p_raw"]
    if args.dry_run or args.vendor is None:
        decision = summarize_assessment(evidence, numeric,
            dict(success=False, seconds=0., skipped_reason="offline_dry_run"))
        print(json.dumps(dict(synthetic_example=True, api_called=False, decision=decision,
                              response_schema=SCHEMA,
                              request=dict(examples=[], queries=[dict(id="qsingle", evidence=evidence)])), indent=2))
        return
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.vendor == "gpt":
        from vrms_cloud.supervisor import assess_evidence
        result = assess_evidence(evidence, [], numeric, args.log)
    else:
        from vrms_deepseek.supervisor import make_provider, assess_evidence
        provider, captured, public = make_provider(max_calls=2)
        result = assess_evidence(provider, captured, public, evidence, [], numeric, args.log)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
