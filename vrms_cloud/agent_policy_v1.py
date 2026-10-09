"""Correction candidates chosen and audited on policy subjects only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agent_tools_v1 import DEFAULT_OUT, correction_counts
from .cloud import write_json
from .second_judgment import sha256

NAMES = ("femba_features", "spectral", "spatial_covariance", "case_retrieval", "equal_average", "fusion")
MARGINS = (0., .05, .10, .15, .20, .25, .30)


def candidates(probability, fusion):
    tools = probability[:, 1:]
    return np.c_[tools, tools.mean(axis=1), fusion]


def apply_rule(baseline, tools, rule):
    if rule is None:
        return np.asarray(baseline)
    selected = tools[:, rule["column"]]
    agree = np.sum((tools[:, :4] >= .5) == (selected[:, None] >= .5), axis=1)
    change = ((selected >= .5) != (baseline >= .5)) & (np.abs(selected - .5) >= rule["margin"]) & (agree >= rule["minimum_agreement"])
    return np.where(change, selected, baseline)


def select_rule(y, baseline, tools, groups):
    best, best_key = None, (0, 0, 0)
    for column in range(len(NAMES)):
        for margin in MARGINS:
            for agreement in (1, 2, 3):
                rule = dict(column=column, name=NAMES[column], margin=margin, minimum_agreement=agreement)
                p = apply_rule(baseline, tools, rule)
                counts = correction_counts(y, baseline, p)
                changed = (baseline >= .5) != (p >= .5)
                subjects = len(np.unique(groups[changed]))
                if counts["changes"] < 4 or subjects < 2 or counts["corrected"] < 2 * counts["damaged"]:
                    continue
                # Penalize harmful changes explicitly; no outer label access.
                utility = counts["corrected"] - 2 * counts["damaged"]
                key = utility, counts["net"], -counts["damaged"]
                if key > best_key:
                    best, best_key = rule, key
    return best


def audit_rule(y, baseline, tools, groups):
    original = select_rule(y, baseline, tools, groups)
    nested = np.asarray(baseline).copy()
    audits = []
    for subject in np.unique(groups):
        select = groups != subject
        test = ~select
        candidate = select_rule(y[select], baseline[select], tools[select], groups[select])
        nested[test] = apply_rule(baseline[test], tools[test], candidate)
        audits.append(dict(heldout_subject=int(subject), selector_subjects=sorted(map(int, set(groups[select]))),
                           candidate=candidate, correction=correction_counts(y[test], baseline[test], nested[test])))
    measured = correction_counts(y, baseline, nested)
    changed = (nested >= .5) != (baseline >= .5)
    supported = len(np.unique(groups[changed]))
    passed = (original is not None and measured["changes"] >= 4 and supported >= 2
              and measured["net"] > 0 and measured["corrected"] >= 2 * measured["damaged"])
    return dict(enabled=passed, rule=original if passed else None, selected_candidate=original,
        selected_correction=correction_counts(y, baseline, apply_rule(baseline, tools, original)),
        nested_correction=measured, nested_changed_subjects=supported, nested_audit=audits,
        rule_scope="independent policy4; LOSO candidate selection and audit; exploratory small-sample guard")


def freeze(out=DEFAULT_OUT):
    out = Path(out)
    directory = out / "policies"
    if directory.exists():
        raise FileExistsError("Preserve frozen policy")
    directory.mkdir()
    splits = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    for split in splits:
        values = np.load(out / "internal_validation" / f"fold_{split['outer_subject']:02d}.npz")
        y, groups = values["policy_labels"], values["policy_subjects"]
        if split["outer_subject"] in set(groups):
            raise ValueError("Test subject entered policy")
        p = values["policy_probability"]
        result = audit_rule(y, p[:, 0], candidates(p, values["policy_fusion"]), groups)
        write_json(directory / f"fold_{split['outer_subject']:02d}.json", result)
    write_json(directory / "manifest.json", dict(frozen_before_outer_evaluation=True, scope="policy4 only",
        margins=MARGINS, minimum_change_cases=4, minimum_change_subjects=2,
        minimum_correction_to_damage=2, miscorrection_utility_penalty=2,
        tool_training_sha256=sha256(out / "training_summary.json"), code_sha256=sha256(__file__),
        policy_hashes={p.name:sha256(p) for p in directory.glob("fold_*.json")}))
    print(json.dumps(dict(policy_folds=len(splits), enabled=sum(
        json.loads(p.read_text(encoding="utf-8"))["enabled"] for p in directory.glob("fold_*.json")))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    freeze(parser.parse_args().out)
