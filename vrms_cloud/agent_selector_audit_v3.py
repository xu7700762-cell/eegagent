"""Nested subject audit of evidence-selector tuning, before any outer query."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from .agent_selector_v3 import CONFIGS,MARGINS,DEFAULT_OUT,inputs,utility,gated_prediction
from .agent_tools_v1 import DEFAULT_OUT as FIRST,binary_metrics,correction_counts,classifier
from .tool_agent_v2 import EXTENDED_TOOLS as SECOND
from .cloud import write_json
from .second_judgment import sha256


def scores(model,x):
    return expit(model.decision_function(x)) if hasattr(model,"decision_function") else model.predict_proba(x)[:,1]


def audit(out=DEFAULT_OUT,first=FIRST,second=SECOND):
    from threadpoolctl import threadpool_limits
    out,first,second=Path(out),Path(first),Path(second)
    directory=out/"nested_audit"
    if directory.exists():raise FileExistsError("Keep previous nested audits")
    directory.mkdir()
    paths=json.loads((first/"local_paths.json").read_text(encoding="utf-8"))
    y=np.asarray([p["label"] for p in paths]);g=np.asarray([p["subject_key"] for p in paths])
    splits=json.loads((first/"split_manifest.json").read_text(encoding="utf-8"));reports=[]
    with threadpool_limits(limits=2):
        for n,split in enumerate(splits,1):
            subject=split["outer_subject"]
            _,a,_,mx,px=inputs(first,second,subject)
            mi,pi=a["meta_indices"],a["policy_indices"]
            baseline=expit(px[:,0]);nested=baseline.copy();raw_nested=np.full(len(pi),np.nan);audits=[]
            for heldout in np.unique(g[pi]):
                select=g[pi]!=heldout;test=~select
                sx,sy,sg=px[select],y[pi[select]],g[pi[select]]
                candidate_predictions=[]
                for config in CONFIGS:
                    p=np.full(len(sy),np.nan)
                    for inner in np.unique(sg):
                        fit=sg!=inner;query=~fit
                        model=classifier(config).fit(np.r_[mx,sx[fit]],np.r_[y[mi],sy[fit]])
                        p[query]=scores(model,sx[query])
                    candidate_predictions.append(p)
                chosen=(0,0.);best=(-10000,-10000,-10000)
                for i,p in enumerate(candidate_predictions):
                    for margin in MARGINS:
                        key=utility(sy,expit(sx[:,0]),gated_prediction(expit(sx[:,0]),p,margin))
                        if key>best:best=key;chosen=i,margin
                i,margin=chosen
                model=classifier(CONFIGS[i]).fit(np.r_[mx,sx],np.r_[y[mi],sy])
                raw_nested[test]=scores(model,px[test]);nested[test]=gated_prediction(baseline[test],raw_nested[test],margin)
                audits.append(dict(heldout_subject=int(heldout),selection_subjects=sorted(map(int,set(sg))),
                    configuration=CONFIGS[i],margin=margin,
                    measured=correction_counts(y[pi[test]],baseline[test],nested[test])))
            measured=correction_counts(y[pi],baseline,nested)
            changes=(baseline>=.5)!=(nested>=.5)
            support=len(np.unique(g[pi][changes]))
            approved=measured["corrected"]>=2*measured["damaged"] and measured["corrected"]>0 and support>=2
            report=dict(outer_subject=subject,approved=approved,correction=measured,changed_subjects=support,
                raw_nested_validation=binary_metrics(y[pi],raw_nested),nested_validation=binary_metrics(y[pi],nested),
                splits=audits,outer_labels_used=False,scope="candidate configuration and margin both exclude each heldout policy subject")
            reports.append(report);write_json(directory/f"fold_{subject:02d}.json",report)
            print(json.dumps(dict(fold=n,total=24,approved=approved,correction=measured)),flush=True)
    write_json(directory/"summary.json",dict(approved_folds=sum(r["approved"] for r in reports),folds=24,
        mean_nested_policy_acc=float(np.mean([r["nested_validation"]["acc"] for r in reports])),
        correction_totals={k:sum(r["correction"][k] for r in reports) for k in ("corrected","damaged","net")},
        code_sha256=sha256(__file__),selector_training_sha256=sha256(out/"training_summary.json"),outer_labels_used=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    audit(parser.parse_args().out)
