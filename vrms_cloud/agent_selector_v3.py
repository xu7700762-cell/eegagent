"""Learn which numeric evidence to trust using internal subject OOF scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .agent_tools_v1 import DEFAULT_OUT as FIRST, binary_metrics, correction_counts, classifier
from .agent_tools_v2 import FAMILIES
from .cloud import write_json
from .second_judgment import sha256
from .tool_agent_v2 import EXTENDED_TOOLS as SECOND

DEFAULT_OUT=Path("outputs/vrms_agent_loso/v6_selector_20261008")
CONFIGS=("lr_0.01","lr_0.1","lr_1","rbf_1","trees")
MARGINS=(0.,.05,.10,.15,.20,.25,.30)


def uncalibrate(p,calibrator):
    scores=logit(np.clip(p,1e-8,1-1e-8))
    if calibrator is None:return scores
    return (scores-float(calibrator.intercept_[0]))/float(calibrator.coef_[0,0])


def inputs(first,second,subject):
    first,second=Path(first),Path(second)
    old=joblib.load(first/"checkpoints"/f"fold_{subject:02d}.joblib")
    a=dict(np.load(first/"internal_validation"/f"fold_{subject:02d}.npz"))
    b=dict(np.load(second/"internal_validation"/f"fold_{subject:02d}.npz"))
    assert np.array_equal(a["meta_indices"],b["meta_indices"]) and np.array_equal(a["policy_indices"],b["policy_indices"])
    columns=[]
    for j,name in enumerate(old["family_order"],1):
        columns.append(uncalibrate(a["meta_probability"][:,j],old["tools"][name]["calibrator"]))
    meta=np.c_[logit(np.clip(a["meta_probability"][:,0],1e-6,1-1e-6)),np.asarray(columns).T,
               logit(np.clip(b["meta_raw"],1e-6,1-1e-6))]
    from .agent_tools_v1 import predict_score
    common=dict(np.load(first/"features/common.npz"));values=dict(np.load(first/"features"/f"fold_{subject:02d}.npz"))
    ii=a["policy_indices"];columns=[]
    for name in old["family_order"][:-1]:
        x=values[name] if name in values else common[name]
        columns.append(predict_score(old["tools"][name]["model"],x[ii]))
    columns.append(uncalibrate(a["policy_probability"][:,4],old["tools"]["case_retrieval"]["calibrator"]))
    policy=np.c_[logit(np.clip(a["policy_probability"][:,0],1e-6,1-1e-6)),np.asarray(columns).T,
                 logit(np.clip(b["policy_raw"],1e-6,1-1e-6))]
    return old,a,b,meta,policy


def utility(y,baseline,p):
    c=correction_counts(y,baseline,p)
    return c["corrected"]-2*c["damaged"],c["net"],-c["damaged"]


def gated_prediction(baseline,probability,margin):
    changed=((baseline>=.5)!=(probability>=.5))&(np.abs(probability-.5)>=margin)
    return np.where(changed,probability,baseline)


def fit(out=DEFAULT_OUT,first=FIRST,second=SECOND):
    from threadpoolctl import threadpool_limits
    out,first,second=Path(out),Path(first),Path(second)
    if out.exists():raise FileExistsError("Preserve previous selector rounds")
    out.mkdir(parents=True);(out/"checkpoints").mkdir();(out/"internal_validation").mkdir()
    splits=json.loads((first/"split_manifest.json").read_text(encoding="utf-8"))
    paths=json.loads((first/"local_paths.json").read_text(encoding="utf-8"))
    y=np.asarray([p["label"] for p in paths]);groups=np.asarray([p["subject_key"] for p in paths])
    write_json(out/"protocol.json",dict(configurations=CONFIGS,margins=MARGINS,
        outer_labels_used=False,selection="policy-subject OOF; fit metaOOF+other3policy, evaluate heldout policy subject",
        utility="corrected - 2*damaged; then net gain; never a label-based query gate",
        final_fit="meta5 OOF evidence plus policy4 independent evidence; nine internal subjects",
        code_sha256=sha256(__file__),first_training_sha256=sha256(first/"training_summary.json"),
        second_training_sha256=sha256(second/"training_summary.json")))
    summary=[];begin=time.perf_counter()
    with threadpool_limits(limits=2):
        for n,split in enumerate(splits,1):
            subject=split["outer_subject"]
            old,a,b,meta_x,policy_x=inputs(first,second,subject)
            mi,pi=a["meta_indices"],a["policy_indices"]
            if subject in set(groups[np.r_[mi,pi]]):raise ValueError("Outer entered correction learning")
            baseline=expit(policy_x[:,0]);models=[]
            for config in CONFIGS:
                p=np.full(len(pi),np.nan)
                for heldout in np.unique(groups[pi]):
                    select=groups[pi]!=heldout;test=~select
                    x=np.r_[meta_x,policy_x[select]];yy=np.r_[y[mi],y[pi[select]]]
                    estimator=classifier(config).fit(x,yy)
                    if hasattr(estimator,"decision_function"):
                        p[test]=expit(estimator.decision_function(policy_x[test]))
                    else:p[test]=estimator.predict_proba(policy_x[test])[:,1]
                models.append(p)
            chosen_config,chosen_margin=0,0.
            key=(-10000,-10000,-10000)
            for i,p in enumerate(models):
                for margin in MARGINS:
                    gated=gated_prediction(baseline,p,margin)
                    candidate=utility(y[pi],baseline,gated)
                    if candidate>key:key=candidate;chosen_config,chosen_margin=i,margin
            p=models[chosen_config];gated=gated_prediction(baseline,p,chosen_margin)
            measured=correction_counts(y[pi],baseline,gated)
            model=classifier(CONFIGS[chosen_config]).fit(np.r_[meta_x,policy_x],np.r_[y[mi],y[pi]])
            record=dict(outer_subject=subject,configuration=CONFIGS[chosen_config],margin=chosen_margin,
                policy_subject_oof=binary_metrics(y[pi],p),policy_gated=binary_metrics(y[pi],gated),
                correction=measured,support_subjects=len(np.unique(groups[pi][(baseline>=.5)!=(gated>=.5)])),
                correction_target_met=measured["corrected"]>=2*measured["damaged"] and measured["corrected"]>0,
                limitation="configuration and margin selected on these OOF rows; selection optimism remains",
                query_labels_used=False)
            joblib.dump(dict(model=model,split=split,report=record,feature_names=["baseline"]+old["family_order"]+list(FAMILIES)),
                        out/"checkpoints"/f"fold_{subject:02d}.joblib")
            write_json(out/"internal_validation"/f"fold_{subject:02d}.json",record)
            summary.append(record)
            print(json.dumps(dict(fold=n,total=24,policy_oof_acc=record["policy_gated"]["acc"],correction=measured)),flush=True)
    write_json(out/"training_summary.json",dict(folds=24,elapsed_seconds=time.perf_counter()-begin,outer_evaluated=False,
        mean_policy_gated_acc=float(np.mean([r["policy_gated"]["acc"] for r in summary])),
        correction_totals={k:sum(r["correction"][k] for r in summary) for k in ("corrected","damaged","net")},
        folds_correction_target_met=sum(r["correction_target_met"] for r in summary),
        checkpoint_sha256={p.name:sha256(p) for p in (out/"checkpoints").glob("*.joblib")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    fit(parser.parse_args().out)
