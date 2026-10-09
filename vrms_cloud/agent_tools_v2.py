"""Additional within-path and compact physiological tools, selected internally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy import linalg, signal
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .agent_tools_v1 import (DEFAULT_OUT as FIRST_TOOLS, CONFIGS, binary_metrics, classifier,
    correction_counts, fit_calibrator, indices, predict_score, selection_key, calibrated)
from .cloud import write_json
from .second_judgment import DEFAULT_MODELS, sha256

DEFAULT_OUT = Path("outputs/vrms_agent_loso/v5_tools_20261008")
FAMILIES = ("femba_mean", "femba_temporal", "spectral_compact", "filterbank_csp")
REGIONS = ((0,1), tuple(range(2,9)), tuple(range(9,14)), tuple(range(14,19)), tuple(range(19,27)), (27,28,29))
BANDS = ((.5,4), (4,8), (8,13), (13,30))
CSP_CONFIGS = ("csp2_lr_0.01", "csp2_lr_0.1", "csp2_lr_1", "csp4_lr_0.01", "csp4_lr_0.1", "csp4_lr_1", "csp4_rbf_1", "csp4_rbf_10")


def compact_spectrum(absolute, reference=None):
    a = np.asarray(absolute, float)
    logpower, relative = a[:, :120].reshape(-1,30,4), a[:, 120:].reshape(-1,30,4)
    def regional(values):
        return np.stack([values[:, group, :].mean(axis=1) for group in REGIONS], axis=1)
    logs, rel = regional(logpower), regional(relative)
    n = max(1,len(a)//3)
    changes = np.zeros(24) if reference is None else regional(np.asarray(reference).reshape(-1,30,4)).mean(axis=0).ravel()
    # Ratios reduce overall amplitude differences across subjects, while
    # preserving initial-reference change as a separately learned feature.
    centered_logs = logs-logs.mean(axis=2,keepdims=True)
    return np.r_[rel.mean(axis=0).ravel(), rel.std(axis=0).ravel(), centered_logs.mean(axis=0).ravel(),
                 centered_logs[-n:].mean(axis=0).ravel()-centered_logs[:n].mean(axis=0).ravel(),
                 changes, float(reference is not None)]


def filterbank_covariance(windows):
    result = []
    for low,high in BANDS:
        sos = signal.butter(3, [low,high], btype="bandpass",fs=256,output="sos")
        values = signal.sosfilt(sos, np.asarray(windows,float), axis=-1)
        values -= values.mean(axis=-1,keepdims=True)
        cov = np.einsum("wct,wdt->wcd",values,values)/(values.shape[-1]-1)
        traces = np.trace(cov,axis1=1,axis2=2)
        cov = cov / np.maximum(traces[:,None,None],1e-10)
        result.append(cov.mean(axis=0))
    return np.asarray(result).ravel()


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Supervised spatial filters refitted inside every subject-heldout head."""
    def __init__(self, components=4, shrinkage=.2):
        self.components, self.shrinkage = components, shrinkage

    def fit(self, x, y):
        values=np.asarray(x).reshape(-1,4,30,30)
        if set(np.unique(y)) != {0,1}:
            raise ValueError("CSP requires both training classes")
        self.filters_=[]
        for band in range(4):
            matrices=[]
            for label in (0,1):
                c=values[np.asarray(y)==label,band].mean(axis=0)
                c=(1-self.shrinkage)*c+self.shrinkage*np.trace(c)/30*np.eye(30)
                matrices.append(c)
            _,vectors=linalg.eigh(matrices[1],matrices[0]+matrices[1])
            n=self.components//2
            self.filters_.append(vectors[:,list(range(n))+list(range(30-n,30))].T)
        self.filters_=np.asarray(self.filters_)
        return self

    def transform(self, x):
        values=np.asarray(x).reshape(-1,4,30,30)
        variances=np.einsum("bkc,nbcd,bkd->nbk",self.filters_,values,self.filters_)
        variances=np.maximum(variances,1e-10)
        return np.log(variances/variances.sum(axis=2,keepdims=True)).reshape(len(values),-1)


def estimator(config):
    if not config.startswith("csp"):
        return classifier(config)
    components=int(config.split("_")[0][3:])
    name,value=config.split("_")[1:]
    model=(LogisticRegression(C=float(value),class_weight="balanced",solver="liblinear",max_iter=2000)
           if name=="lr" else SVC(C=float(value),class_weight="balanced",kernel="rbf"))
    return make_pipeline(FilterBankCSP(components),StandardScaler(),model)


def prepare(out=DEFAULT_OUT, first=FIRST_TOOLS, models=DEFAULT_MODELS):
    from vrms_pilot.model_ablation_v3 import load_dataset,load_features,read_json,verify_frozen
    out,first,models=Path(out),Path(first),Path(models)
    if out.exists():
        raise FileExistsError("Keep completed and failed historical rounds")
    frozen=verify_frozen(models)
    paths,splits,windows,_=load_dataset(frozen["pilot_out"])
    pilot=Path(frozen["pilot_out"])
    absolute=np.load(pilot/"cache/bio_absolute.npy",mmap_mode="r")
    reference=np.load(pilot/"cache/bio_reference.npy",mmap_mode="r")
    compact,spatial=[],[]
    for path in paths:
        sl=slice(path["window_start"],path["window_end"])
        compact.append(compact_spectrum(absolute[sl],reference[sl,240:] if path["reference_available"] else None))
        spatial.append(filterbank_covariance(windows[sl]))
    out.mkdir(parents=True)
    (out/"features").mkdir()
    np.savez_compressed(out/"features/common.npz",spectral_compact=np.asarray(compact),filterbank_csp=np.asarray(spatial))
    write_json(out/"split_manifest.json",splits)
    write_json(out/"local_paths.json",paths)
    extraction=read_json(models/"feature_cache/extraction_audit.json")
    for n,(split,record) in enumerate(zip(splits,extraction["records"]),1):
        features=load_features(models,record,"femba_frozen_mil")
        rows,means=[],[]
        for path in paths:
            a=features[path["window_start"]:path["window_end"]]
            chunks=np.array_split(a,3)
            means.append(a.mean(axis=0))
            rows.append(np.r_[a.mean(axis=0),a.std(axis=0),chunks[-1].mean(axis=0)-chunks[0].mean(axis=0)])
        np.savez_compressed(out/"features"/f"fold_{split['outer_subject']:02d}.npz",femba_mean=np.asarray(means),femba_temporal=np.asarray(rows))
        print(f"extended features {n}/24",flush=True)
    write_json(out/"protocol.json",dict(schema_version="extended_tool_v2",frozen_before_fitting=True,
        acceptance=dict(acc=.70,correction_to_damage=2,final_classes=["High","Low"]),
        tool_families=FAMILIES,configurations=CONFIGS,csp_configurations=CSP_CONFIGS,
        csp="all supervised filters refit inside meta subject OOF; no heldout labels in CSP",
        training="base14+other4 meta for subject OOF; final base14+meta5; policy4 independent",
        tuning="configuration selected by meta subject macro accuracy; raw/calibrated comparison reported on policy4",
        baseline_unchanged=True,first_tools_training_sha256=sha256(first/"training_summary.json"),
        code_sha256=sha256(__file__)))


def train(out=DEFAULT_OUT,first=FIRST_TOOLS):
    from threadpoolctl import threadpool_limits
    out,first=Path(out),Path(first)
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    if protocol["code_sha256"]!=sha256(__file__):
        raise ValueError("Frozen extended source changed")
    if (out/"checkpoints").exists():
        raise FileExistsError("Keep prior training")
    (out/"checkpoints").mkdir();(out/"internal_validation").mkdir()
    paths=json.loads((out/"local_paths.json").read_text(encoding="utf-8"))
    y=np.asarray([p["label"] for p in paths]);groups=np.asarray([p["subject_key"] for p in paths])
    common=dict(np.load(out/"features/common.npz"));records=[];begin=time.perf_counter()
    with threadpool_limits(limits=2):
        for n,split in enumerate(json.loads((out/"split_manifest.json").read_text(encoding="utf-8")),1):
            base,meta,policy=[indices(paths,split[k]) for k in ("base_train","meta_calibration","policy_validation")]
            fold=dict(np.load(out/"features"/f"fold_{split['outer_subject']:02d}.npz"))
            baseline=np.load(first/"features"/f"fold_{split['outer_subject']:02d}.npz")["baseline_probability"]
            selected,reports,meta_raw,meta_cal,policy_raw,policy_cal={}, {}, [], [], [], []
            for family in FAMILIES:
                x=fold[family] if family in fold else common[family]
                configs=CSP_CONFIGS if family=="filterbank_csp" else CONFIGS
                results=[]
                for config in configs:
                    score=np.full(len(meta),np.nan)
                    for subject in np.unique(groups[meta]):
                        fit=np.r_[base,meta[groups[meta]!=subject]];query=meta[groups[meta]==subject]
                        model=estimator(config).fit(x[fit],y[fit])
                        score[np.isin(meta,query)]=predict_score(model,x[query])
                    results.append(score)
                chosen=max(range(len(configs)),key=lambda i:(selection_key(y[meta],expit(results[i]),groups[meta]),-i))
                score=results[chosen]
                cal=fit_calibrator(score,y[meta],groups[meta])
                model=estimator(configs[chosen]).fit(x[np.r_[base,meta]],y[np.r_[base,meta]])
                raw=predict_score(model,x[policy]);pc=calibrated(cal,raw)
                selected[family]=dict(model=model,calibrator=cal,configuration=configs[chosen])
                reports[family]=dict(configuration=configs[chosen],meta_oof_raw=binary_metrics(y[meta],expit(score)),
                    raw_policy=binary_metrics(y[policy],expit(raw)),calibrated_policy=binary_metrics(y[policy],pc),
                    raw_correction=correction_counts(y[policy],baseline[policy],expit(raw)),
                    calibrated_correction=correction_counts(y[policy],baseline[policy],pc))
                meta_raw.append(expit(score));meta_cal.append(calibrated(cal,score));policy_raw.append(expit(raw));policy_cal.append(pc)
            write_json(out/"internal_validation"/f"fold_{split['outer_subject']:02d}.json",dict(outer_subject=split["outer_subject"],reports=reports,outer_labels_used=False))
            np.savez_compressed(out/"internal_validation"/f"fold_{split['outer_subject']:02d}.npz",meta_indices=meta,
                meta_raw=np.asarray(meta_raw).T,meta_calibrated=np.asarray(meta_cal).T,policy_indices=policy,
                policy_raw=np.asarray(policy_raw).T,policy_calibrated=np.asarray(policy_cal).T,
                policy_labels=y[policy],policy_subjects=groups[policy])
            joblib.dump(dict(split=split,tools=selected,reports=reports),out/"checkpoints"/f"fold_{split['outer_subject']:02d}.joblib")
            records.append(reports)
            print(json.dumps(dict(fold=n,total=24,policy_raw_acc={name:value["raw_policy"]["acc"] for name,value in reports.items()})),flush=True)
    write_json(out/"training_summary.json",dict(status="completed_extended_internal_tools",folds=24,outer_evaluated=False,
        elapsed_seconds=time.perf_counter()-begin,
        mean_internal_policy_raw_acc={name:float(np.mean([r[name]["raw_policy"]["acc"] for r in records])) for name in FAMILIES},
        checkpoint_sha256={p.name:sha256(p) for p in (out/"checkpoints").glob("*.joblib")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    prepare(args.out) if args.prepare else train(args.out)
