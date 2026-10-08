"""Audit request sources and calibration-crossfit subject provenance."""
import argparse
import copy
from pathlib import Path

from vrms_cloud.cloud import check_blind, choose_examples
from .common import CLOUD, DEFAULT_OUT, job_hash, message_for, read_json, sha, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    protocol=read_json(args.out/"protocol.json")
    audit=read_json(args.out/"preparation_audit.json")
    jobs=read_json(args.out/"jobs.json")
    folds={f["outer_subject"]:f for f in read_json(CLOUD/"fold_evidence.json")}
    evaluation={r["path_index"]:r for r in read_json(CLOUD/"evaluation.json")}
    diagnostics={d["outer_subject"]:d for d in audit["diagnostics"]}
    keys={(j["mode"],j["outer_subject"],j["path_index"]) for j in jobs}
    if len(jobs)!=1309 or len(keys)!=1309:
        raise ValueError("Duplicate/missing jobs")
    checked_crossfit=0
    for s,fold in folds.items():
        split=fold["split"]
        partitions=[set(split[r]) for r in ("base_train","meta_calibration","policy_validation","outer_test")]
        if sum(map(len,partitions))!=24 or len(set.union(*partitions))!=24 or partitions[-1]!={s}:
            raise ValueError("Subject partitions overlap")
        d=diagnostics[s]
        for model,cf in d["crossfit"].items():
            for f in cf["folds"]:
                target=f["heldout_subject"]
                expected_train=partitions[1]-{target}
                if set(f["training_subjects"])!=expected_train:
                    raise ValueError("Calibration head saw its heldout subject")
                if any(evaluation[i]["subject_key"] not in expected_train for i in f["training_indices"]):
                    raise ValueError("Calibration training source mismatch")
                if any(evaluation[i]["subject_key"]!=target for i in f["target_indices"]):
                    raise ValueError("Calibration target source mismatch")
                checked_crossfit+=1
    for job in jobs:
        import json
        message=json.loads(job["message"])
        if len(message["queries"])!=1 or message["queries"][0]["id"]!="qsingle":
            raise ValueError("Other queries entered a single-path request")
        check_blind(message["queries"])
        for example in message["examples"]:
            check_blind(example["evidence"])
        if job_hash(job["message"])!=job["message_sha256"]:
            raise ValueError("Job hash changed")
        fold=folds[job["outer_subject"]]
        record=next(r for r in fold["records"] if r["path_index"]==job["path_index"])
        if record["role"]!=job["role"] or evaluation[job["path_index"]]["subject_key"]!=job["subject_key"]:
            raise ValueError("Job role/subject mismatch")
        query=copy.deepcopy(message["queries"][0]["evidence"])
        if job["mode"]=="baseline":
            if job["role"]!="policy_validation":
                raise ValueError("Baseline test was redundantly requested")
            examples=[dict(evidence=r["evidence"],observed_class="high" if evaluation[r["path_index"]]["label"] else "low")
                      for r in choose_examples(fold["records"],evaluation,2026+fold["outer_subject"])]
            if job["message"]!=message_for(record["evidence"],examples):
                raise ValueError("Baseline no longer matches the old independent context")
        else:
            context=query.pop("analysis_context")
            if context["model_reliability"]!=diagnostics[job["outer_subject"]]["model_reliability"]:
                raise ValueError("Reliability differs across query subjects")
            provenance=next(r for r in diagnostics[job["outer_subject"]]["contexts"] if r["path_index"]==job["path_index"])
            indices=provenance["selected_example_indices"]
            if len(indices)!=len(message["examples"]):
                raise ValueError("Example mapping changed")
            for i,example in zip(indices,message["examples"]):
                if evaluation[i]["subject_key"] not in fold["split"]["meta_calibration"]:
                    raise ValueError("Retrieved example is outside meta subjects")
                if example["observed_class"]!=("high" if evaluation[i]["label"] else "low"):
                    raise ValueError("Example class changed")
                source=next(r["evidence"] for r in fold["records"] if r["path_index"]==i)
                for name in ("temporal","relative_band_power","reference_log_change","covariance_distance","reference_available","qc_accepted_fraction"):
                    if example["evidence"][name]!=source[name]:
                        raise ValueError("Reference EEG features changed")
        if query!=record["evidence"]:
            raise ValueError("Underlying query measurements changed")
    for name,expected in protocol["original_sha256"].items():
        if sha(name)!=expected:
            raise ValueError("A protected original changed")
    result=dict(status="passed",single_query_jobs_checked=1309,calibration_crossfit_partitions_checked=checked_crossfit,
                raw_query_measurements_unchanged=True,reference_subject_isolation_checked=True,
                reliability_metric_labels_meta_only=True,
                reliability_conditional_on_frozen_models_and_prior_numeric_selection=True,
                baseline_context_reproduced=True,
                protected_original_files_checked=len(protocol["original_sha256"]),
                plan_sha256=sha(args.out/"analysis_plan.json"),jobs_sha256=sha(args.out/"jobs.json"))
    write_json(args.out/"structural_validation.json",result)
    print(__import__("json").dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
