"""Extended real GPT tools with direction-specific internal correction support."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit

from .agent_tools_v1 import DEFAULT_OUT as FIRST_TOOLS, binary_metrics, correction_counts, predict_score, calibrated
from .agent_tools_v2 import FAMILIES, compact_spectrum, filterbank_covariance
from .cloud import check_blind,existing_provider,write_json
from .second_judgment import sha256
from .tool_agent_v1 import EEGToolRuntime,TOOLS as FIRST_API_TOOLS,SCHEMA,SYSTEM as FIRST_SYSTEM,call_agent
from vrms_pilot.data import spectral_features

DEFAULT_OUT=Path("outputs/vrms_agent_loso/v5_gpt_r1_20261008")
EXTENDED_TOOLS=Path("outputs/vrms_agent_loso/v5_tools_portable_20261008")
SYSTEM=FIRST_SYSTEM+"""
Additional tools include a regularized FEMBA mean classifier, within-path
FEMBA changes, compact regional spectral ratios, and a filter-bank CSP spatial
classifier. CSP is supervised and its filters are refitted inside each internal
subject OOF fold; its query inference does not use query labels.
Validation refers to independent policy subjects. Compare BALANCED accuracy,
AUROC and direction-specific correction support as well as ordinary accuracy;
one-class dominance can make ordinary accuracy look good. The target is a
selective correction of the locked baseline, not unconditional model replacement.
For a competing tool class, direction_specific_support counts historical
policy cases with exactly that baseline-to-tool class disagreement. Corrected
and damaged counts are observed on internal subjects, not the current query.
Prefer a correction with at least twice as many corrected as damaged internal
cases and multiple-subject support; seek another complementary tool if evidence
is sparse or conflicting. Do not mistake a failed small-sample guard for proof
that the baseline is right. Resolve weak evidence with the best available
measured tool support and return High or Low with appropriate low confidence.
Acquire the strongest relevant tools first. Avoid fetching every tool by habit.
The original model cannot be independently recalibrated by your confidence.
"""
EXTRA_DESCRIPTIONS=dict(
    femba_mean="Get a regularized classifier of frozen FEMBA mean embeddings and direction-specific correction evidence.",
    femba_temporal="Get learned evidence from within-path FEMBA mean, variance and late-minus-early feature change.",
    spectral_compact="Calculate regional spectral ratios, reference changes and within-path changes; apply the independently trained classifier.",
    filterbank_csp="Calculate delta/theta/alpha/beta covariances and apply supervised CSP filters learned only on internal subjects.")
TOOLS=FIRST_API_TOOLS+[dict(type="function",name=name,description=description,strict=True,
    parameters=dict(type="object",additionalProperties=False,properties={"query":dict(type="string",enum=["qsingle"])},required=["query"]))
    for name,description in EXTRA_DESCRIPTIONS.items()]


class ExtendedRuntime(EEGToolRuntime):
    def __init__(self, first, extended, subject, query_index):
        super().__init__(first,subject,query_index)
        extended=Path(extended)
        source=extended/"checkpoints"/f"fold_{subject:02d}.joblib"
        training=json.loads((extended/"training_summary.json").read_text(encoding="utf-8"))
        if sha256(source)!=training["checkpoint_sha256"][source.name]:
            raise ValueError("Extended checkpoint changed")
        self.extended=joblib.load(source)
        if self.extended["split"]!=self.bundle["split"]:
            raise ValueError("Different auxiliary subject roles")
        self.extra_arrays=dict(np.load(extended/"features"/f"fold_{subject:02d}.npz"))
        self.extra_common=dict(np.load(extended/"features/common.npz"))
        paths=json.loads((extended/"local_paths.json").read_text(encoding="utf-8"))
        groups=np.asarray([p["subject_key"] for p in paths]);truth=np.asarray([p["label"] for p in paths])
        ii=self.bundle["policy_indices"]
        ii=ii[groups[ii]!=self.query_subject]
        self.internal_indices=ii
        self.internal_groups=groups[ii]
        self.internal_truth=truth[ii]
        self.internal_baseline=self.arrays["baseline_probability"][ii]
        self.internal_raw={};self.internal_cal={}
        for family in list(self.bundle["family_order"][:-1])+list(FAMILIES):
            bank=self.extended["tools"] if family in FAMILIES else self.bundle["tools"]
            matrices={**self.extra_common,**self.extra_arrays} if family in FAMILIES else {**self.common,**self.arrays}
            values=matrices[family][ii]
            score=predict_score(bank[family]["model"],values)
            raw=expit(score);cal=calibrated(bank[family]["calibrator"],score)
            self.internal_raw[family],self.internal_cal[family]=raw,cal
            self.profiles[family]=dict(raw_class_validation=binary_metrics(self.internal_truth,raw),
                raw_correction=correction_counts(self.internal_truth,self.internal_baseline,raw),
                calibrated_class_validation=binary_metrics(self.internal_truth,cal),
                calibrated_correction=correction_counts(self.internal_truth,self.internal_baseline,cal))
        self.baseline_validation=binary_metrics(self.internal_truth,self.internal_baseline)

    def initial_evidence(self):
        result=super().initial_evidence()
        result["baseline"]["independent_policy_validation"]=self.baseline_validation
        result["tool_validation_profiles"]=self.profiles
        result["note"]="Direction-specific correction support is returned only after executing the tool"
        check_blind(result)
        return result

    def direction_support(self, family, new_class, representation):
        if family not in self.internal_raw:
            return dict(status="unavailable")
        p=self.internal_raw[family] if representation=="raw" else self.internal_cal[family]
        bclass=int(self.arrays["baseline_probability"][self.query_index]>=.5)
        cclass=int(new_class=="High")
        if bclass==cclass:
            return dict(status="agrees_with_baseline",cases=0)
        subset=((self.internal_baseline>=.5)==bclass)&((p>=.5)==cclass)
        measured=correction_counts(self.internal_truth[subset],self.internal_baseline[subset],p[subset])
        subjects=len(np.unique(self.internal_groups[subset]))
        return dict(status="measured_internal_conflicts",**measured,cases=int(subset.sum()),
            subjects=subjects,correction_to_damage_target_passed=bool(measured["corrected"]>=2*measured["damaged"] and measured["corrected"]>0),
            limited_subject_support=subjects<2)

    def execute(self,name):
        if name not in EXTRA_DESCRIPTIONS:
            result=super().execute(name)
            if name in self.internal_raw:
                result["direction_specific_support"]=dict(raw=self.direction_support(name,result["raw_learned_class"],"raw"),
                    calibrated=None if result["calibrated_high_probability"] is None else self.direction_support(name,
                        "High" if result["calibrated_high_probability"]>=.5 else "Low","calibrated"))
            return result
        return self.tool(name,lambda:self.extra_result(name))

    def extra_result(self,name):
        bank=self.extended["tools"][name]
        description=None
        if name in ("femba_mean","femba_temporal"):
            values=self.extra_arrays[name][self.query_index]
        elif name=="spectral_compact":
            absolute,powers=zip(*(spectral_features(w) for w in self.x))
            changes=np.asarray([np.log((power+1e-10)/(self.reference_power+1e-10)).ravel() for power in powers]) if self.path["reference_available"] else None
            values=compact_spectrum(absolute,changes)
            description=dict(reference_quality="valid" if changes is not None else "unavailable",physiological_direction_validated=False)
        elif name=="filterbank_csp":
            values=filterbank_covariance(self.x)
            description=dict(bands=["delta","theta","alpha","beta"],spatial_filter_fit="internal subjects only",
                interpretation="learned multiband spatial classifier; no universal physiological direction")
        else:
            raise ValueError("Invalid extra tool")
        matrix=self.extra_arrays[name] if name in self.extra_arrays else self.extra_common[name]
        if not np.allclose(values,matrix[self.query_index],atol=1e-5):
            raise ValueError("Actual lazy feature replay differs")
        raw=float(predict_score(bank["model"],values[None])[0])
        cal=bank["calibrator"]
        p=None if cal is None else float(calibrated(cal,[raw])[0])
        result=dict(tool=name,raw_learned_class="High" if raw>=0 else "Low",raw_decision_score=raw,
            calibrated_high_probability=p,internal_validation=self.profiles[name],description=description,
            direction_specific_support=dict(raw=self.direction_support(name,"High" if raw>=0 else "Low","raw"),
                calibrated=None if p is None else self.direction_support(name,"High" if p>=.5 else "Low","calibrated")))
        check_blind(result)
        return result


def run(out=DEFAULT_OUT,first=FIRST_TOOLS,extended=EXTENDED_TOOLS,workers=4,probe=False,policy=False):
    out,first,extended=Path(out),Path(first),Path(extended)
    provider=existing_provider()
    manifest=dict(model=provider["model"],base_url=provider["base_url"],system=SYSTEM,schema=SCHEMA,tools=TOOLS,
        code_sha256=sha256(__file__),client_code_sha256=sha256(Path(__file__).with_name("tool_agent_v1.py")),
        tool_training_sha256=sha256(first/"training_summary.json"),extended_training_sha256=sha256(extended/"training_summary.json"),
        final_classes=["High","Low"],selection="internal meta5 subject OOF configurations; policy4 profiles",
        baseline_correct=89,baseline_paths=146,acceptance=dict(acc=.70,corrections_at_least_twice_damage=True),
        fallback="locked baseline class on API failure",exploratory=True)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text(encoding="utf-8"))!=manifest:
        raise ValueError("Frozen experiment differs")
    write_json(out/"protocol.json",manifest)
    paths=json.loads((first/"local_paths.json").read_text(encoding="utf-8"))
    splits=json.loads((first/"split_manifest.json").read_text(encoding="utf-8"))
    jobs=[(s["outer_subject"],i) for s in splits for i,p in enumerate(paths)
          if p["subject_key"] in (s["policy_validation"] if policy else [s["outer_subject"]])]
    if probe:jobs=jobs[:1]
    directory=out/("policy_calls" if policy else "calls")
    def work(job):
        subject,i=job
        runtime=ExtendedRuntime(first,extended,subject,i)
        name=f"fold_{subject:02d}_{i:03d}.json" if policy else f"{i:03d}.json"
        return call_agent(provider,runtime,directory/name,system=SYSTEM,tools=TOOLS,schema=SCHEMA)
    begin=time.perf_counter();success=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,future in enumerate(as_completed([pool.submit(work,job) for job in jobs]),1):
            log=future.result();success+=int(log["success"])
            print(json.dumps(dict(completed=n,total=len(jobs),success=success,query_index=log["query_index"],tools=log["actual_tool_calls"],seconds=round(log["elapsed_seconds"],1))),flush=True)
    write_json(out/("policy_run.json" if policy else "probe_run.json" if probe else "run.json"),dict(requests=len(jobs),success=success,workers=workers,elapsed_seconds=time.perf_counter()-begin))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--probe",action="store_true");parser.add_argument("--policy",action="store_true")
    args=parser.parse_args();run(out=args.out,probe=args.probe,policy=args.policy)
