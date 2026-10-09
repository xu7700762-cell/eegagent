"""GPT evaluates learned correction advice alongside the actual EEG tools."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit,logit

from .agent_selector_v3 import DEFAULT_OUT as SELECTOR,gated_prediction
from .agent_tools_v1 import DEFAULT_OUT as FIRST
from .tool_agent_v2 import ExtendedRuntime,EXTENDED_TOOLS as SECOND,TOOLS as PREVIOUS_TOOLS,SYSTEM as PREVIOUS_SYSTEM
from .tool_agent_v1 import SCHEMA as BASE_SCHEMA,call_agent
from .cloud import check_blind,existing_provider,write_json
from .second_judgment import sha256

DEFAULT_OUT=Path("outputs/vrms_agent_loso/v6_gpt_r1_20261008")
SYSTEM=PREVIOUS_SYSTEM+"""
The corrective_evidence tool learns how to combine eight numeric EEG evidence
channels and the original MIL score using nine internal subjects. It acquires
these channels lazily and reports both a selected-margin suggestion and the
raw learned combined class. Configurations and margin were chosen with policy
subject OOF; a further fully nested subject audit is reported separately and
may fail. Do not conflate the selected OOF performance with the nested audit.
Request corrective_evidence to check the combined recommendation before your
final decision, and compare its suggestion with the strongest returned tool
evidence. This learned combination handles correlations and raw-score scaling;
do not replace it with an unweighted count of model votes by default.
When the combined recommendation coherently supports a correction, use its
measured internal correction/damage evidence rather than the original model's
extreme uncalibrated score. A failed nested audit weakens its support; it is not
proof that the original class is correct. You must still choose High or Low.
If the combined model, strongest independent physiological tool and the
baseline conflict, state the conflict and choose using the supplied validation
evidence. No query label, quota or source identity is available.
All tools_used values must be exact executed API tool names, not model names
or descriptions. Never cite an unexecuted tool or include bracketed aliases.
"""
TOOLS=PREVIOUS_TOOLS+[dict(type="function",name="corrective_evidence",strict=True,
    description="Acquire the eight EEG tools and obtain a nine-internal-subject learned correction recommendation, selected margin, and honest nested audit.",
    parameters=dict(type="object",additionalProperties=False,properties={"query":dict(type="string",enum=["qsingle"])},required=["query"]))]
SCHEMA=json.loads(json.dumps(BASE_SCHEMA))
SCHEMA["properties"]["tools_used"]["items"]["enum"]=[t["name"] for t in TOOLS]


class CorrectiveRuntime(ExtendedRuntime):
    def __init__(self,first,second,selector,subject,query_index):
        super().__init__(first,second,subject,query_index)
        selector=Path(selector)
        self.selector_directory=selector
        state=selector/"checkpoints"/f"fold_{subject:02d}.joblib"
        training=json.loads((selector/"training_summary.json").read_text(encoding="utf-8"))
        if sha256(state)!=training["checkpoint_sha256"][state.name]:raise ValueError("Changed correction learner")
        self.selector=joblib.load(state)
        if self.selector["split"]!=self.bundle["split"]:raise ValueError("Wrong corrective roles")
        self.selector_audit=json.loads((selector/"nested_audit"/f"fold_{subject:02d}.json").read_text(encoding="utf-8"))

    def initial_evidence(self):
        result=super().initial_evidence()
        r=self.selector["report"]
        result["tool_validation_profiles"]["corrective_evidence"]=dict(selected_oof_class_validation=r["policy_gated"],
            selected_oof_correction=r["correction"],configuration_selection_optimism=True,
            fully_nested_class_validation=self.selector_audit["nested_validation"],
            fully_nested_correction=self.selector_audit["correction"],
            fully_nested_approved=self.selector_audit["approved"],
            learned_on="meta5 OOF evidence and policy4 independent tool evidence; outer excluded")
        check_blind(result)
        return result

    def execute(self,name):
        if name!="corrective_evidence":return super().execute(name)
        return self.tool(name,self.corrective_result)

    def corrective_result(self):
        names=self.selector["feature_names"][1:]
        values=[self.execute(name) for name in names]
        baseline=float(self.arrays["baseline_probability"][self.query_index])
        features=[float(logit(np.clip(baseline,1e-6,1-1e-6)))]
        for name,value in zip(names,values):
            if name=="case_retrieval":score=float(logit(np.clip(value["descriptive_vote"],1e-5,1-1e-5)))
            else:score=value["raw_decision_score"]
            if name in ("femba_mean","femba_temporal","spectral_compact","filterbank_csp"):
                score=float(logit(np.clip(expit(score),1e-6,1-1e-6)))
            features.append(score)
        model=self.selector["model"]
        x=np.asarray(features)[None]
        probability=float(expit(model.decision_function(x))[0]) if hasattr(model,"decision_function") else float(model.predict_proba(x)[0,1])
        report=self.selector["report"]
        guarded=float(gated_prediction(np.asarray([baseline]),np.asarray([probability]),report["margin"])[0])
        result=dict(tool="corrective_evidence",raw_combined_class="High" if probability>=.5 else "Low",
            uncalibrated_combined_score=probability,score_is_calibrated=False,
            selected_margin_suggestion="High" if guarded>=.5 else "Low",selected_margin=report["margin"],
            internal_selected_oof=dict(validation=report["policy_gated"],correction=report["correction"],selection_optimism=True),
            independent_nested_audit=dict(approved=self.selector_audit["approved"],
                validation=self.selector_audit["nested_validation"],correction=self.selector_audit["correction"]),
            actual_individual_evidence=values,
            scope="nine internal subjects only; no query target; auxiliary numeric model, not a clinical probability")
        check_blind(result)
        return result


def run(out=DEFAULT_OUT,first=FIRST,second=SECOND,selector=SELECTOR,workers=4,probe=False,system=SYSTEM):
    out,first,second,selector=map(Path,(out,first,second,selector));provider=existing_provider()
    protocol=dict(model=provider["model"],base_url=provider["base_url"],system=system,schema=SCHEMA,tools=TOOLS,
        code_sha256=sha256(__file__),client_code_sha256=sha256(Path(__file__).with_name("tool_agent_v1.py")),
        tool_training_sha256=sha256(first/"training_summary.json"),extended_training_sha256=sha256(second/"training_summary.json"),
        selector_training_sha256=sha256(selector/"training_summary.json"),selector_nested_audit_sha256=sha256(selector/"nested_audit/summary.json"),
        acceptance=dict(acc=.70,minimum_correction_damage_ratio=2.),final_classes=["High","Low"],
        query_truth_hidden=True,exploratory=True,no_outer_label_parameter_selection=True)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text(encoding="utf-8"))!=protocol:raise ValueError("Frozen corrective Agent changed")
    write_json(out/"protocol.json",protocol)
    paths=json.loads((first/"local_paths.json").read_text(encoding="utf-8"))
    jobs=[(p["subject_key"],i) for i,p in enumerate(paths)]
    if probe:jobs=jobs[:1]
    def work(job):
        subject,i=job;runtime=CorrectiveRuntime(first,second,selector,subject,i)
        return call_agent(provider,runtime,out/"calls"/f"{i:03d}.json",system=system,tools=TOOLS,schema=SCHEMA)
    begin=time.perf_counter();success=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,future in enumerate(as_completed([pool.submit(work,j) for j in jobs]),1):
            result=future.result();success+=int(result["success"])
            print(json.dumps(dict(completed=n,total=len(jobs),success=success,query_index=result["query_index"],
                actual_tools=result["actual_tool_calls"],seconds=round(result["elapsed_seconds"],1))),flush=True)
    write_json(out/("probe_run.json" if probe else "run.json"),dict(requests=len(jobs),success=success,workers=workers,elapsed_seconds=time.perf_counter()-begin))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--probe",action="store_true")
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();run(args.out,probe=args.probe)
