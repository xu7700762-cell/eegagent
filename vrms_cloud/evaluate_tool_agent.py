"""Audit actual function calls and paired 146-path forced-binary decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .agent_tools_v1 import DEFAULT_OUT as TOOLS_OUT
from .cloud import check_blind, write_json
from .evaluate_second_judgment import classification_metrics, paired_subject_ci
from .second_judgment import sha256
from .tool_agent_v1 import DEFAULT_OUT, validate_final


def normalize_tool_citations(value, actual_calls):
    """Expand only explicit slash-separated names of actually executed tools."""
    result=dict(value)
    result["tools_used"]=[]
    for name in value["tools_used"]:
        parts=name.split(" / ")
        if any(part not in actual_calls for part in parts):
            raise ValueError("Cannot normalize an unexecuted or ambiguous citation")
        result["tools_used"].extend(parts)
    return validate_final(result,actual_calls)


def evaluate(out=DEFAULT_OUT, tools_out=TOOLS_OUT):
    out,tools_out=Path(out),Path(tools_out)
    paths=json.loads((tools_out/"local_paths.json").read_text(encoding="utf-8"))
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    if sha256(tools_out/"training_summary.json")!=protocol["tool_training_sha256"]:
        raise ValueError("Changed numeric training")
    rows,tool_counts,patterns,usage=[],Counter(),Counter(),Counter()
    response_models=Counter();attempts=failures=rounds=0
    for i,path in enumerate(paths):
        destination=out/"calls"/f"{i:03d}.json"
        if not destination.exists():
            raise ValueError("All 146 actual calls must finish before scoring")
        main_log=json.loads(destination.read_text(encoding="utf-8"))
        log=main_log
        recovery=out/"recovery"/f"{i:03d}.json"
        if not main_log["success"] and recovery.exists():
            recovered=json.loads(recovery.read_text(encoding="utf-8"))
            if recovered["initial"]!=main_log["initial"]:
                raise ValueError("Recovery changed the original anonymous input")
            if recovered["success"]:log=recovered
        repaired=out/"citation_repair"/f"{i:03d}.json"
        if not log["success"] and repaired.exists():
            fixed=json.loads(repaired.read_text(encoding="utf-8"))
            if fixed["initial"]!=main_log["initial"] or fixed.get("citation_normalization") is not True:
                raise ValueError("Invalid citation-only repair")
            log=fixed
        if log["query_index"]!=i or log["model"]!=protocol["model"] or log["base_url"]!=protocol["base_url"]:
            raise ValueError("Wrong API pairing")
        check_blind(log["initial"])
        raw=log["initial"]["baseline"]["raw_high_probability"]
        source=np.load(tools_out/"features"/f"fold_{path['subject_key']:02d}.npz")["baseline_probability"][i]
        if raw!=source:
            raise ValueError("Locked baseline changed")
        final=None
        recorded_functions=[]
        for r in log["rounds"]:
            rounds+=1
            body=r["body"]
            if body["store"] is not False or body["input"][0]["content"]!=protocol["system"]:
                raise ValueError("Stateless/system contract failed")
            initial=json.loads(body["input"][1]["content"])
            if initial!=log["initial"]:
                raise ValueError("Different query in a later API round")
            for item in body["input"][2:]:
                if item.get("type")=="function_call_output":
                    check_blind(json.loads(item["output"]))
            for a in r["attempts"]:
                attempts+=1
                if a["status"]!="success":
                    failures+=1;continue
                response=a["response"]
                response_models[response.get("model","missing")]+=1
                for key in ("input_tokens","output_tokens","total_tokens"):
                    usage[key]+=response.get("usage",{}).get(key,0)
                calls=[x for x in response.get("output",[]) if x.get("type")=="function_call"]
                for c in calls:
                    if json.loads(c["arguments"])!={"query":"qsingle"}:
                        raise ValueError("Invalid tool arguments")
                    recorded_functions.append(c["name"])
                if not calls:
                    text="".join(part["text"] for item in response.get("output",[]) for part in item.get("content",[]) if part.get("type")=="output_text")
                    if text:
                        try:final=json.loads(text)
                        except ValueError:
                            if log["success"]:raise
        if log["success"]:
            if log.get("citation_normalization"):
                final=normalize_tool_citations(final,log["actual_tool_calls"])
            if validate_final(final,log["actual_tool_calls"])!=log["final"] or not recorded_functions:
                raise ValueError("Decision has no actual GPT tool call")
            decision=int(final["final_class"]=="High")
        else:
            decision=int(raw>=.5)
        tool_counts.update(log["actual_tool_calls"])
        patterns[" -> ".join(log["actual_tool_calls"])]+=1
        rows.append(dict(path_index=i,subject_key=path["subject_key"],true_class=path["label"],
            baseline_probability=raw,baseline_class=int(raw>=.5),agent_class=decision,
            successful=log["success"],tools=";".join(log["actual_tool_calls"]),
            confidence=final["confidence"] if log["success"] else None,
            explanation=final["explanation"] if log["success"] else "API failed; locked binary baseline fallback",
            result_source="citation_repair" if log.get("citation_normalization") else "recovery" if log is not main_log else "main"))
    y=np.asarray([r["true_class"] for r in rows]);a=np.asarray([r["baseline_class"] for r in rows]);b=np.asarray([r["agent_class"] for r in rows])
    groups=np.asarray([r["subject_key"] for r in rows])
    if len(rows)!=146 or sum(a==y)!=89:
        raise ValueError("The locked 89/146 baseline was not reproduced")
    corrected=int(((a!=y)&(b==y)).sum());damaged=int(((a==y)&(b!=y)).sum())
    metrics=classification_metrics(y,b)
    success=sum(r["successful"] for r in rows)
    achieved=metrics["accuracy"]>=.70 and corrected>=2*damaged and corrected>0 and success==146
    # Citation-repair files copy responses and do not represent extra API calls.
    total_usage=Counter();total_models=Counter();total_attempts=total_rounds=total_failures=0
    for directory in ("calls","recovery"):
        for file in (out/directory).glob("*.json"):
            saved=json.loads(file.read_text(encoding="utf-8"))
            for step in saved["rounds"]:
                total_rounds+=1
                for attempt in step["attempts"]:
                    total_attempts+=1
                    if attempt["status"]!="success":total_failures+=1;continue
                    response=attempt["response"];total_models[response.get("model","missing")]+=1
                    for key in ("input_tokens","output_tokens","total_tokens"):
                        total_usage[key]+=response.get("usage",{}).get(key,0)
    summary=dict(status="completed_forced_binary_tool_agent",baseline=classification_metrics(y,a),agent=metrics,
        changed=int(sum(a!=b)),corrected=corrected,damaged=damaged,net=corrected-damaged,
        successful_api_paths=success,fallback_paths=146-success,final_classes=["High","Low"],uncertain_count=0,
        goal=dict(acc_target=.70,minimum_correction_damage_ratio=2,metrics_pass=achieved,
                  completion_requires_protocol_and_tool_audit=True),
        paired_subject_bootstrap=paired_subject_ci(y,a,b,groups),tool_counts=dict(tool_counts),
        actual_tool_sequences=dict(patterns),api=dict(rounds=total_rounds,attempts=total_attempts,failures=total_failures,
            models=dict(total_models),usage=dict(total_usage),
            selected_result_records=dict(rounds=rounds,attempts=attempts,failures=failures,models=dict(response_models),usage=dict(usage))),
        caveats=["exploratory repeatedly examined LOSO data","auxiliary tools use19 subjects vs locked baseline14; report numeric tool controls",
                 "local pretraining overlap not excluded","qualitative GPT confidence is not calibrated"],
        audit=dict(anonymous_query_checked=True,tool_outputs_blind=True,actual_function_calls_reparsed=True,
                   all_classes_binary=True,unchanged_baseline=True,protocol_sha256=sha256(out/"protocol.json"),
                   scorer_sha256=sha256(__file__)))
    write_json(out/"summary.json",summary)
    with (out/"path_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--tools-out",type=Path,default=TOOLS_OUT)
    args=parser.parse_args();evaluate(args.out,args.tools_out)
