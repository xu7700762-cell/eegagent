"""Real Responses function-calling Agent with memoized EEG tool execution.

All tool outputs are constructed from unlabelled query EEG and frozen internal
models. The outer target is available only to the separate evaluator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import joblib
import numpy as np
from scipy.special import expit, logit
import torch

from vrms_pilot.experiment import ToolReplay
from vrms_pilot.data import covariance, spectral_features
from .agent_tools_v1 import DEFAULT_OUT as TOOLS_OUT, FAMILIES, calibrated, predict_score
from .cloud import check_blind, existing_provider, write_json
from .second_judgment import sha256

DEFAULT_OUT = Path("outputs/vrms_agent_loso/v4_gpt_r1_20261008")
PILOT = Path("outputs/vrms_pilot/v2_gpt_flow_20261008")
SYSTEM = """You are an EEG evidence Agent making a forced WHOLE-PATH High/Low
classification. High means the end-of-path questionnaire category >=30, Low
means <30; this is not a diagnosis or instantaneous symptom. There is exactly
one anonymous query and no query truth, identity or sequence information.
The original FEMBA+MIL class is a baseline to retain or correct, not a label.
Its uncalibrated extreme score can be wrong. Final decision MUST be High or
Low, never uncertain. Qualitative confidence may be low.
Use the provided callable EEG tools to acquire evidence before deciding. First
request the tools most relevant to the supplied internal-validation profiles.
If needed request additional tools after inspecting the actual results. You
can request combined_evidence to compare all numeric tool results, but case
examples, feature models and fusion are correlated; never count them as fully
independent votes. Aim to correct errors while avoiding damaging correct
baseline predictions. Treat correction/damage counts measured on independent
policy subjects as stronger evidence than an extreme uncalibrated score.
Each tool has a raw learned class and an OOF calibrated probability. Calibration
may shift the decision and perform worse on independent policy subjects; weigh
the actually measured validation of each representation. Do not assume that a
calibrated score guarantees correctness. Per-tool profiles contain small-sample
internal evidence, not guarantees for the current query.
Use independently trained spectral/spatial predictions when they generalize
better internally; physiological descriptions alone have no established
direction. Delta increase, alpha decrease, or larger unsigned covariance
distance do not universally imply High. Missing reference and low QC are
unknown, never evidence for Low. Neighbour fractions are descriptive, not true
probabilities. Distant or mixed neighbours have weak support. Case selection
has no class quotas, and each selected case has a distinct internal subject.
A nested policy correction rule is a useful risk guard when enabled. When it
is unavailable, decide from the measured tool profiles, agreement/conflict and
the query evidence; do not falsely say that a correction was validated.
Normally revise the baseline only if the more reliable tool evidence gives a
coherent competing class. Do not preserve a known weak baseline merely because
its raw probability is near zero or one. Do not make quota-based flips.
Return strict JSON for id=qsingle, final_class=High or Low, confidence, tools_used,
supporting_evidence, conflicting_evidence, missing_evidence and explanation.
Use concise Chinese explanation citing actual returned numeric evidence.
Keep arrays at most three items and explanation under 200 Chinese characters.
Never output a numerical LLM probability or refer to hidden query targets.
"""
SCHEMA = dict(type="object", additionalProperties=False, properties={
    "id": dict(type="string", enum=["qsingle"]),
    "final_class": dict(type="string", enum=["High", "Low"]),
    "confidence": dict(type="string", enum=["low", "medium", "high"]),
    "tools_used": dict(type="array", items=dict(type="string")),
    "supporting_evidence": dict(type="array", items=dict(type="string")),
    "conflicting_evidence": dict(type="array", items=dict(type="string")),
    "missing_evidence": dict(type="array", items=dict(type="string")),
    "explanation": dict(type="string")}, required=["id", "final_class", "confidence", "tools_used",
        "supporting_evidence", "conflicting_evidence", "missing_evidence", "explanation"])
TOOL_DESCRIPTIONS = dict(
    femba_features="Get the alternative frozen FEMBA path feature classifier, raw class and calibrated probability, with independent policy validation.",
    spectral="Calculate this query's actual spectral features and obtain the learned spectral classifier evidence. No universal band-direction rule.",
    spatial_covariance="Calculate normalized covariance features from the actual query EEG and obtain learned spatial classifier evidence.",
    case_retrieval="Retrieve subject-diverse similar internal EEG cases, distances, actual neighbour classes and calibrated case-vote evidence.",
    combined_evidence="Acquire all four numeric tools lazily, compare classes, and return fitted fusion and independently audited correction advice.")
TOOLS = [dict(type="function", name=name, description=description, strict=True,
    parameters=dict(type="object", additionalProperties=False, properties={
        "query": dict(type="string", enum=["qsingle"])}, required=["query"])) for name, description in TOOL_DESCRIPTIONS.items()]


def raw_validation(bundle, arrays, common, local_labels, policy=None):
    policy = bundle["policy_indices"] if policy is None else policy
    from .agent_tools_v1 import binary_metrics, correction_counts
    result = {}
    for family in FAMILIES:
        x = arrays[family] if family == "femba_features" else common[family]
        raw = expit(predict_score(bundle["tools"][family]["model"], x[policy]))
        calibrated_p = calibrated(bundle["tools"][family]["calibrator"], predict_score(bundle["tools"][family]["model"], x[policy]))
        result[family] = dict(raw_class_validation=binary_metrics(local_labels[policy], raw),
            raw_correction=correction_counts(local_labels[policy], arrays["baseline_probability"][policy], raw),
            calibrated_class_validation=binary_metrics(local_labels[policy], calibrated_p),
            calibrated_correction=correction_counts(local_labels[policy], arrays["baseline_probability"][policy], calibrated_p))
    for family in ("case_retrieval", "fusion", "equal_average"):
        result[family] = dict(calibrated_class_validation=bundle["reports"][family]["policy"],
                              calibrated_correction=bundle["reports"][family]["policy_correction"])
    return result


class EEGToolRuntime(ToolReplay):
    """Reuse ToolReplay.tool so an unrequested classifier/EEG operation never runs."""
    def __init__(self, tools_out, subject, query_index):
        tools_out = Path(tools_out)
        paths = json.loads((tools_out / "local_paths.json").read_text(encoding="utf-8"))
        # Strip query labels before assigning runtime state.
        self.path = {k: paths[query_index][k] for k in ("window_start", "window_end", "accepted_windows", "total_possible_windows", "reference_available")}
        self.query_index, self.subject = query_index, subject
        self.query_subject = paths[query_index]["subject_key"]
        self.x = np.load(PILOT / "cache/windows.npy", mmap_mode="r")[self.path["window_start"]:self.path["window_end"]]
        self.device = torch.device("cpu")
        self.memo, self.calls, self.timings = {}, [], {}
        checkpoint = tools_out / "checkpoints" / f"fold_{subject:02d}.joblib"
        training = json.loads((tools_out / "training_summary.json").read_text(encoding="utf-8"))
        if sha256(checkpoint) != training["checkpoint_sha256"][checkpoint.name]:
            raise ValueError("Frozen tool checkpoint changed")
        self.bundle = joblib.load(checkpoint)
        if subject != self.bundle["split"]["outer_subject"] or subject in self.bundle["split"]["base_train"] + self.bundle["split"]["meta_calibration"]:
            raise ValueError("Wrong query fold")
        self.arrays = dict(np.load(tools_out / "features" / f"fold_{subject:02d}.npz"))
        self.common = dict(np.load(tools_out / "features/common.npz"))
        # Profiles are prepared from policy4 labels only, never query labels.
        yy = np.asarray([p["label"] for p in paths])
        gg = np.asarray([p["subject_key"] for p in paths])
        allowed = self.bundle["split"]["policy_validation"] + [subject]
        if self.query_subject not in allowed:
            raise ValueError("Only policy/outer subjects can be queried")
        policy_indices = self.bundle["policy_indices"]
        self.profiles = raw_validation(self.bundle, self.arrays, self.common, yy)
        self.policy = json.loads((tools_out / "policies" / f"fold_{subject:02d}.json").read_text(encoding="utf-8"))
        if self.query_subject != subject:
            # Internal GPT probes must also exclude their query subject from
            # all labelled validation statistics and candidate selection.
            from .agent_policy_v1 import audit_rule, candidates
            from .agent_tools_v1 import binary_metrics, correction_counts
            values = np.load(tools_out / "internal_validation" / f"fold_{subject:02d}.npz")
            selected = values["policy_subjects"] != self.query_subject
            ii = values["policy_indices"][selected]
            p = values["policy_probability"][selected]
            fusion = values["policy_fusion"][selected]
            self.profiles = raw_validation(self.bundle, self.arrays, self.common, yy, ii)
            for name, score in [("case_retrieval", p[:, 4]), ("fusion", fusion), ("equal_average", p[:, 1:].mean(axis=1))]:
                self.profiles[name] = dict(calibrated_class_validation=binary_metrics(yy[ii], score),
                    calibrated_correction=correction_counts(yy[ii], p[:, 0], score))
            self.policy = audit_rule(yy[ii], p[:, 0], candidates(p, fusion), gg[ii])
        self.reference_changes = None
        if self.path["reference_available"]:
            self.reference_power = np.load(PILOT / "cache/path_reference_power.npy", mmap_mode="r")[query_index]

    def initial_evidence(self):
        raw = float(self.arrays["baseline_probability"][self.query_index])
        cal = self.bundle["tools"]["deep_calibrator"]
        probability = None if cal is None else float(cal.predict_proba([[logit(np.clip(raw, 1e-6, 1-1e-6))]])[0, 1])
        evidence = dict(id="qsingle", baseline=dict(raw_high_probability=raw, raw_class="High" if raw >= .5 else "Low",
            calibrated_high_probability=probability), signal_quality=dict(accepted_windows=len(self.x),
                accepted_window_fraction=len(self.x) / self.path["total_possible_windows"],
                reference_quality="valid" if self.path["reference_available"] else "unavailable"),
            tool_validation_profiles=self.profiles, correction_guard=dict(enabled=self.policy["enabled"],
                nested_correction=self.policy["nested_correction"], rule=self.policy["rule"],
                measured_on="four internal policy subjects; nested subject leave-one-out"))
        check_blind(evidence)
        return evidence

    def _result(self, family, values, description=None):
        tool = self.bundle["tools"][family]
        raw = float(predict_score(tool["model"], values[None])[0])
        cal = tool["calibrator"]
        result = dict(tool=family, raw_learned_class="High" if raw >= 0 else "Low", raw_decision_score=raw,
            calibrated_high_probability=None if cal is None else float(calibrated(cal, [raw])[0]),
            calibration_usable=cal is not None, internal_validation=self.profiles[family],
            prediction_scope="whole-path questionnaire category, not clinical diagnosis")
        if description is not None:
            result["description"] = description
        check_blind(result)
        return result

    def execute(self, name):
        if name not in TOOL_DESCRIPTIONS:
            raise ValueError("Unknown EEG tool")
        return self.tool(name, lambda: self._execute(name))

    def _execute(self, name):
        if name == "femba_features":
            # The expensive frozen encoder cache may be shared; alternative
            # head inference occurs only when requested, not during preparation.
            return self._result(name, self.arrays[name][self.query_index])
        if name == "spectral":
            absolute, powers = zip(*(spectral_features(w) for w in self.x))
            a = np.asarray(absolute, float)
            n = max(1, len(a) // 3)
            changes = np.asarray([np.log((power + 1e-10) / (self.reference_power + 1e-10)).ravel() for power in powers]) if self.path["reference_available"] else None
            x = np.r_[a.mean(axis=0), a.std(axis=0), a[-n:, :120].mean(axis=0)-a[:n, :120].mean(axis=0),
                      np.zeros(120) if changes is None else changes.mean(axis=0), float(changes is not None)]
            if not np.allclose(x, self.common[name][self.query_index], atol=1e-5):
                raise ValueError("Lazy spectral replay differs from tool training feature")
            description = dict(relative_band_power=a[:, 120:].mean(axis=0).reshape(30, 4).mean(axis=0).tolist(),
                reference_log_change=None if changes is None else changes.mean(axis=0).reshape(30, 4).mean(axis=0).tolist(),
                reference_quality="valid" if changes is not None else "unavailable", descriptive_direction_validated=False)
            return self._result(name, x, description)
        if name == "spatial_covariance":
            covariances = np.asarray([covariance(w) for w in self.x])
            scales = np.trace(covariances, axis1=1, axis2=2) / 30
            mean_cov = (covariances / scales[:, None, None]).mean(axis=0)
            e, u = np.linalg.eigh(mean_cov)
            log_cov = (u * np.log(np.maximum(e, 1e-10))) @ u.T
            ii, jj = np.triu_indices(30)
            x = np.r_[log_cov[ii, jj] * np.where(ii == jj, 1., np.sqrt(2.)), np.log(scales).mean(),
                      np.log(scales).std(), np.log(np.diagonal(covariances, axis1=1, axis2=2)).mean(axis=0)]
            if not np.allclose(x, self.common[name][self.query_index], atol=1e-8):
                raise ValueError("Lazy covariance replay differs")
            return self._result(name, x, dict(descriptive_direction_validated=False,
                normalized_log_covariance_norm=float(np.linalg.norm(log_cov)),
                interpretation="learned spatial classifier; covariance magnitude alone has no class direction"))
        if name == "case_retrieval":
            bank = self.bundle["tools"][name]
            reference = bank["scaler"].transform(bank["reference_features"])
            query = bank["scaler"].transform(self.arrays["femba_mean"][[self.query_index]])[0]
            distances = np.sqrt(np.mean((reference-query) ** 2, axis=1))
            order = np.argsort(distances, kind="stable")
            seen, chosen = set(), []
            k, weighted = bank["config"]
            for i in order:
                subject = int(bank["reference_subjects"][i])
                if subject not in seen:
                    chosen.append(int(i)); seen.add(subject)
                if len(chosen) == k:
                    break
            if self.query_subject in seen:
                raise ValueError("The query subject is in the case bank")
            labels = bank["reference_labels"][chosen]
            weights = 1 / np.maximum(distances[chosen], 1e-6) if weighted else np.ones(len(chosen))
            vote = float(np.average(labels, weights=weights))
            cal = bank["calibrator"]
            result = dict(tool=name, k=k, different_subjects=len(seen), high_count=int(labels.sum()),
                low_count=int(len(labels)-labels.sum()), closest_distance=float(distances[chosen[0]]),
                selected_cases=[dict(distance=float(distances[i]), observed_class="High" if bank["reference_labels"][i] else "Low") for i in chosen],
                descriptive_vote=vote, vote_is_calibrated_probability=False,
                raw_learned_class="High" if vote >= .5 else "Low",
                calibrated_high_probability=None if cal is None else float(calibrated(cal, [logit(np.clip(vote, 1e-5, 1-1e-5))])[0]),
                internal_validation=self.profiles[name], reliable_distance_range="unvalidated")
            check_blind(result)
            return result
        if name == "combined_evidence":
            values = [self.execute(tool) for tool in list(FAMILIES) + ["case_retrieval"]]
            probabilities = [v["calibrated_high_probability"] for v in values]
            # Match the training representation if calibration was unusable;
            # a sigmoid score is explicitly uncalibrated, never claimed reliable.
            numeric = [p if p is not None else expit(v.get("raw_decision_score", logit(np.clip(v.get("descriptive_vote", .5), 1e-5, 1-1e-5)))) for p,v in zip(probabilities, values)]
            baseline = float(self.arrays["baseline_probability"][self.query_index])
            fusion = float(self.bundle["tools"]["fusion"].predict_proba(logit(np.clip([[baseline] + numeric], .001, .999)))[0, 1])
            from .agent_policy_v1 import apply_rule
            alternatives = np.asarray([numeric + [float(np.mean(numeric)), fusion]])
            guarded = float(apply_rule(np.asarray([baseline]), alternatives, self.policy["rule"])[0])
            result = dict(tool=name, evidence=values, fitted_fusion_high_probability=fusion,
                fusion_validation=self.profiles["fusion"], equal_average_validation=self.profiles["equal_average"],
                nested_guard_enabled=self.policy["enabled"], nested_guard_suggested_class="High" if guarded >= .5 else "Low",
                nested_guard_correction=self.policy["nested_correction"],
                warning="tools/cases share EEG and are correlated; fusion is an auxiliary numerical control")
            check_blind(result)
            return result
        raise ValueError("Unimplemented tool")


def validate_final(value, actual_calls):
    if not isinstance(value, dict) or set(value) != set(SCHEMA["required"]):
        raise ValueError("Invalid final fields")
    if value["id"] != "qsingle" or value["final_class"] not in ("High", "Low") or value["confidence"] not in ("low", "medium", "high"):
        raise ValueError("Final class must be High/Low for one anonymous query")
    for name in ("tools_used", "supporting_evidence", "conflicting_evidence", "missing_evidence"):
        if not isinstance(value[name], list) or any(not isinstance(x, str) for x in value[name]):
            raise ValueError("Invalid evidence array")
    if not value["tools_used"] or not set(value["tools_used"]).issubset(set(actual_calls)):
        raise ValueError("Final decision cites an unexecuted tool")
    if not isinstance(value["explanation"], str) or not value["explanation"].strip():
        raise ValueError("Missing decision explanation")
    return value


def call_agent(provider, runtime, destination, *, system=SYSTEM, tools=TOOLS, schema=SCHEMA):
    destination = Path(destination)
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    initial = runtime.initial_evidence()
    conversation = [dict(role="system", content=system), dict(role="user", content=json.dumps(initial, ensure_ascii=False, separators=(",", ":")))]
    log = dict(query_index=runtime.query_index, model=provider["model"], base_url=provider["base_url"], initial=initial,
               success=False, rounds=[], actual_tool_calls=[], tool_timings_ms={})
    begin = time.perf_counter()
    for step in range(4):
        tools_allowed = step < 3
        body = dict(model=provider["model"], input=conversation, store=False,
            reasoning=dict(effort="medium"), max_output_tokens=4096,
            text=dict(format=dict(type="json_schema", name="eeg_agent_final", schema=schema, strict=True)))
        if tools_allowed:
            body.update(tools=tools, parallel_tool_calls=True, tool_choice="required" if step == 0 else "auto")
        round_log = dict(step=step + 1, body=json.loads(json.dumps(body)), attempts=[])
        raw = None
        for attempt in range(2):
            started = time.perf_counter()
            try:
                request = urllib.request.Request(provider["base_url"] + "/responses", data=json.dumps(body,ensure_ascii=False).encode(),
                    headers={"Authorization": "Bearer " + provider["key"], "Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = json.load(response)
                if raw.get("status") != "completed":
                    raise ValueError("Agent response incomplete")
                round_log["attempts"].append(dict(attempt=attempt+1, status="success", seconds=time.perf_counter()-started, response=raw))
                break
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, KeyError, TypeError) as exc:
                error = dict(attempt=attempt+1, status="failure", seconds=time.perf_counter()-started, error_type=type(exc).__name__)
                if isinstance(exc, urllib.error.HTTPError):
                    error["http_status"] = exc.code
                round_log["attempts"].append(error)
                raw = None
                if attempt == 0:
                    time.sleep(1)
        log["rounds"].append(round_log)
        if raw is None:
            break
        tool_calls = [x for x in raw.get("output", []) if x.get("type") == "function_call"]
        if tool_calls:
            conversation.extend(raw["output"])
            for call in tool_calls:
                try:
                    arguments = json.loads(call["arguments"])
                    if arguments != {"query": "qsingle"}:
                        raise ValueError("Invalid query tool arguments")
                    result = runtime.execute(call["name"])
                except (ValueError, KeyError, TypeError) as exc:
                    result = dict(tool=call.get("name", "unknown"), available=False, error_type=type(exc).__name__)
                check_blind(result)
                conversation.append(dict(type="function_call_output", call_id=call["call_id"],
                    output=json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)))
            continue
        text = "".join(p["text"] for x in raw.get("output", []) for p in x.get("content", []) if p.get("type") == "output_text")
        try:
            final = validate_final(json.loads(text), runtime.calls)
            log.update(success=True, final=final)
        except (ValueError, KeyError, TypeError) as exc:
            log["final_error_type"] = type(exc).__name__
        break
    log.update(actual_tool_calls=runtime.calls, tool_timings_ms=runtime.timings, elapsed_seconds=time.perf_counter()-begin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, log)
    return log


def run(out=DEFAULT_OUT, tools_out=TOOLS_OUT, workers=4, probe=False, policy=False):
    out, tools_out = Path(out), Path(tools_out)
    provider = existing_provider()
    manifest = dict(model=provider["model"], base_url=provider["base_url"], schema=SCHEMA, system=SYSTEM, tools=TOOLS,
        reasoning_effort="medium", maximum_rounds=4, code_sha256=sha256(__file__),
        tool_training_sha256=sha256(tools_out / "training_summary.json"),
        numeric_policy_sha256=sha256(tools_out / "policies/manifest.json"), final_classes=["High", "Low"],
        outer_labels_sent=False, fallback="original locked baseline class after failed API")
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Frozen Agent protocol differs; use a new directory")
    write_json(protocol, manifest)
    paths = json.loads((tools_out / "local_paths.json").read_text(encoding="utf-8"))
    splits = json.loads((tools_out / "split_manifest.json").read_text(encoding="utf-8"))
    jobs = [(split["outer_subject"], i) for split in splits for i,p in enumerate(paths)
            if p["subject_key"] in (split["policy_validation"] if policy else [split["outer_subject"]])]
    if probe:
        jobs = jobs[:1]
    directory = out / ("policy_calls" if policy else "calls")
    def work(job):
        subject, i = job
        path = directory / (f"fold_{subject:02d}_{i:03d}.json" if policy else f"{i:03d}.json")
        runtime = EEGToolRuntime(tools_out, subject, i)
        return call_agent(provider, runtime, path)
    begin = time.perf_counter()
    success = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            log = future.result()
            success += int(log["success"])
            print(json.dumps(dict(completed=number, total=len(jobs), successful=success,
                query_index=log["query_index"], actual_tools=log["actual_tool_calls"],
                seconds=round(log["elapsed_seconds"], 1))), flush=True)
    write_json(out / ("policy_run.json" if policy else "probe_run.json" if probe else "run.json"),
               dict(requests=len(jobs), successful=success, workers=workers, elapsed_seconds=time.perf_counter()-begin))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tools-out", type=Path, default=TOOLS_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--policy", action="store_true")
    args = parser.parse_args()
    run(args.out, args.tools_out, args.workers, args.probe, args.policy)


if __name__ == "__main__":
    main()
