"""Prepare V2 deep/QC evidence; physiological tools remain lazy at runtime."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from vrms_pilot.data import digest, write_json
from vrms_pilot.experiment import DEFAULT_OUT as PILOT_OUT, ToolReplay
from vrms_pilot.model import CompactEEGCNN

DEFAULT_OUT = Path("outputs/vrms_cloud/v2_seed2026")


def rounded(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


class ReplayData:
    """Load only V2 eligible caches and independently fitted frozen components."""
    def __init__(self, pilot_out):
        self.pilot_out = Path(pilot_out)
        self.cache, self.result = self.pilot_out / "cache", self.pilot_out / "results"
        self.audit = json.loads((self.cache / "dataset_audit.json").read_text(encoding="utf-8"))
        if self.audit.get("schema_version") != "raw_event_eligibility_v2":
            raise ValueError("V2 requires new raw-event-complete caches; historical checkpoints cannot be reused")
        self.paths = json.loads((self.cache / "evaluation_manifest.json").read_text(encoding="utf-8"))
        self.splits = json.loads((self.pilot_out / "split_manifest.json").read_text(encoding="utf-8"))
        self.windows = np.load(self.cache / "windows.npy", mmap_mode="r")[:self.audit["accepted_windows"]]
        rows = [json.loads(s) for s in (self.cache / "window_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
        self.starts = np.asarray([r["start_sample"] / 1024 for r in rows])
        self.references = np.load(self.cache / "path_references.npy", mmap_mode="r")
        self.powers = np.load(self.cache / "path_reference_power.npy", mmap_mode="r")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.protocol = json.loads((self.result / "protocol.json").read_text(encoding="utf-8"))
        if self.protocol.get("schema_version") != "pilot_v2":
            raise ValueError("Expected a frozen V2 pilot protocol")
        if len(self.paths) != self.audit["paths"] or self.audit["eligibility_sha256"] != digest(self.cache / "eligibility_audit.json"):
            raise ValueError("Eligibility/cache manifest changed")
        for filename, expected in self.protocol["code_sha256"].items():
            if digest(Path("vrms_pilot") / filename) != expected:
                raise ValueError(f"Frozen V2 source changed: {filename}")
        for filename, expected in self.protocol["external_code_sha256"].items():
            if digest(Path(filename)) != expected:
                raise ValueError(f"Frozen V2 external source changed: {filename}")

    def load_bundle(self, split):
        stem = self.result / "checkpoints" / f"outer_subject_{split['outer_subject']:02d}"
        state = torch.load(str(stem) + ".pt", map_location=self.device, weights_only=True)
        components = joblib.load(str(stem) + ".joblib")
        if components["split"] != split or state["training_subjects"] != split["base_train"]:
            raise ValueError("Frozen subject split changed")
        if "reliability" not in components:
            raise ValueError("Missing independently fitted V2 reliability checkpoint")
        model = CompactEEGCNN().to(self.device)
        model.load_state_dict(state["model"])
        model.eval()
        return dict(cnn=model, mean=state["mean"].to(self.device), scale=state["scale"].to(self.device),
                    **{k: v for k, v in components.items() if k != "split"})

    def replay(self, path_index, bundle):
        path = self.paths[path_index]
        sl = slice(path["window_start"], path["window_end"])
        return ToolReplay(path, self.windows[sl], self.starts[sl], self.references[path_index],
                          self.powers[path_index], bundle, self.device)


def prepare_v2(out=DEFAULT_OUT, pilot_out=PILOT_OUT):
    out = Path(out)
    if (out / "protocol.json").exists() or (out / "fold_evidence.json").exists():
        raise ValueError("Preparation already exists; use a fresh V2 run directory")
    data = ReplayData(pilot_out)
    out.mkdir(parents=True, exist_ok=True)
    folds = []
    for split in data.splits:
        bundle = data.load_bundle(split)
        records = []
        for i, path in enumerate(data.paths):
            s = path["subject_key"]
            role = "outer_test" if s == split["outer_subject"] else "meta" if s in split["meta_calibration"] else "policy_validation" if s in split["policy_validation"] else None
            if role is None:
                continue
            replay = data.replay(i, bundle)
            evidence = replay.evidence_snapshot()
            if any(t in replay.calls for t in ("BiomarkerCalculator", "CovarianceAnalyzer", "CaseRetriever")):
                raise AssertionError("Preparation eagerly computed an optional tool")
            records.append(dict(path_index=i, role=role, evidence=evidence))
        folds.append(dict(outer_subject=split["outer_subject"], split=split, records=records))
        print(f"V2 deep/QC evidence fold {split['outer_subject']:02d}: {len(records)} records", flush=True)
    write_json(out / "fold_evidence.json", folds)
    write_json(out / "evaluation.json", [dict(path_index=i, subject_key=p["subject_key"], label=p["label"]) for i, p in enumerate(data.paths)])
    protocol = dict(schema_version="uncertainty_agent_v2", total_paths=len(data.paths), subjects=data.audit["subjects"],
        pilot_out=str(Path(pilot_out).resolve()), endpoint="whole_path_score_ge30", seed=2026,
        probability_source="independently calibrated deep model only", llm_changes_probability=False,
        tool_execution="ToolReplay lazy; preparation computes deep/QC/temporal descriptors only",
        example_source="5 internal meta subjects; subject-diverse genuine Top K=5",
        partial_run=bool(data.protocol.get("smoke", False)), exploratory=True,
        query_labels_sent=False, raw_eeg_sent=False, source_identifiers_sent=False,
        eligibility_sha256=data.audit["eligibility_sha256"],
        split_manifest_sha256=digest(Path(pilot_out) / "split_manifest.json"),
        source_artifact_sha256={name: digest(data.result / name) for name in ("path_oof.csv", "inner_predictions.csv", "protocol.json")},
        checkpoint_sha256={p.name: digest(p) for p in sorted((data.result / "checkpoints").glob("*"))},
        cache_artifact_sha256={name: digest(data.cache / name) for name in
            ("windows.npy", "path_references.npy", "path_reference_power.npy", "window_evidence.jsonl",
             "evaluation_manifest.json", "dataset_audit.json", "eligibility_audit.json")},
        evidence_sha256=digest(out / "fold_evidence.json"), evaluation_sha256=digest(out / "evaluation.json"),
        implementation_sha256={name: digest(Path(name)) for name in
            ("vrms_refine/retrieval.py", "vrms_deepseek/supervisor.py", "vrms_deepseek/provider.py", "eegagent_config.py")},
        code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")})
    write_json(out / "protocol.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-out", type=Path, default=PILOT_OUT)
    args = parser.parse_args()
    prepare_v2(args.out, args.pilot_out)


if __name__ == "__main__":
    main()
