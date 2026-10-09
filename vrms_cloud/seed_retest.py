"""One predeclared seed repeat, preserving the original experiment files."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from vrms_pilot import model_ablation_v3 as mil
from vrms_pilot.experiment import save_csv
from vrms_pilot.model import seed_everything
from . import agent_tools_v1 as first_tools
from . import agent_tools_v2 as extra_tools
from .agent_policy_v1 import freeze as freeze_policy
from .agent_selector_v3 import fit as fit_selector
from .agent_selector_audit_v3 import audit as audit_selector
from .cloud import write_json
from .evaluate_second_judgment import classification_metrics
from .second_judgment import DEFAULT_MODELS, sha256
from .tool_agent_v2 import EXTENDED_TOOLS
from .tool_agent_v3 import run as run_agent
from .tool_agent_v4 import DEFAULT_OUT as OLD_GPT, SYSTEM

DEFAULT_OUT = Path("outputs/vrms_agent_seed_retest/seed2027_20261008")
SOURCE_NAMES = ("seed_retest.py", "agent_tools_v1.py", "agent_tools_v2.py", "agent_policy_v1.py",
                "agent_selector_v3.py", "agent_selector_audit_v3.py", "tool_agent_v1.py",
                "tool_agent_v2.py", "tool_agent_v3.py", "tool_agent_v4.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_training_seed(seed):
    # These are explicit process-local overrides, recorded in the protocol.
    # No source, pretrained tensor, old checkpoint or split is modified.
    mil.SEED = first_tools.SEED = int(seed)
    seed_everything(int(seed))


def freeze_protocol(out, seed, scope):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a fresh seed output; keep prior runs")
    sources = Path(__file__).parent
    original = read(first_tools.DEFAULT_OUT / "protocol.json")
    old_agent = read(OLD_GPT / "protocol.json")
    assert sha256(DEFAULT_MODELS / "model_results/path_oof.csv") == original["source_hashes"]["baseline"]
    payload = dict(schema_version="seed_repeat_v1", seed=seed, scope=scope,
        frozen_before_training=True, seed_selection="2027 chosen before any new result; one repeat only",
        preserved_reference=dict(seed=2026,baseline_correct=89,agent_correct=105,paths=146),
        unchanged=["EEG/labels", "24 outer LOSO and base14/meta5/policy4 roles", "pretrained FEMBA and cached features",
                   "MIL architecture/epochs/optimizer", "tool candidate grids and internal selection recipe", "GPT system/tools/schema"],
        retrained=["FEMBA path MIL head"] if scope == "full" else [],
        numeric_models_retrained=True,gpt_seed_parameter=None,
        gpt_randomness="Fresh calls; the existing Responses request has no seed parameter; training seed does not fix GPT sampling",
        original_split_sha256=original["source_hashes"]["original_split"],
        original_baseline_csv_sha256=original["source_hashes"]["baseline"],
        original_tool_training_sha256=sha256(first_tools.DEFAULT_OUT / "training_summary.json"),
        original_extended_training_sha256=sha256(EXTENDED_TOOLS / "training_summary.json"),
        original_agent_protocol_sha256=sha256(OLD_GPT / "protocol.json"),
        gpt_system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
        model=old_agent["model"],base_url=old_agent["base_url"],
        source_sha256={name:sha256(sources / name) for name in SOURCE_NAMES},
        mil_source_sha256=sha256(Path(mil.__file__)),
        no_outer_label_parameter_selection=True,final_classes=["High","Low"],exploratory=True)
    out.mkdir(parents=True)
    write_json(out / "protocol.json", payload)
    snapshot = out / "sources"
    snapshot.mkdir()
    for name in SOURCE_NAMES:
        shutil.copy2(sources / name, snapshot / name)
    shutil.copy2(Path(mil.__file__), snapshot / "model_ablation_v3.py")
    return payload


def verify_protocol(out):
    out = Path(out)
    payload = read(out / "protocol.json")
    for name, expected in payload["source_sha256"].items():
        assert sha256(Path(__file__).parent / name) == expected, f"Changed seed source: {name}"
        assert sha256(out / "sources" / name) == expected
    assert sha256(Path(mil.__file__)) == payload["mil_source_sha256"]
    assert sha256(DEFAULT_MODELS / "model_results/path_oof.csv") == payload["original_baseline_csv_sha256"]
    assert sha256(OLD_GPT / "protocol.json") == payload["original_agent_protocol_sha256"]
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == payload["gpt_system_sha256"]
    return payload


def models(out):
    out = Path(out)
    payload = verify_protocol(out)
    destination = out / "models"
    if destination.exists():
        raise FileExistsError("Keep completed or partial model training")
    set_training_seed(payload["seed"])
    original = mil.verify_frozen(DEFAULT_MODELS)
    paths, splits, _, _ = mil.load_dataset(original["pilot_out"])
    extraction = read(DEFAULT_MODELS / "feature_cache/extraction_audit.json")
    assert len(paths) == 146 and len(splits) == 24
    assert sha256(Path(original["pilot_out"]) / "split_manifest.json") == payload["original_split_sha256"]
    destination.mkdir()
    (destination / "checkpoints").mkdir()
    (destination / "fold_predictions").mkdir()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the original WSL CUDA runtime for a fair MIL seed repeat")
    rows, audits = {}, []
    begin = time.perf_counter()
    for n, (split, record) in enumerate(zip(splits, extraction["records"]), 1):
        subject = split["outer_subject"]
        assert record["outer_subject"] == subject
        base = mil.role_indices(paths, split, "base_train")
        test = mil.role_indices(paths, split, "outer_test")
        features = mil.load_features(DEFAULT_MODELS, record, "femba_frozen_mil")
        if payload["scope"] == "full":
            head, training = mil.fit_mil(features, paths, base, device)
            training["seed"] = payload["seed"]
            old_state = torch.load(DEFAULT_MODELS / "model_results/checkpoints" / f"outer_subject_{subject:02d}_femba_frozen_mil.pt", map_location="cpu", weights_only=True)
            assert training["initial_state_sha256"] != old_state["training"]["initial_state_sha256"]
            state = dict(method="femba_frozen_mil",input_dim=features.shape[1],state_dict=head.cpu().state_dict(),split=split,training=training,
                         feature_sha256=record["feature_files"]["femba_frozen_mil"]["sha256"])
        else:
            state = torch.load(DEFAULT_MODELS / "model_results/checkpoints" / f"outer_subject_{subject:02d}_femba_frozen_mil.pt", map_location="cpu", weights_only=True)
            training = state["training"]
        assert training["train_subjects"] == sorted(split["base_train"])
        stem = destination / "checkpoints" / f"fold_{subject:02d}.pt"
        torch.save(state, stem)
        head = mil.PathMIL(state["input_dim"]).to(device)
        head.load_state_dict(state["state_dict"])
        probability = mil.predict_mil(head, features, paths, np.arange(146), device)
        assert np.isfinite(probability).all()
        reload_head = mil.PathMIL(state["input_dim"]).to(device)
        reload_head.load_state_dict(torch.load(stem, map_location=device, weights_only=True)["state_dict"])
        replay = mil.predict_mil(reload_head, features, paths, test, device)
        error = float(np.max(np.abs(probability[test] - replay[test])))
        assert error == 0
        np.savez_compressed(destination / "fold_predictions" / f"fold_{subject:02d}.npz",baseline_probability=probability)
        audits.append(dict(outer_subject=subject,training=training,checkpoint_sha256=sha256(stem),reload_max_abs_error=error,
                           frozen_feature_sha256=record["feature_files"]["femba_frozen_mil"]["sha256"]))
        for i in test:
            rows[int(i)] = dict(path_index=int(i),subject_key=subject,label=paths[i]["label"],accepted_windows=paths[i]["accepted_windows"],
                                femba_frozen_mil_probability=float(probability[i]))
        save_csv(destination / "path_oof.csv", [rows[i] for i in sorted(rows)])
        write_json(destination / "fold_audit.json", audits)
        print(json.dumps(dict(stage="MIL",fold=n,total=24,outer_subject=subject,seed=payload["seed"],reload_error=error)),flush=True)
    assert set(rows) == set(range(146))
    y = np.asarray([rows[i]["label"] for i in range(146)])
    p = np.asarray([rows[i]["femba_frozen_mil_probability"] for i in range(146)])
    write_json(destination / "summary.json",dict(status="completed",seed=payload["seed"],scope=payload["scope"],
        metrics=classification_metrics(y,p >= .5),folds=24,seconds=time.perf_counter()-begin,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda),
        input_and_encoder_unchanged=True,base14_only=True,epochs=mil.EPOCHS,
        feature_extraction_audit_sha256=sha256(DEFAULT_MODELS / "feature_cache/extraction_audit.json")))


def copy_file(source, destination):
    assert not destination.exists()
    shutil.copy2(source, destination)
    assert sha256(source) == sha256(destination)


def numeric(out):
    out = Path(out)
    payload = verify_protocol(out)
    model_summary = read(out / "models/summary.json")
    first, second = out / "tools1", out / "tools2"
    if first.exists() or second.exists():
        raise FileExistsError("Keep prior numeric training")
    first.mkdir()
    (first / "features").mkdir()
    for name in ("local_paths.json", "split_manifest.json", "descriptions.json"):
        copy_file(first_tools.DEFAULT_OUT / name, first / name)
    copy_file(first_tools.DEFAULT_OUT / "features/common.npz", first / "features/common.npz")
    p1 = copy.deepcopy(read(first_tools.DEFAULT_OUT / "protocol.json"))
    p1.update(seed=payload["seed"],seed_retest_protocol_sha256=sha256(out / "protocol.json"),
        baseline=dict(seed=payload["seed"] if payload["scope"] == "full" else 2026,locked_correct=model_summary["metrics"]["correct"],
                      paths=146,threshold=.5,accuracy=model_summary["metrics"]["accuracy"]),feature_folds=[])
    p1["source_hashes"]["baseline"] = sha256(out / "models/path_oof.csv")
    for split in read(first / "split_manifest.json"):
        subject = split["outer_subject"]
        values = dict(np.load(first_tools.DEFAULT_OUT / "features" / f"fold_{subject:02d}.npz"))
        values["baseline_probability"] = np.load(out / "models/fold_predictions" / f"fold_{subject:02d}.npz")["baseline_probability"]
        destination = first / "features" / f"fold_{subject:02d}.npz"
        np.savez_compressed(destination, **values)
        p1["feature_folds"].append(dict(outer_subject=subject,original_head_sha256=sha256(out / "models/checkpoints" / f"fold_{subject:02d}.pt"),
            feature_sha256=sha256(destination),baseline_max_abs_error=0.))
    write_json(first / "protocol.json", p1)
    set_training_seed(payload["seed"])
    first_tools.train(first)
    freeze_policy(first)

    second.mkdir()
    (second / "features").mkdir()
    for name in ("local_paths.json", "split_manifest.json"):
        copy_file(EXTENDED_TOOLS / name, second / name)
    for source in (EXTENDED_TOOLS / "features").glob("*.npz"):
        copy_file(source, second / "features" / source.name)
    p2 = copy.deepcopy(read(EXTENDED_TOOLS / "protocol.json"))
    p2.update(seed=payload["seed"],first_tools_training_sha256=sha256(first / "training_summary.json"),
              seed_retest_protocol_sha256=sha256(out / "protocol.json"))
    write_json(second / "protocol.json", p2)
    set_training_seed(payload["seed"])
    # Imported training keeps FilterBankCSP importable, avoiding __main__ pickle names.
    extra_tools.train(second, first)
    set_training_seed(payload["seed"])
    fit_selector(out / "selector", first, second)
    set_training_seed(payload["seed"])
    audit_selector(out / "selector", first, second)
    import sklearn
    write_json(out / "numeric_complete.json",dict(status="completed",seed=payload["seed"],folds=24,
        environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
        first_training_sha256=sha256(first / "training_summary.json"),second_training_sha256=sha256(second / "training_summary.json"),
        selector_training_sha256=sha256(out / "selector/training_summary.json"),
        nested_audit_sha256=sha256(out / "selector/nested_audit/summary.json"),outer_labels_used_for_selection=False))


def gpt(out, probe=False):
    out = Path(out)
    verify_protocol(out)
    assert read(out / "numeric_complete.json")["status"] == "completed"
    run_agent(out=out / "gpt",first=out / "tools1",second=out / "tools2",selector=out / "selector",system=SYSTEM,probe=probe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=["freeze","models","numeric","gpt"])
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--seed",type=int,default=2027)
    parser.add_argument("--scope",choices=["full","locked"],default="full")
    parser.add_argument("--probe",action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze_protocol(args.out,args.seed,args.scope)
    elif args.stage == "models":
        models(args.out)
    elif args.stage == "numeric":
        numeric(args.out)
    else:
        gpt(args.out,args.probe)
