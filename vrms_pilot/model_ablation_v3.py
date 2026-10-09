"""Independent frozen CNN/FEMBA path-MIL comparison on the frozen V2 LOSO split.

No labels are inherited by windows: each complete path is one supervised bag.
Both encoders receive the same base-subject-fitted input normalization. MIL
heads fit base14 only for a fixed 12 epochs; policy4 is reported, never selected.
This comparison does not isolate pretraining from architecture/source effects.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import digest, read_csv, write_json
from .experiment import metrics, save_csv
from .model import CompactEEGCNN, seed_everything

SEED = 2026
EPOCHS = 12
METHODS = ("cnn_frozen_mil", "femba_frozen_mil")
EXPECTED_FEMBA_SHA256 = "0e2ab9109d87a32c6b25f0c307fb8b1102ef5e0e83e86b9b07f7dee166daaa27"
DEFAULT_PILOT = Path("outputs/vrms_pilot/v2_gpt_flow_20261008")
DEFAULT_OUT = Path("outputs/vrms_model_ablation/v3_20261008")
DEFAULT_ENCODER = None
DEFAULT_CHECKPOINT = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_digest(model):
    sha = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        sha.update(key.encode())
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def load_dataset(pilot_out):
    pilot_out = Path(pilot_out)
    cache = pilot_out / "cache"
    audit = read_json(cache / "dataset_audit.json")
    paths = read_json(cache / "evaluation_manifest.json")
    splits = read_json(pilot_out / "split_manifest.json")
    protocol = read_json(pilot_out / "results/protocol.json")
    validation = read_json(pilot_out / "results/validation.json")
    if (audit.get("schema_version") != "raw_event_eligibility_v2" or protocol.get("smoke")
            or validation.get("status") != "passed" or len(paths) != 146 or len(splits) != 24):
        raise ValueError("A completed validated 146-path/24-fold V2 pilot is required")
    if digest(cache / "eligibility_audit.json") != audit["eligibility_sha256"]:
        raise ValueError("Changed eligible data")
    for name, expected in protocol["code_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"Frozen pilot source changed: {name}")
    for split in splits:
        keys = ("base_train", "head_fit", "probability_calibration", "policy_validation", "outer_test")
        groups = [set(split[key]) for key in keys]
        if [len(group) for group in groups] != [14, 3, 2, 4, 1] or any(
                a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise ValueError("V2 subject roles overlap or changed")
    windows = np.load(cache / "windows.npy", mmap_mode="r")[:audit["accepted_windows"]]
    if windows.shape[1:] != (30, 1280):
        raise ValueError("Expected causal 30-channel 5-second EEG windows")
    return paths, splits, windows, audit


def load_femba(source, checkpoint, device):
    if digest(checkpoint) != EXPECTED_FEMBA_SHA256:
        raise ValueError("The declared local recovered FEMBA checkpoint hash differs")
    spec = importlib.util.spec_from_file_location("eegagent_v3_femba_encoder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.build_encoder(device, backend="native")
    loaded = module.load_pretrained_checkpoint(encoder, checkpoint)
    if loaded["loaded_keys"] != 83 or loaded["missing_keys"] or loaded["unexpected_keys"] or loaded["skipped_keys"]:
        raise ValueError("FEMBA encoder tensors were not completely loaded")
    encoder.eval().requires_grad_(False)
    return encoder, loaded


def role_indices(paths, split, role):
    return np.asarray([i for i, path in enumerate(paths) if path["subject_key"] in split[role]], int)


def base_window_indices(paths, base):
    parts = [np.arange(paths[i]["window_start"], paths[i]["window_end"]) for i in base]
    return np.concatenate(parts).astype(int)


def preflight(out, pilot_out, encoder_source, pretrained_checkpoint):
    out = Path(out)
    if (out / "preflight.json").exists():
        raise FileExistsError("Preserve the existing preflight; use a fresh output")
    paths, splits, windows, audit = load_dataset(pilot_out)
    if not torch.cuda.is_available():
        raise RuntimeError("Use native mamba-ssm in the verified WSL CUDA runtime")
    import mamba_ssm
    device = torch.device("cuda")
    encoder, loaded = load_femba(encoder_source, pretrained_checkpoint, device)
    checkpoint = torch.load(Path(pilot_out) / "results/checkpoints/outer_subject_01.pt", map_location=device, weights_only=True)
    if checkpoint["training_subjects"] != splits[0]["base_train"]:
        raise ValueError("CNN normalization/checkpoint source subjects changed")
    sample = torch.from_numpy(np.array(windows[:2])).to(device)
    sample = ((sample - checkpoint["mean"]) / checkpoint["scale"]).clamp(-20, 20)
    # Two eligible windows only verify the runtime; extraction remains a separate stage.
    before = state_digest(encoder)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = encoder.forward_tokens(sample)
        repeat = encoder.forward_tokens(sample)
    torch.cuda.synchronize()
    if tokens.shape != (2, 80, 525) or not torch.isfinite(tokens).all() or before != state_digest(encoder):
        raise ValueError("Frozen native FEMBA forward failed")
    core = Path(__file__).parent
    source_names = ("model_ablation_v3.py", "model.py", "data.py", "experiment.py", "engine.py", "reliability.py")
    cache_names = ("windows.npy", "evaluation_manifest.json", "dataset_audit.json", "eligibility_audit.json")
    payload = dict(schema_version="frozen_encoder_path_mil_v3", status="passed", seed=SEED, epochs=EPOCHS,
                   pilot_out=str(Path(pilot_out).resolve()), encoder_source=str(Path(encoder_source).resolve()),
                   pretrained_checkpoint=str(Path(pretrained_checkpoint).resolve()),
                   encoder_source_sha256=digest(encoder_source), pretrained_checkpoint_sha256=digest(pretrained_checkpoint),
                   implementation_sha256={name: digest(core / name) for name in source_names},
                   pilot_artifact_sha256={name: digest(Path(pilot_out) / name) for name in
                                         ("split_manifest.json", "results/protocol.json", "results/validation.json", "results/path_oof.csv", "results/summary.json")},
                   cache_sha256={name: digest(Path(pilot_out) / "cache" / name) for name in cache_names},
                   cnn_checkpoint_sha256={p.name: digest(p) for p in sorted((Path(pilot_out) / "results/checkpoints").glob("*.pt"))},
                   paths=len(paths), subjects=audit["subjects"], windows=len(windows), folds=len(splits),
                   checkpoint_load=loaded, femba_parameters=sum(p.numel() for p in encoder.parameters()),
                   frozen_parameters_require_grad=False, input_shape=[2, 30, 1280], token_shape=list(tokens.shape),
                   repeated_forward_max_abs_error=float((tokens - repeat).abs().max()),
                   probe_seconds=time.perf_counter() - begin, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
                   environment=dict(python=platform.python_version(), torch=torch.__version__, mamba_ssm=mamba_ssm.__version__,
                                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
                   cnn_encoder="existing freshly trained V2 base14 CNN; feature mean/std gives 48 dimensions",
                   femba_encoder="local auxiliary guoshanche+monifeixing recovered encoder-only checkpoint; not official TUH weights",
                   pretraining_overlap="not_excluded_missing_original_pretraining_manifest",
                   pretraining_recovery="best_checkpoint.json records restored_encoder_only=true from a frozen selection checkpoint",
                   input_normalization="both encoders use each fold's V2 CNN base14-only mean/scale, clipped +/-20; no subject/test EA or reference adaptation",
                   input_transfer_limit="original auxiliary pretraining used channelwise robust statistics; those original statistics/manifest are unavailable",
                   training="entire path is one bag; train MIL on base14 only; fixed final epoch12; policy4 reported without selection; meta5 unused by MIL",
                   comparison_limit="encoder architecture, supervised CNN source fitting, auxiliary pretraining and dimensions differ; this is not an isolated causal pretraining ablation")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "preflight.json", payload)
    return payload


def verify_frozen(out):
    out = Path(out)
    payload = read_json(out / "preflight.json")
    for name, expected in payload["implementation_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"Changed frozen comparison source: {name}")
    for key, sha_key in (("encoder_source", "encoder_source_sha256"), ("pretrained_checkpoint", "pretrained_checkpoint_sha256")):
        if digest(payload[key]) != payload[sha_key]:
            raise ValueError("Changed frozen pretrained source/weights")
    pilot = Path(payload["pilot_out"])
    for name, expected in payload["pilot_artifact_sha256"].items():
        if digest(pilot / name) != expected:
            raise ValueError("Changed frozen V2 pilot artifacts")
    for name, expected in payload["cache_sha256"].items():
        if digest(pilot / "cache" / name) != expected:
            raise ValueError("Changed frozen EEG cache")
    for name, expected in payload["cnn_checkpoint_sha256"].items():
        if digest(pilot / "results/checkpoints" / name) != expected:
            raise ValueError("Changed frozen CNN checkpoint")
    return payload


def extract(out, batch_size=16):
    out = Path(out)
    payload = verify_frozen(out)
    directory = out / "feature_cache"
    if directory.exists():
        raise FileExistsError("Preserve the existing extracted features")
    paths, splits, windows, _ = load_dataset(payload["pilot_out"])
    device = torch.device("cuda")
    encoder, _ = load_femba(payload["encoder_source"], payload["pretrained_checkpoint"], device)
    frozen_sha = state_digest(encoder)
    directory.mkdir()
    records = []
    overall_start = time.perf_counter()
    for split in splits:
        begin = time.perf_counter()
        subject = split["outer_subject"]
        stem = Path(payload["pilot_out"]) / "results/checkpoints" / f"outer_subject_{subject:02d}.pt"
        checkpoint = torch.load(stem, map_location=device, weights_only=True)
        if checkpoint["training_subjects"] != split["base_train"] or checkpoint["heldout_subject"] != subject:
            raise ValueError("Wrong fold normalization/encoder")
        cnn = CompactEEGCNN().to(device)
        cnn.load_state_dict(checkpoint["model"])
        cnn.eval().requires_grad_(False)
        cnn_sha = state_digest(cnn)
        features = {name: np.empty((len(windows), dim), np.float32) for name, dim in zip(METHODS, (48, 525))}
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.array(windows[start:start + batch_size], dtype=np.float32)).to(device)
            x = ((x - checkpoint["mean"]) / checkpoint["scale"]).clamp(-20, 20)
            with torch.inference_mode():
                z = cnn.features(x)
                cnn_feature = torch.cat((z.mean(dim=-1), z.std(dim=-1, correction=0)), dim=1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    femba_feature = encoder.forward_tokens(x).float().mean(dim=1)
            features[METHODS[0]][start:start + len(x)] = cnn_feature.cpu().numpy()
            features[METHODS[1]][start:start + len(x)] = femba_feature.cpu().numpy()
        torch.cuda.synchronize()
        if cnn_sha != state_digest(cnn) or frozen_sha != state_digest(encoder):
            raise ValueError("A frozen encoder changed during extraction")
        record = dict(outer_subject=subject, base_subjects=split["base_train"],
                      normalization_source="frozen CNN checkpoint base14 statistics shared by both encoders",
                      cnn_state_sha256=cnn_sha, femba_state_sha256=frozen_sha,
                      seconds=time.perf_counter() - begin, gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20,
                      feature_files={})
        for method, values in features.items():
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite frozen features")
            name = f"outer_subject_{subject:02d}_{method}.npy"
            np.save(directory / name, values)
            record["feature_files"][method] = dict(name=name, shape=list(values.shape), sha256=digest(directory / name))
        records.append(record)
        print(f"extract fold {len(records)}/24 subject {subject:02d}: {time.perf_counter()-begin:.2f}s", flush=True)
        del cnn, checkpoint, features
    result = dict(status="completed", frozen_encoders_verified=True, folds=len(records), windows=len(windows),
                  batch_size=batch_size, seconds=time.perf_counter() - overall_start, records=records)
    write_json(directory / "extraction_audit.json", result)
    return result


class PathMIL(nn.Module):
    """One logit per complete variable-length path, with a shared attention head."""
    def __init__(self, input_dim, center=None, scale=None):
        super().__init__()
        self.register_buffer("center", torch.zeros(input_dim) if center is None else torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(input_dim) if scale is None else torch.as_tensor(scale, dtype=torch.float32))
        self.projection = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Linear(64, 1)

    def forward(self, window_features):
        if window_features.ndim != 2 or not len(window_features):
            raise ValueError("MIL needs at least one valid window, never an empty path")
        z = self.projection(((window_features.float() - self.center) / self.scale).clamp(-20, 20))
        attention = self.attention(z).softmax(dim=0)
        return self.classifier((attention * z).sum(dim=0)).squeeze(-1)


def fit_mil(features, paths, base, device, epochs=EPOCHS):
    seed_everything(SEED)
    usable = np.asarray([i for i in base if paths[i]["accepted_windows"]], int)
    yy = np.asarray([paths[i]["label"] for i in usable])
    counts = np.bincount(yy, minlength=2)
    if len(usable) < 6 or min(counts) == 0:
        raise ValueError("Base14 MIL training requires both path classes")
    reference = features[base_window_indices(paths, usable)]
    center, scale = reference.mean(axis=0), np.maximum(reference.std(axis=0), 1e-4)
    head = PathMIL(features.shape[1], center, scale).to(device)
    initial_sha = state_digest(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.001)
    feature_tensor = torch.from_numpy(np.asarray(features)).to(device)
    rng = np.random.default_rng(SEED)
    history = []
    begin = time.perf_counter()
    head.train()
    for epoch in range(epochs):
        order = rng.permutation(usable)
        losses = []
        for start in range(0, len(order), 4):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for i in order[start:start + 4]:
                path = paths[i]
                logit = head(feature_tensor[path["window_start"]:path["window_end"]])
                target = torch.tensor(float(path["label"]), device=device)
                weight = len(usable) / (2 * counts[path["label"]])
                batch_losses.append(nn.functional.binary_cross_entropy_with_logits(logit, target) * weight)
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            losses.extend(float(v.detach()) for v in batch_losses)
        history.append(dict(epoch=epoch + 1, path_weighted_train_bce=float(np.mean(losses))))
    head.eval()
    return head, dict(epochs=epochs, selected_epoch=epochs, selection="fixed_final_epoch_no_validation_selection",
                      training_indices=usable.tolist(), train_paths=len(usable),
                      train_subjects=sorted({paths[i]["subject_key"] for i in usable}),
                      feature_normalization_subjects=sorted({paths[i]["subject_key"] for i in usable}),
                      parameters=sum(p.numel() for p in head.parameters()),
                      initial_state_sha256=initial_sha, final_state_sha256=state_digest(head),
                      history=history, seconds=time.perf_counter() - begin)


def predict_mil(head, features, paths, indices, device):
    predictions = np.full(len(paths), np.nan)
    values = torch.from_numpy(np.asarray(features)).to(device)
    head.eval()
    with torch.inference_mode():
        for i in indices:
            path = paths[i]
            if path["accepted_windows"]:
                logit = head(values[path["window_start"]:path["window_end"]])
                predictions[i] = float(logit.sigmoid())
    return predictions


def load_features(out, record, method):
    artifact = record["feature_files"][method]
    path = Path(out) / "feature_cache" / artifact["name"]
    if digest(path) != artifact["sha256"]:
        raise ValueError("Frozen feature cache changed")
    values = np.load(path)
    if list(values.shape) != artifact["shape"] or not np.isfinite(values).all():
        raise ValueError("Invalid frozen features")
    return values


def report_summary(rows, paths, pilot_out):
    yy = [path["label"] for path in paths]
    result = dict(total_paths=len(paths), methods={}, v2_current_run_diagnostics={})
    for method in METHODS:
        p = np.asarray([np.nan if rows[i][f"{method}_probability"] is None else rows[i][f"{method}_probability"] for i in range(len(paths))], float)
        result["methods"][method] = metrics(yy, p, np.isfinite(p))
    original = read_csv(Path(pilot_out) / "results/path_oof.csv")
    original = {int(row["path_index"]): row for row in original}
    if set(original) != set(range(len(paths))):
        raise ValueError("V2 diagnostics do not match the complete eligible manifest")
    for method in ("deep", "deep_temporal", "full"):
        p = [float(original[i][f"{method}_probability"]) if original[i][f"{method}_probability"] else np.nan for i in range(len(paths))]
        result["v2_current_run_diagnostics"][method] = metrics(yy, p, [original[i][f"{method}_state"] in ("high", "low") for i in range(len(paths))])
    return result


def run(out):
    out = Path(out)
    payload = verify_frozen(out)
    directory = out / "model_results"
    if directory.exists():
        raise FileExistsError("Preserve existing complete or partial training")
    paths, splits, _, _ = load_dataset(payload["pilot_out"])
    extraction = read_json(out / "feature_cache/extraction_audit.json")
    if extraction["status"] != "completed" or extraction["folds"] != len(splits):
        raise ValueError("Incomplete frozen feature extraction")
    directory.mkdir()
    (directory / "checkpoints").mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits, internal = {}, [], []
    begin = time.perf_counter()
    for split, record in zip(splits, extraction["records"]):
        if record["outer_subject"] != split["outer_subject"]:
            raise ValueError("Feature fold order changed")
        subject = split["outer_subject"]
        base, policy, test = [role_indices(paths, split, role) for role in ("base_train", "policy_validation", "outer_test")]
        for method in METHODS:
            features = load_features(out, record, method)
            head, training = fit_mil(features, paths, base, device)
            p = predict_mil(head, features, paths, np.r_[policy, test], device)
            stem = directory / "checkpoints" / f"outer_subject_{subject:02d}_{method}.pt"
            torch.save(dict(schema_version="frozen_encoder_path_mil_v3", method=method, outer_subject=subject,
                            input_dim=features.shape[1], state_dict=head.cpu().state_dict(), split=split,
                            training=training, feature_sha256=record["feature_files"][method]["sha256"]), stem)
            reloaded = PathMIL(features.shape[1]).to(device)
            reloaded.load_state_dict(torch.load(stem, map_location=device, weights_only=True)["state_dict"])
            replay = predict_mil(reloaded, features, paths, np.r_[policy, test], device)
            error = float(np.max(np.abs(p[np.r_[policy, test]] - replay[np.r_[policy, test]])))
            if error > 1e-6:
                raise ValueError("MIL checkpoint reload differs")
            policy_p = p[policy]
            labels = np.asarray([paths[i]["label"] for i in policy])
            valid = np.isfinite(policy_p)
            policy_bce = float(np.mean(-(labels[valid] * np.log(np.clip(policy_p[valid], 1e-6, 1-1e-6)) + (1-labels[valid]) * np.log(np.clip(1-policy_p[valid], 1e-6, 1-1e-6))))) if valid.any() else None
            audits.append(dict(outer_subject=subject, method=method, training=training,
                               policy_bce_diagnostic=policy_bce, policy_used_for_selection=False,
                               checkpoint_sha256=digest(stem), reload_max_abs_error=error))
            for i in policy:
                internal.append(dict(outer_subject=subject, method=method, path_index=int(i),
                                     role="policy_validation_diagnostic_only", subject_key=paths[i]["subject_key"],
                                     label=paths[i]["label"], probability=None if not np.isfinite(p[i]) else float(p[i])))
            for i in test:
                row = rows.setdefault(int(i), dict(path_index=int(i), subject_key=paths[i]["subject_key"],
                                                   label=paths[i]["label"], accepted_windows=paths[i]["accepted_windows"]))
                row[f"{method}_probability"] = None if not np.isfinite(p[i]) else float(p[i])
                row[f"{method}_state"] = "insufficient_data" if not np.isfinite(p[i]) else "high" if p[i] >= .5 else "low"
            print(f"MIL fold {subject:02d} {method}: train={training['train_paths']} epoch12loss={training['history'][-1]['path_weighted_train_bce']:.4f} reload={error:.1g}", flush=True)
        save_csv(directory / "path_oof.csv", [rows[i] for i in sorted(rows)])
        save_csv(directory / "policy_predictions.csv", internal)
        write_json(directory / "fold_audit.json", audits)
    if set(rows) != set(range(len(paths))):
        raise ValueError("Missing outer predictions")
    summary = report_summary(rows, paths, payload["pilot_out"])
    summary.update(status="completed_single_seed_exploratory_loso", folds=len(splits), epochs=EPOCHS, seed=SEED,
                   elapsed_seconds=time.perf_counter() - begin, pretraining_overlap_excluded=False,
                   llm_tested=False, pi_measured=False, bag_unit="entire_path_all_QC_accepted_windows",
                   checkpoint_count=len(audits), model_selection="fixed_final_epoch12", probability_calibrated=False)
    write_json(directory / "summary.json", summary)
    return summary


def validate(out):
    out = Path(out)
    payload = verify_frozen(out)
    paths, splits, _, _ = load_dataset(payload["pilot_out"])
    summary = read_json(out / "model_results/summary.json")
    rows = {int(r["path_index"]): r for r in read_csv(out / "model_results/path_oof.csv")}
    if set(rows) != set(range(len(paths))) or summary["folds"] != 24:
        raise ValueError("Incomplete OOF")
    for row in rows.values():
        for method in METHODS:
            row[f"{method}_probability"] = float(row[f"{method}_probability"]) if row[f"{method}_probability"] else None
    recomputed = report_summary(rows, paths, payload["pilot_out"])
    if any(recomputed[key] != summary[key] for key in ("methods", "v2_current_run_diagnostics")):
        raise ValueError("Metrics differ from OOF recomputation")
    extraction = read_json(out / "feature_cache/extraction_audit.json")
    audits = read_json(out / "model_results/fold_audit.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    errors = []
    for split, record in zip(splits, extraction["records"]):
        test = role_indices(paths, split, "outer_test")
        for method in METHODS:
            features = load_features(out, record, method)
            stem = out / "model_results/checkpoints" / f"outer_subject_{split['outer_subject']:02d}_{method}.pt"
            state = torch.load(stem, map_location=device, weights_only=True)
            if state["split"] != split or state["training"]["train_subjects"] != sorted(split["base_train"]) or state["training"]["epochs"] != EPOCHS:
                raise ValueError("Head training provenance changed")
            head = PathMIL(state["input_dim"]).to(device)
            head.load_state_dict(state["state_dict"])
            p = predict_mil(head, features, paths, test, device)
            for i in test:
                expected = rows[int(i)][f"{method}_probability"]
                if np.isfinite(p[i]) != (expected is not None):
                    raise ValueError("Checkpoint refusal differs")
                if expected is not None:
                    error = abs(p[i] - expected)
                    if error > 1e-6:
                        raise ValueError("Checkpoint OOF probability differs")
                    errors.append(error)
    result = dict(status="passed", paths=len(rows), folds=len(splits), checkpoint_count=len(audits),
                  prediction_reload_comparisons=len(errors), max_reload_error=max(errors, default=0),
                  base14_only_mil_training=True, meta_and_outer_not_used_for_mil_training=True,
                  policy_not_used_for_selection=True, frozen_input_and_encoder_hashes_verified=True,
                  common_v2_manifest=True, metrics_recomputed=True, pretraining_overlap_excluded=False)
    write_json(out / "model_results/validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "extract", "run", "validate"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-out", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--encoder-source", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.stage == "preflight":
        if args.encoder_source is None or args.pretrained_checkpoint is None:
            parser.error("preflight requires --encoder-source and --pretrained-checkpoint; weights are not bundled")
        result = preflight(args.out, args.pilot_out, args.encoder_source, args.pretrained_checkpoint)
    elif args.stage == "extract":
        result = extract(args.out, args.batch_size)
    elif args.stage == "run":
        result = run(args.out)
    else:
        result = validate(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
