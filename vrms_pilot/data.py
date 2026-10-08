"""Causal raw CDT replay, initial reference, and label-separated feature cache."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy import signal

FS_RAW, FS, WINDOW = 1024, 256, 1280
BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_dpo(path):
    text = Path(path).read_text(encoding="utf-8", errors="strict")
    def value(key):
        m = re.search(rf"^\s*{key}\s*=\s*(.*?)\s*$", text, re.M)
        if not m:
            raise ValueError(f"Missing DPO key: {key}")
        return m.group(1)
    names = []
    for block in ("LABELS", "LABELS_OTHERS"):
        m = re.search(rf"{block} START_LIST[^\n]*\n(.*?){block} END_LIST", text, re.S)
        if m:
            names.extend(s.strip() for s in m.group(1).splitlines() if s.strip())
    meta = dict(fs=float(value("SampleFreqHz")), samples=int(value("NumSamples")),
                channels=int(value("NumChannels")), names=names,
                byte_order=value("DataByteOrder"), sample_order=value("DataSampOrder"))
    if meta["fs"] != FS_RAW or meta["byte_order"] != "INTEL" or meta["sample_order"] != "SAMP":
        raise ValueError(f"Unsupported raw format: {meta}")
    if len(names) != meta["channels"]:
        raise ValueError("DPO channel list mismatch")
    return meta


class CausalP0:
    """Input and output are microvolts. Global decimation phase is preserved."""
    def __init__(self, channels=30):
        b, a = signal.iirnotch(50.0, 30.0, fs=FS_RAW)
        self.sos = np.vstack((signal.tf2sos(b, a),
                              signal.butter(4, [0.5, 45], btype="bandpass", fs=FS_RAW, output="sos"),
                              signal.butter(8, 70, btype="lowpass", fs=FS_RAW, output="sos")))
        self.state = np.zeros((len(self.sos), 2, channels))
        self.input_count = 0

    def process(self, x):
        y, self.state = signal.sosfilt(self.sos, np.asarray(x, dtype=np.float64), axis=0, zi=self.state)
        offset = (-self.input_count) % 4
        self.input_count += len(x)
        return y[offset::4].astype(np.float32)


def qc(x):
    finite = bool(np.isfinite(x).all())
    if not finite:
        return False, {"finite": False, "max_ptp_uv": None, "nonflat_channels": 0}
    ptp = np.ptp(x, axis=1)
    nonflat = int(np.count_nonzero(np.std(x, axis=1) >= 0.05))
    good = bool(np.max(ptp) <= 2000 and nonflat >= 28)
    return good, dict(finite=True, max_ptp_uv=float(np.max(ptp)), nonflat_channels=nonflat)


def spectral_features(x):
    f, psd = signal.welch(x, fs=FS, nperseg=512, noverlap=256, axis=1)
    power = np.stack([np.trapezoid(psd[:, (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=1)
                      for lo, hi in BANDS], axis=1)
    power = np.maximum(power, 1e-10)
    relative = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-10)
    return np.concatenate((np.log(power).ravel(), relative.ravel())).astype(np.float32), power


def covariance(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    c = x @ x.T / (x.shape[1] - 1)
    return 0.9 * c + 0.1 * max(float(np.trace(c)) / len(c), 1e-10) * np.eye(len(c))


def reference_invroot(reference):
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    return (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10)))) @ eigenvectors.T


def covariance_features(x, reference, invroot=None):
    if invroot is None:
        invroot = reference_invroot(reference)
    whitened = invroot @ covariance(x) @ invroot
    e, u = np.linalg.eigh((whitened + whitened.T) / 2)
    logc = (u * np.log(np.maximum(e, 1e-10))) @ u.T
    ii, jj = np.triu_indices(30)
    tangent = logc[ii, jj] * np.where(ii == jj, 1.0, np.sqrt(2.0))
    return np.r_[tangent, np.linalg.norm(logc, "fro")].astype(np.float32)


def task_events(raw_path, meta):
    """Keep CEO events beyond EOF for auditing; fall back only if no task marks exist."""
    ceo = Path(str(raw_path) + ".ceo")
    text = ceo.read_text(encoding="utf-8", errors="strict") if ceo.exists() else ""
    block = re.search(r"NUMBER_LIST START_LIST[^\n]*\n(.*?)NUMBER_LIST END_LIST", text, re.S)
    events = []
    if block:
        for line in block.group(1).splitlines():
            fields = line.split()
            if len(fields) >= 3:
                sample, mark = int(float(fields[0])), int(float(fields[2]))
                if mark in (20, 22):
                    events.append((sample, mark))
    if events:
        return sorted(set(events)), "ceo_number_list"
    if "Trigger" not in meta["names"]:
        raise ValueError("No task events or Trigger channel available")
    actual = Path(raw_path).stat().st_size // (meta["channels"] * 4)
    raw = np.memmap(raw_path, dtype="<f4", mode="r", shape=(actual, meta["channels"]))
    previous = 0
    col = meta["names"].index("Trigger")
    for start in range(0, actual, 65536):
        values = np.rint(raw[start:start + 65536, col]).astype(np.int64)
        prior = np.r_[previous, values[:-1]]
        hits = np.flatnonzero((values != prior) & np.isin(values, [20, 22]))
        events.extend((start + int(i), int(values[i])) for i in hits)
        previous = int(values[-1])
    del raw
    return events, "trigger_channel_fallback"


def boundary_audit(row, events, actual_samples):
    a, b = int(row["start_sample_1024"]), int(row["end_sample_1024"])
    marks = sorted(set(events))
    reasons = []
    if not 0 <= a < b <= actual_samples:
        reasons.append("boundary_outside_raw")
    if (a, 20) not in marks:
        reasons.append("start_event_missing")
    following = next(((s, m) for s, m in marks if s > a), None)
    actual_end = following[0] if following and following[1] == 22 else None
    if actual_end != b:
        reasons.append("end_event_mismatch")
    if actual_end is not None and actual_end > actual_samples:
        reasons.append("end_event_after_eof")
    return dict(raw_complete=not reasons, exclusion_reasons=reasons,
                actual_end_event_sample=actual_end, csv_end_sample=b, raw_samples=actual_samples)


def audit_eligibility(data_root):
    """Read-only event audit before cache allocation or any training split."""
    root = Path(data_root)
    labels = root / "labels/task_segments_with_path_scores.csv"
    tasks = [r for r in read_csv(labels) if r["is_complete"].lower() == "true"
             and r["path_score_available"].lower() == "true" and r["path_score"]
             and float(r["duration_sec"]) >= 15]
    tasks.sort(key=lambda r: (int(r["subject_id"]), int(r["start_sample_1024"])))
    sources, candidates, eligible = {}, [], []
    for row in tasks:
        filename = row["file"]
        if filename not in sources:
            raw = root / "raw" / (filename + ".cdt")
            meta = parse_dpo(str(raw) + ".dpo")
            if raw.stat().st_size % (meta["channels"] * 4):
                raise ValueError(f"Partial raw frame: {raw}")
            events, event_source = task_events(raw, meta)
            sources[filename] = (events, raw.stat().st_size // (meta["channels"] * 4), event_source)
        events, actual, event_source = sources[filename]
        check = boundary_audit(row, events, actual)
        candidates.append(dict(candidate_index=len(candidates), file=filename,
                               subject_key=int(row["subject_id"]),
                               start_sample=int(row["start_sample_1024"]),
                               event_source=event_source, **check))
        if check["raw_complete"]:
            eligible.append(row)
    if not eligible:
        raise ValueError("No scored paths have complete raw task events")
    return eligible, dict(schema_version="raw_event_eligibility_v2", candidate_paths=len(tasks),
                          paths=len(eligible), subjects=len({r["subject_id"] for r in eligible}),
                          excluded_paths=len(tasks) - len(eligible), candidates=candidates,
                          label_sha256=digest(labels), source_modified=False)


def prepare(data_root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = Path(data_root) / "labels/task_segments_with_path_scores.csv"
    states = Path(data_root) / "labels/state_segments_by_mark.csv"
    if (out / "dataset_audit.json").exists() or (out / "windows.npy").exists():
        raise ValueError("Cache already exists; use a fresh V2 output directory")
    tasks, eligibility = audit_eligibility(data_root)
    write_json(out / "eligibility_audit.json", eligibility)
    all_states = read_csv(states)
    subjects = sorted({int(r["subject_id"]) for r in tasks})
    if len(subjects) != 24:
        raise ValueError("This experiment requires 24 subjects; inspect the eligibility audit")
    max_windows = sum((int(r["end_sample_1024"]) - int(r["start_sample_1024"])) // (5 * FS_RAW) for r in tasks)
    cache = np.lib.format.open_memmap(out / "windows.npy", mode="w+", dtype=np.float32,
                                     shape=(max_windows, 30, WINDOW))
    window_records, evaluation, bio_abs, bio_ref, cov_features, timings = [], [], [], [], [], []
    subject_audit, source_info = [], []
    path_references = np.full((len(tasks), 30, 30), np.nan, dtype=np.float32)
    path_reference_power = np.full((len(tasks), 30, 4), np.nan, dtype=np.float32)
    cursor = 0
    channel_order = None
    for subject in subjects:
        start_time = time.perf_counter()
        subject_tasks = [(i, r) for i, r in enumerate(tasks) if int(r["subject_id"]) == subject]
        filename = subject_tasks[0][1]["file"]
        raw_path = Path(data_root) / "raw" / f"{filename}.cdt"
        dpo = Path(str(raw_path) + ".dpo")
        meta = parse_dpo(dpo)
        if raw_path.stat().st_size % (meta["channels"] * 4):
            raise ValueError(f"Raw file does not contain complete samples: {raw_path}")
        actual_samples = raw_path.stat().st_size // (meta["channels"] * 4)
        eeg_indices = [i for i, name in enumerate(meta["names"])
                       if i < 32 and name not in ("M1", "M2")]
        names = [meta["names"][i] for i in eeg_indices]
        if len(names) != 30 or (channel_order is not None and names != channel_order):
            raise ValueError("30-channel input contract mismatch")
        channel_order = names
        reference_indices = [meta["names"].index(n) for n in ("M1", "M2")]
        raw = np.memmap(raw_path, dtype="<f4", mode="r", shape=(actual_samples, meta["channels"]))
        last_needed = max(int(r["end_sample_1024"]) for _, r in subject_tasks)
        if last_needed > actual_samples:
            raise ValueError(f"Task boundary extends beyond available raw samples: {raw_path}")
        stream = CausalP0()
        chunks = []
        for offset in range(0, last_needed, 16384):
            block = np.asarray(raw[offset:min(offset + 16384, last_needed)], dtype=np.float64)
            block = block[:, eeg_indices] - block[:, reference_indices].mean(axis=1, keepdims=True)
            chunks.append(stream.process(block))
        processed = np.concatenate(chunks).T
        del chunks, raw
        ss = [r for r in all_states if int(r["subject_id"]) == subject]
        first_task_start = min(int(r["start_sample_1024"]) for r in ss if r["state"] == "task")
        rests = [r for r in ss if r["state"] == "rest" and r["is_complete"].lower() == "true"
                 and int(r["end_sample_1024"]) <= first_task_start]
        reference, ref_power = None, None
        reference_interval = None
        if rests:
            rest = min(rests, key=lambda r: int(r["start_sample_1024"]))
            a = max(int(rest["start_sample_1024"]), 5 * FS_RAW)
            b = min(int(rest["end_sample_1024"]), first_task_start, a + 30 * FS_RAW)
            reference_interval = [a, b]
            if b - a >= 10 * FS_RAW:
                segment = processed[:, (a + 3) // 4:(b + 3) // 4]
                if qc(segment)[0]:
                    reference = covariance(segment)
                    ref_power = spectral_features(segment)[1]
        subject_count = 0
        invroot = reference_invroot(reference) if reference is not None else None
        for path_index, row in subject_tasks:
            if reference is not None:
                path_references[path_index] = reference
                path_reference_power[path_index] = ref_power
            a, b = int(row["start_sample_1024"]), int(row["end_sample_1024"])
            begin_cursor = cursor
            path_qc = []
            for raw_start in range(a, b - 5 * FS_RAW + 1, 5 * FS_RAW):
                start = (raw_start + 3) // 4
                x = processed[:, start:start + WINDOW]
                if x.shape != (30, WINDOW):
                    raise ValueError("Incomplete decimation window")
                q0 = time.perf_counter()
                good, q = qc(x)
                qc_ms = (time.perf_counter() - q0) * 1000
                path_qc.append(q)
                if raw_start < 5 * FS_RAW or not good:
                    continue
                cache[cursor] = x
                t0 = time.perf_counter()
                absolute, power = spectral_features(x)
                changes = None if ref_power is None else np.log((power + 1e-10) / (ref_power + 1e-10)).ravel()
                bm = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                c = None if reference is None else covariance_features(x, reference, invroot)
                cm = (time.perf_counter() - t0) * 1000
                bio_abs.append(absolute)
                bio_ref.append(np.r_[absolute, changes] if changes is not None else np.full(360, np.nan))
                cov_features.append(c if c is not None else np.full(466, np.nan))
                timings.append([qc_ms, bm, cm])
                window_records.append(dict(window_index=cursor, episode_handle=f"episode-{path_index:03d}",
                                           path_index=path_index, start_sample=raw_start,
                                           end_sample=4 * (start + WINDOW - 1) + 1,
                                           cutoff_sec=(4 * (start + WINDOW - 1) + 1) / FS_RAW,
                                           qc=q, reference_available=reference is not None))
                cursor += 1
                subject_count += 1
            evaluation.append(dict(path_index=path_index, subject_key=subject,
                                   episode_handle=f"episode-{path_index:03d}",
                                   score=float(row["path_score"]), label=int(float(row["path_score"]) >= 30),
                                   label_scope="whole_path", path_start_sec=a / FS_RAW, path_end_sec=b / FS_RAW,
                                   window_start=begin_cursor, window_end=cursor,
                                   total_possible_windows=(b - a) // (5 * FS_RAW),
                                   accepted_windows=cursor - begin_cursor,
                                   reference_available=reference is not None))
        subject_audit.append(dict(subject_key=subject, windows=subject_count,
                                  reference_available=reference is not None,
                                  reference_interval_raw_samples=reference_interval,
                                  replay_compute_sec=time.perf_counter() - start_time))
        source_info.append(dict(path=str(raw_path), size_bytes=raw_path.stat().st_size,
                                mtime_ns=raw_path.stat().st_mtime_ns, dpo_sha256=digest(dpo)))
        source_info[-1].update(dpo_samples=meta["samples"], actual_samples=actual_samples,
                               metadata_count_mismatch=meta["samples"] != actual_samples,
                               requested_end_sample=last_needed,
                               ceo_sha256=digest(str(raw_path) + ".ceo") if Path(str(raw_path) + ".ceo").exists() else None)
        print(f"prepared subject {subject:02d}: {subject_count} windows; reference={reference is not None}; "
              f"{time.perf_counter() - start_time:.1f}s", flush=True)
        del processed
    cache.flush()
    del cache
    np.save(out / "bio_absolute.npy", np.asarray(bio_abs, dtype=np.float32))
    np.save(out / "bio_reference.npy", np.asarray(bio_ref, dtype=np.float32))
    np.save(out / "covariance.npy", np.asarray(cov_features, dtype=np.float32))
    np.save(out / "tool_timings_ms.npy", np.asarray(timings, dtype=np.float32))
    np.save(out / "path_references.npy", path_references)
    np.save(out / "path_reference_power.npy", path_reference_power)
    with (out / "window_evidence.jsonl").open("w", encoding="utf-8") as f:
        for r in window_records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(out / "evaluation_manifest.json", evaluation)
    write_json(out / "dataset_audit.json", dict(schema_version="raw_event_eligibility_v2",
                                               candidate_paths=eligibility["candidate_paths"],
                                               excluded_paths=eligibility["excluded_paths"],
                                               eligibility_sha256=digest(out / "eligibility_audit.json"),
                                               label_sha256=digest(labels), state_sha256=digest(states),
                                               paths=len(evaluation), subjects=len(subjects),
                                               accepted_windows=cursor, allocated_windows=max_windows,
                                               channel_order=channel_order, subjects_audit=subject_audit,
                                               sources=source_info, source_modified=False))
    return cursor
