# -*- coding: utf-8 -*-
"""Raw EEG recording tools. No questionnaire, label, MAT or prediction-table reader.

Identity resolves a local file only. Specialists receive anonymous measurements.
Model inputs are freshly derived from the same waveform as descriptive tools.
"""
import copy
import json
import math
import re
import threading
import uuid
from pathlib import Path

import numpy as np
from scipy import signal

from .config import EEG_CHANNELS, file_hash
from .data import metadata
from .tools import _spectral

MAX_UPLOAD = 512 * 1024 * 1024
GROUPS = {'frontal': ('F3', 'Fz', 'F4'), 'parietal': ('P3', 'Pz', 'P4')}
AMBIGUOUS = re.compile(r'可能|或许|大概|似乎|貌似|不排除|倾向于|一定程度|证据不足|有待进一步')
PROFILE = 'raw-vrms-causal-v1'

# The repository does not ship electrode coordinates.  This deterministic
# adjacency map is therefore deliberately conservative: interpolation only
# uses named neighbouring electrodes from the 10-20 layout and never invents
# a spatial position from the current signal.
CHANNEL_NEIGHBORS = {
    'Fp1': ('Fp2', 'F7', 'F3'), 'Fp2': ('Fp1', 'F4', 'F8'),
    'F11': ('F7', 'F3', 'FT11'), 'F12': ('F8', 'F4', 'FT12'),
    'F7': ('Fp1', 'F11', 'F3', 'FT11'), 'F8': ('Fp2', 'F12', 'F4', 'FT12'),
    'F3': ('Fp1', 'F7', 'Fz', 'F4', 'FC3'), 'Fz': ('F3', 'F4', 'FCz'),
    'F4': ('Fp2', 'F8', 'Fz', 'F3', 'FC4'), 'FT11': ('F11', 'F7', 'T7', 'FC3'),
    'FT12': ('F12', 'F8', 'T8', 'FC4'), 'FC3': ('F3', 'FCz', 'C3', 'FT11'),
    'FCz': ('Fz', 'FC3', 'FC4', 'Cz'), 'FC4': ('F4', 'FCz', 'C4', 'FT12'),
    'T7': ('FT11', 'C3', 'P7'), 'T8': ('FT12', 'C4', 'P8'),
    'C3': ('FC3', 'Cz', 'CP3', 'T7'), 'Cz': ('FCz', 'C3', 'C4', 'CPz'),
    'C4': ('FC4', 'Cz', 'CP4', 'T8'), 'CP3': ('C3', 'CPz', 'P3', 'P7'),
    'CPz': ('Cz', 'CP3', 'CP4', 'Pz'), 'CP4': ('C4', 'CPz', 'P4', 'P8'),
    'P7': ('T7', 'CP3', 'P3', 'O1'), 'P8': ('T8', 'CP4', 'P4', 'O2'),
    'P3': ('CP3', 'Pz', 'P7', 'O1'), 'Pz': ('CPz', 'P3', 'P4', 'Oz'),
    'P4': ('CP4', 'Pz', 'P8', 'O2'), 'O1': ('P7', 'P3', 'Oz'),
    'Oz': ('O1', 'O2', 'Pz'), 'O2': ('P8', 'P4', 'Oz'),
}


def normalize_repair_plan(plan):
    """Validate a local channel-repair request without accepting file paths."""
    plan = {} if plan is None else dict(plan)
    strategy = str(plan.get('strategy', 'none')).lower()
    if strategy not in ('none', 'drop', 'interpolate'):
        raise ValueError('channel repair strategy must be none, drop or interpolate')
    bad = plan.get('bad_channels', [])
    if not isinstance(bad, (list, tuple)) or any(not isinstance(x, str) or not x.strip() for x in bad):
        raise ValueError('bad_channels must be a list of channel names')
    bad = list(dict.fromkeys(x.strip() for x in bad))
    auto_detect = plan.get('auto_detect', False)
    minimum = plan.get('min_neighbors', 2)
    if not isinstance(auto_detect, bool):
        raise ValueError('auto_detect must be a boolean')
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 2 <= minimum <= 30:
        raise ValueError('min_neighbors must be an integer between 2 and 30')
    return {'strategy': strategy, 'bad_channels': bad,
            'auto_detect': auto_detect, 'min_neighbors': minimum}


def _quality_thresholds(z_threshold, corr_threshold, amplitude_threshold, flat_threshold):
    values = (z_threshold, corr_threshold, amplitude_threshold, flat_threshold)
    if (not all(math.isfinite(float(value)) for value in values) or
            z_threshold <= 0 or not -1 <= corr_threshold <= 1 or
            not 0 < flat_threshold < amplitude_threshold):
        raise ValueError('Invalid EEG quality thresholds')


def _robust_z(values):
    values = np.asarray(values, float)
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = max(1.4826 * mad, 1e-12)
    return (values - center) / scale


def assess_channel_quality(processed, names, *, z_threshold=3.0,
                           corr_threshold=.4, amplitude_threshold=2000.0,
                           flat_threshold=.05):
    """Return transparent channel metrics and conservative bad-channel flags."""
    x = np.asarray(processed, float)
    _quality_thresholds(z_threshold, corr_threshold, amplitude_threshold, flat_threshold)
    if x.ndim != 2 or x.shape[0] != len(names) or len(names) < 2:
        raise ValueError('processed EEG must have shape channels by samples')
    # All-channel NaNs are deliberately inserted during filter warmup and
    # acquisition gaps. They affect window coverage, not individual lead faults.
    observable = np.isfinite(x).any(axis=0)
    if not observable.any():
        return {'status': 'unavailable', 'score': 0., 'bad_channels': list(names),
                'blocking_bad_channels': list(names),
                'channels': [{'name': name, 'finite_fraction': 0., 'bad': True,
                              'reasons': ['no_finite_samples']} for name in names]}
    values = x[:, observable]
    finite_fraction = np.mean(np.isfinite(values), axis=1)
    finite_rows = [row[np.isfinite(row)] for row in values]
    ptp = np.array([np.ptp(row) if len(row) else 0. for row in finite_rows])
    std = np.array([np.std(row) if len(row) else 0. for row in finite_rows])
    log_std_z = _robust_z(np.log(np.maximum(std, 1e-12)))
    reference = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    correlations = []
    for row in values:
        common = np.isfinite(row) & np.isfinite(reference)
        centered = row[common] - np.mean(row[common]) if common.any() else np.array([])
        reference_centered = reference[common] - np.mean(reference[common]) if common.any() else np.array([])
        denom = float(np.linalg.norm(centered) * np.linalg.norm(reference_centered))
        correlations.append(None if denom <= 1e-12 else float(np.dot(centered, reference_centered) / denom))
    channels = []
    bad = []
    blocking = []
    warnings = []
    blocking_reasons = {'missing_or_nonfinite_samples', 'peak_to_peak_above_threshold',
                        'flat_channel', 'robust_amplitude_outlier'}
    for i, name in enumerate(names):
        reasons = []
        if finite_fraction[i] < .99:
            reasons.append('missing_or_nonfinite_samples')
        if ptp[i] > amplitude_threshold:
            reasons.append('peak_to_peak_above_threshold')
        if std[i] < flat_threshold:
            reasons.append('flat_channel')
        if abs(log_std_z[i]) > z_threshold:
            reasons.append('robust_amplitude_outlier')
        if correlations[i] is not None and correlations[i] < corr_threshold:
            reasons.append('low_median_reference_correlation')
        if any(reason in blocking_reasons for reason in reasons):
            blocking.append(name)
        elif reasons:
            warnings.append(name)
        is_bad = bool(reasons)
        if is_bad:
            bad.append(name)
        channels.append({'name': name, 'finite_fraction': round(float(finite_fraction[i]), 6),
                         'peak_to_peak_uv': round(float(ptp[i]), 6),
                         'std_uv': round(float(std[i]), 6),
                         'median_reference_correlation': None if correlations[i] is None else round(correlations[i], 6),
                         'robust_amplitude_z': round(float(log_std_z[i]), 6),
                         'bad': is_bad, 'reasons': reasons})
    return {'status': 'pass' if not blocking else 'review',
            'score': round(100. * (len(names) - len(bad)) / max(1, len(names)), 4),
            'bad_channels': bad, 'blocking_bad_channels': blocking, 'warning_channels': warnings,
            'channels': channels,
            'thresholds': {'z_threshold': float(z_threshold), 'corr_threshold': float(corr_threshold),
                           'amplitude_threshold_uv': float(amplitude_threshold),
                           'flat_threshold_uv': float(flat_threshold)}}


def reclassify_channel_quality(report, *, z_threshold=3.0, corr_threshold=.4,
                               amplitude_threshold=2000.0, flat_threshold=.05):
    """Reapply thresholds to cached metrics without rereading raw EEG."""
    _quality_thresholds(z_threshold, corr_threshold, amplitude_threshold, flat_threshold)
    if report.get('status') == 'unavailable':
        return copy.deepcopy(report)
    channels = []
    bad = []
    blocking = []
    warnings = []
    blocking_reasons = {'missing_or_nonfinite_samples', 'peak_to_peak_above_threshold',
                        'flat_channel', 'robust_amplitude_outlier'}
    for old in report.get('channels', []):
        reasons = []
        if old['finite_fraction'] < .99:
            reasons.append('missing_or_nonfinite_samples')
        if old['peak_to_peak_uv'] > amplitude_threshold:
            reasons.append('peak_to_peak_above_threshold')
        if old['std_uv'] < flat_threshold:
            reasons.append('flat_channel')
        if abs(old['robust_amplitude_z']) > z_threshold:
            reasons.append('robust_amplitude_outlier')
        corr = old.get('median_reference_correlation')
        if corr is not None and corr < corr_threshold:
            reasons.append('low_median_reference_correlation')
        if any(reason in blocking_reasons for reason in reasons):
            blocking.append(old['name'])
        elif reasons:
            warnings.append(old['name'])
        item = {**old, 'bad': bool(reasons), 'reasons': reasons}
        channels.append(item)
        if reasons:
            bad.append(item['name'])
    return {'status': 'pass' if not blocking else 'review',
            'score': round(100. * (len(channels) - len(bad)) / max(1, len(channels)), 4),
            'bad_channels': bad, 'blocking_bad_channels': blocking, 'warning_channels': warnings,
            'channels': channels,
            'thresholds': {'z_threshold': float(z_threshold), 'corr_threshold': float(corr_threshold),
                           'amplitude_threshold_uv': float(amplitude_threshold),
                           'flat_threshold_uv': float(flat_threshold)}}


def apply_channel_repair(processed, names, bad_channels, strategy='none', min_neighbors=2):
    """Drop or interpolate explicitly identified channels in local processed EEG."""
    strategy = str(strategy).lower()
    names = list(names)
    bad = list(dict.fromkeys(bad_channels))
    unknown = sorted(set(bad) - set(names))
    if unknown:
        raise ValueError('Unknown bad EEG channels: ' + ', '.join(unknown))
    if strategy == 'none' or not bad:
        return np.asarray(processed, dtype=np.float32), names, {'strategy': 'none', 'bad_channels': [], 'donors': {}}
    if strategy == 'drop':
        keep = [i for i, name in enumerate(names) if name not in set(bad)]
        if len(keep) < 2:
            raise ValueError('Channel drop must retain at least two scalp EEG channels')
        return np.asarray(processed, dtype=np.float32)[keep], [names[i] for i in keep], {
            'strategy': 'drop', 'bad_channels': bad, 'donors': {}, 'remaining_channels': [names[i] for i in keep]}
    if strategy != 'interpolate':
        raise ValueError('channel repair strategy must be none, drop or interpolate')
    if isinstance(min_neighbors, bool) or not isinstance(min_neighbors, int) or not 2 <= min_neighbors <= 30:
        raise ValueError('min_neighbors must be an integer between 2 and 30')
    result = np.asarray(processed, dtype=np.float32).copy()
    bad_set = set(bad)
    index = {name: i for i, name in enumerate(names)}
    donors = {}
    for name in bad:
        candidates = [x for x in CHANNEL_NEIGHBORS.get(name, ()) if x in index and x not in bad_set]
        if len(candidates) < min_neighbors:
            raise ValueError('Not enough named neighbouring donor channels for interpolation: ' + name)
        donor_indices = [index[x] for x in candidates]
        donor_values = result[donor_indices]
        # Preserve shared gaps; interpolation must not bridge missing epochs.
        replacement = np.mean(donor_values, axis=0)
        result[index[name]] = replacement.astype(np.float32)
        donors[name] = candidates
    return result, names, {'strategy': 'interpolate', 'bad_channels': bad, 'donors': donors,
                           'remaining_channels': names}


def clock(seconds):
    millis = round(seconds * 1000)
    minutes, remainder = divmod(millis, 60000)
    return f'{minutes:02d}:{remainder / 1000:06.3f}'


def _ratio(a, b):
    return float(a / b) if a is not None and b is not None and b > 1e-12 else None


def _change(value, reference):
    ratio = _ratio(value, reference)
    if ratio is None:
        return {'relative_change': None, 'direction': '不可计算', 'text': '参考指标缺失或为零，不能计算相对变化'}
    change = ratio - 1
    direction = '上升' if change > 1e-8 else '下降' if change < -1e-8 else '持平'
    return {'relative_change': change, 'direction': direction,
            'text': direction + (f'{abs(change) * 100:.2f}%' if direction != '持平' else '')}


def _finite_measurement(value):
    return value is not None and math.isfinite(value)


def trigger_events(raw, col):
    """Rising/change edges from the embedded waveform, never the CEO event table."""
    found, previous = [], 0
    for start in range(0, len(raw), 65536):
        part = np.asarray(raw[start:start + 65536, col])
        valid = np.isfinite(part)
        rounded = np.rint(np.where(valid, part, 0)).astype(np.int64)
        prior = np.r_[previous, rounded[:-1]]
        hits = np.flatnonzero(valid & (rounded != prior) & np.isin(rounded, (20, 22)))
        found.extend((start + int(i), int(rounded[i])) for i in hits)
        previous = int(rounded[-1])
    return found


def state_segments_from_events(marks, total, fs):
    """Adjacent 20→22 tasks and 22→20 rests, with no implied initial rest.

    Repeated marks terminate the previous candidate as incomplete. The new
    marker starts a new candidate, so a completed interval never bridges an
    anomalous marker. An unclosed final candidate is explicitly incomplete.
    """
    tasks, rests, warnings, pending = [], [], [], None

    def append(start, end, mark, complete, reason=None):
        if end <= start:
            return
        target = tasks if mark == 20 else rests
        row = {'ordinal': len(target) + 1, 'start_sample': start, 'end_sample': end,
               'start_seconds': start / fs, 'end_seconds': end / fs,
               'start_clock': clock(start / fs), 'end_clock': clock(end / fs),
               'complete': complete, 'state': 'task' if mark == 20 else 'rest'}
        if reason:
            row['incomplete_reason'] = reason
        target.append(row)

    for stamp, mark in marks:
        if not 0 <= stamp < total or mark not in (20, 22):
            warnings.append('无效事件：当前候选片段不跨越该事件')
            if pending is not None:
                start, previous = pending
                append(start, min(max(stamp, start), total), previous, False, '遇到无效事件')
            pending = None
            continue
        if pending is None and mark == 22:
            warnings.append('结束事件前未找到开始事件')
        elif pending is not None:
            start, previous = pending
            if stamp <= start:
                warnings.append('非递增事件：当前候选片段作废')
                pending = None
                continue
            elif previous == mark:
                reason = '重复开始事件：前片段缺少结束标记' if mark == 20 else '重复结束事件：前休息段缺少下一开始标记'
                warnings.append(reason)
                append(start, stamp, previous, False, reason)
            else:
                append(start, stamp, previous, True)
        pending = (stamp, mark)
    if pending is not None:
        start, mark = pending
        reason = '记录结束前缺少结束标记' if mark == 20 else '记录结束前缺少下一开始标记'
        append(start, total, mark, False, reason)
    return tasks, rests, warnings


def segments_from_events(marks, total, fs):
    tasks, _, warnings = state_segments_from_events(marks, total, fs)
    return tasks, warnings


def preprocess(raw, channels, fs):
    """Fresh stateful filtering; the 1024-Hz branch matches the frozen VRMS input.

    Other EDF rates support descriptive tools only. Gaps reset the filter and
    discard the next five seconds, rather than connecting discontinuous data.
    """
    names = [name for name in EEG_CHANNELS if name in channels]
    picks = [channels.index(name) for name in names]
    reference = [channels.index(name) for name in ('M1', 'M2') if name in channels]
    if len(names) < 2:
        raise ValueError('原始文件须包含至少两个名称明确的头皮 EEG 通道')
    sections = []
    if fs > 110:
        b, a = signal.iirnotch(50, 30, fs=fs)
        sections.append(signal.tf2sos(b, a))
    sections.append(signal.butter(4, [.5, min(45, fs * .4)], btype='bandpass', fs=fs, output='sos'))
    if fs > 160:
        sections.append(signal.butter(8, 70, btype='lowpass', fs=fs, output='sos'))
    sos = np.vstack(sections)
    # Invalid samples keep their location; they are not removed or interpolated.
    full = np.full((len(raw), len(names)), np.nan, dtype=np.float32)
    state = np.zeros((len(sos), 2, len(names)))
    warm_until, last_finite = int(5 * fs), True
    for offset in range(0, len(raw), 16384):
        block = np.asarray(raw[offset:offset + 16384], dtype=np.float64)
        x = block[:, picks]
        if len(reference) == 2:
            x = x - block[:, reference].mean(axis=1, keepdims=True)
        finite = np.isfinite(x).all(axis=1)
        edges = np.r_[0, np.flatnonzero(finite[1:] != finite[:-1]) + 1, len(x)]
        for left, right in zip(edges[:-1], edges[1:]):
            if not finite[left]:
                state.fill(0)
                last_finite = False
                continue
            if not last_finite:
                warm_until = offset + left + int(5 * fs)
            y, state = signal.sosfilt(sos, x[left:right], axis=0, zi=state)
            full[offset + left:offset + right] = y
            full[offset + left:min(offset + right, warm_until)] = np.nan
            last_finite = True
    model_compatible = fs == 1024 and names == EEG_CHANNELS and len(reference) == 2
    if fs == 1024:
        processed = full[::4].T
    else:
        from fractions import Fraction
        fraction = Fraction(256 / fs).limit_denominator(4096)
        finite = np.isfinite(full).all(axis=1)
        # Resample each finite run separately to preserve gap boundaries.
        processed = np.full((len(names), int(math.ceil(len(full) * 256 / fs))), np.nan, np.float32)
        edges = np.r_[0, np.flatnonzero(finite[1:] != finite[:-1]) + 1, len(full)]
        for left, right in zip(edges[:-1], edges[1:]):
            if finite[left]:
                part = signal.resample_poly(full[left:right], fraction.numerator, fraction.denominator, axis=0).T
                begin = int(math.ceil(left * 256 / fs))
                stop = min(processed.shape[1], begin + part.shape[1])
                processed[:, begin:stop] = part[:, :stop - begin]
    return processed, names, model_compatible


def window_stats(windows, channels, total):
    powers = [_spectral(w)[2] for w in windows]
    if not powers:
        return {'available': False, 'accepted_windows': 0, 'total_windows': total,
                'reason': '没有通过质量检查的完整五秒 EEG 窗口'}
    mean = {band: np.mean([p[band] for p in powers], axis=0) for band in powers[0]}
    all_power = {band: float(values.mean()) for band, values in mean.items()}
    regions = {}
    for group, names in GROUPS.items():
        indices = [channels.index(name) for name in names if name in channels]
        # Do not silently substitute missing frontal/parietal channels.
        regions[group] = {band: float(values[indices].mean()) for band, values in mean.items()} if len(indices) == len(names) else {}
    faa = None
    if 'F3' in channels and 'F4' in channels:
        f3, f4 = mean['alpha'][channels.index('F3')], mean['alpha'][channels.index('F4')]
        if f3 > 1e-12 and f4 > 1e-12:
            faa = float(math.log(f4) - math.log(f3))
    tar_values = [_ratio(float(p['theta'].mean()), float(p['alpha'].mean())) for p in powers]
    return {'available': True, 'accepted_windows': len(windows), 'total_windows': total,
            'analyzed_seconds': len(windows) * 5, 'window_coverage': len(windows) / max(1, total),
            'theta_alpha': _ratio(all_power['theta'], all_power['alpha']),
            'theta_beta': _ratio(all_power['theta'], all_power['beta']),
            'workload_index': _ratio(regions['frontal'].get('theta'), regions['parietal'].get('alpha')),
            'faa': faa, 'mean_band_power_uv2': all_power, 'regions': regions,
            'tar_window_cv': float(np.std(tar_values) / np.mean(tar_values)) if all(v is not None for v in tar_values) and np.mean(tar_values) > 1e-12 else None}


class RecordingRepository:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_root = Path(cfg['data_root'])
        self.root = Path(cfg['output_root']) / 'brain_recordings'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache = {}

    def catalog(self):
        local = [{'id': f'subject-{n:02d}', 'title': f'被试 {n} · 原始 EEG', 'format': 'CDT'}
                 for n in range(1, 27) if (self.data_root / f'data/raw/Acquisition {n:02d}.cdt').is_file()]
        uploads = []
        for path in sorted(self.root.glob('edf-*/registration.json')):
            item = json.loads(path.read_text(encoding='utf-8'))
            uploads.append({key: item[key] for key in ('id', 'title', 'format')})
        return local + uploads

    def validate(self, recording_id):
        if recording_id not in {item['id'] for item in self.catalog()}:
            raise ValueError('没有找到该被试的原始 EEG 文件；可上传 EDF 后在对话中提问')
        return recording_id

    def current_evidence(self, recording_id):
        """Return the latest locally repaired evidence for this session."""
        self.validate(recording_id)
        with self.lock:
            cached = self.cache.get(recording_id)
            if cached is not None:
                return copy.deepcopy(cached['evidence'])
        return self.analyze(recording_id)

    def resolve(self, text):
        candidates = set()
        for item in self.catalog():
            if item['format'] == 'EDF' and (item['title'].casefold() in text.casefold() or item['id'] in text):
                candidates.add(item['id'])
        if candidates:
            if len(candidates) != 1:
                raise ValueError('本轮请指定一个 EDF 文件')
            return self.validate(next(iter(candidates)))
        if re.search(r'\S+\.edf\b', text, re.I):
            raise ValueError('该 EDF 尚未上传；请先用网页的 EDF 上传入口提供原始文件')
        for match in re.finditer(r'(?:被试|受试者|subject\s*|Acquisition\s*|subject-)\s*(?:编号\s*)?0*(\d{1,2})(?!\d)', text, re.I):
            candidates.add(f'subject-{int(match.group(1)):02d}')
        if len(candidates) > 1:
            raise ValueError('本轮请指定一个被试或一个 EDF 文件；完成后可继续提问下一个')
        return self.validate(next(iter(candidates))) if candidates else None

    @staticmethod
    def anonymous_question(text):
        text = re.sub(r'(?:被试|受试者|subject\s*|Acquisition\s*|subject-)\s*(?:编号\s*)?0*\d{1,2}(?!\d)', '当前记录', text, flags=re.I)
        return re.sub(r'(?:SSQ|问卷|自评|真实(?:标签|评分)|实际评分)[^，。；\n]*', '事后评价信息已隔离', text, flags=re.I)

    @staticmethod
    def _edf(path):
        import mne
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose='ERROR')
        canonical = {name.casefold(): name for name in EEG_CHANNELS + ['M1', 'M2', 'Trigger']}
        names = [canonical.get(re.sub(r'^(?:EEG\s+)|(?:[-\s](?:REF|LE|AVG))$', '', name.strip(), flags=re.I).casefold(), name)
                 for name in raw.ch_names]
        if len(names) != len(set(names)) or sum(name in EEG_CHANNELS for name in names) < 2:
            raw.close()
            raise ValueError('EDF 须包含至少两个名称明确且不重复的头皮 EEG 通道')
        if not 100 <= raw.info['sfreq'] <= 4096:
            raw.close()
            raise ValueError('EDF 采样率须在 100 至 4096 Hz 之间')
        return raw, names

    def register_edf(self, path, filename):
        raw, _ = self._edf(path)
        raw.close()
        token = 'edf-' + uuid.uuid4().hex[:12]
        directory = self.root / token
        directory.mkdir()
        Path(path).replace(directory / 'signal.edf')
        item = {'id': token, 'title': Path(filename.replace('\\', '/')).name[:160], 'format': 'EDF'}
        (directory / 'registration.json').write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')
        return item

    def analyze(self, recording_id, repair_plan=None):
        self.validate(recording_id)
        repair_plan = normalize_repair_plan(repair_plan)
        with self.lock:
            if recording_id.startswith('subject-'):
                path = self.data_root / f'data/raw/Acquisition {int(recording_id[-2:]):02d}.cdt'
                meta = metadata(path)
                raw = np.memmap(path, mode='r', dtype='<f4', shape=(meta['actual_samples'], meta['n_channels']))
                channels, fs = meta['channels'], 1024
                signature = (file_hash(path), file_hash(str(path) + '.dpo'))
                marks = trigger_events(raw, channels.index('Trigger'))
                event_source = 'embedded_trigger_waveform'
            else:
                path = self.root / recording_id / 'signal.edf'
                signature = (file_hash(path),)
                source, channels = self._edf(path)
                try:
                    fs = float(source.info['sfreq'])
                    raw = source.get_data().T * 1e6
                finally:
                    source.close()
                marks = trigger_events(raw / 1e6, channels.index('Trigger')) if 'Trigger' in channels else []
                event_source = 'embedded_trigger_waveform' if marks else 'no_task_events'
            if (recording_id in self.cache and self.cache[recording_id]['signature'] == signature and
                    self.cache[recording_id].get('repair_plan') == repair_plan):
                return copy.deepcopy(self.cache[recording_id]['evidence'])
            segments, rest_segments, warnings = state_segments_from_events(marks, len(raw), fs)
            task_count = len(segments) if marks else None
            if not marks:
                segments = [{'ordinal': 1, 'start_sample': 0, 'end_sample': len(raw),
                             'start_seconds': 0, 'end_seconds': len(raw) / fs,
                             'start_clock': clock(0), 'end_clock': clock(len(raw) / fs), 'complete': True}]
            total_samples = len(raw)
            processed, names, compatible = preprocess(raw, channels, fs)
            del raw
            pre_quality = assess_channel_quality(processed, names)
            requested_bad = list(repair_plan['bad_channels'])
            if repair_plan['auto_detect'] and not requested_bad:
                requested_bad = list(pre_quality.get('blocking_bad_channels',
                                                     pre_quality.get('bad_channels', [])))
            repaired, repaired_names, repair_report = apply_channel_repair(
                processed, names, requested_bad, repair_plan['strategy'], repair_plan['min_neighbors'])
            if repair_report['strategy'] != 'none':
                processed, names = repaired, repaired_names
                # ``preprocess`` already checked the sampling rate and M1/M2
                # reference.  A drop changes the channel contract; an
                # interpolation keeps the original names and can therefore
                # preserve compatibility when the pre-repair contract passed.
                compatible = compatible and names == EEG_CHANNELS
            post_quality = assess_channel_quality(processed, names)
            # A model readout is not published when the quality gate needs
            # review. Descriptive tools still receive the repaired signal and
            # can explain the failing channels.
            compatible = compatible and post_quality.get('status') == 'pass'
            def windows_between(a, b):
                result, stamps = [], []
                total = max(0, int((b - a) // (5 * fs)))
                for i in range(total):
                    stamp = a + int(i * 5 * fs)
                    start = int(math.ceil(stamp * 256 / fs))
                    x = processed[:, start:start + 1280]
                    required = 28 if len(names) == 30 else max(2, math.ceil(len(names) * .9))
                    if x.shape == (len(names), 1280) and np.isfinite(x).all() and np.ptp(x, axis=1).max() <= 2000 and np.count_nonzero(x.std(axis=1) >= .05) >= required:
                        result.append(x)
                        stamps.append(stamp / fs)
                return result, stamps, total
            baseline = {'available': False, 'reason': '本协议不使用记录开头作为静息基线'}
            all_windows, model_windows, analyzed_segments = [], [], []
            for segment in segments:
                windows, stamps, total = windows_between(segment['start_sample'], segment['end_sample'])
                stats = window_stats(windows, names, total)
                analyzed_segments.append({**segment, 'stats': stats})
                if segment['complete']:
                    all_windows.extend(windows)
                model_windows.append((windows, stamps))
            overall = window_stats(all_windows, names, sum(s['stats']['total_windows'] for s in analyzed_segments if s['complete']))
            all_rest_windows, analyzed_rests = [], []
            for segment in rest_segments:
                windows, _, total = windows_between(segment['start_sample'], segment['end_sample'])
                analyzed_rests.append({**segment, 'stats': window_stats(windows, names, total)})
                if segment['complete']:
                    all_rest_windows.extend(windows)
            rest_overall = window_stats(all_rest_windows, names, sum(s['stats']['total_windows'] for s in analyzed_rests if s['complete']))
            record = {'input_policy': 'raw_eeg_only', 'profile': PROFILE,
                      'event_source': event_source, 'task_count': task_count,
                      'segment_scope': 'task_paths' if marks else 'whole_recording',
                      'complete_task_count': sum(s['complete'] for s in segments) if marks else None,
                      'incomplete_task_count': sum(not s['complete'] for s in segments) if marks else None,
                      'duration_seconds': total_samples / fs, 'sampling_hz': fs,
                      'eeg_channels': names, 'signal_sha256': signature[0], 'baseline': baseline,
                      'preprocessing': {'profile': PROFILE, 'source_sampling_hz': fs,
                                        'output_sampling_hz': 256, 'reference': 'M1/M2 when both present',
                                        'notch_hz': 50 if fs > 110 else None,
                                        'bandpass_hz': [.5, min(45, fs * .4)],
                                        'lowpass_hz': 70 if fs > 160 else None,
                                        'warmup_seconds_after_start_or_gap': 5,
                                        'window_seconds': 5},
                      'channel_quality': {'before_repair': pre_quality, 'after_repair': post_quality},
                      'channel_repair': repair_report,
                      'segments': analyzed_segments, 'overall': overall,
                      'rest_segments': analyzed_rests, 'rest_overall': rest_overall,
                      'complete_rest_count': sum(s['complete'] for s in rest_segments) if marks else None,
                      'incomplete_rest_count': sum(not s['complete'] for s in rest_segments) if marks else None,
                      'warnings': warnings,
                      'model_compatible': compatible,
                      'model_compatibility_reason':
                          '输入合同与逐导联质量检查均通过' if compatible else
                          ('逐导联质量检查未通过，禁止发布 VRMSModel 分类'
                           if post_quality.get('status') != 'pass' else
                           '通道、采样率或参考不满足 VRMSModel 输入合同'),
                      'quality_policy': 'VRMS训练输入协议：五秒窗，峰峰值≤2000µV，30通道至少28个非平直；断流后预热五秒'}
            evidence = {'status': 'arrived', 'arrived': {}, 'recording': record,
                        'measurement_context': {'source': 'raw_recording', 'input_policy': 'raw_eeg_only',
                            'signal_sha256': signature[0], 'preprocessing_profile': PROFILE,
                            'comparison_policy': 'event_segments_only', 'baseline_policy': 'disabled',
                            'aggregation': 'fresh raw waveform; task 20→22 and rest 22→20 grouped separately; complete segments only; no questionnaire or target labels'}}
            self.cache[recording_id] = {'signature': signature, 'repair_plan': repair_plan,
                                       'evidence': evidence, 'model_windows': model_windows,
                                       'model_baseline': None}
            return copy.deepcopy(evidence)

    def model_inputs(self, recording_id):
        with self.lock:
            value = self.cache[recording_id]
            return [(np.asarray(w, dtype=np.float32), list(stamps)) for w, stamps in value['model_windows']]

    def model_baseline(self, recording_id):
        with self.lock:
            return self.cache[recording_id]['model_baseline']


def _segment_comparisons(rows, events_available, noun):
    completed = [s for s in rows if s['complete']]
    comparison = {'first_to_last': None, 'max_complete': None}
    if not events_available:
        comparison['first_to_last_unavailable_reason'] = f'原始记录没有任务事件，首末完整{noun}比较未定义'
        comparison['max_complete_unavailable_reason'] = f'原始记录没有任务事件，完整{noun}排名未定义'
        return comparison
    if len(completed) < 2:
        comparison['first_to_last_unavailable_reason'] = f'只有 {len(completed)} 条完整{noun}，首末比较需要至少两条'
    elif not all(_finite_measurement(s['value']) for s in (completed[0], completed[-1])):
        missing = [f"第 {s['ordinal']} 条" for s in (completed[0], completed[-1]) if not _finite_measurement(s['value'])]
        comparison['first_to_last_unavailable_reason'] = '、'.join(missing) + f'完整{noun}缺少有效指标，不能替换首末端点进行比较'
    else:
        first, last = completed[0], completed[-1]
        comparison['first_to_last'] = {'first_ordinal': first['ordinal'], 'last_ordinal': last['ordinal'],
            'first_value': first['value'], 'last_value': last['value'],
            'first_start_clock': first['start_clock'], 'first_end_clock': first['end_clock'],
            'last_start_clock': last['start_clock'], 'last_end_clock': last['end_clock'],
            'change': _change(last['value'], first['value'])}
    ranked = [s for s in completed if _finite_measurement(s['value'])]
    if ranked:
        maximum = max(s['value'] for s in ranked)
        comparison['max_complete'] = {'value': maximum,
            'segments': [{k: s[k] for k in ('ordinal', 'value', 'start_clock', 'end_clock', 'complete')}
                         for s in ranked if s['value'] == maximum],
            'compared_count': len(ranked), 'missing_count': len(completed) - len(ranked)}
    else:
        comparison['max_complete_unavailable_reason'] = f'没有取得有效指标的完整{noun}，无法计算最高指标'
    return comparison


def recording_tool(tool, evidence):
    record = evidence['recording']
    domain = tool.split('.')[0]
    if domain == 'vrms':
        measurements = {key: copy.deepcopy(record[key]) for key in ('task_count', 'complete_task_count', 'incomplete_task_count', 'event_source')}
        measurements['segments'] = [{k: s[k] for k in ('ordinal', 'start_seconds', 'end_seconds', 'start_clock', 'end_clock', 'complete')} for s in record['segments']]
        measurements['rest_segments'] = [{k: s[k] for k in ('ordinal', 'start_seconds', 'end_seconds', 'start_clock', 'end_clock', 'complete')} for s in record.get('rest_segments', [])]
        measurements['complete_rest_count'] = record.get('complete_rest_count')
        measurements['incomplete_rest_count'] = record.get('incomplete_rest_count')
        available = record['task_count'] is not None
        reason = '已从原始事件波形统计任务片段' if available else '原始文件没有任务事件通道，路径数未定义'
    else:
        key = 'theta_alpha' if domain == 'fatigue' else 'workload_index'
        def measured_rows(rows):
            return [{'ordinal': s['ordinal'], 'complete': s['complete'], 'start_clock': s['start_clock'],
                     'end_clock': s['end_clock'], 'value': s['stats'].get(key),
                     'accepted_windows': s['stats']['accepted_windows']} for s in rows]
        measurements = {'indicator': '全头皮 θ/α（TAR）' if domain == 'fatigue' else '额区 θ / 顶区 α',
                        'segment_scope': record['segment_scope'],
                        'aggregation': ('路径20→22与休息段22→20分开汇总；各状态完整片段的合格五秒窗口先汇总频带功率后求比；未完成片段不进入整体指标'
                                        if record['segment_scope'] == 'task_paths' else
                                        '整段记录的合格五秒窗口先汇总频带功率后求比；原始记录没有任务事件'),
                        'baseline_available': False,
                        'baseline_unavailable_reason': '本协议不使用记录开头作为静息基线',
                        'task_value': record['overall'].get(key),
                        'segments': measured_rows(record['segments']),
                        'rest_value': record.get('rest_overall', {}).get(key),
                        'rest_segments': measured_rows(record.get('rest_segments', []))}
        if domain == 'emotion':
            measurements['faa'] = record['overall'].get('faa')
            measurements['rest_faa'] = record.get('rest_overall', {}).get('faa')
        events_available = record['segment_scope'] == 'task_paths'
        measurements.update(_segment_comparisons(measurements['segments'], events_available, '路径'))
        measurements.update({'rest_' + name: value for name, value in
                             _segment_comparisons(measurements['rest_segments'], events_available, '休息段').items()})
        for prefix, rows, stats, noun in (
                ('task', measurements['segments'], record['overall'], '完整路径' if events_available else '整段记录'),
                ('rest', measurements['rest_segments'], record.get('rest_overall', {}), '完整休息段')):
            if _finite_measurement(measurements[prefix + '_value']):
                continue
            if prefix == 'rest' and (not events_available or not any(s['complete'] for s in rows)):
                missing_reason = '原始记录没有完整休息事件段，休息指标未定义'
            elif prefix == 'task' and not any(s['complete'] for s in rows):
                missing_reason = '原始记录没有完整路径事件段，路径指标未定义'
            elif stats.get('accepted_windows') == 0:
                missing_reason = noun + '没有通过质量检查的完整五秒 EEG 窗口'
            elif domain == 'emotion' and 'eeg_channels' in record and any(
                    name not in record['eeg_channels'] for names in GROUPS.values() for name in names):
                missing_reason = noun + '缺少工作负荷指标所需的额区 F3/Fz/F4 或顶区 P3/Pz/P4 通道'
            else:
                missing_reason = noun + '缺少本域指标所需的有限频带功率或有效分母'
            measurements[prefix + '_unavailable_reason'] = missing_reason
        available = any(_finite_measurement(measurements[k]) for k in ('task_value', 'rest_value'))
        reason = '已从原始 EEG 按路径与休息状态计算指标' if available else '缺少合格窗口或指标所需通道'
    return {'tool': tool, 'domain': domain, 'available': available, 'status': 'ok' if available else 'unavailable',
            'reason': reason, 'measurements': measurements, 'input_policy': 'raw_eeg_only',
            'signal_sha256': record['signal_sha256'], 'classifier_required': domain == 'vrms',
            'classifier_available': False}


def direct_summary(domain, results):
    """An honest deterministic summary, also used when cloud generation fails."""
    parts = []
    for item in results:
        # Shared preparation tools deliberately return different, local-only
        # schemas (for example ``channels`` or ``quality`` rather than
        # ``measurements``).  Keep their summaries useful while ensuring the
        # domain-specific branches below never assume every tool has a
        # measurement payload.
        value = item.get('result') or {}
        tool = item.get('tool', '')
        measured = value.get('measurements') or {}
        if tool == 'EEGFileLoader':
            if value.get('available') is False:
                parts.append(value.get('reason', 'EEG 文件加载未完成') + '。')
            else:
                channels = value.get('channel_count')
                sampling = value.get('sampling_hz')
                duration = value.get('duration_seconds')
                details = []
                if channels is not None:
                    details.append(f'{channels} 个通道')
                if sampling is not None:
                    details.append(f'{sampling:g} Hz 采样率' if isinstance(sampling, float) else f'{sampling} Hz 采样率')
                if duration is not None:
                    details.append(f'{duration:.2f} 秒记录' if isinstance(duration, (int, float)) else f'{duration} 秒记录')
                parts.append('EEG 文件已在本地受限目录加载' + ('，' + '、'.join(details) if details else '') + '。')
            continue
        if tool == 'EEGPreprocessor':
            if value.get('available') is False:
                parts.append(value.get('reason', 'EEG 预处理未完成') + '。')
            else:
                profile = (value.get('preprocessing') or {}).get('profile', '冻结预处理合同')
                accepted, total, coverage = (value.get('accepted_windows'), value.get('total_windows'),
                                             value.get('window_coverage'))
                window_text = ''
                if accepted is not None and total is not None:
                    window_text = f'，得到 {accepted}/{total} 个合格五秒窗口'
                    if _finite_measurement(coverage):
                        window_text += f'（覆盖率 {coverage:.4f}）'
                parts.append(f'已按 {profile} 执行本地预处理{window_text}。')
            continue
        if tool == 'EEGQualityAssessor':
            if value.get('available') is False:
                parts.append(value.get('reason', 'EEG 质量检查未完成') + '。')
            else:
                quality = value.get('quality') or {}
                bad = quality.get('bad_channels') or []
                accepted, total, coverage = (value.get('accepted_windows'), value.get('total_windows'),
                                             value.get('window_coverage'))
                status = '发现待复核坏导联：' + '、'.join(bad) if bad else '未发现需要处理的坏导联'
                window_text = ''
                if accepted is not None and total is not None:
                    window_text = f'；合格窗口 {accepted}/{total}'
                    if _finite_measurement(coverage):
                        window_text += f'，覆盖率 {coverage:.4f}'
                parts.append(f'逐导联质量检查{status}{window_text}。')
            continue
        if tool == 'EEGChannelRepair':
            if value.get('available') is False:
                parts.append(value.get('reason', '坏导联处理未完成') + '。')
            elif value.get('applied'):
                repair = value.get('repair') or {}
                strategy = repair.get('strategy', '已请求处理')
                bad = repair.get('bad_channels') or []
                donors = repair.get('donors') or {}
                donor_text = ''
                if donors:
                    donor_text = '；插值供体：' + '，'.join(f'{channel}←{", ".join(names)}'
                                                       for channel, names in donors.items())
                parts.append(f'已执行坏导联处理（{strategy}）：' + ('、'.join(bad) if bad else '按质量结果') + donor_text + '。')
            else:
                parts.append(value.get('reason', '未执行坏导联删除或插值') + '。')
            continue
        if tool == 'vrms.raw_recording':
            if measured.get('task_count') is not None:
                parts.append(f"原始事件记录包含 {measured['task_count']} 个任务片段，{measured['complete_task_count']} 个完整，{measured['incomplete_task_count']} 个缺少结束标记。")
            else:
                parts.append(value['reason'] + '。')
        elif tool == 'vrms.raw_model':
            rows = measured.get('segments', [])
            classified = [r for r in rows if r.get('predicted_class')]
            if classified:
                source = 'GPT 工具 Agent' if value.get('gpt_requested') else 'VRMSModel'
                parts.append(source + '的路径分类：' + '；'.join(f"第 {r['ordinal']} 条为{'高类' if r['predicted_class'] == 'High' else '低类'}" for r in classified) + '。')
                first = measured.get('first_high')
                parts.append(f"首条高类路径为第 {first['ordinal']} 条，起止时间 {first['start_clock']}–{first['end_clock']}；整路径判定在该路径结束后形成。" if first else '全部已分类完整路径均为低类，未检出高类路径。')
                parts.append('该输出定位高类路径，不能定位路径内部的症状首次发生秒数。')
            else:
                parts.append(value.get('reason', 'VRMS 模型没有输出路径分类') + '。')
            if value.get('gpt_failed_paths'):
                parts.append(f"{value['gpt_failed_paths']} 条路径的 GPT 判断未完成；本地数值结果单独保留，没有替代 GPT 分类。")
            for row in rows:
                if not row.get('predicted_class'):
                    parts.append(f"第 {row['ordinal']} 个片段：{row['reason']}。")
        elif domain in ('fatigue', 'emotion'):
            indicator = measured.get('indicator', 'EEG指标')
            if _finite_measurement(measured.get('task_value')) or _finite_measurement(measured.get('rest_value')):
                label = '疲劳相关 EEG 指标' if domain == 'fatigue' else 'EEG 工作负荷指标'
                if measured.get('segment_scope') == 'whole_recording':
                    parts.append(f"整段记录的{label}（{indicator}）为 {measured['task_value']:.4f}；未提供任务事件，状态分段未定义。")
                    continue
                for prefix, noun in (('', '路径'), ('rest_', '休息段')):
                    overall = measured.get('task_value' if not prefix else 'rest_value')
                    if _finite_measurement(overall):
                        parts.append(f"完整{noun}的{label}（{indicator}）汇总为 {overall:.4f}。")
                    comparison = measured.get(prefix + 'first_to_last')
                    if comparison:
                        parts.append(f"从第 {comparison['first_ordinal']} 条完整{noun}（{comparison['first_start_clock']}–{comparison['first_end_clock']}，{comparison['first_value']:.4f}）"
                                     f"到第 {comparison['last_ordinal']} 条完整{noun}（{comparison['last_start_clock']}–{comparison['last_end_clock']}，{comparison['last_value']:.4f}），"
                                     f"指标{comparison['change']['text']}。")
                    else:
                        parts.append(measured.get(prefix + 'first_to_last_unavailable_reason', f'未取得完整{noun}首末比较') + '。')
                    maximum = measured.get(prefix + 'max_complete')
                    if maximum:
                        peak = '、'.join(f"第 {r['ordinal']} 条（{r['start_clock']}–{r['end_clock']}）" for r in maximum['segments'])
                        parts.append(f"完整{noun}中指标最高的是{peak}，值为 {maximum['value']:.4f}。")
                    rows = [r for r in measured.get(prefix + 'segments', []) if _finite_measurement(r['value'])]
                    if rows:
                        parts.append(f'各{noun}指标：' + '；'.join(f"第 {r['ordinal']} 条 {r['value']:.4f}（{r['start_clock']}–{r['end_clock']}，{'完整' if r['complete'] else '不完整'}）" for r in rows) + '。')
            else:
                parts.append(value['reason'] + '。')
    return '\n\n'.join(parts)
