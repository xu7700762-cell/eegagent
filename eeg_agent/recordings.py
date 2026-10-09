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

    def analyze(self, recording_id):
        self.validate(recording_id)
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
            if recording_id in self.cache and self.cache[recording_id]['signature'] == signature:
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
                      'segments': analyzed_segments, 'overall': overall,
                      'rest_segments': analyzed_rests, 'rest_overall': rest_overall,
                      'complete_rest_count': sum(s['complete'] for s in rest_segments) if marks else None,
                      'incomplete_rest_count': sum(not s['complete'] for s in rest_segments) if marks else None,
                      'warnings': warnings,
                      'model_compatible': compatible,
                      'quality_policy': 'VRMS训练输入协议：五秒窗，峰峰值≤2000µV，30通道至少28个非平直；断流后预热五秒'}
            evidence = {'status': 'arrived', 'arrived': {}, 'recording': record,
                        'measurement_context': {'source': 'raw_recording', 'input_policy': 'raw_eeg_only',
                            'signal_sha256': signature[0], 'preprocessing_profile': PROFILE,
                            'comparison_policy': 'event_segments_only', 'baseline_policy': 'disabled',
                            'aggregation': 'fresh raw waveform; task 20→22 and rest 22→20 grouped separately; complete segments only; no questionnaire or target labels'}}
            self.cache[recording_id] = {'signature': signature, 'evidence': evidence, 'model_windows': model_windows,
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
        value, tool = item['result'], item['tool']
        measured = value.get('measurements', {})
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
