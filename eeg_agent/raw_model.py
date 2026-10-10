# -*- coding: utf-8 -*-
"""Bounded local CUDA worker for raw-only conversation tools."""
import copy
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LOCK = threading.Lock()
ROOT = Path(__file__).resolve().parents[1]


def _wsl(path):
    path = Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]


def public_result(result):
    """Expose GPT conclusions; numerical comparisons and full API logs stay local."""
    result = copy.deepcopy(result)
    for row in result.get('measurements', {}).get('segments', []):
        row.pop('numeric_fusion', None)
        for key in ('mil_class', 'mil_raw_high_probability', 'threshold') if result.get('gpt_requested') else ():
            row.pop(key, None)
        for key in ('rounds', 'tool_outputs', 'tool_timings_ms'):
            row.get('cloud_judgment', {}).pop(key, None)
    return result


def assess(repository, recording_id, evidence, provider=None, cloud=False):
    record = evidence['recording']
    result = {'tool': 'vrms.raw_model', 'domain': 'vrms', 'available': False,
              'input_policy': 'raw_eeg_only', 'classifier_required': True, 'classifier_available': False,
              'signal_sha256': record['signal_sha256'], 'measurements': {'segments': [], 'first_high': None}}
    if record['task_count'] is None:
        result['reason'] = '原始文件没有任务开始与结束事件，未形成整路径模型输入；指标工具继续分析'
        return result
    if not record['model_compatible']:
        result['reason'] = record.get('model_compatibility_reason') or (
            '该文件不满足 VRMSModel 输入合同；其他工具继续分析')
        return result
    fold = int(recording_id[-2:]) if recording_id.startswith('subject-') else 1
    manifest = Path(repository.cfg['model_manifest'])
    if not manifest.is_file():
        result['reason'] = '本地尚未配置冻结 VRMSModel 权重；可继续执行描述性 EEG 工具'
        return result
    assets = json.loads(manifest.read_text(encoding='utf-8'))
    if assets.get('seed') != 2026 or str(fold) not in assets.get('folds', {}):
        result['reason'] = '没有对应的 seed2026 留一被试权重；本轮未发布分类'
        return result
    inputs = repository.model_inputs(recording_id)
    directory = repository.root / 'inference' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    request = {'fold': fold,
               'signal_sha256': record['signal_sha256'], 'segments': [], 'baseline_file': False}
    # Recording starts during VR exposure. Its opening waveform is not an
    # established resting reference; use the trained missing-reference branch.
    for segment, (windows, stamps) in zip(record['segments'], inputs):
        row = {k: segment[k] for k in ('ordinal', 'start_seconds', 'end_seconds', 'start_clock', 'end_clock', 'complete')}
        stats = segment['stats']
        row.update(predicted_class=None, reason='')
        if not segment['complete']:
            row['reason'] = '缺少结束标记，未发布整路径分类'
        elif len(windows) < repository.cfg['minimum_path_windows'] or stats.get('window_coverage', 0) < repository.cfg['minimum_path_coverage']:
            row['reason'] = '合格窗口数量或覆盖率未达到模型发布规则'
        else:
            filename = f"segment-{segment['ordinal']:03d}.npy"
            np.save(directory / filename, windows, allow_pickle=False)
            request['segments'].append({'ordinal': segment['ordinal'], 'file': filename})
        result['measurements']['segments'].append(row)
    if not request['segments']:
        result['reason'] = '没有满足完整性、质量和覆盖率要求的路径'
        return result
    request_path = directory / 'request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    command = list(repository.cfg['worker_command'])
    using_wsl = Path(command[0]).name.lower() in ('wsl', 'wsl.exe')
    if using_wsl:
        command += ['--cd', _wsl(ROOT), '--exec', repository.cfg['worker_python']]
    command += ['-B', '-m', 'vrms_model.inference',
                _wsl(request_path) if using_wsl else str(request_path),
                _wsl(manifest) if using_wsl else str(manifest)]
    try:
        with LOCK:
            completed = subprocess.run(command, capture_output=True,
                cwd=ROOT, timeout=240, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        value = json.loads(completed.stdout.decode('utf-8'))
        (directory / 'worker_result.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        if not value.get('available'):
            result['reason'] = value.get('reason', '原始波形编码未完成')
            return result
        actual = {row['ordinal']: row for row in value['segments']}
        expected = {row['ordinal'] for row in request['segments']}
        if set(actual) != expected or value['signal_sha256'] != record['signal_sha256'] or value.get('fresh_raw_encoding') is not True:
            raise ValueError('Worker/source mismatch')
        for row in result['measurements']['segments']:
            if row['ordinal'] in actual:
                prediction = actual[row['ordinal']]
                p = prediction['numeric_fusion']['raw_combined_score']
                if not isinstance(p, (int, float)) or not 0 <= p <= 1 or prediction['predicted_class'] != ('High' if p >= .5 else 'Low'):
                    raise ValueError('Invalid raw model prediction')
                row.update(prediction)
                if not cloud:
                    row.update(predicted_class=prediction['mil_class'],classification_layer='vrmsmodel')
        if cloud:
            from .raw_tool_agent import classify
            predictions = [row for row in result['measurements']['segments'] if row['ordinal'] in actual]
            with ThreadPoolExecutor(max_workers=min(3, len(predictions)), thread_name_prefix='raw-gpt') as pool:
                judgments = list(pool.map(lambda row: classify(provider, row), predictions))
            for row, judgment in zip(predictions, judgments):
                # Scores and intermediate classes remain intact even when GPT
                # disagrees. A failed GPT request never becomes a GPT class.
                row['cloud_judgment'] = judgment
                row['predicted_class'] = judgment['final_class'] if judgment['available'] else None
                row['classification_layer'] = 'gpt_tool_agent' if judgment['available'] else 'unavailable_gpt_tool_agent'
                row['reason'] = '' if judgment['available'] else judgment['reason']
                (directory / f"gpt_path_{row['ordinal']:03d}.json").write_text(
                    json.dumps(judgment, ensure_ascii=False, indent=2), encoding='utf-8')
            succeeded = sum(judgment['available'] for judgment in judgments)
            result.update(gpt_requested=True, gpt_successful_paths=succeeded,
                          gpt_failed_paths=len(judgments) - succeeded)
        high = next((row for row in result['measurements']['segments'] if row['predicted_class'] == 'High'), None)
        if high:
            result['measurements']['first_high'] = {k: high[k] for k in ('ordinal', 'start_seconds', 'end_seconds', 'start_clock', 'end_clock')}
            result['measurements']['first_high']['decision_available_seconds'] = high['end_seconds']
        result.update(available=True, classifier_available=True, status='ok', reason='VRMS 模型已重新编码原始 EEG 并完成整路径分类',
                      fresh_raw_encoding=True, model_sha256=value['model_sha256'],
                      encoder_sha256=value['encoder_sha256'], normalization_sha256=value['normalization_sha256'])
        result.update({key: value[key] for key in ('numeric_first_sha256', 'numeric_second_sha256', 'numeric_selector_sha256')})
        if cloud:
            result.update(status='partial_fallback' if result['gpt_failed_paths'] else 'ok',
                reason='已重新编码原始 EEG 并完成 GPT 工具 Agent 判断' if not result['gpt_failed_paths'] else '部分 GPT 判断未完成，数值结果已保留且未代替 GPT 分类')
        result['measurements'].update(readout='whole_path_gpt_tool_agent' if cloud else 'whole_path_vrmsmodel', probability_calibrated=False,
            baseline_policy='disabled', comparison_policy='event_segments_only',
            onset_scope='首条高类完整路径的时间区间；判定使用整条路径数据，不是路径内首次症状发生时刻')
    except subprocess.TimeoutExpired:
        result['reason'] = '原始波形编码超过四分钟，本轮未发布分类；工具测量已保留'
    except Exception:
        result['reason'] = '原始波形推理进程或输出校验失败，本轮未发布分类'
    return result
