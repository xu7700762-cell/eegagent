# -*- coding: utf-8 -*-
"""Evaluate the current raw EEG GPT tool Agent; labels enter the scoring stage only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = None
from eeg_agent import raw_model, raw_tool_agent
from eeg_agent.agents import CloudProvider
from eeg_agent.config import load_config, file_hash
from eeg_agent.data import metadata
from eeg_agent.recordings import RecordingRepository, trigger_events, state_segments_from_events
import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_paths():
    return [Path(__file__), ROOT / 'scripts/recover_raw_gpt_failures.py',
        Path(CONFIG_PATH) if CONFIG_PATH else ROOT / 'configs/local.yaml', *[ROOT / f'eeg_agent/{name}.py' for name in
        ('recordings', 'data', 'tools', 'raw_model', 'raw_tool_agent', 'agents', 'cloud_config', 'config')],
        *[ROOT / f'vrms_model/{name}.py' for name in ('inference','numeric','features','assets','model','encoder')]]


def prepare(out):
    cfg = load_config(CONFIG_PATH)
    if out.exists():
        raise ValueError('A fresh output directory is required')
    # Read only scored-path eligibility and bounds here, not score values or classes.
    label_path = Path(cfg['data_root']) / 'data/labels/task_segments_with_path_scores.csv'
    with label_path.open(encoding='utf-8-sig', newline='') as stream:
        candidates = [{key: row[key] for key in ('subject_id', 'start_sample_1024', 'end_sample_1024')}
                      for row in csv.DictReader(stream) if row['is_complete'].lower() == 'true'
                      and row['path_score_available'].lower() == 'true' and row['path_score']
                      and float(row['duration_sec']) >= 15]
    eligible, excluded, subjects = [], [], {}
    for subject in sorted({int(row['subject_id']) for row in candidates}):
        path = Path(cfg['data_root']) / f'data/raw/Acquisition {subject:02d}.cdt'
        meta = metadata(path)
        raw = np.memmap(path, mode='r', dtype='<f4', shape=(meta['actual_samples'], meta['n_channels']))
        tasks, _, _ = state_segments_from_events(trigger_events(raw, meta['channels'].index('Trigger')),
                                                 meta['actual_samples'], 1024)
        del raw
        complete = {(row['start_sample'], row['end_sample']): row for row in tasks if row['complete']}
        subject_candidates = [r for r in candidates if int(r['subject_id']) == subject]
        # The CSV/CEO and raw Trigger have a record-specific fixed offset.
        # Infer it from EXACT path durations, without score/class information.
        offsets = Counter(task['start_sample'] - int(row['start_sample_1024'])
            for row in subject_candidates for task in complete.values()
            if task['end_sample'] - task['start_sample'] == int(row['end_sample_1024']) - int(row['start_sample_1024']))
        if not offsets or (len(offsets) > 1 and offsets.most_common(2)[0][1] == offsets.most_common(2)[1][1]):
            raise ValueError('Ambiguous raw/CSV event offset')
        offset = offsets.most_common(1)[0][0]
        for row in subject_candidates:
            start, end = int(row['start_sample_1024']), int(row['end_sample_1024'])
            entry = {'subject': subject, 'start_sample': start, 'end_sample': end,
                     'path_key': f'{subject:02d}:{start}'}
            raw_bounds = (start + offset, end + offset)
            if raw_bounds not in complete:
                excluded.append({**entry, 'reason': 'no exact-duration raw-trigger match under the record fixed offset'})
            else:
                eligible.append({**entry, 'ordinal': complete[raw_bounds]['ordinal'],
                                 'raw_start_sample': raw_bounds[0], 'raw_end_sample': raw_bounds[1]})
        subjects[str(subject)] = {'raw_sha256': file_hash(path), 'dpo_sha256': file_hash(str(path) + '.dpo'),
                                   'raw_tasks': len(tasks), 'complete_raw_tasks': len(complete),
                                   'csv_to_raw_offset_samples': offset, 'offset_exact_duration_matches': offsets[offset]}
        print(json.dumps({'stage': 'prepare', 'subject': subject, 'complete_tasks': len(complete)}, ensure_ascii=False), flush=True)
    eligible.sort(key=lambda row: (row['subject'], row['start_sample']))
    if len(eligible) != 146 or len(subjects) != 24:
        raise ValueError(f'Unexpected eligible denominator: {len(eligible)} paths, {len(subjects)} subjects')
    out.mkdir(parents=True)
    sources = {str(path): file_hash(path) for path in source_paths()}
    for path in source_paths():
        target = out / 'sources' / (path.parent.name + '_' + path.name)
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    save(out / 'protocol.json', {'schema': 'raw_gpt_all_paths_v1', 'created_at': now(),
        'prediction_entrypoint': 'eeg_agent.raw_model.assess(cloud=True)', 'provider_model': 'gpt-6.1-sol',
        'model_seed': 2026, 'label_threshold': 30, 'total': 146, 'subject_count': 24,
        'label_file': str(label_path), 'label_sha256': file_hash(label_path),
        'paths': eligible, 'excluded_candidates': excluded, 'subjects': subjects,
        'source_sha256': sources, 'baseline_policy': 'disabled',
        'model_manifest_sha256': file_hash(cfg['model_manifest']), 'model_manifest': cfg['model_manifest'],
        'fitting_calls': 0, 'fresh_raw_encoding': True, 'reuse_prior_gpt': False,
        'scoring': 'after all predictions; all146 denominator; missing/failed outputs count incorrect',
        'additional_raw_paths': 'same recording tool runs all publishable tasks; unlabelled tasks are reported separately',
        'api_failure_policy': 'no retry of valid decisions; no numerical class fallback',
        'raw_csv_alignment': 'record fixed offset from exact duration matches; all eligible paths have exact start/end after translation'})
    save(out / 'progress.json', {'stage': 'prepared', 'total': 146, 'completed_subjects': 0, 'completed_paths': 0})


def verify_protocol(protocol):
    for source, expected in protocol['source_sha256'].items():
        if file_hash(source) != expected:
            raise ValueError('Frozen inference source changed')
    if file_hash(protocol['model_manifest']) != protocol['model_manifest_sha256']:
        raise ValueError('Frozen model manifest changed')
    if file_hash(protocol['label_file']) != protocol['label_sha256']:
        raise ValueError('Evaluation labels changed')


def verify_runtime_config(cfg, protocol):
    if file_hash(cfg['model_manifest']) != protocol['model_manifest_sha256']:
        raise ValueError('Configured model manifest differs from frozen evaluation')
    label_file = Path(cfg['data_root']) / 'data/labels/task_segments_with_path_scores.csv'
    if label_file.resolve() != Path(protocol['label_file']).resolve():
        raise ValueError('Configured dataset differs from frozen evaluation')


def predict(out):
    protocol = read(out / 'protocol.json')
    verify_protocol(protocol)
    cfg = load_config(CONFIG_PATH)
    verify_runtime_config(cfg, protocol)
    cfg['output_root'] = str(out / 'runtime')
    # Per-record shared budget is the same as the webpage, no classifier training.
    cfg['api_timeout_seconds'] = 60
    repository = RecordingRepository(cfg)
    begin, all_rows, calls, failures, encoded = time.perf_counter(), [], 0, 0, 0
    for n, subject in enumerate(map(int, protocol['subjects']), 1):
        recording_id = f'subject-{subject:02d}'
        log_path = out / 'predictions' / f'subject_{subject:02d}.json'
        if log_path.exists():
            raise ValueError('Saved predictions must not be silently replayed as a new evaluation')
        events = []
        provider = CloudProvider('api', cfg, lambda kind, payload: events.append({'kind': kind, 'payload': payload}))
        if provider.model != protocol['provider_model']:
            raise ValueError('Selected provider differs from frozen GPT model')
        evidence = repository.analyze(recording_id)
        expected = protocol['subjects'][str(subject)]
        if evidence['recording']['signal_sha256'] != expected['raw_sha256']:
            raise ValueError('Raw waveform changed')
        raw_path = Path(cfg['data_root']) / f'data/raw/Acquisition {subject:02d}.cdt'
        if file_hash(str(raw_path) + '.dpo') != expected['dpo_sha256']:
            raise ValueError('Raw metadata changed')
        raw_segments = {row['ordinal']: row for row in evidence['recording']['segments']}
        for path in [row for row in protocol['paths'] if row['subject'] == subject]:
            segment = raw_segments[path['ordinal']]
            if (not segment['complete'] or segment['start_sample'] != path['raw_start_sample']
                    or segment['end_sample'] != path['raw_end_sample']):
                raise ValueError('Frozen raw path alignment changed')
        # Questionnaire/label table and evaluation labels are never passed here.
        result = raw_model.assess(repository, recording_id, evidence, provider=provider, cloud=True)
        summary = provider.summary()
        save(log_path, {'subject': subject, 'result': result, 'provider': summary, 'events': events, 'finished_at': now()})
        segments = result['measurements']['segments']
        by_ordinal = {row['ordinal']: row for row in segments}
        for path in [row for row in protocol['paths'] if row['subject'] == subject]:
            actual = by_ordinal.get(path['ordinal'], {})
            judgment = actual.get('cloud_judgment', {})
            all_rows.append({**path, 'vrmsmodel_class': actual.get('mil_class'),
                'vrmsmodel_score': actual.get('mil_raw_high_probability'), 'gpt_class': judgment.get('final_class') if judgment.get('available') else None,
                'generation_status': judgment.get('generation_status', 'unavailable'),
                'confidence': judgment.get('confidence'), 'explanation': judgment.get('explanation', ''),
                'reason': actual.get('reason') or result.get('reason', ''),
                'accepted_windows': actual.get('accepted_windows'),
                'actual_tool_calls': judgment.get('actual_tool_calls', [])})
        calls += summary['calls']
        failures += summary['failures']
        encoded += sum(row.get('fresh_encoded_windows', 0) for row in segments)
        save(out / 'unscored_predictions.json', all_rows)
        progress = {'stage': 'predicting', 'total': 146, 'completed_subjects': n, 'total_subjects': 24,
                    'completed_paths': len(all_rows), 'valid_gpt_paths': sum(row['gpt_class'] is not None for row in all_rows),
                    'api_calls': calls, 'api_failed_calls': failures, 'fresh_encoded_windows': encoded,
                    'elapsed_seconds': time.perf_counter() - begin, 'updated_at': now()}
        save(out / 'progress.json', progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        repository.cache.clear()
        del evidence, result, provider, segments
    verify_protocol(protocol)
    if len(all_rows) != protocol['total']:
        raise ValueError('Incomplete evaluation denominator')
    save(out / 'prediction_complete.json', {**progress, 'stage': 'prediction_complete', 'labels_used_for_prediction': False})


def score_rows(predictions, truths):
    if len({row['path_key'] for row in predictions}) != len(predictions):
        raise ValueError('Duplicate path predictions')
    rows = []
    for prediction in predictions:
        if prediction['path_key'] not in truths:
            raise ValueError('Missing path truth')
        row = {**prediction, **truths[prediction['path_key']]}
        row['correct'] = row['gpt_class'] is not None and row['gpt_class'] == row['true_class']
        rows.append(row)
    valid = [row for row in rows if row['gpt_class'] is not None]
    correct = sum(row['correct'] for row in rows)
    matrix = [[sum(row['true_class'] == actual and row['gpt_class'] == predicted for row in rows)
               for predicted in ('Low', 'High')] for actual in ('Low', 'High')]
    classes = {cls: sum(row['true_class'] == cls for row in rows) for cls in ('Low', 'High')}
    result = {'total': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
              'valid_gpt_paths': len(valid), 'failed_or_unpublished_paths': len(rows) - len(valid),
              'coverage': len(valid) / len(rows), 'conditional_accuracy': correct / len(valid) if valid else None,
              'true_class_counts': classes, 'confusion_matrix_order': ['Low', 'High'], 'confusion_matrix': matrix,
              'balanced_accuracy': .5 * (matrix[0][0] / classes['Low'] + matrix[1][1] / classes['High'])
                  if all(classes.values()) else None}
    return rows, result


def score(out):
    protocol = read(out / 'protocol.json')
    verify_protocol(protocol)
    completion = read(out / 'prediction_complete.json')
    predictions = read(out / 'unscored_predictions.json')
    if len(predictions) != protocol['total']:
        raise ValueError('Do not report partial accuracy')
    # This is the first stage that reads target score/class values.
    truths = {}
    with Path(protocol['label_file']).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['path_score_available'].lower() == 'true' and row['path_score']:
                key = f"{int(row['subject_id']):02d}:{int(row['start_sample_1024'])}"
                truths[key] = {'path_score': float(row['path_score']),
                               'true_class': 'High' if float(row['path_score']) >= 30 else 'Low'}
    rows, metrics = score_rows(predictions, truths)
    with (out / 'path_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'actual_tool_calls': ';'.join(row['actual_tool_calls'])})
    subjects = [{'subject': subject, 'paths': sum(row['subject'] == subject for row in rows),
                 'correct': sum(row['subject'] == subject and row['correct'] for row in rows),
                 'valid_gpt_paths': sum(row['subject'] == subject and row['gpt_class'] is not None for row in rows)}
                for subject in sorted({row['subject'] for row in rows})]
    model_predictions = [{**row, 'gpt_class': row.get('vrmsmodel_class')} for row in predictions]
    _, model_metrics = score_rows(model_predictions, truths)
    summary = {'status': 'completed_and_scored', 'protocol': 'current_raw_eeg_gpt_tool_agent_no_reference',
        'model': protocol['provider_model'], 'model_seed': 2026, 'metrics': metrics,
        'vrmsmodel_metrics': model_metrics, 'api_calls': completion['api_calls'],
        'api_failed_calls': completion['api_failed_calls'], 'fresh_encoded_windows': completion['fresh_encoded_windows'],
        'by_subject': subjects, 'scored_at': now(),
        'limitations': ['same repeatedly developed local data; no independent external generalization',
                        'no resting reference; target labels used only after all predictions; no retraining']}
    save(out / 'summary.json', summary)
    text = (f"# 当前原始 EEG 全路径评价\n\n"
        f"真实 GPT ACC：{metrics['accuracy']*100:.2f}%（{metrics['correct']}/{metrics['total']}）。\n\n"
        f"VRMSModel ACC：{model_metrics['accuracy']*100:.2f}%（{model_metrics['correct']}/{model_metrics['total']}）。\n\n"
        f"有效 GPT 分类：{metrics['valid_gpt_paths']}/{metrics['total']}。失败输出计错误，分母固定。\n\n"
        "仅 seed2026；原始 EEG、无静息参考、整路径标签<30为Low，≥30为High。目标评分只在全部预测后读取。\n")
    (out / 'report.md').write_text(text, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'predict', 'score'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    CONFIG_PATH = args.config
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage](args.out.resolve())
