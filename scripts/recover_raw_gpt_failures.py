# -*- coding: utf-8 -*-
"""Retry unavailable GPT decisions without labels or changing successful decisions."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_agent.agents import CloudProvider
from eeg_agent.config import load_config, file_hash
from eeg_agent.raw_tool_agent import classify
from scripts.evaluate_raw_gpt_all import read, save, verify_protocol, verify_runtime_config


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(source, out):
    protocol = read(source / 'protocol.json')
    verify_protocol(protocol)
    if out.exists():
        manifest = read(out / 'recovery_protocol.json')
        assert manifest['original_output'] == str(source)
        for name, expected in manifest['original_file_sha256'].items():
            assert file_hash(source / name) == expected, 'Original first-pass artifact changed'
        return protocol
    assert read(source / 'prediction_complete.json')['completed_paths'] == 146
    out.mkdir(parents=True)
    names = ['protocol.json', 'prediction_complete.json', 'unscored_predictions.json',
             *[str(path.relative_to(source)) for path in sorted((source / 'predictions').glob('subject_*.json'))]]
    for name in names:
        target = out / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source / name, target)
    shutil.copytree(source / 'sources', out / 'sources')
    save(out / 'recovery_protocol.json', {
        'created_at': now(), 'original_output': str(source),
        'original_file_sha256': {name: file_hash(source / name) for name in names},
        'policy': 'only unavailable GPT decisions; never retry or replace valid decisions',
        'maximum_recovery_attempts_per_path': 2,
        'consecutive_failure_circuit_break': 3,
        'evidence_policy': 'reuse fresh EEG evidence computed in this same evaluation; no new fitting',
        'labels_loaded': False, 'inference_source_changes': False,
    })
    return protocol


class DiagnosticProvider(CloudProvider):
    def call(self, *args, **kwargs):
        try:
            return super().call(*args, **kwargs)
        except Exception as exc:
            body = getattr(exc, 'body', {})
            error = body.get('error', body) if isinstance(body, dict) else {}
            message = str(error.get('message', '') if isinstance(error, dict) else '').lower()
            diagnostic = {'exception': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None),
                'indicators': [term for term in ('quota', 'rate', 'limit', 'exhaust', 'capacity',
                    'credential', 'authentication', 'unauthorized', 'channel', 'model', 'token',
                    'timeout', 'unavailable', 'account', 'no auth', 'no available') if term in message]}
            self.emit('recovery_api_diagnostic', diagnostic)
            raise


def refresh(source, out, protocol):
    rows = copy.deepcopy(read(source / 'unscored_predictions.json'))
    index = {(row['subject'], row['ordinal']): row for row in rows}
    totals = {'api_calls': 0, 'api_failed_calls': 0, 'fresh_encoded_windows': 0}
    for subject in map(int, protocol['subjects']):
        original = read(source / 'predictions' / f'subject_{subject:02d}.json')
        saved = read(out / 'predictions' / f'subject_{subject:02d}.json')
        original_segments = {row['ordinal']: row for row in original['result']['measurements']['segments']}
        for segment in saved['result']['measurements']['segments']:
            if original_segments[segment['ordinal']].get('cloud_judgment', {}).get('available'):
                assert segment == original_segments[segment['ordinal']], 'A successful first-pass decision changed'
            totals['fresh_encoded_windows'] += segment.get('fresh_encoded_windows', 0)
            row = index.get((subject, segment['ordinal']))
            if row is not None:
                judgment = segment.get('cloud_judgment', {})
                row.update(gpt_class=judgment.get('final_class') if judgment.get('available') else None,
                    generation_status=judgment.get('generation_status', 'unavailable'),
                    confidence=judgment.get('confidence'), explanation=judgment.get('explanation', ''),
                    reason=segment.get('reason', ''), actual_tool_calls=judgment.get('actual_tool_calls', []))
        segments = saved['result']['measurements']['segments']
        judgments = [row['cloud_judgment'] for row in segments if 'cloud_judgment' in row]
        succeeded = sum(judgment['available'] for judgment in judgments)
        saved['result'].update(gpt_successful_paths=succeeded,
            gpt_failed_paths=len(judgments) - succeeded,
            status='ok' if succeeded == len(judgments) else 'partial_fallback')
        high = next((row for row in segments if row['predicted_class'] == 'High'), None)
        first_high = None
        if high:
            first_high = {key: high[key] for key in ('ordinal', 'start_seconds', 'end_seconds', 'start_clock', 'end_clock')}
            first_high['decision_available_seconds'] = high['end_seconds']
        saved['result']['measurements']['first_high'] = first_high
        save(out / 'predictions' / f'subject_{subject:02d}.json', saved)
        totals['api_calls'] += saved['provider']['calls']
        totals['api_failed_calls'] += saved['provider']['failures']
    save(out / 'unscored_predictions.json', rows)
    complete = read(source / 'prediction_complete.json')
    complete.update(totals, valid_gpt_paths=sum(row['gpt_class'] is not None for row in rows),
                    recovery_attempts=len(list((out / 'recovery_attempts').glob('*.json'))),
                    updated_at=now(), original_prediction_complete_at=complete['updated_at'],
                    labels_used_for_prediction=False)
    save(out / 'prediction_complete.json', complete)
    save(out / 'progress.json', complete)
    return complete


def recover(source, out, limit, config=None):
    protocol = initialize(source, out)
    cfg = load_config(config)
    verify_runtime_config(cfg, protocol)
    cfg['api_timeout_seconds'] = 60
    attempt_directory = out / 'recovery_attempts'
    attempt_directory.mkdir(exist_ok=True)
    scored = {(row['subject'], row['ordinal']) for row in protocol['paths']}
    candidates = []
    for subject in map(int, protocol['subjects']):
        saved = read(out / 'predictions' / f'subject_{subject:02d}.json')
        for row in saved['result']['measurements']['segments']:
            if row.get('numeric_fusion') and not row.get('cloud_judgment', {}).get('available'):
                attempts = len(list(attempt_directory.glob(f'{subject:02d}_{row["ordinal"]:03d}_*.json')))
                if attempts < 2:
                    candidates.append((subject, row['ordinal'], attempts + 1))
    candidates.sort(key=lambda item: ((item[0], item[1]) not in scored, item[0], item[1]))
    failures = 0
    for number, (subject, ordinal, attempt) in enumerate(candidates[:limit] if limit else candidates, 1):
        path = out / 'predictions' / f'subject_{subject:02d}.json'
        saved = read(path)
        row = next(row for row in saved['result']['measurements']['segments'] if row['ordinal'] == ordinal)
        assert not row.get('cloud_judgment', {}).get('available')
        events = []
        provider = DiagnosticProvider('api', cfg, lambda kind, payload: events.append({'kind': kind, 'payload': payload}))
        assert provider.model == protocol['provider_model']
        evidence_hash = hashlib.sha256(json.dumps(row['numeric_fusion'], sort_keys=True,
                                                 ensure_ascii=False).encode('utf-8')).hexdigest()
        judgment = classify(provider, row)
        usage = provider.summary()
        save(attempt_directory / f'{subject:02d}_{ordinal:03d}_{attempt}.json', {
            'subject': subject, 'ordinal': ordinal, 'attempt': attempt, 'finished_at': now(),
            'fresh_evidence_sha256': evidence_hash, 'judgment': judgment, 'provider': usage,
            'events': events, 'labels_loaded': False})
        aggregate = saved['provider']
        for key in ('calls', 'failures', 'input_tokens', 'output_tokens', 'total_call_seconds', 'budget_rejections'):
            aggregate[key] += usage[key]
        aggregate['records'].extend(usage['records'])
        aggregate['retries'] = aggregate.get('retries', 0) + 1
        saved['events'].extend(events)
        if judgment['available']:
            row.update(cloud_judgment=judgment, predicted_class=judgment['final_class'],
                       classification_layer='gpt_tool_agent', reason='')
            failures = 0
        else:
            failures += 1
        save(path, saved)
        complete = refresh(source, out, protocol)
        print(json.dumps({'subject': subject, 'ordinal': ordinal, 'attempt': attempt,
            'available': judgment['available'], 'error_type': judgment.get('error_type'),
            'diagnostics': [e['payload'] for e in events if e['kind'] == 'recovery_api_diagnostic'],
            'valid_gpt_paths': complete['valid_gpt_paths'], 'attempted_this_pass': number},
            ensure_ascii=False), flush=True)
        if failures >= 3:
            print(json.dumps({'stage': 'recovery_circuit_break', 'reason': 'three consecutive API failures'}), flush=True)
            break
    verify_protocol(protocol)
    refresh(source, out, protocol)
    save(out / 'recovery_runner_sha256.json', {'path': str(Path(__file__).resolve()),
                                             'sha256': file_hash(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    recover(args.source.resolve(), args.out.resolve(), args.limit, args.config)
