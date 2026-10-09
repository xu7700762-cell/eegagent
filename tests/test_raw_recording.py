# -*- coding: utf-8 -*-
"""Raw-only inference contracts: isolate labels, events, gaps, and truthful prose."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eeg_agent.brain import BrainManager, question_domains
from eeg_agent.config import EEG_CHANNELS, load_config
from eeg_agent.domain_knowledge import HierarchicalKnowledge
from eeg_agent.recordings import RecordingRepository, preprocess, trigger_events, recording_tool, state_segments_from_events, direct_summary
from eeg_agent.service import create_app


@pytest.fixture
def raw_cfg(tmp_path):
    cfg = load_config()
    cfg['output_root'] = str(tmp_path / 'output')
    cfg['data_root'] = str(tmp_path / 'data')
    cfg['cloud_config_path'] = str(tmp_path / '.env')
    folder = Path(cfg['data_root']) / 'data/raw'
    folder.mkdir(parents=True)
    channels = EEG_CHANNELS + ['M1', 'M2', 'VEOG', 'HEOG', 'ECG', 'EGG', 'Trigger']
    count, fs = 70 * 1024, 1024
    t = np.arange(count) / fs
    x = np.zeros((count, len(channels)), dtype='<f4')
    for i in range(30):
        x[:, i] = 15 * np.sin(2 * np.pi * (5 + i * .03) * t) + 9 * np.sin(2 * np.pi * (10 + i * .02) * t)
    for stamp, code in ((10 * fs, 20), (25 * fs, 22), (35 * fs, 20), (50 * fs, 22), (60 * fs, 20)):
        x[stamp:stamp + 10, -1] = code
    path = folder / 'Acquisition 10.cdt'
    x.tofile(path)
    text = ('NumChannels = 37\nNumSamples = ' + str(count) + '\nSampleFreqHz = 1024\n'
            'DataByteOrder = INTEL\nDataSampOrder = SAMP\nDataFormat = 6\nDataUnit = uV\nCommonScale = 1\nCommonOffset = 0\n'
            'LABELS START_LIST\n' + '\n'.join(channels) + '\nLABELS END_LIST\n')
    Path(str(path) + '.dpo').write_text(text, encoding='utf-8')
    return cfg


def test_raw_reads_only_waveform_and_header(raw_cfg, monkeypatch):
    repository = RecordingRepository(raw_cfg)
    original_open = Path.open
    read_paths = []
    def guarded_open(path, mode='r', *args, **kwargs):
        if 'r' in mode:
            assert path.suffix.lower() not in ('.csv', '.xlsx', '.mat', '.ceo')
            read_paths.append(path)
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr('eeg_agent.data.events', lambda *a: pytest.fail('Must scan the embedded waveform'), raising=False)
    evidence = repository.analyze('subject-10')
    record = evidence['recording']
    assert (record['task_count'], record['complete_task_count'], record['incomplete_task_count']) == (3, 2, 1)
    assert record['segments'][0]['start_seconds'] == 10
    assert record['segments'][-1]['end_seconds'] == 70
    assert record['event_source'] == 'embedded_trigger_waveform'
    assert record['overall']['accepted_windows'] == 6
    assert [(s['start_seconds'], s['end_seconds'], s['complete']) for s in record['rest_segments']] == [(25, 35, True), (50, 60, True)]
    assert record['rest_overall']['accepted_windows'] == 4
    assert record['baseline'] == {'available': False, 'reason': '本协议不使用记录开头作为静息基线'}
    assert repository.model_baseline('subject-10') is None
    assert evidence['measurement_context']['comparison_policy'] == 'event_segments_only'
    assert evidence['measurement_context']['baseline_policy'] == 'disabled'
    assert not any(key.endswith('_baseline_change') for row in record['segments'] for key in row['stats'])
    assert record['model_compatible']
    assert not any(p.suffix in ('.mat', '.csv', '.xlsx') for p in read_paths)
    forbidden = {'label', 'path_score', 'subject_id', 'subject_key', 'questionnaire', 'ssq', 'fatigue_pre'}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
    check(evidence)
    for tool in ('vrms.raw_recording', 'fatigue.raw_spectrum', 'emotion.raw_workload'):
        check(recording_tool(tool, evidence))


def test_trigger_edges_and_gap_reset(raw_cfg):
    values = np.zeros((70000, 1), np.float32)
    values[65535:65540] = 20
    values[68000:68005] = 22
    assert trigger_events(values, 0) == [(65535, 20), (68000, 22)]
    path = Path(raw_cfg['data_root']) / 'data/raw/Acquisition 10.cdt'
    raw = np.fromfile(path, dtype='<f4').reshape(-1, 37)
    raw[27 * 1024:27 * 1024 + 10, 0] = np.nan
    processed, channels, compatible = preprocess(raw, EEG_CHANNELS + ['M1', 'M2', 'VEOG', 'HEOG', 'ECG', 'EGG', 'Trigger'], 1024)
    assert compatible and len(channels) == 30
    assert np.isfinite(processed[:, 20 * 256:21 * 256]).all()
    assert np.isnan(processed[:, 28 * 256:32 * 256]).all()
    assert np.isfinite(processed[:, 40 * 256:41 * 256]).all()


def test_event_states_require_marked_boundaries_and_do_not_assume_initial_rest():
    tasks, rests, warnings = state_segments_from_events([(10, 20), (25, 22), (35, 20), (50, 22)], 60, 1)
    assert [(s['start_seconds'], s['end_seconds'], s['complete']) for s in tasks] == [(10, 25, True), (35, 50, True)]
    assert [(s['start_seconds'], s['end_seconds'], s['complete']) for s in rests] == [(25, 35, True), (50, 60, False)]
    assert all(s['start_seconds'] >= 25 for s in rests)
    assert '下一开始标记' in rests[-1]['incomplete_reason']
    assert warnings == []
    assert state_segments_from_events([], 60, 1) == ([], [], [])


def test_duplicate_events_split_incomplete_states_instead_of_bridging_anomaly():
    tasks, rests, warnings = state_segments_from_events([(10, 20), (15, 20), (25, 22),
                                                        (30, 22), (40, 20)], 50, 1)
    assert [(s['start_seconds'], s['end_seconds'], s['complete']) for s in tasks] == [(10, 15, False), (15, 25, True), (40, 50, False)]
    assert [(s['start_seconds'], s['end_seconds'], s['complete']) for s in rests] == [(25, 30, False), (30, 40, True)]
    assert len(warnings) == 2
    assert '重复结束事件' in rests[0]['incomplete_reason']


def test_initial_orphan_end_starts_only_marked_rest_and_invalid_event_breaks_state():
    tasks, rests, warnings = state_segments_from_events([(5, 22), (15, 20), (18, 99), (25, 22)], 35, 1)
    assert [(s['start_seconds'], s['end_seconds'], s['complete']) for s in rests] == [(5, 15, True), (25, 35, False)]
    assert [(s['start_seconds'], s['end_seconds'], s['complete']) for s in tasks] == [(15, 18, False)]
    assert '结束事件前未找到开始事件' in warnings
    assert any('无效事件' in reason for reason in warnings)


def test_state_overall_pools_all_complete_windows_then_forms_ratio(raw_cfg, monkeypatch):
    import eeg_agent.recordings as module
    # Distinct powers per marked state make averaging segment ratios incorrect.
    def prepared(raw, channels, fs):
        marker = np.full(70 * 256, 777.0)
        for start, end, value in ((10, 25, 8), (25, 35, 12), (35, 50, 2), (50, 60, 6), (60, 70, 99)):
            marker[start * 256:end * 256] = value
        t = np.arange(len(marker)) / 256
        waves = np.tile(marker + np.sin(2 * np.pi * 10 * t), (6, 1))
        return waves, ['F3', 'Fz', 'F4', 'P3', 'Pz', 'P4'], False
    def powers(window):
        theta = round(float(window[0, 0]))
        alpha = {8: 1, 12: 3, 2: 2, 6: 1, 99: 1, 777: 1}[theta]
        return None, None, {'theta': np.full(6, theta), 'alpha': np.full(6, alpha), 'beta': np.ones(6)}
    monkeypatch.setattr(module, 'preprocess', prepared)
    monkeypatch.setattr(module, '_spectral', powers)
    repository = RecordingRepository(raw_cfg)
    analysis = repository.analyze('subject-10')
    record = analysis['recording']
    assert record['overall']['accepted_windows'] == 6
    assert record['overall']['theta_alpha'] == pytest.approx(10 / 3)
    assert record['overall']['workload_index'] == pytest.approx(10 / 3)
    assert record['rest_overall']['accepted_windows'] == 4
    assert record['rest_overall']['theta_alpha'] == pytest.approx(4.5)
    assert record['rest_overall']['workload_index'] == pytest.approx(4.5)
    assert record['segments'][-1]['stats']['theta_alpha'] == 99
    assert repository.model_baseline('subject-10') is None
    result = recording_tool('fatigue.raw_spectrum', analysis)
    measured = result['measurements']
    assert measured['task_value'] == pytest.approx(10 / 3)
    assert measured['rest_value'] == pytest.approx(4.5)
    assert measured['first_to_last']['last_ordinal'] == 2
    assert measured['rest_max_complete']['segments'][0]['ordinal'] == 2
    explanation = direct_summary('fatigue', [{'tool': 'fatigue.raw_spectrum', 'result': result}])
    assert '基线' not in explanation
    assert '完整休息段' in explanation and '00:25.000' in explanation


def test_natural_file_binding_and_missing_file(raw_cfg):
    repository = RecordingRepository(raw_cfg)
    assert repository.resolve('分析被试10：路径数、疲劳感、心理负荷和眩晕度如何？') == 'subject-10'
    assert repository.resolve('请分析 Acquisition 10 的原始波形') == 'subject-10'
    with pytest.raises(ValueError): repository.resolve('分析被试27')
    with pytest.raises(ValueError): repository.resolve('分析未上传的subject10.edf')
    question = repository.anonymous_question('被试10，问卷分数60，SSQ 80，分析脑电')
    assert '被试10' not in question and '60' not in question and '80' not in question


def test_raw_rag_excludes_uploaded_subject_answers(raw_cfg):
    knowledge = HierarchicalKnowledge(raw_cfg)
    knowledge.add_document(domain='shared', title='被试答案', source='questionnaire-test', text='被试10答案 高类 高类 高类 fatigue.raw_spectrum 心理负荷严重')
    result = knowledge.search('fatigue.raw_spectrum 高类 心理负荷', domains=['fatigue'], method_only=True)
    assert result['results']
    assert all(r['library'] == 'bundled' and r['source'] != 'questionnaire-test' for r in result['results'])


class Tasks:
    def list(self): return []
    def get(self, key): raise KeyError(key)


class Provider:
    calls = []
    instructions = []
    def __init__(self, backend, cfg, emit): self.client = None
    def call(self, role, payload, **kwargs):
        self.calls.append(copy.deepcopy(payload))
        self.instructions.append((role, copy.deepcopy(kwargs.get('messages', []))))
        if 'available_agents' in payload:
            return {'tasks': [{'agent': d, 'task': '分析当前原始 EEG'} for d in payload['available_agents']], 'explanation': '三个领域分别调用原始波形工具。'}
        if 'available_tools' in payload:
            return {'requested_tools': [s['id'] for s in payload['available_tools']]}
        results = payload.get('tool_results') or [r for a in payload['agent_reports'] for r in a['tool_results']]
        refs = payload.get('knowledge') or [r for a in payload.get('agent_reports', []) for r in a['references']]
        keys = {'vrms.raw_recording': ('task_count', '原始事件包含'), 'fatigue.raw_spectrum': ('task_value', '疲劳完整路径汇总指标为'), 'emotion.raw_workload': ('task_value', '心理负荷完整路径汇总指标为')}
        phrases, claims = [], []
        for item in results:
            if item['tool'] in keys:
                key, prefix = keys[item['tool']]
                value = item['result']['measurements'][key]
                phrases.append(prefix + str(value) + ('个任务片段' if key == 'task_count' else '') + '。来源：' + item['tool'])
                claims.append({'source': item['tool'], 'path': 'measurements.' + key, 'value': value})
        return {'explanation': '。\n\n'.join(phrases) + '。', 'citations': list(dict.fromkeys(r['id'] for r in refs))[:3], 'measurement_claims': claims}
    def summary(self): return {'backend': 'api', 'calls': len(self.calls), 'simulated': False}


def test_chat_resolves_recording_without_selection_and_restores(raw_cfg, monkeypatch):
    import eeg_agent.brain as module
    captured = []
    def model(repository, recording_id, evidence, provider=None, cloud=False):
        assert provider is not None and cloud
        captured.append((recording_id, evidence['recording']['signal_sha256']))
        return {'tool': 'vrms.raw_model', 'available': False, 'reason': '此测试不执行模型', 'measurements': {}}
    monkeypatch.setattr(module, 'assess_raw_model', model)
    Provider.calls = []
    Provider.instructions = []
    manager = BrainManager(raw_cfg, Tasks(), provider_factory=Provider)
    session_id = manager.create({'backend': 'api', 'domains': ['fatigue', 'emotion'], 'input_policy': 'raw_eeg_only'})['id']
    session = manager.get(session_id)
    session.message('请让三个 Agent 分析被试10，跑了多少条路径，疲劳、心理负荷指标如何？')
    deadline = time.monotonic() + 10
    while session.snapshot()['state'] == 'running' and time.monotonic() < deadline: time.sleep(.01)
    assert session.state == 'idle' and session.result['workflow_status'] == 'cloud_verified'
    assert session.result['measurement_context']['input_policy'] == 'raw_eeg_only'
    assert {r['tool'] for a in session.result['agent_reports'] for r in a['tool_results']} == {'vrms.raw_recording', 'vrms.raw_model', 'fatigue.raw_spectrum', 'emotion.raw_workload'}
    assert len(captured) == 1 and captured[0][0] == 'subject-10'
    assert session.domains == ['vrms', 'fatigue', 'emotion']
    assert any(e['kind'] == 'domains_selected' and e['payload']['added_domains'] == ['vrms'] for e in session.events)
    assert '被试10' not in json.dumps(Provider.calls, ensure_ascii=False)
    for role in ('FatigueAgent', 'EmotionAgent'):
        instruction = next(messages[0]['content'] for name, messages in Provider.instructions
                           if name == role and 'Answer only your assigned domain' in messages[0]['content'])
        assert 'those tools are outside your scope' in instruction
        assert 'definitive local model readout' not in instruction
        assert 'give the first High path interval' not in instruction
    restored = BrainManager(raw_cfg, Tasks(), provider_factory=Provider).get(session_id)
    assert restored.recording_id == 'subject-10' and restored.input_policy == 'raw_eeg_only'
    assert restored.domains == ['vrms', 'fatigue', 'emotion']


@pytest.mark.parametrize('text', ['疲劳相对基线下降。', '疲劳基线指标为3.2。'])
def test_raw_report_rejects_reintroduced_opening_baseline(tmp_path, text):
    from eeg_agent.brain import BrainSession
    session = object.__new__(BrainSession)
    session.recording_id, session.directory, session.turn = 'synthetic', tmp_path, 1
    with pytest.raises(ValueError, match='invalid raw baseline comparison'):
        session._checked_language({'explanation': text, 'citations': [], 'measurement_claims': []}, [], [], 'fatigue')


def test_raw_report_can_explain_disabled_baseline_and_report_direct_measurement(tmp_path):
    from eeg_agent.brain import BrainSession
    session = object.__new__(BrainSession)
    session.recording_id, session.directory, session.turn = 'synthetic', tmp_path, 1
    tools = [{'tool': 'fatigue.raw_spectrum', 'result': {'available': True, 'measurements': {'task_value': 3.2}}}]
    report = {'explanation': '本协议不使用记录开头作为静息基线。疲劳路径汇总指标为3.2，来源fatigue.raw_spectrum。',
              'citations': [], 'measurement_claims': [{'source': 'fatigue.raw_spectrum', 'path': 'measurements.task_value', 'value': 3.2}]}
    assert session._checked_language(report, [], tools, 'fatigue')[0] == report['explanation']


@pytest.mark.parametrize('text,selected,expected', [
    ('请让三个 Agent 协作分析', ['fatigue', 'emotion'], ['vrms', 'fatigue', 'emotion']),
    ('请让 3 个 Agent 协作', ['emotion'], ['vrms', 'fatigue', 'emotion']),
    ('让全部领域 Agent 协作', ['fatigue'], ['vrms', 'fatigue', 'emotion']),
    ('all three agents', ['emotion'], ['vrms', 'fatigue', 'emotion']),
    ('眩晕和疲劳指标如何变化？', ['emotion'], ['vrms', 'fatigue', 'emotion']),
    ('请让VRMSAgent分析', ['fatigue'], ['vrms', 'fatigue']),
    ('只让FatigueAgent回答，其他领域不用', ['vrms', 'fatigue', 'emotion'], ['fatigue']),
    ('仅分析心理负荷', ['vrms', 'fatigue'], ['emotion']),
    ('仅分析疲劳和心理负荷，先忽略眩晕', ['vrms'], ['fatigue', 'emotion']),
    ('请继续解释这个指标', ['fatigue'], ['fatigue']),
    ('第二条路径呢？', ['emotion'], ['emotion']),
])
def test_question_routes_explicit_agents_and_keeps_generic_followup(text, selected, expected):
    assert question_domains(text, selected) == expected


def test_vague_or_empty_answer_is_rejected(raw_cfg):
    manager = BrainManager(raw_cfg, Tasks(), provider_factory=Provider)
    session = manager.get(manager.create({'backend': 'mock', 'domains': ['fatigue'], 'recording_id': 'subject-10'})['id'])
    with pytest.raises(ValueError):
        session._checked_language({'explanation': '可能疲劳，证据不足。', 'citations': []}, [], [], 'fatigue')
    result = recording_tool('fatigue.raw_spectrum', manager.recordings.analyze('subject-10'))
    with pytest.raises(ValueError):
        session._checked_language({'explanation': '工具分析已经完成。', 'citations': [], 'measurement_claims': []}, [], [{'tool': 'fatigue.raw_spectrum', 'result': result}], 'fatigue')


def test_upload_validation_does_not_change_existing_sources(raw_cfg):
    manager = BrainManager(raw_cfg, Tasks())
    client = TestClient(create_app(raw_cfg, task_manager=Tasks(), brain_manager=manager, knowledge_index=object()))
    assert client.get('/api/brain/recordings').json()['recordings'][0]['id'] == 'subject-10'
    assert client.post('/api/brain/recordings/edf?filename=bad.txt', content=b'bad').status_code == 400
    assert client.post('/api/brain/recordings/edf?filename=empty.edf', content=b'').status_code == 400
    assert client.post('/api/brain/recordings/edf?filename=bad.edf', content=b'not an EDF').status_code == 400
    assert not list(manager.recordings.root.glob('_upload_*'))


def test_valid_edf_upload_uses_waveforms_and_does_not_invent_paths(raw_cfg):
    def field(value, size): return str(value).ljust(size).encode('ascii')
    ns, fs, duration = 3, 256, 60
    header = (field('0', 8) + field('anonymous', 80) + field('EEG test', 80) + field('01.01.26', 8) +
              field('00.00.00', 8) + field(256 + 256 * ns, 8) + field('', 44) + field(duration, 8) + field(1, 8) + field(ns, 4))
    for values, width in ((['F3', 'F4', 'Oz'], 16), ([''] * ns, 80), (['uV'] * ns, 8), ([-100] * ns, 8),
                          ([100] * ns, 8), ([-32768] * ns, 8), ([32767] * ns, 8), ([''] * ns, 80), ([fs] * ns, 8), ([''] * ns, 32)):
        header += b''.join(field(v, width) for v in values)
    t = np.arange(duration * fs) / fs
    waves = np.array([5000 * (np.sin(2 * np.pi * 6 * t) + np.sin(2 * np.pi * 10 * t)) * (1 + i * .1) for i in range(ns)], dtype='<i2')
    data = b''.join(waves[:, second * fs:(second + 1) * fs].tobytes() for second in range(duration))
    manager = BrainManager(raw_cfg, Tasks())
    client = TestClient(create_app(raw_cfg, task_manager=Tasks(), brain_manager=manager, knowledge_index=object()))
    response = client.post('/api/brain/recordings/edf?filename=example.edf', content=header + data)
    assert response.status_code == 200
    token = response.json()['id']
    assert manager.recordings.resolve('请分析 example.edf 的疲劳指标') == token
    record = manager.recordings.analyze(token)['recording']
    assert record['task_count'] is None and record['event_source'] == 'no_task_events'
    assert record['overall']['theta_alpha'] is not None
    assert record['rest_segments'] == []
    assert not record['rest_overall']['available']
    assert manager.recordings.model_baseline(token) is None
    assert record['overall']['workload_index'] is None and not record['model_compatible']
    from eeg_agent.raw_model import assess
    analysis = manager.recordings.analyze(token)
    assert not assess(manager.recordings, token, analysis)['available']
    assert '未形成整路径' in assess(manager.recordings, token, analysis)['reason']


def test_raw_answer_cannot_swap_path_class_or_invent_psychological_grade(raw_cfg):
    manager = BrainManager(raw_cfg, Tasks())
    session = manager.get(manager.create({'backend': 'mock', 'domains': ['vrms'], 'recording_id': 'subject-10'})['id'])
    result = {'available': True, 'measurements': {'segments': [{'ordinal': 1, 'predicted_class': 'Low'}, {'ordinal': 2, 'predicted_class': 'High'}], 'first_high': {'ordinal': 2}}}
    claims = [{'source': 'vrms.raw_model', 'path': 'measurements.segments.0.ordinal', 'value': 1},
              {'source': 'vrms.raw_model', 'path': 'measurements.segments.1.predicted_class', 'value': 'High'}]
    for text in ('VRMS 模型第1条判为High。', '首条高类为第1条。', '心理负荷很高。'):
        with pytest.raises(ValueError):
            session._checked_language({'explanation': text, 'citations': [], 'measurement_claims': claims}, [], [{'tool': 'vrms.raw_model', 'result': result}], 'vrms')
