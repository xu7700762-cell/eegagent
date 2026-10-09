# -*- coding: utf-8 -*-
"""Real tool-protocol routing and class provenance, using a deterministic API double."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from eeg_agent.agents import CloudProvider
from eeg_agent import raw_model, raw_tool_agent


def prediction(ordinal=1):
    return {'ordinal': ordinal, 'mil_raw_high_probability': .4, 'mil_class': 'Low',
            'accepted_windows': 3, 'predicted_class': 'High', 'classification_layer': 'numeric_fusion',
            'numeric_fusion': {'raw_combined_score': .6, 'predicted_class': 'High',
                'actual_tools': [{'tool': name, 'predicted_class': 'High', 'raw_decision_score': .2}
                                 for name in raw_tool_agent.TOOL_NAMES]}}


def final(cls='Low', tools=('corrective_evidence',)):
    return {'id': 'qsingle', 'final_class': cls, 'confidence': 'low', 'tools_used': list(tools),
            'supporting_evidence': ['实际频谱证据'], 'conflicting_evidence': ['工具方向存在冲突'],
            'missing_evidence': [], 'explanation': '最终判为Low，保留工具冲突并降低定性置信度。'}


class Provider:
    model = 'gpt-6.1-sol'

    def __init__(self, final_value=None, fail=False, direct=False, name='corrective_evidence'):
        self.calls = []
        self.final_value = final_value or final()
        self.fail, self.direct, self.name = fail, direct, name

    def call(self, role, payload, tools=None, messages=None):
        self.calls.append(copy.deepcopy({'role': role, 'payload': payload, 'tools': tools, 'messages': messages}))
        if self.fail:
            raise RuntimeError('secret-from-a-provider-exception')
        if len(self.calls) == 1 and not self.direct:
            output = [{'type': 'function_call', 'name': self.name, 'call_id': 'call1',
                       'arguments': json.dumps({'query': 'qsingle'})}]
        else:
            output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(self.final_value)}]}]
        return {'status': 'completed', 'output': output}


def test_gpt_discrete_class_can_differ_without_mutating_numeric_evidence_or_sending_identity():
    row = prediction()
    row.update(subject_id='secret-subject', questionnaire=99, target_label='High', waveform=np.ones((30, 1280)))
    numeric = copy.deepcopy(row['numeric_fusion'])
    provider = Provider()
    result = raw_tool_agent.classify(provider, row)
    assert result['available'] and result['final_class'] == 'Low'
    assert result['generation_status'] == 'cloud_verified'
    assert result['actual_tool_calls'] == [*raw_tool_agent.TOOL_NAMES, 'corrective_evidence']
    assert row['numeric_fusion'] == numeric and row['predicted_class'] == 'High'
    outbound = json.dumps(provider.calls)
    assert not any(value in outbound for value in ('secret-subject', 'questionnaire": 99', 'target_label', 'waveform'))
    tool_output = provider.calls[1]['messages'][-1]
    assert tool_output['type'] == 'function_call_output'
    assert json.loads(tool_output['output'])['raw_combined_class'] == 'High'


@pytest.mark.parametrize('provider', [Provider(fail=True), Provider(direct=True),
    Provider(name='unregistered'), Provider(final_value={**final(), 'llm_probability': .7}),
    Provider(final_value=final(tools=('unexecuted',)))])
def test_failed_invalid_or_unexecuted_tool_reply_never_becomes_a_final_class(provider):
    result = raw_tool_agent.classify(provider, prediction())
    assert not result['available'] and result['final_class'] is None
    assert result['generation_status'] == 'fallback'
    assert 'secret-from-a-provider-exception' not in json.dumps(result)


def test_non_gpt_configuration_is_not_silently_used_as_gpt():
    provider = Provider()
    provider.model = 'deepseek-flash'
    result = raw_tool_agent.classify(provider, prediction())
    assert not result['available'] and not provider.calls and 'CPA GPT' in result['reason']


def test_public_report_omits_numerical_fusion_and_keeps_internal_records():
    row = prediction()
    row['cloud_judgment'] = {'available': True, 'final_class': 'Low', 'confidence': 'low',
                             'rounds': [{}], 'tool_outputs': {'corrective_evidence': {'raw_combined_class': 'High'}}}
    row['predicted_class'] = 'Low'
    row['classification_layer'] = 'gpt_tool_agent'
    result = {'tool': 'vrms.raw_model', 'gpt_requested': True, 'measurements': {'segments': [row]}}
    public = raw_model.public_result(result)
    assert public['measurements']['segments'][0]['predicted_class'] == 'Low'
    assert not any(key in json.dumps(public) for key in ('numeric_fusion', 'mil_class', 'raw_combined_class', 'rounds'))
    assert result['measurements']['segments'][0]['numeric_fusion']['predicted_class'] == 'High'


def test_sdk_uses_responses_function_calls_and_shared_budget():
    provider = object.__new__(CloudProvider)
    provider.backend, provider.model = 'api', 'gpt-6.1-sol'
    provider.cfg = {'api_max_calls': 1}
    provider.calls, provider.lock, provider.budget_rejections = [], threading.Lock(), 0
    provider.emit = lambda *args: None
    observed = []

    def create(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(status='completed', usage=None,
            model_dump=lambda **_: {'status': 'completed', 'output': []})

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = provider.call('VRMSAgent', {'generation_phase': 'vrms_tool_agent', 'step': 0},
                           tools=raw_tool_agent.TOOLS, messages=[{'role': 'user', 'content': 'anonymous'}])
    assert result['status'] == 'completed'
    assert observed[0]['tool_choice'] == 'required' and observed[0]['store'] is False
    assert observed[0]['text']['format']['type'] == 'json_schema'
    assert provider.calls[0]['api_protocol'] == 'responses'
    with pytest.raises(RuntimeError, match='budget'):
        provider.call('VRMSAgent', {'generation_phase': 'vrms_tool_agent', 'step': 1})
    assert len(observed) == 1 and provider.budget_rejections == 1
