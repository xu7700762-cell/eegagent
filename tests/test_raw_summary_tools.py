# -*- coding: utf-8 -*-
"""Comparison contracts for raw EEG summaries; no API, labels or model run."""
import pytest

from eeg_agent.recordings import recording_tool


TOOLS = [('fatigue.raw_spectrum', 'theta_alpha'), ('emotion.raw_workload', 'workload_index')]


def evidence(rows, key, scope='task_paths', rests=()):
    segments = []
    rest_segments = []
    for values, target in ((rows, segments), (rests, rest_segments)):
        for ordinal, value, complete in values:
            target.append({'ordinal': ordinal, 'complete': complete,
                           'start_clock': f'{ordinal:02d}:00.000', 'end_clock': f'{ordinal:02d}:45.000',
                           'stats': {key: value, 'accepted_windows': 9}})
    return {'recording': {'signal_sha256': 'raw-test-waveform', 'segment_scope': scope,
                         'baseline': {'available': False, 'reason': '本协议不使用记录开头作为静息基线'},
                         'overall': {key: 2.0}, 'rest_overall': {key: 3.0 if rests else None},
                         'segments': segments, 'rest_segments': rest_segments}}


@pytest.mark.parametrize('tool,key', TOOLS)
def test_normal_comparison_includes_values_and_actual_intervals(tool, key):
    result = recording_tool(tool, evidence([(1, 4.0, True), (2, 3.0, True), (3, 2.0, True)], key))
    measured = result['measurements']
    assert measured['segment_scope'] == 'task_paths'
    assert measured['baseline_available'] is False
    assert not {'baseline_interval', 'baseline_value', 'baseline_scope', 'baseline_change'}.intersection(measured)
    assert '先汇总频带功率后求比' in measured['aggregation']
    comparison = measured['first_to_last']
    assert (comparison['first_ordinal'], comparison['last_ordinal']) == (1, 3)
    assert (comparison['first_value'], comparison['last_value']) == (4.0, 2.0)
    assert comparison['change']['relative_change'] == pytest.approx(-.5)
    assert comparison['change']['direction'] == '下降'
    assert (comparison['first_start_clock'], comparison['first_end_clock']) == ('01:00.000', '01:45.000')
    assert (comparison['last_start_clock'], comparison['last_end_clock']) == ('03:00.000', '03:45.000')
    assert 'first_to_last_unavailable_reason' not in measured


@pytest.mark.parametrize('tool,key', TOOLS)
def test_maximum_excludes_unfinished_segment_and_keeps_all_ties(tool, key):
    measured = recording_tool(tool, evidence([(1, 2.0, True), (2, 4.0, True),
                                              (3, 4.0, True), (4, 99.0, False)], key))['measurements']
    maximum = measured['max_complete']
    assert maximum == {'value': 4.0,
                       'segments': [{'ordinal': 2, 'value': 4.0, 'start_clock': '02:00.000', 'end_clock': '02:45.000', 'complete': True},
                                    {'ordinal': 3, 'value': 4.0, 'start_clock': '03:00.000', 'end_clock': '03:45.000', 'complete': True}],
                       'compared_count': 3, 'missing_count': 0}
    assert measured['first_to_last']['last_ordinal'] == 3


@pytest.mark.parametrize('rows,missing_ordinal', [
    ([(1, None, True), (2, 3.0, True), (3, 2.0, True)], 1),
    ([(1, 2.0, True), (2, 3.0, True), (3, float('nan'), True)], 3),
])
def test_endpoint_without_value_never_substitutes_middle_path(rows, missing_ordinal):
    measured = recording_tool('fatigue.raw_spectrum', evidence(rows, 'theta_alpha'))['measurements']
    assert measured['first_to_last'] is None
    assert f'第 {missing_ordinal} 条' in measured['first_to_last_unavailable_reason']
    assert '不能替换首末端点' in measured['first_to_last_unavailable_reason']
    assert measured['max_complete']['compared_count'] == 2
    assert measured['max_complete']['missing_count'] == 1


def test_ranking_rejects_missing_and_nonfinite_values():
    measured = recording_tool('emotion.raw_workload', evidence(
        [(1, 2.0, True), (2, None, True), (3, float('inf'), True),
         (4, float('nan'), True), (5, 1.0, True), (6, 500.0, False)], 'workload_index'))['measurements']
    assert measured['max_complete']['value'] == 2.0
    assert measured['max_complete']['compared_count'] == 2
    assert measured['max_complete']['missing_count'] == 3
    assert measured['first_to_last']['last_ordinal'] == 5


@pytest.mark.parametrize('tool,key', TOOLS)
def test_whole_recording_never_invents_path_comparison(tool, key):
    measured = recording_tool(tool, evidence([(1, 2.0, True), (2, 3.0, True)], key,
                                             scope='whole_recording'))['measurements']
    assert measured['segment_scope'] == 'whole_recording'
    assert measured['first_to_last'] is None and measured['max_complete'] is None
    assert '没有任务事件' in measured['first_to_last_unavailable_reason']
    assert '没有任务事件' in measured['max_complete_unavailable_reason']
    assert '路径' not in measured['aggregation']
    assert measured['rest_first_to_last'] is None and measured['rest_max_complete'] is None
    assert '没有任务事件' in measured['rest_first_to_last_unavailable_reason']


def test_single_complete_path_has_rank_but_no_first_last_comparison():
    measured = recording_tool('fatigue.raw_spectrum', evidence(
        [(1, 3.0, True), (2, 6.0, False)], 'theta_alpha'))['measurements']
    assert measured['first_to_last'] is None
    assert '至少两条' in measured['first_to_last_unavailable_reason']
    assert measured['max_complete']['value'] == 3.0


def test_no_valid_complete_indicator_has_specific_unavailability():
    measured = recording_tool('emotion.raw_workload', evidence(
        [(1, None, True), (2, float('nan'), True)], 'workload_index'))['measurements']
    assert measured['max_complete'] is None
    assert '没有取得有效指标的完整路径' in measured['max_complete_unavailable_reason']


@pytest.mark.parametrize('tool,key', TOOLS)
def test_rest_comparisons_have_independent_endpoints_peak_and_intervals(tool, key):
    measured = recording_tool(tool, evidence([(1, 2.0, True), (2, 3.0, True)], key,
        rests=[(1, 5.0, True), (2, 4.0, True), (3, 10.0, False)]))['measurements']
    assert measured['rest_value'] == 3.0
    assert (measured['rest_first_to_last']['first_value'], measured['rest_first_to_last']['last_value']) == (5.0, 4.0)
    assert measured['rest_first_to_last']['change']['direction'] == '下降'
    assert measured['rest_max_complete']['value'] == 5.0
    assert measured['rest_max_complete']['segments'][0]['complete'] is True
    assert measured['rest_max_complete']['compared_count'] == 2


def test_missing_rest_endpoint_does_not_substitute_measured_middle():
    measured = recording_tool('fatigue.raw_spectrum', evidence([(1, 2.0, True)], 'theta_alpha',
        rests=[(1, None, True), (2, 4.0, True), (3, 5.0, True)]))['measurements']
    assert measured['rest_first_to_last'] is None
    assert '第 1 条完整休息段' in measured['rest_first_to_last_unavailable_reason']
    assert measured['rest_max_complete']['missing_count'] == 1


def test_available_task_has_specific_reason_for_missing_rest_measurement():
    result = recording_tool('fatigue.raw_spectrum', evidence([(1, 2.0, True)], 'theta_alpha'))
    assert result['available']
    measured = result['measurements']
    assert measured['rest_value'] is None
    assert measured['rest_unavailable_reason'] == '原始记录没有完整休息事件段，休息指标未定义'
    assert 'task_unavailable_reason' not in measured


def test_only_rest_available_keeps_tool_available_and_explains_missing_task():
    analysis = evidence([(1, None, False)], 'theta_alpha', rests=[(1, 3.0, True)])
    analysis['recording']['overall'] = {'accepted_windows': 0}
    result = recording_tool('fatigue.raw_spectrum', analysis)
    assert result['available'] and result['status'] == 'ok'
    measured = result['measurements']
    assert measured['task_value'] is None and measured['rest_value'] == 3.0
    assert measured['task_unavailable_reason'] == '原始记录没有完整路径事件段，路径指标未定义'
    assert 'rest_unavailable_reason' not in measured


def test_complete_rest_without_quality_windows_has_quality_reason():
    analysis = evidence([(1, 2.0, True)], 'theta_alpha', rests=[(1, None, True)])
    analysis['recording']['rest_overall'] = {'accepted_windows': 0}
    measured = recording_tool('fatigue.raw_spectrum', analysis)['measurements']
    assert measured['rest_unavailable_reason'] == '完整休息段没有通过质量检查的完整五秒 EEG 窗口'


def test_quality_rest_without_region_channels_has_channel_reason():
    analysis = evidence([(1, 2.0, True)], 'workload_index', rests=[(1, None, True)])
    analysis['recording']['eeg_channels'] = ['F3', 'F4', 'Oz']
    analysis['recording']['rest_overall'] = {'accepted_windows': 9}
    measured = recording_tool('emotion.raw_workload', analysis)['measurements']
    assert '缺少工作负荷指标所需' in measured['rest_unavailable_reason']
    assert '质量检查' not in measured['rest_unavailable_reason']


def test_complete_task_without_quality_windows_has_quality_reason():
    analysis = evidence([(1, None, True)], 'theta_alpha', rests=[(1, 3.0, True)])
    analysis['recording']['overall'] = {'accepted_windows': 0}
    result = recording_tool('fatigue.raw_spectrum', analysis)
    assert result['available']
    assert result['measurements']['task_unavailable_reason'] == '完整路径没有通过质量检查的完整五秒 EEG 窗口'
