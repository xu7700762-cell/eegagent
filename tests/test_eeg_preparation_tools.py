import numpy as np
import pytest

from eeg_agent.brain import _normalize_tool_requests
from eeg_agent.domain_tools import execute_tool, toolkit_catalog
from eeg_agent.recordings import (apply_channel_repair, assess_channel_quality,
                                  normalize_repair_plan)


def _signals():
    t = np.linspace(0, 20, 2000, endpoint=False)
    base = np.vstack([
        np.sin(2 * np.pi * 1.0 * t),
        np.sin(2 * np.pi * 1.0 * t + .1),
        np.sin(2 * np.pi * 1.0 * t - .1),
        np.sin(2 * np.pi * 1.0 * t + .2),
    ]).astype(np.float32)
    base[0] = 0.0
    return base


def test_quality_assessor_flags_flat_channel_and_repair_interpolates_named_donors():
    names = ['F3', 'Fz', 'F4', 'Cz']
    report = assess_channel_quality(_signals(), names)
    assert report['status'] == 'review'
    assert 'F3' in report['bad_channels']
    repaired, returned_names, details = apply_channel_repair(
        _signals(), names, ['F3'], strategy='interpolate', min_neighbors=2)
    assert returned_names == names
    assert details['donors']['F3'] == ['Fz', 'F4']
    assert np.allclose(repaired[0], np.mean(repaired[[1, 2]], axis=0))


def test_quality_assessor_does_not_turn_one_shared_transient_into_bad_leads():
    samples = 3 * 1280
    t = np.arange(samples, dtype=float) / 256.0
    signal = np.vstack([
        np.sin(2 * np.pi * 1.0 * t),
        np.sin(2 * np.pi * 1.0 * t + .1),
        np.sin(2 * np.pi * 1.0 * t - .1),
        np.sin(2 * np.pi * 1.0 * t + .2),
    ])
    # A short common artifact is a window event, not four permanently broken
    # electrodes.  The window aggregator should ignore it for interpolation.
    signal[:, 100] += 5000.0
    report = assess_channel_quality(signal, ['F3', 'Fz', 'F4', 'Cz'])
    assert report['status'] == 'pass'
    assert report['persistent_bad_channels'] == []
    assert report['blocking_bad_channels'] == []
    assert report['window_qc']['shared_artifact_windows'] == 1
    assert report['window_qc']['shared_artifact_fraction'] < .5


def test_quality_assessor_marks_a_flat_lead_persistent_across_windows():
    samples = 3 * 1280
    t = np.arange(samples, dtype=float) / 256.0
    signal = np.vstack([
        np.zeros(samples),
        np.sin(2 * np.pi * 1.0 * t),
        np.sin(2 * np.pi * 1.0 * t + .1),
        np.sin(2 * np.pi * 1.0 * t + .2),
    ])
    report = assess_channel_quality(signal, ['F3', 'Fz', 'F4', 'Cz'])
    assert report['status'] == 'review'
    assert 'F3' in report['persistent_bad_channels']
    assert 'F3' in report['blocking_bad_channels']
    assert report['channels'][0]['bad_window_fraction'] == 1.0


def test_channel_drop_removes_bad_lead_and_marks_model_contract_as_noncompatible_by_shape():
    names = ['F3', 'Fz', 'F4']
    repaired, returned_names, details = apply_channel_repair(
        _signals()[:3], names, ['F3'], strategy='drop')
    assert returned_names == ['Fz', 'F4']
    assert details['strategy'] == 'drop'
    assert repaired.shape == (2, 2000)


def test_interpolation_rejects_missing_named_donors_instead_of_using_arbitrary_channels():
    names = ['F3', 'Cz']
    with pytest.raises(ValueError, match='neighbouring donor'):
        apply_channel_repair(_signals()[:2], names, ['F3'], strategy='interpolate', min_neighbors=2)


def test_shared_tools_are_visible_to_all_three_specialists():
    catalog = toolkit_catalog(include_recordings=True)
    expected = {'EEGFileLoader', 'EEGPreprocessor', 'EEGQualityAssessor', 'EEGChannelRepair'}
    assert all(expected.issubset({item['id'] for item in catalog[domain]})
               for domain in ('vrms', 'fatigue', 'emotion'))


def test_preparation_tools_return_only_local_summaries_and_repair_reloads_contract():
    processed = _signals()
    quality = assess_channel_quality(processed, ['F3', 'Fz', 'F4', 'Cz'])
    record = {
        'sampling_hz': 256, 'eeg_channels': ['F3', 'Fz', 'F4', 'Cz'],
        'event_source': 'embedded_trigger_waveform', 'duration_seconds': 20.,
        'signal_sha256': 'x', 'preprocessing': {'profile': 'raw-vrms-causal-v1', 'output_sampling_hz': 256},
        'channel_quality': {'after_repair': quality},
        'overall': {'accepted_windows': 3, 'total_windows': 4, 'window_coverage': .75},
        'model_compatible': False, 'channel_repair': {'strategy': 'none'},
    }
    evidence = {'recording': record}
    loaded = execute_tool('EEGFileLoader', evidence)
    quality_result = execute_tool('EEGQualityAssessor', evidence)
    preprocessed = execute_tool('EEGPreprocessor', evidence)
    assert loaded['available'] and 'signal_sha256' in loaded
    assert quality_result['quality']['bad_channels'] == ['F3']
    assert preprocessed['preprocessing']['profile'] == 'raw-vrms-causal-v1'

    class Repository:
        def analyze(self, recording_id, repair_plan):
            assert recording_id == 'subject-01'
            assert repair_plan == normalize_repair_plan({'strategy': 'interpolate', 'bad_channels': ['F3']})
            return {'recording': {**record, 'channel_repair': {'strategy': 'interpolate', 'bad_channels': ['F3']},
                                  'model_compatible': False,
                                  'channel_quality': {'after_repair': quality},
                                  'overall': record['overall']}}

    repaired = execute_tool('EEGChannelRepair', evidence, repository=Repository(),
                            recording_id='subject-01',
                            args={'strategy': 'interpolate', 'bad_channels': ['F3']})
    assert repaired['available'] and repaired['applied']


def test_planner_accepts_tool_arguments_without_allowing_unknown_tools():
    requests = _normalize_tool_requests([
        'EEGFileLoader',
        {'tool_name': 'EEGChannelRepair',
         'tool_args': {'strategy': 'drop', 'bad_channels': ['F3']}},
    ], {'EEGFileLoader', 'EEGChannelRepair'})
    assert requests[1][0] == 'EEGChannelRepair'
    assert requests[1][1]['strategy'] == 'drop'
