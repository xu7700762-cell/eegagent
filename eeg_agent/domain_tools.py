# -*- coding: utf-8 -*-
import copy

from .recordings import recording_tool


_GENERAL = [
    {'id': 'EEGFileLoader', 'name': 'EEG文件加载器',
     'description': '在本地受限目录加载一个原始EEG记录，并返回匿名格式、通道、采样率和完整性摘要。不得向模型返回文件路径或原始波形。',
     'input': '已选定的本地原始EEG记录', 'output': '格式、通道、采样率、事件和文件完整性摘要',
     'capability': 'data_loading', 'parameters': {'recording_id': '当前记录，由控制器绑定'}},
    {'id': 'EEGPreprocessor', 'name': 'EEG预处理器',
     'description': '执行当前输入合同规定的重参考、陷波、带通、低通、重采样、断流预热和五秒分窗，并报告实际采用的参数。',
     'input': '本地原始EEG', 'output': '预处理配置、输出采样率和可分析窗口摘要',
     'capability': 'preprocessing', 'parameters': {'profile': 'raw-vrms-causal-v1'}},
    {'id': 'EEGQualityAssessor', 'name': 'EEG质量评估器',
     'description': '按五秒窗口聚合逐导联有限值、峰峰值、标准差、稳健振幅离群和中位参考相关性；区分持续坏导联与短暂伪迹，返回质量分数和窗口覆盖率。',
     'input': '预处理后的本地EEG', 'output': '逐导联质量、坏导联、合格窗口和覆盖率',
     'capability': 'quality_control',
     'parameters': {'z_threshold': 3.0, 'corr_threshold': .4, 'amplitude_threshold_uv': 2000.0,
                    'flat_threshold_uv': .05, 'window_seconds': 5.0,
                    'persistence_threshold': .5}},
    {'id': 'EEGChannelRepair', 'name': 'EEG坏导联处理器',
     'description': '根据质量评估结果对坏导联执行保守处理：drop 删除该导联，interpolate 使用明确邻近导联均值插值；处理后重新执行质量检查和模型输入合同核验。',
     'input': '本地预处理EEG及坏导联列表', 'output': '处理方式、坏导联、插值供体、剩余通道和后处理质量摘要',
     'capability': 'channel_repair',
     'parameters': {'strategy': 'none|drop|interpolate', 'bad_channels': [], 'auto_detect': False,
                    'min_neighbors': 2}},
]

_CATALOG = {
    'vrms': [
        {'id': 'vrms.raw_recording', 'name': '原始事件与路径统计',
         'description': '从原始波形内的触发通道统计完整与未结束路径，并返回时间区间。',
         'input': '原始 EEG 及内嵌事件通道', 'output': '路径数量、完整性、起止时间',
         'capability': 'measurement', 'classifier_required': False},
        {'id': 'vrms.raw_model', 'name': '原始 EEG 与 GPT 工具 Agent',
         'description': '重新编码原始 EEG，执行冻结模型与八路工具；GPT 实际调用工具后给出最终 High/Low。',
         'input': '质量合格的原始 EEG 五秒窗口与已配置 CPA GPT',
         'output': 'GPT 最终分类、真实工具调用和首条高类路径区间',
         'capability': 'raw_vrms_gpt_tool_agent'},
    ],
    'fatigue': [
        {'id': 'fatigue.raw_spectrum', 'name': '整记录疲劳相关指标',
         'description': '按事件边界计算完整路径及休息段的 θ/α，直接比较同类片段的首末值与最高值。',
         'input': '原始 EEG 及内嵌任务事件', 'output': 'TAR、路径与休息段值、首末变化、最高片段及时间',
         'capability': 'measurement'},
    ],
    'emotion': [
        {'id': 'emotion.raw_workload', 'name': '整记录心理负荷与额区指标',
         'description': '按事件边界计算路径和休息段的额区 θ/顶区 α 及 FAA，直接比较同类片段的首末值与最高值。',
         'input': '原始 EEG，F3/Fz/F4 与 P3/Pz/P4 通道',
         'output': 'EEG工作负荷指标及额区不对称性、路径与休息段汇总',
         'capability': 'measurement'},
    ],
}

GENERAL_IDS = {spec['id'] for spec in _GENERAL}


def toolkit_catalog(include_recordings=True):
    result = {domain: [{**spec, 'domain': domain, 'input_scope': 'raw_eeg_only',
        'classifier_required': spec.get('classifier_required', domain == 'vrms'),
        'classifier_available': spec.get('classifier_required', domain == 'vrms')} for spec in specs]
        for domain, specs in _CATALOG.items()}
    if include_recordings:
        for domain in result:
            general = [dict(spec, domain=domain, input_scope='raw_eeg_only',
                            classifier_required=False, classifier_available=False, shared=True)
                       for spec in _GENERAL]
            result[domain] = general + result[domain]
    return copy.deepcopy(result)


def _quality_result(record, args):
    from .recordings import reclassify_channel_quality
    args = args or {}
    quality = record.get('channel_quality', {}).get('after_repair')
    if quality is None:
        return {'tool': 'EEGQualityAssessor', 'available': False,
                'reason': '当前记录没有预处理质量摘要'}
    try:
        result = reclassify_channel_quality(
            quality, z_threshold=float(args.get('z_threshold', 3.0)),
            corr_threshold=float(args.get('corr_threshold', .4)),
            amplitude_threshold=float(args.get('amplitude_threshold_uv', 2000.0)),
            flat_threshold=float(args.get('flat_threshold_uv', .05)))
    except (TypeError, ValueError) as exc:
        return {'tool': 'EEGQualityAssessor', 'available': False, 'reason': str(exc)}
    overall = record.get('overall', {})
    return {'tool': 'EEGQualityAssessor', 'available': True, 'quality': result,
            'accepted_windows': overall.get('accepted_windows'),
            'total_windows': overall.get('total_windows'),
            'window_coverage': overall.get('window_coverage'),
            'model_compatible': record.get('model_compatible')}


def execute_tool(tool, evidence, *, repository=None, recording_id=None, args=None):
    registered = {s['id'] for specs in _CATALOG.values() for s in specs} | GENERAL_IDS
    if tool not in registered or tool == 'vrms.raw_model':
        raise ValueError('Unknown descriptive recording tool')
    if 'recording' not in evidence:
        return {'tool': tool, 'available': False,
                'reason': '当前没有原始 EEG，可先进行方法问答', 'measurements': {}}
    record = evidence['recording']
    args = {} if args is None else dict(args)
    if tool == 'EEGFileLoader':
        return {'tool': tool, 'available': True, 'format': 'CDT/DPO or EDF',
                'sampling_hz': record.get('sampling_hz'),
                'channel_count': len(record.get('eeg_channels', [])),
                'channels': record.get('eeg_channels', []),
                'event_source': record.get('event_source'),
                'duration_seconds': record.get('duration_seconds'),
                'signal_sha256': record.get('signal_sha256')}
    if tool == 'EEGPreprocessor':
        profile = record.get('preprocessing', {})
        requested = args.get('profile', profile.get('profile'))
        if requested != profile.get('profile'):
            return {'tool': tool, 'available': False,
                    'reason': '当前模型只允许冻结的本地预处理合同',
                    'requested_profile': requested,
                    'applied_profile': profile.get('profile')}
        return {'tool': tool, 'available': True, 'preprocessing': profile,
                'channel_count': len(record.get('eeg_channels', [])),
                'accepted_windows': record.get('overall', {}).get('accepted_windows'),
                'total_windows': record.get('overall', {}).get('total_windows'),
                'window_coverage': record.get('overall', {}).get('window_coverage')}
    if tool == 'EEGQualityAssessor':
        return _quality_result(record, args)
    if tool == 'EEGChannelRepair':
        if repository is None or recording_id is None:
            return {'tool': tool, 'available': False,
                    'reason': '坏导联处理需要本地记录控制器'}
        from .recordings import normalize_repair_plan
        try:
            plan = normalize_repair_plan(args)
            if plan['strategy'] == 'none':
                return {'tool': tool, 'available': True, 'applied': False,
                        'reason': '未请求丢弃或插值；仅保留质量报告',
                        'current_repair': record.get('channel_repair')}
            current = record.get('channel_repair', {})
            if current.get('strategy') not in (None, 'none') and current != plan:
                return {'tool': tool, 'available': False, 'applied': False,
                        'reason': '当前记录已经采用另一种坏导联处理方案，拒绝在同一轮混用'}
            updated = repository.analyze(recording_id, repair_plan=plan)
            repaired = updated['recording'].get('channel_repair', {})
            return {'tool': tool, 'available': True,
                    'applied': repaired.get('strategy') != 'none', 'repair': repaired,
                    'quality_after': updated['recording'].get('channel_quality', {}).get('after_repair'),
                    'model_compatible': updated['recording'].get('model_compatible'),
                    'accepted_windows': updated['recording'].get('overall', {}).get('accepted_windows'),
                    'window_coverage': updated['recording'].get('overall', {}).get('window_coverage')}
        except (TypeError, ValueError) as exc:
            return {'tool': tool, 'available': False, 'applied': False, 'reason': str(exc)}
    return recording_tool(tool, evidence)
