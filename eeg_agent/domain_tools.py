# -*- coding: utf-8 -*-
import copy
from .recordings import recording_tool

_CATALOG = {'vrms': [{'id': 'vrms.raw_recording', 'name': '原始事件与路径统计', 'description': '从原始波形内的触发通道统计完整与未结束路径，并返回时间区间。', 'input': '原始 EEG 及内嵌事件通道', 'output': '路径数量、完整性、起止时间', 'capability': 'measurement', 'classifier_required': False}, {'id': 'vrms.raw_model', 'name': '原始 EEG 与 GPT 工具 Agent', 'description': '重新编码原始 EEG，执行冻结 MIL 与八路工具；GPT 实际调用工具后给出最终 High/Low 并定位首条高类路径。', 'input': '质量合格的原始 EEG 五秒窗口与已配置 CPA GPT', 'output': 'GPT 最终分类、真实工具调用和首条高类路径区间', 'capability': 'raw_vrms_gpt_tool_agent'}], 'fatigue': [{'id': 'fatigue.raw_spectrum', 'name': '整记录疲劳相关指标', 'description': '按事件边界计算完整路径及休息段的 θ/α，直接比较同类片段的首末值与最高值。', 'input': '原始 EEG 波形与内嵌任务事件', 'output': 'TAR、路径与休息段值、首末变化、最高片段及时间', 'capability': 'measurement'}], 'emotion': [{'id': 'emotion.raw_workload', 'name': '整记录心理负荷与额区指标', 'description': '按事件边界计算路径和休息段的额区 θ/顶区 α 及 FAA，直接比较同类片段。', 'input': '原始 EEG，F3/Fz/F4 与 P3/Pz/P4 通道', 'output': 'EEG工作负荷指标、路径与休息段变化、FAA', 'capability': 'measurement'}]}

def toolkit_catalog(include_recordings=True):
    return copy.deepcopy({domain: [{**spec, 'domain': domain, 'input_scope': 'raw_eeg_only',
        'classifier_required': spec.get('classifier_required', domain == 'vrms'),
        'classifier_available': spec.get('classifier_required', domain == 'vrms')} for spec in specs]
        for domain, specs in _CATALOG.items()})

def execute_tool(tool, evidence):
    if tool not in {s['id'] for specs in _CATALOG.values() for s in specs} or tool == 'vrms.raw_model':
        raise ValueError('Unknown descriptive recording tool')
    if 'recording' not in evidence:
        return {'tool': tool, 'available': False, 'reason': '当前没有原始 EEG，可先进行方法问答', 'measurements': {}}
    return recording_tool(tool, evidence)
