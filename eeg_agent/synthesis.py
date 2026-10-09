# -*- coding: utf-8 -*-
"""Question-focused coverage checks over executed raw EEG tool facts."""
import math
import re
from decimal import Decimal, ROUND_HALF_UP


TOPICS = {
    'fatigue': ('疲劳', 'fatigue.raw_spectrum', r'fatigue|疲劳'),
    'emotion': ('心理负荷', 'emotion.raw_workload', r'emotion|心理负荷|工作负荷|情绪|情感'),
}


def question_scopes(question):
    blocks = re.split(r'(?:^|[\n：:？?；;])\s*\d+\s*[.、)]\s*', question)
    scopes = {domain: [] for domain in TOPICS}
    for block in blocks:
        active_domains = set()
        for clause in re.split(r'[。？！?；;\n]', block):
            explicit = {d for d, (_, _, pattern) in TOPICS.items() if re.search(pattern, clause, re.I)}
            if explicit:
                active_domains = explicit
            if explicit or re.search(r'路径|休息|比较|变化|上升|下降|首末|最高|最大|峰值|趋势|path|rest', clause, re.I):
                for domain in active_domains:
                    scopes[domain].append(clause)
    return {domain: '\n'.join(items) for domain, items in scopes.items() if items}


def requirements(question, tool_results):
    """Select existing local facts for the comparisons actually requested."""
    scopes = question_scopes(question)
    by_tool = {item['tool']: item['result'] for item in tool_results}
    result = []
    for domain, scope in scopes.items():
        label, source, _ = TOPICS[domain]
        tool = by_tool.get(source)
        if not tool:
            continue
        measured = tool.get('measurements', {})

        def add(kind, paths, unavailable=None, state='task'):
            facts = []
            for path in paths:
                value = tool
                try:
                    for key in path.split('.'):
                        value = value[int(key)] if isinstance(value, list) else value[key]
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                if value is not None and not isinstance(value, bool):
                    facts.append({'source': source, 'path': path, 'value': value})
            result.append({'id': domain + '.' + kind, 'topic': label,
                           'source': source, 'state': state, 'required_facts': facts,
                           'unavailable_reason': unavailable})

        clauses = re.split(r'[。？！?；;\n]', scope)
        has_rest = bool(re.search(r'休息|rest', scope, re.I))
        state_clauses = {'task': [], 'rest': []}
        active_states = {'rest'} if has_rest and not re.search(r'路径|任务|path|task', scope, re.I) else {'task'}
        for clause in clauses:
            explicit = set()
            if re.search(r'路径|任务|path|task', clause, re.I):
                explicit.add('task')
            if re.search(r'休息|rest', clause, re.I):
                explicit.add('rest')
            if explicit:
                active_states = explicit
            if explicit or re.search(TOPICS[domain][2] + r'|比较|变化|上升|下降|首末|最高|最大|峰值|趋势', clause, re.I):
                for state in active_states:
                    state_clauses[state].append(clause)
        state_scopes = {state: '\n'.join(items) for state, items in state_clauses.items()}
        for state, state_scope in state_scopes.items():
            if not state_scope.strip():
                continue
            prefix = 'rest_' if state == 'rest' else ''
            suffix = 'rest_' if state == 'rest' else ''
            if not tool.get('available'):
                add(suffix + 'measurement', ['reason'], tool.get('reason'), state=state)
                continue
            value_key = 'rest_value' if state == 'rest' else 'task_value'
            if measured.get(value_key) is not None:
                add(suffix + 'measurement', ['measurements.' + value_key], state=state)
            else:
                reason_key = state + '_unavailable_reason'
                add(suffix + 'measurement', ['measurements.' + reason_key], measured.get(reason_key), state=state)
            if ((re.search(r'首(?:条|段|末)|第一(?:条|个|段)?|first', state_scope, re.I) and
                 re.search(r'末(?:条|段|路径)|最后|last', state_scope, re.I)) or
                    re.search(r'比较|变化|上升|下降|趋势|compare|change', state_scope, re.I)):
                key = prefix + 'first_to_last'
                comparison = measured.get(key)
                if comparison:
                    add(suffix + 'first_last', ['measurements.' + key + '.' + part for part in (
                        'first_ordinal', 'last_ordinal', 'first_value', 'last_value',
                        'first_start_clock', 'first_end_clock', 'last_start_clock', 'last_end_clock', 'change.text')], state=state)
                else:
                    add(suffix + 'first_last', ['measurements.' + key + '_unavailable_reason'],
                        measured.get(key + '_unavailable_reason'), state=state)
            if re.search(r'最高|最大|峰值|highest|maximum|peak', state_scope, re.I):
                key = prefix + 'max_complete'
                peak = measured.get(key)
                if peak:
                    paths = ['measurements.' + key + '.value']
                    for index in range(len(peak['segments'])):
                        paths += [f'measurements.{key}.segments.{index}.{part}'
                                  for part in ('ordinal', 'start_clock', 'end_clock')]
                    if peak.get('missing_count'):
                        paths.append('measurements.' + key + '.missing_count')
                    add(suffix + 'maximum', paths, state=state)
                else:
                    add(suffix + 'maximum', ['measurements.' + key + '_unavailable_reason'],
                        measured.get(key + '_unavailable_reason'), state=state)
    return result


def validate_answer(value, required, numbers):
    """Claims must be used in the matching topic paragraph, not merely listed."""
    text = value.get('explanation', '')
    if re.search(r'(?m)^\s*(?:VRMSAgent|FatigueAgent|EmotionAgent)\s*[:：]', text):
        raise ValueError('supervisor repeated specialist reports')
    claims = {(c.get('source'), c.get('path')): c.get('value')
              for c in value.get('measurement_claims', []) if isinstance(c, dict)}
    paragraphs = re.split(r'\n\s*\n|\n(?=\s*(?:\d+[.、)]\s*)?(?:疲劳|心理负荷|工作负荷|情感|情绪))', text)
    topic_paragraphs = {(domain, state): [] for domain in TOPICS for state in ('task', 'rest')}
    for paragraph in paragraphs:
        if re.match(r'^\s*综合结论', paragraph):
            continue
        # A paragraph belongs to its first topic. Mixed prose cannot supply
        # swapped values to two domains simply by containing both number sets.
        matches = [(match.start(), domain) for domain, (_, _, pattern) in TOPICS.items()
                   if (match := re.search(pattern, paragraph, re.I))]
        if matches:
            state_match = re.search(r'休息|路径|任务|rest|path|task', paragraph, re.I)
            state = 'rest' if state_match and re.fullmatch(r'休息|rest', state_match[0], re.I) else 'task'
            topic_paragraphs[min(matches)[1], state].append(paragraph)
    for item in required:
        domain = item['id'].split('.')[0]
        state = item.get('state', 'task')
        topic_text = '\n'.join(topic_paragraphs[domain, state])
        if not topic_text:
            raise ValueError('supervisor omitted requested comparison')
        source_names = {'fatigue': r'fatigue\.raw_spectrum|(?:疲劳)?原始频谱工具|TAR工具',
                        'emotion': r'emotion\.raw_workload|(?:原始EEG)?(?:心理|工作)负荷工具'}
        if not re.search(source_names[domain], topic_text, re.I):
            raise ValueError('supervisor omitted requested tool attribution')
        compact = re.sub(r'\s+', '', topic_text).replace('％', '%')
        present = numbers(re.sub(r'\d+:\d{2}\.\d{3}', '', topic_text))
        for fact in item['required_facts']:
            if claims.get((fact['source'], fact['path'])) != fact['value']:
                raise ValueError('supervisor omitted requested comparison')
            actual = fact['value']
            if isinstance(actual, (int, float)) and math.isfinite(actual):
                number = Decimal(str(actual))
                displays = {number}
                if isinstance(actual, float):
                    displays.update(number.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
                                    for digits in range(2, 7))
                if not displays & present:
                    raise ValueError('supervisor omitted requested comparison')
            elif isinstance(actual, str):
                change = re.fullmatch(r'(上升|下降|持平)([\d.]+%)', actual)
                if change:
                    if not re.search(change[1] + r'(?:了|约|幅度|百分比|为|是|[：:（(])*' + re.escape(change[2]), compact):
                        raise ValueError('supervisor comparison direction conflicts with tool')
                elif re.fullmatch(r'\d+:\d{2}\.\d{3}', actual) and actual not in compact:
                    raise ValueError('supervisor omitted requested comparison')
                elif (fact['path'] == 'reason' or fact['path'].endswith(('_unavailable_reason', '.text'))) and re.sub(r'\s+', '', actual) not in compact:
                    raise ValueError('supervisor omitted requested comparison')
        by_path = {f['path']: f['value'] for f in item['required_facts']}
        if item['id'].endswith('measurement'):
            actual = by_path.get('measurements.rest_value' if state == 'rest' else 'measurements.task_value')
            if actual is not None:
                summaries = [sentence for sentence in re.split(r'[。；;\n]', topic_text)
                             if re.search(r'汇总|总体|整体|任务期|任务值|任务指标|休息值|休息指标', sentence)]
                if not any(_displayed(actual, sentence, numbers) for sentence in summaries):
                    raise ValueError('supervisor comparison binding conflicts with tool')
        prefix = 'rest_' if state == 'rest' else ''
        comparison_key = 'measurements.' + prefix + 'first_to_last'
        peak_key = 'measurements.' + prefix + 'max_complete'
        if item['id'].endswith('first_last') and comparison_key + '.first_ordinal' in by_path:
            for prefix in ('first', 'last'):
                base = comparison_key + '.' + prefix
                _check_path(by_path[base + '_ordinal'], by_path[base + '_value'],
                            by_path.get(base + '_start_clock'), by_path.get(base + '_end_clock'), topic_text, numbers)
        if item['id'].endswith('maximum') and peak_key + '.value' in by_path:
            if not re.search(r'最高|最大|峰值', topic_text):
                raise ValueError('supervisor comparison binding conflicts with tool')
            peak_value = by_path[peak_key + '.value']
            shared_peak = any(_displayed(peak_value, span, numbers) for span in re.findall(
                r'(?:最高|最大|峰值)[^。；;\n]*?(?=第\s*\d|[。；;\n]|$)', topic_text))
            for path, ordinal in by_path.items():
                if path.endswith('.ordinal'):
                    base = path.rsplit('.', 1)[0]
                    _check_path(ordinal, peak_value,
                                by_path.get(base + '.start_clock'), by_path.get(base + '.end_clock'), topic_text, numbers,
                                shared_value=shared_peak)
            if peak_key + '.missing_count' in by_path and not re.search(r'有效|合格|有测量|有数据|已测得|可用', topic_text):
                raise ValueError('supervisor comparison binding conflicts with tool')
    return {'requested_items': len(required), 'verified_items': len(required),
            'topics': list(dict.fromkeys(item['topic'] for item in required))}


def _displayed(actual, text, numbers):
    number = Decimal(str(actual))
    displays = {number}
    if isinstance(actual, float):
        displays.update(number.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
                        for digits in range(2, 7))
    measurement_text = re.sub(r'\d+:\d{2}\.\d{3}|第\s*\d+(?:\s*[、,，]\s*\d+)*\s*(?:条|段|个)', '', text)
    return bool(displays & numbers(measurement_text))


def _check_path(ordinal, value, start, end, text, numbers, shared_value=False):
    mentions = list(re.finditer(r'第\s*(\d+(?:\s*[、,，]\s*\d+)*)\s*(?:条|段|个)', text))
    for index, mention in enumerate(mentions):
        if ordinal not in [int(v) for v in re.findall(r'\d+', mention[1])]:
            continue
        scope = text[mention.start():mentions[index + 1].start() if index + 1 < len(mentions) else len(text)]
        if (shared_value or _displayed(value, scope, numbers)) and all(not clock or clock in scope for clock in (start, end)):
            return
    raise ValueError('supervisor comparison binding conflicts with tool')
