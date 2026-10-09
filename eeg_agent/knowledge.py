# -*- coding: utf-8 -*-
import re
_STOP = {'采用', '一个', 'for', 'the', 'with', '进行', '可以', '这个', '以及', 'and', 'not', '不能', '因为', 'this', 'was', '根据', 'that', '当前', 'can', 'from', '我们', 'are'}

def _terms(text):
    text = str(text).lower()
    result = re.findall(r'[a-z][a-z0-9_\-]*', text)
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        # Deterministic Chinese bigrams avoid a Pi-side segmenter dependency.
        result.extend(run[i:i+2] for i in range(max(0, len(run)-1)))
    return [term for term in result if term not in _STOP]


def _sections(text):
    title, lines = '正文', []
    for line in text.splitlines():
        if re.match(r'^#{1,6}\s+', line):
            if lines and '\n'.join(lines).strip():
                yield title, '\n'.join(lines).strip()
            title, lines = re.sub(r'^#{1,6}\s+', '', line).strip(), []
        else:
            lines.append(line)
    if lines and '\n'.join(lines).strip():
        yield title, '\n'.join(lines).strip()
