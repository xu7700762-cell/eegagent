# -*- coding: utf-8 -*-
import re
from pathlib import Path
from .config import EEG_CHANNELS

def list_block(text, name):
    match = re.search(r'(?ms)^\s*' + re.escape(name) + r' START_LIST[^\n]*\n(.*?)^\s*' + re.escape(name) + r' END_LIST', text)
    return [x.strip() for x in match.group(1).splitlines() if x.strip() and not x.lstrip().startswith('#')] if match else []


def metadata(path):
    path = Path(path)
    text = Path(str(path) + '.dpo').read_text(encoding='utf-8', errors='replace')
    def key(name):
        m = re.search(r'(?m)^\s*' + name + r'\s*=\s*(.*?)\s*$', text)
        if not m: raise ValueError('Missing DPO field: ' + name)
        return m.group(1).strip()
    channels = list_block(text, 'LABELS') + list_block(text, 'LABELS_OTHERS')
    n = int(key('NumChannels'))
    if key('DataByteOrder') != 'INTEL' or key('DataSampOrder') != 'SAMP' or key('DataFormat') != '6':
        raise ValueError('Unsupported CDT byte/sample/format contract')
    if int(float(key('SampleFreqHz'))) != 1024: raise ValueError('Expected 1024 Hz')
    if len(channels) != n or len(set(channels)) != n: raise ValueError('Invalid channel names')
    required = EEG_CHANNELS + ['M1','M2','VEOG','HEOG','Trigger']
    if any(ch not in channels for ch in required): raise ValueError('Missing required channels')
    units = re.findall(r'(?m)^\s*DataUnit\s*=\s*(\S+)', text)
    scales = re.findall(r'(?m)^\s*CommonScale\s*=\s*(\S+)', text)
    offsets = re.findall(r'(?m)^\s*CommonOffset\s*=\s*(\S+)', text)
    if not units or any(u != 'uV' for u in units) or any(float(s) != 1 for s in scales) or any(float(s) != 0 for s in offsets):
        raise ValueError('Unsupported unit/scale/offset')
    size = path.stat().st_size
    if size % (n * 4): raise ValueError('Partial trailing sample frame')
    actual = size // (n * 4); declared = int(key('NumSamples'))
    return {'channels': channels, 'n_channels': n, 'actual_samples': actual, 'declared_samples': declared,
            'warnings': ['DPO sample count differs; read actual complete frames'] if actual != declared else [], 'unit': 'uV', 'sfreq': 1024}
