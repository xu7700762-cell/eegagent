# -*- coding: utf-8 -*-
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
import yaml

PROJECT = Path(__file__).resolve().parents[1]
PIPELINE_ID = 'raw_eeg_gpt_tools_seed2026'
EEG_CHANNELS = ['Fp1','Fp2','F11','F7','F3','Fz','F4','F8','F12','FT11','FC3','FCz','FC4','FT12','T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3','Pz','P4','P8','O1','Oz','O2']

def load_config(path=None):
    defaults = {'data_root': './private_data', 'output_root': './outputs',
        'model_manifest': './private_assets/seed2026/manifest.json', 'seed': 2026,
        'worker_command': [sys.executable], 'api_max_calls': 50, 'api_timeout_seconds': 60,
        'window_seconds': 5, 'minimum_path_windows': 3, 'minimum_path_coverage': .8}
    configured = Path(path) if path else PROJECT / 'configs/local.yaml'
    if configured.exists():
        defaults.update(yaml.safe_load(configured.read_text(encoding='utf-8')) or {})
    defaults['data_root'] = os.environ.get('EEG_DATA_ROOT', defaults['data_root'])
    defaults['model_manifest'] = os.environ.get('EEG_MODEL_MANIFEST', defaults['model_manifest'])
    if defaults['seed'] != 2026 or defaults['window_seconds'] != 5:
        raise ValueError('The active contract requires seed2026 and five-second windows')
    if not 0 < defaults['minimum_path_coverage'] <= 1 or defaults['minimum_path_windows'] < 3:
        raise ValueError('Invalid complete-path publication policy')
    if not isinstance(defaults['worker_command'], list) or not all(isinstance(v,str) and v for v in defaults['worker_command']):
        raise ValueError('worker_command must be a nonempty list of executable arguments')
    for key in ('data_root', 'output_root', 'model_manifest'):
        value = Path(defaults[key]).expanduser()
        defaults[key] = str((value if value.is_absolute() else PROJECT / value).resolve())
    return defaults

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
