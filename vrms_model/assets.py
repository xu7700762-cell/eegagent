# -*- coding: utf-8 -*-
"""Trusted local seed2026 assets; no data, labels or cached query predictions."""
import hashlib
import json
from pathlib import Path
import joblib

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()

class Assets:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.value = json.loads(self.path.read_text(encoding='utf-8'))
        if self.value.get('seed') != 2026 or self.value.get('baseline_policy') != 'disabled':
            raise ValueError('Expected frozen seed2026 assets with reference disabled')

    def item(self, fold, name):
        record = self.value['encoder'] if name == 'encoder' else self.value['folds'][str(fold)][name]
        path = Path(record['file'])
        path = path if path.is_absolute() else self.path.parent / path
        if sha(path) != record['sha256']:
            raise ValueError('Frozen model asset hash differs')
        return path.resolve(), record['sha256']

    def optional(self, fold, name):
        return self.item(fold, name)[0] if name in self.value['folds'][str(fold)] else None

    def bundle(self, fold, name):
        path, expected = self.item(fold, name)
        bundle = joblib.load(path)
        split = bundle['split']
        if split['outer_subject'] != fold or any(fold in split.get(key, []) for key in
                ('base_train','meta_calibration','policy_validation')):
            raise ValueError('Query overlaps numerical fitting subjects')
        return bundle, expected
