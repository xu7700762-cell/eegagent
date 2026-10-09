# -*- coding: utf-8 -*-
"""Export trusted existing model assets without fitting or reading query data.

The private import JSON supplies source paths, optional string aliases and
module aliases. Pickle/joblib sources must come from a trusted local owner.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import joblib
import torch
from vrms_model.assets import sha


def rename(value, names):
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, dict):
        return {rename(k,names): rename(v,names) for k,v in value.items()}
    if isinstance(value, list):
        return [rename(v,names) for v in value]
    if isinstance(value, tuple):
        return tuple(rename(v,names) for v in value)
    return value


def export(spec_path, out):
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if spec['seed'] != 2026 or out.exists():
        raise ValueError('Use seed2026 and a fresh asset directory')
    for old, new in spec.get('module_aliases',{}).items():
        for end in range(1,len(old.split('.'))):
            parent = '.'.join(old.split('.')[:end])
            sys.modules.setdefault(parent,types.ModuleType(parent))
        sys.modules[old] = importlib.import_module(new)
    names = spec.get('string_aliases',{})
    out.mkdir(parents=True)
    manifest = {'seed':2026,'baseline_policy':'disabled','folds':{}}
    source_hashes = {}

    def convert(source, target, kind):
        source = Path(source)
        source_hashes[str(target.relative_to(out))] = sha(source)
        if kind in ('head','normalization','encoder'):
            value = torch.load(source,map_location='cpu',weights_only=False)
            if kind == 'head':
                if value.get('seed',spec['seed']) != spec['seed']:
                    raise ValueError('Conflicting frozen head seed')
                value['seed'] = spec['seed']
                value = {key:value[key] for key in ('state_dict','input_dim','split','seed')}
            elif kind == 'normalization':
                value = {key:value[key] for key in ('mean','scale','heldout_subject','training_subjects')}
            else:
                state = value.get('state_dict',value)
                value = {'state_dict':{key:tensor for key,tensor in state.items()
                    if key.removeprefix('model.').startswith(('patch_embed','pos_embed','mamba_blocks','norm_layers'))}}
                if len(value['state_dict']) != 83:
                    raise ValueError('Expected all 83 frozen encoder tensors')
            torch.save(value,target)
        elif kind == 'nested_audit':
            value = rename(json.loads(source.read_text(encoding='utf-8')),names)
            target.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            joblib.dump(rename(joblib.load(source),names),target)
        return {'file':str(target.relative_to(out)).replace('\\','/'),'sha256':sha(target)}

    manifest['encoder'] = convert(spec['encoder'],out/'encoder.pt','encoder')
    for fold, sources in spec['folds'].items():
        manifest['folds'][fold] = {}
        for kind, source in sources.items():
            suffix = '.pt' if kind in ('head','normalization') else '.json' if kind=='nested_audit' else '.joblib'
            target = out/(f'fold_{int(fold):02d}_{kind}'+suffix)
            manifest['folds'][fold][kind] = convert(source,target,kind)
    manifest['import_source_sha256'] = source_hashes
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'folds':len(manifest['folds']),'manifest_sha256':sha(out/'manifest.json')}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    export(args.import_manifest,args.out)
