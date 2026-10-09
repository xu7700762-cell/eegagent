# -*- coding: utf-8 -*-
"""Fresh waveform encoding and frozen local EEG tools; target labels never enter."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .assets import Assets, sha
from .encoder import build_encoder, load_pretrained_checkpoint
from .model import PathMIL
from .numeric import FrozenNumeric

def infer(request_path, manifest):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding='utf-8'))
    if set(request) != {'fold','signal_sha256','segments','baseline_file'} or type(request['fold']) is not int:
        raise ValueError('Invalid raw model request')
    if request['baseline_file'] is not False:
        raise ValueError('Opening waveform cannot serve as resting reference')
    fold, assets = request['fold'], Assets(manifest)
    head_path, head_sha = assets.item(fold,'head')
    normalization_path, normalization_sha = assets.item(fold,'normalization')
    encoder_path, encoder_sha = assets.item(fold,'encoder')
    if not torch.cuda.is_available():
        raise RuntimeError('The frozen native VRMSModel encoder requires CUDA and mamba-ssm')
    torch.set_num_threads(2)
    device, start = torch.device('cuda'), time.perf_counter()
    encoder = build_encoder(device, backend='native')
    loaded = load_pretrained_checkpoint(encoder, encoder_path)
    if loaded['loaded_keys'] != 83 or loaded['missing_keys'] or loaded['unexpected_keys'] or loaded['skipped_keys']:
        raise ValueError('VRMSModel encoder tensors were not fully loaded')
    encoder.eval().requires_grad_(False)
    normalization = torch.load(normalization_path,map_location='cpu',weights_only=True)
    state = torch.load(head_path,map_location='cpu',weights_only=True)
    if state['split']['outer_subject'] != fold or normalization['heldout_subject'] != fold or state.get('seed',2026) != 2026:
        raise ValueError('Wrong held-out model or seed')
    if state['input_dim'] != 525 or fold in normalization['training_subjects']:
        raise ValueError('Invalid model dimensions or normalization subject overlap')
    for name in ('base_train','head_fit','probability_calibration','policy_validation'):
        if fold in state['split'].get(name,[]):
            raise ValueError('Query identity overlaps model fitting')
    head = PathMIL(state['input_dim']).to(device)
    head.load_state_dict(state['state_dict'])
    head.eval().requires_grad_(False)
    mean, scale = normalization['mean'].to(device), normalization['scale'].to(device)
    numeric, results = FrozenNumeric(fold, manifest), []
    with torch.inference_mode():
        for segment in request['segments']:
            filename = segment['file']
            if not filename.startswith('segment-') or Path(filename).name != filename or not filename.endswith('.npy'):
                raise ValueError('Invalid waveform filename')
            windows = np.load(request_path.parent/filename,allow_pickle=False)
            if windows.ndim != 3 or windows.shape[1:] != (30,1280) or not 3 <= len(windows) <= 10000 or not np.isfinite(windows).all():
                raise ValueError('Invalid EEG windows')
            features = []
            for offset in range(0,len(windows),16):
                x = torch.from_numpy(windows[offset:offset+16]).to(device)
                x = ((x-mean)/scale).clamp(-20,20)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    features.append(encoder.forward_tokens(x).float().mean(dim=1))
            encoded = torch.cat(features)
            probability = float(head(encoded).sigmoid().cpu())
            evidence = numeric.predict(windows,encoded.cpu().numpy(),probability,baseline=None)
            results.append({'ordinal':segment['ordinal'],'predicted_class':evidence['predicted_class'],
                'mil_raw_high_probability':probability,'threshold':.5,
                'mil_class':'High' if probability >= .5 else 'Low','numeric_fusion':evidence,
                'classification_layer':'numeric_fusion','accepted_windows':len(windows),
                'fresh_encoded_windows':len(windows)})
    torch.cuda.synchronize()
    return {'available':True,'input_policy':'raw_eeg_only','fresh_raw_encoding':True,
        'baseline_policy':'disabled','comparison_policy':'event_segments_only',
        'signal_sha256':request['signal_sha256'],'model_sha256':head_sha,
        'encoder_sha256':encoder_sha,'normalization_sha256':normalization_sha,
        'seconds':time.perf_counter()-start,'segments':results,
        'readout':'whole_path_model_and_internal_evidence',**numeric.hashes,'probability_calibrated':False}

if __name__ == '__main__':
    try:
        result = infer(sys.argv[1],sys.argv[2])
    except Exception as exc:
        result = {'available':False,'error_type':type(exc).__name__,
                  'reason':'本地原始波形编码未完成；本轮未发布 VRMS 分类'}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
