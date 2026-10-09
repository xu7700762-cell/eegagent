"""Reliability V3 on frozen V2 CNNs and identical outer subject splits.

No CNN training, cloud requests, RAG changes, or outer-label threshold selection.
Only the scalar readout is cross-fitted on the five existing meta subjects.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from .data import digest, qc, read_csv, write_json
from .engine_v3 import reliability_decision
from .experiment import fit_logistic, probabilities, save_csv
from .model import CompactEEGCNN, predict_windows
from .reliability import signal_descriptor
from .risk_coverage_v3 import paired_subject_bootstrap, risk_curve, score_metrics

DEFAULT_SOURCE = Path('outputs/vrms_pilot/v2_gpt_flow_20261008')
DEFAULT_OUT = Path('outputs/vrms_reliability/v3_20261008')
DEFAULT_ANCHOR = Path('outputs/vrms_cloud/v2_gpt_flow_20261008/protocol.json')


def load_source(source, anchor=DEFAULT_ANCHOR):
    source = Path(source)
    protocol = json.loads((source/'results/protocol.json').read_text(encoding='utf-8'))
    if protocol['schema_version'] != 'pilot_v2' or protocol['smoke'] or protocol['seed'] != 2026:
        raise ValueError('Requires the frozen, full seed2026 V2 run')
    for name, expected in protocol['code_sha256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError('Frozen V2 source changed: '+name)
    for name, expected in protocol['external_code_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError('Frozen V2 external source changed: '+name)
    # Compare against the already-frozen pre-GPT preparation; recording only
    # today's hashes would not detect a modification before this V3 run.
    anchor=Path(anchor)
    frozen=json.loads(anchor.read_text(encoding='utf-8'))
    if digest(source/'split_manifest.json')!=frozen['split_manifest_sha256']:
        raise ValueError('Source split differs from the frozen V2 anchor')
    for key,folder in (('source_artifact_sha256',source/'results'),
                       ('checkpoint_sha256',source/'results/checkpoints'),
                       ('cache_artifact_sha256',source/'cache')):
        for name,expected in frozen[key].items():
            if digest(folder/name)!=expected:
                raise ValueError('Source differs from the frozen V2 anchor: '+name)
    audit = json.loads((source/'cache/dataset_audit.json').read_text(encoding='utf-8'))
    paths = json.loads((source/'cache/evaluation_manifest.json').read_text(encoding='utf-8'))
    splits = json.loads((source/'split_manifest.json').read_text(encoding='utf-8'))
    if audit['paths'] != len(paths) or len(splits) != audit['subjects']:
        raise ValueError('Frozen V2 manifest mismatch')
    windows = np.load(source/'cache/windows.npy', mmap_mode='r')[:audit['accepted_windows']]
    files = [anchor,source/'split_manifest.json', source/'results/protocol.json', source/'results/path_oof.csv']
    files += list((source/'cache').glob('*')) + list((source/'results/checkpoints').glob('*'))
    source_hashes = {str(p.resolve()):digest(p) for p in files if p.is_file()}
    return paths, splits, windows, audit, source_hashes


def crossfit_meta_head(features, y, groups, meta_subjects):
    """Each meta score comes from a head that did not fit its subject."""
    meta_subjects = list(meta_subjects)
    scores = np.full(len(y), np.nan)
    lineage = []
    for heldout in meta_subjects:
        fit_subjects = [s for s in meta_subjects if s != heldout]
        fitting = np.flatnonzero(np.isin(groups, fit_subjects))
        query = np.flatnonzero(groups == heldout)
        head = fit_logistic(features, y, fitting, c=.3, balanced=False)
        values = probabilities(head, features)
        scores[query] = values[query]
        lineage.append(dict(heldout_subject=int(heldout), head_fit_subjects=fit_subjects,
                            query_indices=query.tolist(), head_fit_indices=fitting.tolist(),
                            head_available=head is not None))
    return scores, lineage


def run(source=DEFAULT_SOURCE, out=DEFAULT_OUT,anchor=DEFAULT_ANCHOR):
    from .reliability_v3 import fit_reliability_v3
    source, out = Path(source), Path(out)
    result = out/'results'
    if result.exists():
        raise FileExistsError('Preserve V3 runs; choose a fresh output directory')
    paths, splits, windows, audit, source_hashes = load_source(source,anchor)
    result.mkdir(parents=True)
    (result/'checkpoints').mkdir()
    y = np.asarray([p['label'] for p in paths], int)
    groups = np.asarray([p['subject_key'] for p in paths], int)
    fractions = np.asarray([p['accepted_windows']/max(p['total_possible_windows'],1) for p in paths])
    descriptors = np.full((len(paths),60),np.nan)
    for i,p in enumerate(paths):
        descriptor=signal_descriptor(windows[p['window_start']:p['window_end']])
        if descriptor is not None:
            descriptors[i]=descriptor
    quality = np.asarray([p['accepted_windows']>0 and all(qc(x)[0] for x in windows[p['window_start']:p['window_end']]) for p in paths], bool)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    protocol = dict(schema_version='reliability_v3', seed=2026, source_pilot=str(source.resolve()),
        source_hashes=source_hashes,source_anchor=str(Path(anchor).resolve()),outer_split_unchanged=True, cnn_weights_unchanged=True,
        scalar_head='V2 same LR C=.3 unbalanced; 5meta final, 4meta leave-one-subject-out scoring',
        calibration='Platt fit on 5meta subject-wise OOF head predictions',
        meta_diagnostic_scope='OOF head scores; calibration fit metrics are not independent calibration validation',
        classification_threshold=.5, target_risk=.20, minimum_coverage=.35,
        operating_goal='Increase coverage only after independent risk/support validation; never relax from outer labels',
        policy='fixed 2 selection + 2 audit; additional 3/1 subject nested audit; no post-audit reselection',
        clinical_risk_guarantee=False, cnn_training=False, llm_called=False, rag_changed=False,
        code_sha256={name:digest(Path(__file__).parent/name) for name in
                     ('engine_v3.py','reliability_v3.py','reliability_experiment_v3.py','risk_coverage_v3.py')},
        device=str(device), torch=torch.__version__)
    write_json(result/'protocol.json', protocol)
    save_csv(out/'split_manifest.csv', [dict(outer_subject=s['outer_subject'],base=';'.join(map(str,s['base_train'])),
             meta=';'.join(map(str,s['meta_calibration'])),policy=';'.join(map(str,s['policy_validation']))) for s in splits])
    baseline = {int(r['path_index']):r for r in read_csv(source/'results/path_oof.csv')}
    rows, inner, audits = [], [], []
    started = time.perf_counter()
    for fold_number, split in enumerate(splits, 1):
        base = np.flatnonzero(np.isin(groups,split['base_train']))
        meta = np.flatnonzero(np.isin(groups,split['meta_calibration']))
        policy = np.flatnonzero(np.isin(groups,split['policy_validation']))
        test = np.flatnonzero(groups == split['outer_subject'])
        stem = source/'results/checkpoints'/f"outer_subject_{split['outer_subject']:02d}"
        state = torch.load(str(stem)+'.pt', map_location=device, weights_only=True)
        components = joblib.load(str(stem)+'.joblib')
        if state['training_subjects'] != split['base_train'] or components['split'] != split:
            raise ValueError('Frozen CNN roles differ from the preserved LOSO split')
        cnn = CompactEEGCNN().to(device)
        cnn.load_state_dict(state['model'], strict=True)
        cnn.eval()
        features = np.full((len(paths),1), np.nan)
        for i in np.r_[meta,policy,test]:
            p = paths[i]
            z = predict_windows(cnn,state['mean'].to(device),state['scale'].to(device),
                                windows[p['window_start']:p['window_end']], device)
            if len(z):
                features[i,0] = float(z.mean())
        old_probability = probabilities(components['meta']['deep'],features)
        if any(abs(old_probability[i]-float(baseline[int(i)]['deep_probability']))>2e-5 for i in test):
            raise ValueError('Unchanged CNN/head failed V2 probability reproduction')
        head = fit_logistic(features,y,meta,c=.3,balanced=False)
        raw = probabilities(head,features)
        oof, lineage = crossfit_meta_head(features,y,groups,split['meta_calibration'])
        reliability = fit_reliability_v3(raw,y,groups,descriptors,fractions,
            base_indices=base,meta_indices=meta,policy_indices=policy,
            meta_oof_probability=oof,meta_oof_provenance=lineage,qc_passed=quality)
        target = result/'checkpoints'/f"outer_subject_{split['outer_subject']:02d}.joblib"
        joblib.dump(dict(head=head,reliability=reliability,split=split,meta_oof_provenance=lineage),target)
        np.save(target.with_suffix('.features.npy'),features)
        np.save(target.with_suffix('.meta_oof.npy'),oof)
        for i in np.r_[meta,policy]:
            inner.append(dict(outer_subject=split['outer_subject'],path_index=int(i),subject_key=int(groups[i]),
                role='meta' if i in meta else 'policy', label=int(y[i]),
                final_head_probability=float(raw[i]) if np.isfinite(raw[i]) else None,
                meta_oof_probability=float(oof[i]) if np.isfinite(oof[i]) else None,
                calibrated_final_head_probability=reliability.calibrate(raw[i])))
        for i in test:
            gate = reliability.assess(raw[i],descriptors[i],fractions[i],qc_passed=bool(quality[i]),
                                      reference_available=bool(paths[i]['reference_available']))
            decision = reliability_decision(gate)
            rows.append(dict(path_index=int(i),subject_key=int(groups[i]),label=int(y[i]),
                p_raw_v2=float(baseline[int(i)]['deep_probability']),p_cal_v2=float(baseline[int(i)]['adaptive_probability'])
                if baseline[int(i)]['adaptive_probability'] else None,state_v2=baseline[int(i)]['adaptive_state'],
                p_raw=float(raw[i]) if np.isfinite(raw[i]) else None,p_cal=gate['p_cal'],state=decision['state'],
                rejection_reasons=';'.join(gate['rejection_reasons']),
                calibration_status=gate['calibration_status'],policy_status=gate['policy_status'],
                valid_window_fraction=float(fractions[i]),qc_passed=bool(quality[i]),
                coverage_warning=gate.get('signal_quality',{}).get('coverage_warning',False),
                ood=gate['ood'],ood_score=gate['ood_score'],prediction_reliable=gate['prediction_reliable']))
        audits.append(dict(outer_subject=split['outer_subject'],meta_oof_provenance=lineage,
            reliability_provenance=reliability.provenance,reliability_validation=reliability.validation))
        save_csv(result/'path_oof.csv',sorted(rows,key=lambda r:r['path_index']))
        save_csv(result/'inner_predictions.csv',inner)
        write_json(result/'fold_audit.json',audits)
        print(json.dumps(dict(fold=fold_number,outer_subject=split['outer_subject'],
                calibration=reliability.calibrator is not None,policy_validated=reliability.thresholds is not None,
                completed_paths=len(rows))),flush=True)
        del cnn,components
    summary = summarize(sorted(rows,key=lambda r:r['path_index']),audits)
    summary.update(status='completed_single_seed_exploratory_loso',seconds=time.perf_counter()-started,
                   llm_called=False,cnn_retrained=False,outer_folds=len(splits))
    write_json(result/'summary.json',summary)
    if any(digest(path)!=value for path,value in source_hashes.items()):
        raise ValueError('Source artifacts changed during the experiment')
    artifact_paths=[p for p in (result/'checkpoints').glob('*') if p.is_file()]
    artifact_paths += [result/name for name in ('path_oof.csv','inner_predictions.csv','fold_audit.json','protocol.json','summary.json')]
    write_json(result/'artifact_manifest.json',{str(p.relative_to(result)):digest(p) for p in artifact_paths})
    validate(source,out,anchor)
    print(json.dumps({k:summary[k] for k in ('state_counts','publication','rejection_reason_counts','calibration_folds')},indent=2))
    return summary


def summarize(rows,audits):
    y = np.asarray([r['label'] for r in rows],int)
    groups = np.asarray([r['subject_key'] for r in rows],int)
    scores = {key:np.asarray([np.nan if r[key] is None else r[key] for r in rows],float)
              for key in ('p_raw_v2','p_cal_v2','p_raw','p_cal')}
    published = np.asarray([r['state'] in ('high','low') for r in rows])
    common = np.isfinite(scores['p_raw']) & np.isfinite(scores['p_cal'])
    folds = []
    for subject in np.unique(groups):
        mask = (groups == subject) & common
        raw,cal = score_metrics(y[mask],scores['p_raw'][mask]),score_metrics(y[mask],scores['p_cal'][mask])
        folds.append(dict(subject=int(subject),matched_paths=int(mask.sum()),raw=raw,calibrated=cal,
             auroc_difference=cal['auroc']-raw['auroc'] if cal['auroc'] is not None else None))
    return dict(total_paths=len(rows),state_counts=dict(Counter(r['state'] for r in rows)),
        calibration_folds=sum(a['reliability_validation'].get('calibration_status') == 'fitted' for a in audits),
        validated_policy_folds=sum(a['reliability_validation'].get('status') == 'validated' for a in audits),
        rejection_reason_counts=dict(Counter(reason for r in rows for reason in r['rejection_reasons'].split(';') if reason)),
        rejection_counts_overlap=True,
        publication=dict(paths=int(published.sum()),coverage=float(published.mean()),
             metrics=score_metrics(y[published],scores['p_cal'][published]) if published.any() else None),
        all_score_metrics={key:score_metrics(y,p) for key,p in scores.items()},
        matched_raw_calibration=dict(paths=int(common.sum()),raw=score_metrics(y[common],scores['p_raw'][common]),
                                   calibrated=score_metrics(y[common],scores['p_cal'][common])),
        per_fold_calibration=folds,
        paired_calibration_diagnostic=paired_subject_bootstrap(y,scores['p_raw'],scores['p_cal'],groups),
        risk_coverage={key:risk_curve(y,p,groups) for key,p in scores.items()},
        limitations=['Scalar head refit on 5meta changes readout fitting protocol; not a pure Platt-only ablation',
            'OOF meta head fit excludes query subject; calibrated meta re-prediction is not independent calibration validation',
            'Nested policy estimates evaluate the selection procedure; deployment uses the independently audited primary candidate',
            'Risk-coverage ranks outer scores descriptively; no threshold is chosen from outer labels',
            'Operational risk targets are exploratory, not finite-sample or clinical guarantees'])


def validate(source=DEFAULT_SOURCE,out=DEFAULT_OUT,anchor=DEFAULT_ANCHOR):
    source,out=Path(source),Path(out)
    result=out/'results'
    protocol=json.loads((result/'protocol.json').read_text(encoding='utf-8'))
    artifacts=json.loads((result/'artifact_manifest.json').read_text(encoding='utf-8'))
    for name,value in artifacts.items():
        if digest(result/name)!=value:
            raise ValueError('Frozen V3 output artifact changed: '+name)
    for path,value in protocol['source_hashes'].items():
        if digest(path)!=value:
            raise ValueError('Frozen source artifact changed: '+path)
    for name,value in protocol['code_sha256'].items():
        if digest(Path(__file__).parent/name)!=value:
            raise ValueError('Frozen V3 source changed: '+name)
    paths=json.loads((source/'cache/evaluation_manifest.json').read_text(encoding='utf-8'))
    splits=json.loads((source/'split_manifest.json').read_text(encoding='utf-8'))
    windows=np.load(source/'cache/windows.npy',mmap_mode='r')
    rows=read_csv(result/'path_oof.csv')
    lookup={int(r['path_index']):r for r in rows}
    if len(rows)!=len(paths) or len(lookup)!=len(paths):
        raise ValueError('Incomplete outer OOF paths')
    comparisons=0
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y=np.asarray([p['label'] for p in paths],int)
    groups=np.asarray([p['subject_key'] for p in paths],int)
    for split in splits:
        stem=result/'checkpoints'/f"outer_subject_{split['outer_subject']:02d}"
        bundle=joblib.load(str(stem)+'.joblib')
        if bundle['split']!=split:
            raise ValueError('V3 split changed')
        for lineage in bundle['meta_oof_provenance']:
            if lineage['heldout_subject'] in lineage['head_fit_subjects']:
                raise ValueError('OOF head saw its query subject')
            if set(lineage['head_fit_subjects'])|{lineage['heldout_subject']}!=set(split['meta_calibration']):
                raise ValueError('OOF head roles changed')
        features=np.load(str(stem)+'.features.npy')
        recomputed_oof,_=crossfit_meta_head(features,y,groups,split['meta_calibration'])
        stored_oof=np.load(str(stem)+'.meta_oof.npy')
        if not np.allclose(stored_oof,recomputed_oof,equal_nan=True,atol=1e-12,rtol=1e-12):
            raise ValueError('Meta OOF head predictions could not be reproduced')
        raw=probabilities(bundle['head'],features)
        source_state=torch.load(source/'results/checkpoints'/f"outer_subject_{split['outer_subject']:02d}.pt",map_location=device,weights_only=True)
        cnn=CompactEEGCNN().to(device)
        cnn.load_state_dict(source_state['model'],strict=True)
        cnn.eval()
        for i,p in enumerate(paths):
            if p['subject_key']!=split['outer_subject']:
                continue
            sl=windows[p['window_start']:p['window_end']]
            z=predict_windows(cnn,source_state['mean'].to(device),source_state['scale'].to(device),sl,device)
            if len(z) and not np.isclose(z.mean(),features[i,0],atol=2e-5,rtol=2e-5):
                raise ValueError('Saved features differ from the frozen CNN inference')
            quality=bool(len(sl)) and all(qc(x)[0] for x in sl)
            gate=bundle['reliability'].assess(raw[i],signal_descriptor(sl),
                p['accepted_windows']/max(p['total_possible_windows'],1),qc_passed=quality,
                reference_available=bool(p['reference_available']))
            decision=reliability_decision(gate)
            expected=lookup[i]
            if decision['state']!=expected['state'] or ';'.join(gate['rejection_reasons'])!=expected['rejection_reasons']:
                raise ValueError('Saved decision differs from reloaded gate')
            for key,value in (('p_raw',raw[i]),('p_cal',gate['p_cal'])):
                saved=float(expected[key]) if expected[key] else None
                if (value is None)!=(saved is None) or (value is not None and abs(value-saved)>2e-8):
                    raise ValueError('Reloaded probability mismatch')
            comparisons+=1
    write_json(result/'validation.json',dict(status='passed',paths=comparisons,outer_folds=len(splits),
        meta_oof_subject_disjoint=True,source_artifacts_unchanged=True,cnn_weights_unchanged=True,
        reloaded_numeric_decisions_checked=True,outer_threshold_selection=False,llm_called=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['run','validate'],default='run')
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--anchor',type=Path,default=DEFAULT_ANCHOR)
    args=parser.parse_args()
    (run if args.stage=='run' else validate)(args.source,args.out,args.anchor)


if __name__=='__main__':
    main()
