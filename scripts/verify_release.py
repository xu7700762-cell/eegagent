# -*- coding: utf-8 -*-
"""Verify the public package, paired metrics, links and private-file exclusions."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def public_files():
    listing = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT)
    return sorted({ROOT/name for name in listing.decode('utf-8').splitlines() if (ROOT/name).is_file()})


def verify():
    files = public_files()
    forbidden_name = ''.join(chr(code) for code in (102,101,109,98,97))
    private_suffixes = {'.pt','.ckpt','.joblib','.npy','.npz','.cdt','.dpo','.ceo','.edf','.mat','.log'}
    for path in files:
        relative = path.relative_to(ROOT)
        assert relative.parts[0] not in ('outputs','private_assets','private_data','.venv'),relative
        assert path.suffix.lower() not in private_suffixes,relative
        assert relative.as_posix() not in ('.env','cloud_profiles.json'),relative
        assert path.stat().st_size < 2_000_000,relative
        text = path.read_text(encoding='utf-8-sig')
        assert forbidden_name not in (str(relative)+'\n'+text).lower(),f'Superseded model name: {relative}'
        assert not re.search(r'(?:sk-|ghp_|gho_)[A-Za-z0-9_-]{20,}',text),f'Credential-like text: {relative}'
        assert not re.search(r'[A-Z]:[/\\]|/mnt/[a-z]/',text),f'Machine-specific path: {relative}'
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if re.match(r'(?:https?://|#|mailto:)',target):continue
                assert (path.parent/target.split('#')[0]).exists(),(relative,target)
        if path.suffix == '.py':compile(text,str(relative),'exec')
        if path.suffix == '.json':json.loads(text)
    for name in ('README.md','docs/PROTOCOL.md','docs/REPRODUCTION.md','eeg_agent/service.py'):
        assert (ROOT/name).is_file(),name
    for name in ('vrms_cloud','vrms_pilot','vrms_refine','vrms_deepseek','vrms_weights'):
        assert not any(path.relative_to(ROOT).parts[0]==name for path in files),f'Obsolete entrypoint: {name}'
    result = json.loads((ROOT/'results/current_seed2026.json').read_text(encoding='utf-8'))
    assert result['seed']==2026 and result['model']=='gpt-6.1-sol'
    source=ROOT/'results/current_seed2026_paths.csv'
    rows=list(csv.DictReader(source.open(encoding='utf-8')))
    assert len(rows)==146 and len({row['path_id'] for row in rows})==146
    assert hashlib.sha256(source.read_bytes()).hexdigest()==result['provenance']['paired_public_csv_sha256']
    for name,expected_correct in (('gpt',108),('vrmsmodel',94)):
        metric=result['metrics'][name]
        correct=sum(row[name+'_class']==row['true_class'] for row in rows)
        assert correct==metric['correct']==expected_correct
        assert metric['accuracy']==correct/146 and metric['total']==146 and metric['coverage']==1
        matrix=[[sum(row['true_class']==a and row[name+'_class']==b for row in rows)
                 for b in ('Low','High')] for a in ('Low','High')]
        assert matrix==metric['confusion_matrix']
    assert all(row['gpt_generation_status']=='cloud_verified' for row in rows)
    archived=json.loads((ROOT/'results/seed_sensitivity_archive.json').read_text(encoding='utf-8'))
    assert [row['seed'] for row in archived['records']]==[2026,2027,2028]
    validation=json.loads((ROOT/'results/packaging_validation.json').read_text(encoding='utf-8'))
    assert validation['status']=='passed'
    assert validation['frozen_model_equivalence']['folds']==24
    provenance=json.loads((ROOT/'source_provenance.json').read_text(encoding='utf-8'))
    for name,expected in provenance['published_code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,f'Public code hash: {name}'
    from scripts.verify_optimized_rag import verify_results
    rag = verify_results()
    print(json.dumps({'status':'passed','public_files':len(files),'gpt_correct':108,'vrmsmodel_correct':94,
        'scoreable_paths':146,'archived_seeds':3,'optimized_rag_correct':rag['metrics']['percentile_group']['correct'],
        'matched_without_rag_correct':rag['metrics']['without_rag']['correct'],'rag_contexts_verified':rag['contexts_verified']}))


if __name__=='__main__':verify()
