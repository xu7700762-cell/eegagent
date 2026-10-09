# -*- coding: utf-8 -*-
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from eeg_agent.config import load_config, atomic_json
from eeg_agent.evaluation import EvaluationManager
from scripts import recover_raw_gpt_failures as recovery
from scripts.evaluate_raw_gpt_all import score_rows
from vrms_model.training import fit_mil
import numpy as np
import torch


def test_accuracy_keeps_fixed_denominator_and_missing_gpt_is_wrong():
    predictions=[{'path_key':'a','gpt_class':'High'},{'path_key':'b','gpt_class':None}]
    truths={'a':{'true_class':'High'},'b':{'true_class':'Low'}}
    _,metric=score_rows(predictions,truths)
    assert metric['accuracy']==.5 and metric['coverage']==.5
    assert metric['conditional_accuracy']==1 and metric['total']==2


def manager(tmp_path):
    cfg=load_config()
    cfg['output_root']=str(tmp_path)
    return EvaluationManager(cfg)


def test_invalid_resume_does_not_hold_cloud_lock(tmp_path):
    value=manager(tmp_path)
    with pytest.raises(ValueError,match='未知'):
        value.start(resume_from='bad')
    assert not value.cloud_in_use() and not list(value.root.iterdir())


def test_restart_marks_stale_evaluation_interrupted(tmp_path):
    root=tmp_path/'evaluations'/'0123456789ab'
    atomic_json(root/'web_record.json',{'id':root.name,'state':'running','terminal':False})
    value=manager(tmp_path)
    assert value.get(root.name)['state']=='interrupted'
    assert not value.cloud_in_use()


def test_resume_preserves_original_config_and_propagates_it_to_score(tmp_path,monkeypatch):
    value=manager(tmp_path)
    original=value.root/'0123456789ab'
    original.mkdir()
    config=original/'execution.yaml'
    config.write_text('seed: 2026\n',encoding='utf-8')
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        assert command[command.index('--config')+1]==str(config)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('eeg_agent.evaluation.subprocess.run',run)
    directory=value.root/'abcdef012345'
    directory.mkdir()
    value._run(directory,{'id':directory.name,'resume_from':original.name,'state':'running','terminal':False})
    assert len(commands)==2 and 'scripts.recover_raw_gpt_failures' in commands[0]
    assert commands[1][commands[1].index('--stage')+1]=='score'
    assert config.read_text()=='seed: 2026\n'
    assert value.get(directory.name)['state']=='completed'


def test_recovery_does_not_retry_or_change_valid_decisions(tmp_path,monkeypatch):
    source,out=tmp_path/'source',tmp_path/'out'
    numeric={'raw_combined_score':.7,'predicted_class':'High','actual_tools':[]}
    valid={'ordinal':1,'predicted_class':'Low','numeric_fusion':numeric,
           'cloud_judgment':{'available':True,'final_class':'Low'}}
    missing={'ordinal':2,'predicted_class':None,'numeric_fusion':numeric,
             'cloud_judgment':{'available':False,'final_class':None}}
    totals={'calls':2,'failures':1,'input_tokens':0,'output_tokens':0,'total_call_seconds':0,'budget_rejections':0,'records':[]}
    for base in (source,out):
        atomic_json(base/'predictions/subject_01.json',{'result':{'measurements':{'segments':[valid,missing]}},'provider':totals,'events':[]})
    calls=[]
    protocol={'subjects':{'1':{}},'paths':[{'subject':1,'ordinal':1},{'subject':1,'ordinal':2}],'provider_model':'gpt-6.1-sol'}
    monkeypatch.setattr(recovery,'initialize',lambda *args:protocol)
    monkeypatch.setattr(recovery,'verify_protocol',lambda *args:None)
    monkeypatch.setattr(recovery,'verify_runtime_config',lambda *args:None)
    monkeypatch.setattr(recovery,'refresh',lambda *args:{'valid_gpt_paths':2})
    class Provider:
        model='gpt-6.1-sol'
        def __init__(self,*args):pass
        def summary(self):return copy.deepcopy(totals)
    monkeypatch.setattr(recovery,'DiagnosticProvider',Provider)
    def classify(provider,row):
        calls.append(row['ordinal'])
        return {'available':True,'final_class':'High'}
    monkeypatch.setattr(recovery,'classify',classify)
    recovery.recover(source,out,0)
    rows=json.loads((out/'predictions/subject_01.json').read_text())['result']['measurements']['segments']
    assert calls==[2] and rows[0]==valid and rows[1]['predicted_class']=='High'
    assert rows[1]['numeric_fusion']==numeric


def test_active_mil_training_recipe_runs_on_base_paths():
    features=np.arange(36,dtype=np.float32).reshape(12,3)/10
    paths=[{'window_start':i*2,'window_end':i*2+2,'accepted_windows':2,'label':i%2,'subject_key':i//2} for i in range(6)]
    head,report=fit_mil(features,paths,list(range(6)),torch.device('cpu'),epochs=1)
    assert report['train_paths']==6 and torch.isfinite(head(torch.from_numpy(features[:2])))
