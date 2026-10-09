# -*- coding: utf-8 -*-
"""Asynchronous webpage batches over the same raw EEG classifier as conversations."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import yaml
from .config import PROJECT, atomic_json

class EvaluationManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg['output_root']) / 'evaluations'
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock = threading.RLock()
        self.active = None
        for path in self.root.glob('*/web_record.json'):
            record = json.loads(path.read_text(encoding='utf-8'))
            if not record.get('terminal'):
                record.update(state='interrupted',terminal=True,error='process_restarted')
                atomic_json(path,record)

    def cloud_in_use(self):
        with self.lock:
            return self.active is not None

    def list(self):
        records = [self.get(path.parent.name) for path in self.root.glob('*/web_record.json')]
        return sorted(records,key=lambda record:record.get('created',0),reverse=True)

    def get(self, identity):
        if not isinstance(identity,str) or len(identity) != 12 or any(c not in '0123456789abcdef' for c in identity):
            raise KeyError(identity)
        directory = self.root/identity
        record = json.loads((directory/'web_record.json').read_text(encoding='utf-8'))
        progress = directory/'run/progress.json'
        if not progress.exists():
            progress = directory/'progress.json'
        if progress.exists():
            value = json.loads(progress.read_text(encoding='utf-8'))
            record.update(total=value['total'],completed_paths=value.get('completed_paths',0),
                          cloud_completed=value.get('valid_gpt_paths',0),
                          cloud_failed=value.get('completed_paths',0)-value.get('valid_gpt_paths',0))
        summary = directory/'summary.json'
        if summary.exists():
            value = json.loads(summary.read_text(encoding='utf-8'))
            def visible(metrics):
                return {'total':metrics['total'],'correct':metrics['correct'],'accuracy':metrics['accuracy'],
                        'coverage':metrics['coverage'],'available':metrics['valid_gpt_paths']}
            record['metrics'] = {'vrmsmodel':visible(value['vrmsmodel_metrics']),'gpt':visible(value['metrics'])}
            record['cloud_metrics_coverage'] = value['metrics']['coverage']
        record['can_resume'] = (record['terminal'] and record.get('cloud_failed',0)>0 and
                                (directory/'run/prediction_complete.json').is_file())
        return record

    def start(self, seed=2026, resume_from=None):
        if seed != 2026:
            raise ValueError('当前网页只运行 seed2026')
        with self.lock:
            if self.active:
                raise RuntimeError('已有评价正在运行')
            if resume_from:
                try:
                    previous = self.get(resume_from)
                except (KeyError,FileNotFoundError):
                    raise ValueError('未知补跑来源')
                if not previous['can_resume']:
                    raise ValueError('原评价必须完成全部首次预测且仍有失败分类，才能补跑')
            identity = uuid.uuid4().hex[:12]
            directory = self.root/identity
            directory.mkdir()
            record = {'id':identity,'seed':2026,'model_id':'gpt-6.1-sol','state':'running','terminal':False,
                      'total':146,'completed_paths':0,'cloud_completed':0,'cloud_failed':0,'created':time.time()}
            if resume_from:
                record['resume_from'] = resume_from
            atomic_json(directory/'web_record.json',record)
            self.active = identity
            threading.Thread(target=self._run,args=(directory,record),daemon=True).start()
            return record

    def _run(self, directory, record):
        run = directory/'run'
        try:
            config_path = directory/'execution.yaml'
            if record.get('resume_from'):
                # Preserve the original frozen config bytes. A path-only copy
                # must not replace or bypass the original source hash contract.
                config_path = self.root/record['resume_from']/'execution.yaml'
                (directory/'execution.yaml').write_bytes(config_path.read_bytes())
            else:
                config_path.write_text(yaml.safe_dump(self.cfg,allow_unicode=True),encoding='utf-8')
            if record.get('resume_from'):
                command = [sys.executable,'-X','utf8','-B','-m','scripts.recover_raw_gpt_failures',
                    '--source',str(self.root/record['resume_from']/'run'),'--out',str(run),
                    '--config',str(config_path)]
                with (directory/'worker.log').open('a',encoding='utf-8') as stream:
                    if subprocess.run(command,cwd=PROJECT,stdout=stream,stderr=stream).returncode:
                        raise RuntimeError('失败调用补跑进程未完成')
                stages = ('score',)
            else:
                stages = ('prepare','predict','score')
            for stage in stages:
                command = [sys.executable,'-X','utf8','-B','-m','scripts.evaluate_raw_gpt_all',
                           '--stage',stage,'--out',str(run),'--config',str(config_path)]
                with (directory/'worker.log').open('a',encoding='utf-8') as stream:
                    completed = subprocess.run(command,cwd=PROJECT,stdout=stream,stderr=stream)
                if completed.returncode:
                    raise RuntimeError('评价进程未完成，详见本地日志')
            record.update(state='completed',terminal=True)
        except Exception as exc:
            record.update(state='failed',terminal=True,error=type(exc).__name__)
        finally:
            # Files stay in the run directory; expose metadata without API logs.
            for name in ('progress.json','summary.json','report.md'):
                if (run/name).exists():
                    (directory/name).write_bytes((run/name).read_bytes())
            atomic_json(directory/'web_record.json',record)
            with self.lock:
                self.active = None
