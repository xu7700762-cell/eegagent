# -*- coding: utf-8 -*-
"""Only the current three-specialist raw EEG workspace and its matching evaluation."""
import asyncio
import json
from pathlib import Path
import threading
from typing import Literal, Optional
from urllib.parse import urlsplit
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .brain import BrainManager
from .cloud_config import ConfigError, public_settings, save_settings, test_connection
from .config import PROJECT, PIPELINE_ID, load_config
from .evaluation import EvaluationManager
from .recordings import MAX_UPLOAD

class SessionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    backend: Literal['api','mock'] = 'api'
    domains: list[Literal['vrms','fatigue','emotion']] = Field(default_factory=lambda:['vrms','fatigue','emotion'],min_length=1,max_length=3)
    recording_id: Optional[str] = None
    input_policy: Literal['raw_eeg_only'] = 'raw_eeg_only'
    pipeline_id: Literal['raw_eeg_gpt_tools_seed2026'] = PIPELINE_ID
    vrms_path_id: None = None
    task_id: None = None

class MessageRequest(BaseModel):
    text: str = Field(min_length=1,max_length=2000)

class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    seed: Literal[2026] = 2026
    resume_from: Optional[str] = None

class DocumentRequest(BaseModel):
    domain: Literal['shared','vrms','fatigue','emotion']
    title: str = Field(min_length=1,max_length=200)
    text: str = Field(min_length=1,max_length=80000)
    source: str = Field(min_length=1,max_length=1000)
    doi: str = ''
    page: str = ''

def create_app(cfg=None, brain_manager=None, task_manager=None, knowledge_index=None):
    cfg = cfg or load_config()
    app = FastAPI(title='EEGAgent raw EEG collaboration')
    brain = brain_manager or BrainManager(cfg)
    evaluations = EvaluationManager(cfg)
    config_path = cfg.get('cloud_config_path')
    lock = threading.RLock()
    app.state.brain = brain
    app.state.evaluations = evaluations
    app.state.cloud_testing = False
    web = Path(__file__).with_name('web')
    app.mount('/assets',StaticFiles(directory=web),name='assets')

    def busy():
        return brain.cloud_in_use() or evaluations.cloud_in_use() or app.state.cloud_testing

    def check_origin(request):
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.url.netloc:
            raise HTTPException(403,'只接受同源网页请求')

    def find(identity):
        try:
            return brain.get(identity)
        except KeyError:
            raise HTTPException(404,'未知对话')

    @app.get('/')
    def index():
        return FileResponse(web/'brain.html')

    @app.get('/settings')
    def settings_page():
        return FileResponse(web/'settings.html')

    @app.get('/api/brain/catalog')
    def catalog():
        return brain.catalog()

    @app.get('/api/brain/vrms/catalog')
    def model_catalog(seed: int=2026):
        if seed != 2026:
            raise HTTPException(400,'当前入口只使用 seed2026')
        configured = public_settings(config_path)
        manifest = Path(cfg['model_manifest'])
        results_file = PROJECT/'results/current_seed2026.json'
        metrics = json.loads(results_file.read_text(encoding='utf-8')) if results_file.exists() else {}
        return {'status':'ready','pipeline_id':PIPELINE_ID,'paths':[],
            'capabilities':{'cloud_ready':configured['configured'] and configured['model_id'].startswith('gpt-'),
                            'raw_model_assets_available':manifest.is_file(),'active_model':configured['model_id']},
            'boundary':'原始 EEG；目标标签只用于事后评价；无记录开头静息参考；GPT失败不发布最终类别',
            'current_metrics':metrics,'seed':2026}

    @app.get('/api/brain/recordings')
    def recordings():
        return {'recordings':brain.recordings.catalog(),'input_policy':'raw_eeg_only'}

    @app.post('/api/brain/recordings/edf')
    async def upload(request:Request,filename:str='recording.edf'):
        check_origin(request)
        if not filename.lower().endswith('.edf') or len(filename)>200:
            raise HTTPException(400,'请提供原始 EDF 文件')
        if busy():
            raise HTTPException(409,'当前分析正在运行')
        temporary = brain.recordings.root/('_upload_'+uuid.uuid4().hex+'.edf')
        try:
            size = 0
            with temporary.open('wb') as stream:
                async for block in request.stream():
                    size += len(block)
                    if size>MAX_UPLOAD:
                        raise HTTPException(413,'EDF 文件须不超过512MB')
                    stream.write(block)
            if not size:
                raise HTTPException(400,'EDF 文件为空')
            return brain.recordings.register_edf(temporary,filename)
        except (ValueError,OSError):
            raise HTTPException(400,'无法读取该 EDF 的原始格式和 EEG 通道')
        finally:
            temporary.unlink(missing_ok=True)

    @app.post('/api/brain/sessions')
    def create_session(document:SessionRequest):
        if len(set(document.domains)) != len(document.domains):
            raise HTTPException(400,'领域选择不能重复')
        with lock:
            if document.backend=='api' and not public_settings(config_path)['configured']:
                raise HTTPException(400,'请先保存云端接口、模型与密钥')
            if document.backend=='api' and (evaluations.cloud_in_use() or app.state.cloud_testing):
                raise HTTPException(409,'云端接口正被评价或连接测试占用')
            try:
                return brain.create(document.model_dump())
            except ValueError as exc:
                raise HTTPException(400,str(exc))

    @app.get('/api/brain/sessions')
    def sessions():
        return brain.list()

    @app.get('/api/brain/sessions/{identity}')
    def session_get(identity:str):
        return find(identity).snapshot()

    @app.post('/api/brain/sessions/{identity}/messages')
    def message(identity:str,document:MessageRequest):
        with lock:
            session = find(identity)
            if session.backend=='api' and (evaluations.cloud_in_use() or app.state.cloud_testing):
                raise HTTPException(409,'评价或连接测试正在使用接口')
            try:
                return session.message(document.text.strip())
            except (RuntimeError,ValueError) as exc:
                raise HTTPException(409,str(exc))

    @app.get('/api/brain/sessions/{identity}/events')
    async def events(identity:str,request:Request,after:int=-1):
        session = find(identity)
        async def stream():
            cursor=after
            while not await request.is_disconnected():
                snapshot=session.snapshot()
                for event in snapshot['events']:
                    if event['sequence']>cursor:
                        cursor=event['sequence']
                        yield 'id: '+str(cursor)+'\ndata: '+json.dumps(event,ensure_ascii=False)+'\n\n'
                if snapshot['state']!='running':
                    break
                await asyncio.sleep(.25)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache'})

    @app.get('/api/brain/sessions/{identity}/report')
    def session_report(identity:str):
        return find(identity).report() or {'status':'pending'}

    @app.get('/api/brain/knowledge/search')
    def search(q:str='',domains:str='vrms,fatigue,emotion',limit:int=5):
        try:
            return brain.knowledge.search(q[:2000],domains=domains.split(','),limit=limit)
        except ValueError as exc:
            raise HTTPException(400,str(exc))

    @app.post('/api/brain/knowledge/documents')
    def add_document(request:Request,document:DocumentRequest):
        check_origin(request)
        return brain.knowledge.add_document(**document.model_dump())

    @app.get('/api/cloud-settings')
    def read_cloud():
        return public_settings(config_path,in_use=busy())

    @app.put('/api/cloud-settings')
    async def save_cloud(request:Request):
        check_origin(request)
        with lock:
            if busy():
                raise HTTPException(409,'接口使用期间不能修改配置')
            try:
                return save_settings(await request.json(),config_path)
            except ConfigError as exc:
                raise HTTPException(400,str(exc))

    @app.post('/api/cloud-settings/test')
    def test_cloud(request:Request):
        check_origin(request)
        with lock:
            if busy():
                raise HTTPException(409,'接口使用期间不能连接测试')
            app.state.cloud_testing=True
        try:
            return test_connection(config_path)
        finally:
            with lock:
                app.state.cloud_testing=False

    @app.post('/api/brain/vrms/evaluations')
    def start_evaluation(document:EvaluationRequest):
        with lock:
            if busy():
                raise HTTPException(409,'请等待当前分析完成')
            cloud=public_settings(config_path)
            if not cloud['configured'] or cloud['model_id']!='gpt-6.1-sol':
                raise HTTPException(400,'当前冻结评价要求配置 gpt-6.1-sol')
            if not Path(cfg['model_manifest']).is_file():
                raise HTTPException(400,'请先配置本地冻结 VRMSModel 权重')
            try:
                return evaluations.start(document.seed, document.resume_from)
            except (RuntimeError,ValueError) as exc:
                raise HTTPException(409,str(exc))

    @app.get('/api/brain/vrms/evaluations')
    def evaluation_list():
        return {'evaluations':evaluations.list()}

    @app.get('/api/brain/vrms/evaluations/{identity}')
    def evaluation_get(identity:str):
        try:
            return evaluations.get(identity)
        except (KeyError,FileNotFoundError):
            raise HTTPException(404,'未知评价')

    @app.get('/api/brain/vrms/evaluations/{identity}/report')
    def evaluation_report(identity:str):
        record=evaluation_get(identity)
        if record['state']!='completed':
            raise HTTPException(409,'评价尚未完成')
        return record
    return app
