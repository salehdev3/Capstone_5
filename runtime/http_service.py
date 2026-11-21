"""Local HTTP service and recorded demo for an already initialized Recovery v2 pipeline."""
from pathlib import Path
import hashlib
import json
import socket
import tempfile
import threading
import time
import uuid
from datetime import datetime,timezone
import requests
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from starlette.concurrency import run_in_threadpool
import uvicorn

MANIFEST_SHA='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'


def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def build_app(pipeline,instance=None):
    instance=instance or uuid.uuid4().hex
    app=FastAPI(title='Camera-Trap Event Triage',version='integrated-species-cv')
    lock=threading.Lock()
    @app.get('/health')
    def health():
        return dict(status='ready',instance=instance,segmentation_policy_sha256=pipeline.policy_sha256)
    def invoke(paths,records):
        with lock:return pipeline.process_batch(paths,records)
    @app.post('/batch')
    async def batch_endpoint(images:list[UploadFile]=File(...),metadata:str=Form(...)):
        try:
            records=json.loads(metadata)
            if not isinstance(records,list) or not images or len(records)!=len(images):
                raise ValueError('One metadata record is required per image')
            if not all(isinstance(r,dict) and r.get('id') and r.get('event') for r in records):
                raise ValueError('Each record requires id and event')
            if len({str(r['id']) for r in records})!=len(records):
                raise ValueError('Image IDs must be unique within a batch')
            with tempfile.TemporaryDirectory() as folder:
                paths=[]
                for i,image in enumerate(images):
                    raw=await image.read()
                    if not raw:raise ValueError('Empty image upload')
                    path=Path(folder)/f'{i}.jpg';path.write_bytes(raw);paths.append(path)
                return await run_in_threadpool(invoke,paths,records)
        except (ValueError,AssertionError,KeyError) as exc:
            raise HTTPException(status_code=400,detail=str(exc)) from exc
    return app,instance


def start_service(pipeline):
    app,instance=build_app(pipeline)
    listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True);thread.start()
    base=f'http://127.0.0.1:{port}'
    for _ in range(100):
        if not thread.is_alive():raise RuntimeError('HTTP server exited during startup')
        try:
            response=requests.get(base+'/health',timeout=1)
            if response.ok and response.json().get('instance')==instance:
                return server,thread,base,instance
        except requests.RequestException:pass
        time.sleep(.1)
    server.should_exit=True
    raise RuntimeError('HTTP server did not become ready')


