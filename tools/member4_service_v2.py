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
    app=FastAPI(title='Camera-Trap Event Triage',version='recovery-v2')
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


def select_demo(root):
    root=Path(root);manifest=root/'phase1_source/data/cct20_team_v2/manifest.json'
    if not manifest.is_file():raise FileNotFoundError('Missing frozen manifest: '+str(manifest))
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==MANIFEST_SHA
    rows=json.loads(manifest.read_text());images=root/'images'
    candidates=[r for r in rows if r['split']=='val_select' and (images/(r['id']+'.jpg')).is_file()]
    # Two positive events plus one empty event when available, selected before inference.
    chosen=[];events=set()
    for animal,limit in [(1,2),(0,1)]:
        selected=0
        for r in candidates:
            if int(r['animal'])==animal and r['event'] not in events:
                chosen.append(r);events.add(r['event']);selected+=1
                if selected==limit:break
    if len(chosen)<3:
        for r in candidates:
            if r['event'] not in events:
                chosen.append(r);events.add(r['event'])
                if len(chosen)==3:break
    if len(chosen)!=3:raise ValueError('Need three available validation events for the recorded demo')
    return chosen,images


def record_demo(pipeline,root,output):
    from PIL import Image,ImageDraw
    import pandas as pd
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    demo,images=select_demo(root)
    folder=output/'member4_recovery_v2/http_demos'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    metadata=[dict(id=r['id'],event=r['event'],timestamp=r.get('timestamp')) for r in demo]
    save_json(folder/'request_metadata.json',metadata)
    server,thread,base,instance=start_service(pipeline)
    print('Local HTTP service ready:',base,flush=True)
    print('Running three validation images through detection, species and Recovery v2 masks. CPU inference may take several minutes.',flush=True)
    started=time.perf_counter()
    try:
        response=requests.post(base+'/batch',
            files=[('images',(r['id']+'.jpg',(images/(r['id']+'.jpg')).read_bytes(),'image/jpeg')) for r in demo],
            data={'metadata':json.dumps(metadata)},timeout=(10,1200))
        if not response.ok:raise RuntimeError(f'HTTP {response.status_code}: {response.text[:2000]}')
        result=response.json()
        assert [r['id'] for r in result['frames']]==[r['id'] for r in demo]
        elapsed=time.perf_counter()-started
        save_json(folder/'service_test.json',dict(status=response.status_code,instance=instance,base_url=base,result=result))
        record=dict(kind='executed local HTTP request',split='val_select',images=3,
            frame_ids=[r['id'] for r in demo],source_manifest_sha256=MANIFEST_SHA,
            segmentation_policy_sha256=pipeline.policy_sha256,http_status=response.status_code,
            elapsed_seconds=elapsed,scope='Three-image functional HTTP demo including transport; not a throughput benchmark or a video recording.')
        save_json(folder/'demo_record.json',record)
        provenance=Path(pipeline.output_dir)/'integration_provenance.json'
        if provenance.is_file():save_json(folder/'integration_provenance.json',json.loads(provenance.read_text()))
        for r,frame in zip(demo,result['frames']):
            im=Image.open(images/(r['id']+'.jpg')).convert('RGB');draw=ImageDraw.Draw(im)
            for obj in frame['objects']:
                x1,y1,x2,y2=obj['box'];color='orange' if obj['review'] else 'red'
                draw.rectangle((x1*im.width,y1*im.height,x2*im.width,y2*im.height),outline=color,width=3)
                draw.text((x1*im.width,y1*im.height),f"{obj['species']} {obj['confidence']:.2f}",fill='yellow')
                points=[(float(x*im.width),float(y*im.height)) for x,y in obj.get('polygon_normalized',[])]
                if len(points)>=3:draw.line(points+[points[0]],fill='lime',width=3)
            im.save(folder/(r['id']+'_overlay.png'))
        pd.DataFrame(result['events']).to_csv(folder/'events.csv',index=False)
        # Keep legacy audit filenames pointing to this actually executed demo.
        save_json(output/'service_test.json',dict(status=response.status_code,result=result,evidence_dir=str(folder),instance=instance))
        save_json(output/'demo_record.json',dict(record,evidence_dir=str(folder)))
        print('HTTP batch demo: PASS',flush=True);print('Evidence saved:',folder,flush=True)
        return dict(server=server,thread=thread,base_url=base,app=server.config.app,demo_output=result,demo=demo,images=images,evidence_dir=folder)
    except Exception:
        server.should_exit=True
        raise
