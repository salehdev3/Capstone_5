"""Measured experimental runtime for v3; detector executes once, masks use direct M5."""
from pathlib import Path
import json,time,hashlib,importlib.util
import cv2,numpy as np,torch
from ultralytics import YOLO

def load_optimizer():
 p=Path(__file__).with_name('optimize.py');spec=importlib.util.spec_from_file_location('v3_optimizer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Pipeline:
 def __init__(self,root,device='cpu'):
  self.root=Path(root);self.run=self.root/'phase2_results/optimization_v3';self.config=json.loads((self.run/'selection/release_candidate.json').read_text());self.freeze=json.loads((self.run/'selection/freeze.json').read_text());self.op=load_optimizer()
  assert self.op.sha(self.run/'selection/release_candidate.json')==self.freeze['candidate_sha256']
  self.policy_sha256=self.freeze['candidate_sha256'];self.device=device;torch.set_num_threads(4)
  for path,key in [(self.config['detector']['path'],'detector_sha256'),(self.config['segmentation']['path'],'segmenter_sha256'),(self.config['species']['path'],'species_sha256'),(self.config['species']['policy'],'species_policy_sha256')]:assert self.op.sha(path)==self.freeze[key]
  self.detector=YOLO(self.config['detector']['path']);self.segmenter=YOLO(self.config['segmentation']['path'])
  ck=torch.load(self.config['species']['path'],map_location='cpu',weights_only=True);self.classes=ck['classes'];self.classifier=self.op.species_model(self.classes,ck).to(device).eval();self.temperature=json.loads(Path(self.config['species']['policy']).read_text())['temperature']
 def process_frame(self,path,meta):
  begin=time.perf_counter();im=cv2.imread(str(path))
  if im is None:raise ValueError('Unreadable image')
  d=self.config['detector'];s=self.config['segmentation'];result=self.detector.predict(im,conf=d['conf'],iou=d['nms'],imgsz=d['size'],rect=False,device=self.device,verbose=False)[0]
  boxes=result.boxes.xyxyn.cpu().numpy().tolist();scores=result.boxes.conf.cpu().numpy().tolist();det_end=time.perf_counter();objects=[]
  for box,confidence in zip(boxes,scores):
   rgb=self.op.crop(im,box);x=(rgb.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
   with torch.inference_mode():p=(self.classifier(torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).unsqueeze(0).to(self.device))/self.temperature).softmax(1)[0]
   score,index=p.max(0);objects.append(dict(box=box,detector_confidence=confidence,species=self.classes[int(index)],confidence=float(score),review=float(score)<.6))
  cls_end=time.perf_counter();polygons=[];segboxes=[]
  if boxes:
   result=self.segmenter.predict(im,conf=s['conf'],iou=.7,imgsz=s['size'],rect=False,device=self.device,verbose=False)[0]
   if result.masks is not None:polygons=[p.tolist() for p in result.masks.xyn];segboxes=result.boxes.xyxyn.cpu().numpy().tolist()
  pairs=self.op.box_matches(boxes,segboxes)
  for i,j in pairs:
   mask=self.op.raster([polygons[j]],im.shape[1],im.shape[0]);objects[i].update(relative_body_area=float(mask.mean()),polygon_normalized=polygons[j])
  for obj in objects:obj['review']=obj['review'] or 'relative_body_area' not in obj
  end=time.perf_counter()
  return dict(id=meta['id'],event=meta['event'],timestamp=meta.get('timestamp'),count=len(objects),objects=objects,segmentation=dict(matched_masks=len(pairs),policy_sha256=self.policy_sha256),timing=dict(detection=det_end-begin,classification=cls_end-det_end,segmentation=end-cls_end,total=end-begin))
 def process_batch(self,paths,metadata):
  if len(paths)!=len(metadata):raise ValueError('One metadata record per image required')
  frames=[self.process_frame(p,m) for p,m in zip(paths,metadata)];events=[]
  for event in sorted({r['event'] for r in frames}):
   group=[r for r in frames if r['event']==event];species={}
   for r in group:
    counts={}
    for o in r['objects']:counts[o['species']]=counts.get(o['species'],0)+1
    for key,value in counts.items():species[key]=max(species.get(key,0),value)
   if max(r['count'] for r in group)>0:events.append(dict(event=event,count=max(r['count'] for r in group),species_counts=species))
  return dict(frames=frames,events=events)

def benchmark(root,cache,device='cpu'):
 p=Pipeline(root,device);op=p.op;_,run,rows,_=op.init(root)
 ids=['59d0e396-23d2-11e8-a6a3-ec086b02610b','59a62b26-23d2-11e8-a6a3-ec086b02610b','5860ed49-23d2-11e8-a6a3-ec086b02610b','58629431-23d2-11e8-a6a3-ec086b02610b','5888bc6a-23d2-11e8-a6a3-ec086b02610b'];byid={r['id']:r for r in rows};meta=[byid[i] for i in ids];paths=[Path(cache)/'images'/(i+'.jpg') for i in ids]
 from datetime import datetime,timezone
 folder=run/'runtime_evaluations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
 for i in range(3):p.process_frame(paths[i],meta[i])
 timing=[]
 for i in range(20):
  frame=p.process_frame(paths[i%5],meta[i%5]);timing.append(dict(iteration=i,id=frame['id'],count=frame['count'],**frame['timing']));print('Timing',i+1,'/20',frame['timing']['total'],flush=True)
 seconds=[r['total'] for r in timing];op.write_csv(folder/'latencies.csv',timing)
 import resource,platform
 op.save(folder/'benchmark.json',dict(p50=float(np.quantile(seconds,.5)),p95=float(np.quantile(seconds,.95)),p99=float(np.quantile(seconds,.99)),throughput=1/float(np.mean(seconds)),peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,device=device,batch=1,warmup=3,repeats=20,scope='Same five validation functional cases as v2, sequential frame processing, excludes HTTP. Cumulative process RSS.',python=platform.python_version(),torch=torch.__version__,candidate_sha256=p.policy_sha256))
 # A real local HTTP request with the same staged API contracts.
 import requests,socket,threading,uvicorn
 from fastapi import FastAPI,UploadFile,File,Form,HTTPException
 import tempfile
 app=FastAPI(title='Camera-Trap v3 Candidate');lock=threading.Lock()
 @app.get('/health')
 def health():return {'status':'ready','candidate':p.policy_sha256}
 @app.post('/batch')
 def batch(images:list[UploadFile]=File(...),metadata:str=Form(...)):
  try:
   records=json.loads(metadata)
   if not isinstance(records,list) or len(records)!=len(images) or not images:raise ValueError('One record per image required')
   if not all(isinstance(r,dict) and r.get('id') and r.get('event') for r in records):raise ValueError('id and event required')
   if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate image IDs')
   with tempfile.TemporaryDirectory() as tmp:
    paths=[]
    for i,im in enumerate(images):
     path=Path(tmp)/f'{i}.jpg';path.write_bytes(im.file.read());paths.append(path)
    with lock:return p.process_batch(paths,records)
  except (ValueError,KeyError,AssertionError) as error:raise HTTPException(400,str(error))
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 server=uvicorn.Server(uvicorn.Config(app,log_level='warning'));thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
 try:
  base=f'http://127.0.0.1:{port}'
  for _ in range(100):
   try:
    if requests.get(base+'/health',timeout=1).ok:break
   except requests.RequestException:pass
   time.sleep(.1)
  response=requests.post(base+'/batch',files=[('images',(path.name,path.read_bytes(),'image/jpeg')) for path in paths],data={'metadata':json.dumps([dict(id=r['id'],event=r['event'],timestamp=r.get('timestamp')) for r in meta])},timeout=600)
  response.raise_for_status();result=response.json();op.save(folder/'service_test.json',dict(status=response.status_code,result=result,candidate_sha256=p.policy_sha256))
  for path,frame in zip(paths,result['frames']):
   im=cv2.imread(str(path));h,w=im.shape[:2]
   for obj in frame['objects']:
    x1,y1,x2,y2=obj['box'];cv2.rectangle(im,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),(0,0,255),2);cv2.putText(im,obj['species'],(int(x1*w),max(16,int(y1*h))),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
    if obj.get('polygon_normalized'):cv2.polylines(im,[(np.asarray(obj['polygon_normalized'])*[w,h]).astype(np.int32)],True,(0,255,0),2)
   cv2.imwrite(str(folder/(frame['id']+'_overlay.png')),im)
 finally:server.should_exit=True;thread.join(timeout=10);sock.close()
 print('Saved runtime evaluation:',folder);return folder

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--cache',default='/content/phase2_v3_cache');parser.add_argument('--device',default='cpu');a=parser.parse_args();benchmark(a.root,a.cache,a.device)
