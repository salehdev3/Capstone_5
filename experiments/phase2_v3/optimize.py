"""Versioned GPU optimization of camera-trap detection, masks and crop classification."""
from pathlib import Path
import argparse,csv,hashlib,json,math,random,shutil,time,platform
import numpy as np
from PIL import Image,ImageDraw

MANIFEST='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'
PLAN=dict(version='optimization_v3',seed=42,detector='yolo11s.pt',segmenter='yolo11s-seg.pt',imgsz=960,epochs=70,patience=12,batch=4,species_epochs=30,species_patience=7,species_batch=24,
 targets=dict(species_accuracy=.90,species_macro_f1=.90,detector_map50=.90,mask_mean_iou=.90),
 acceptance=dict(animal_recall_min=.99,empty_removal_min=.2,species_macro_f1_min=.65,species_ece_max=.1,mask_iou_min=.7,event_count_mae_max=1,p95_seconds_max=2),
 selection='val_select only; temperature on val_calibrate; historical test excluded',baseline_full_pipeline_validation_iou=.8239696630179356)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def write_csv(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def init(root):
 root=Path(root);run=root/'phase2_results/optimization_v3';run.mkdir(parents=True,exist_ok=True)
 if (run/'plan.json').exists():assert read(run/'plan.json')==PLAN,'Changed plan: create a new version'
 else:save(run/'plan.json',PLAN)
 manifest=root/'phase1_source/data/cct20_team_v2/manifest.json';assert sha(manifest)==MANIFEST
 rows=read(manifest);classes=read(root/'phase1_source/data/cct20_team_v2/species.json');assert len(classes)==14
 for key in ('camera','event','pixel_sha256'):
  groups={}
  for r in rows:groups.setdefault(r[key],set()).add(r['split'])
  assert all(len(v)==1 for v in groups.values()),key
 return root,run,rows,classes

def setup_device():
 import torch
 assert torch.cuda.is_available(),'GPU required for v3 training. Connect a Colab GPU; CPU training is intentionally blocked.'
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 return 'cuda'

def prepare(root,cache):
 root,run,rows,classes=init(root);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
 chosen=[r for r in rows if r['split'] in ('train','val_select','val_calibrate')];by_id={r['id']:r for r in chosen};audits=[]
 for i,r in enumerate(chosen,1):
  source=root/'images'/(r['id']+'.jpg');assert source.is_file(),str(source)
  raw=source.read_bytes()
  with Image.open(source) as im:
   rgb=im.convert('RGB');assert rgb.size==(r['width'],r['height']);assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256'],r['id']
  target=cache/'images'/source.name;target.parent.mkdir(exist_ok=True)
  if not target.exists() or sha(target)!=hashlib.sha256(raw).hexdigest():target.write_bytes(raw)
  if r['split'] in ('train','val_select') and not(r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes'])):
   for kind in ('images','labels'):(cache/'detection'/kind/r['split']).mkdir(parents=True,exist_ok=True)
   shutil.copyfile(target,cache/'detection/images'/r['split']/source.name)
   labels=[]
   for b in r['boxes']:
    x1,y1,x2,y2=b['xyxy'];assert 0<=x1<x2<=1 and 0<=y1<y2<=1
    labels.append(f'0 {(x1+x2)/2} {(y1+y2)/2} {x2-x1} {y2-y1}')
   (cache/'detection/labels'/r['split']/(r['id']+'.txt')).write_text('\n'.join(labels))
  audits.append(dict(id=r['id'],split=r['split'],sha256=sha(target)))
  if i%250==0:print('Verified/copied',i,'/',len(chosen),flush=True)
 from shapely.geometry import Polygon
 from shapely.validation import explain_validity
 issues=[];mask_sources={};counts={}
 derived_path=root/'releases/Segmentation_M4_Guided_Recovery_v2/derived_training_manifest.json'
 assert derived_path.is_file(),'Restore the Recovery-v2 derived training manifest first'
 derived={r['id']:r for r in read(derived_path)}
 # Preserve original reference rasterization; flag geometric defects without silently rewriting labels.
 for split in ('train','val_select'):
  labels=sorted((root/'segmentation/labels'/split).glob('*.txt'));assert len(labels)==(150 if split=='train' else 40)
  counts[split]=len(labels)
  for p in labels:
   assert p.stem in by_id and by_id[p.stem]['split']==split
   lines=p.read_text().splitlines();assert lines,p.name
   if split=='train':
    entry=derived[p.stem];assert entry['source_label_sha256']==sha(p),'Training reference version differs'
    lines=[line for i,line in enumerate(lines) if i not in set(entry['removed_indices'])]
    assert lines
   text='\n'.join(lines)+'\n'
   for k,line in enumerate(lines):
    values=[float(v) for v in line.split()];assert len(values)>=7 and len(values)%2==1 and values[0]==0
    assert all(math.isfinite(v) and 0<=v<=1 for v in values[1:])
    poly=Polygon(np.array(values[1:]).reshape(-1,2))
    if not poly.is_valid:issues.append(dict(id=p.stem,split=split,polygon=k,reason=explain_validity(poly)))
   for kind in ('images','labels'):(cache/'segmentation'/kind/split).mkdir(parents=True,exist_ok=True)
   label_target=cache/'segmentation/labels'/split/p.name
   if split=='train':
    label_target.write_text(text);assert sha(label_target)==derived[p.stem]['derived_label_sha256']
   else:shutil.copyfile(p,label_target)
   shutil.copyfile(cache/'images'/(p.stem+'.jpg'),cache/'segmentation/images'/split/(p.stem+'.jpg'))
   mask_sources[f'{split}/{p.name}']=dict(original=sha(p),training_copy=sha(label_target))
 for task in ('detection','segmentation'):
  save(cache/task/'data.yaml',dict(path=str((cache/task).resolve()),train='images/train',val='images/val_select',names={0:'animal'}))
 
 if (run/'mask_sources.json').exists():assert read(run/'mask_sources.json')==mask_sources,'Mask release changed: start a new experiment version'
 save(run/'image_audit.json',audits);save(run/'mask_sources.json',mask_sources);save(run/'mask_geometry_findings.json',issues)
 save(run/'prepared.json',dict(cache=str(cache.resolve()),frames=len(chosen),mask_counts=counts,geometry_findings=len(issues),manifest_sha256=MANIFEST))
 print('Preparation saved. Geometry findings:',len(issues),flush=True)

def train_yolo(root,cache,task):
 import torch
 from ultralytics import YOLO
 root,run,rows,classes=init(root);setup_device();cache=Path(cache)
 assert (run/'prepared.json').is_file(),'Run prepare first'
 assert (cache/task/'data.yaml').exists(),'Local cache expired; rerun prepare'
 dest=run/task;done=dest/'completed.json'
 if done.exists():
  assert sha(dest/'weights/best.pt')==read(done)['best_sha256'];print(task,'already trained');return
 last=dest/'weights/last.pt'
 if last.exists():
  ck=torch.load(last,map_location='cpu',weights_only=False)
  if ck.get('epoch',-1)>=0 and ck.get('optimizer') is not None:
   model=YOLO(str(last));model.train(resume=True,device=0)
  else:raise RuntimeError(f'Completed/stripped last checkpoint found without completion receipt: inspect {dest} before proceeding')
 else:
  model=YOLO(PLAN['detector' if task=='detection' else 'segmenter'])
  model.train(data=str(cache/task/'data.yaml'),project=str(run),name=task,exist_ok=True,device=0,epochs=PLAN['epochs'],patience=PLAN['patience'],batch=PLAN['batch'],imgsz=PLAN['imgsz'],
   optimizer='AdamW',lr0=.0005 if task=='segmentation' else .001,weight_decay=.0005,cos_lr=True,warmup_epochs=3,seed=42,workers=2,amp=True,save=True,save_period=5,
   mosaic=.25 if task=='segmentation' else .5,close_mosaic=15,mixup=0,fliplr=.5,flipud=0,hsv_h=.01,hsv_s=.25,hsv_v=.25,scale=.3,translate=.1,plots=True)
 save(done,dict(best_sha256=sha(dest/'weights/best.pt'),task=task,selection='native validation AP; later overlap/operating-point comparison required'))

def species_model(classes,checkpoint):
 import torch
 from transformers import ResNetConfig,ResNetForImageClassification
 class Net(torch.nn.Module):
  def __init__(self):
   super().__init__();self.net=ResNetForImageClassification(ResNetConfig());self.net.classifier=torch.nn.Sequential(torch.nn.Flatten(start_dim=1),torch.nn.Linear(self.net.config.hidden_sizes[-1],len(classes)))
  def forward(self,x):return self.net(x).logits
 model=Net();model.load_state_dict(checkpoint['state_dict'],strict=True);return model

def crop(im,box):
 import cv2
 x1,y1,x2,y2=box;h,w=im.shape[:2];dx=(x2-x1)*.05;dy=(y2-y1)*.05
 l,t=max(0,int((x1-dx)*w)),max(0,int((y1-dy)*h));r,b=min(w,max(l+1,int((x2+dx)*w))),min(h,max(t+1,int((y2+dy)*h)))
 rgb=cv2.cvtColor(im[t:b,l:r],cv2.COLOR_BGR2RGB);ih,iw=rgb.shape[:2]
 return cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA if ih>224 or iw>224 else cv2.INTER_LINEAR)

def species_data(rows,classes,cache,split,augment=False):
 import torch,cv2
 from torchvision.transforms import ColorJitter,RandomHorizontalFlip,Compose,ToTensor,Normalize
 class Crops(torch.utils.data.Dataset):
  def __init__(self):
   self.items=[(r,b) for r in rows if r['split']==split for b in r['boxes'] if b['species'] in classes]
   self.transforms=Compose(([RandomHorizontalFlip(),ColorJitter(.15,.15,.1,.01)] if augment else [])+[ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
  def __len__(self):return len(self.items)
  def __getitem__(self,i):
   r,b=self.items[i];im=cv2.imread(str(Path(cache)/'images'/(r['id']+'.jpg')));assert im is not None
   return self.transforms(Image.fromarray(crop(im,b['xyxy']))),classes.index(b['species'])
 return Crops()

def classification_metrics(z,y,temp=1):
 from scipy.special import softmax
 from sklearn.metrics import f1_score,confusion_matrix
 p=softmax(np.asarray(z)/temp,axis=1);y=np.asarray(y);pred=p.argmax(1);confidence=p.max(1);correct=pred==y;ece=0.
 for i in range(15):
  mask=(confidence>=i/15)&(confidence<(i+1)/15 if i<14 else confidence<=1)
  if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
 return dict(n=len(y),accuracy=float(correct.mean()),macro_f1=float(f1_score(y,pred,labels=list(range(p.shape[1])),average='macro',zero_division=0)),ece=float(ece),confusion_matrix=confusion_matrix(y,pred,labels=list(range(p.shape[1]))).tolist())

def logits(model,loader,device):
 import torch
 model.eval();zs=[];ys=[]
 with torch.inference_mode():
  for x,y in loader:zs.append(model(x.to(device)).cpu().numpy());ys.extend(y.tolist())
 return np.concatenate(zs),np.array(ys)

def train_species(root,cache):
 import torch
 from torch.utils.data import DataLoader,Subset
 root,run,rows,classes=init(root);device=setup_device();dest=run/'species';dest.mkdir(exist_ok=True)
 source=root/'models/species_recovery_v1/best.pt';ck=torch.load(source,map_location='cpu',weights_only=True);assert ck['classes']==classes and ck['task']=='species'
 if (dest/'completed.json').exists():assert sha(dest/'best.pt')==read(dest/'completed.json')['best_sha256'];print('species already trained');return
 model=species_model(classes,ck).to(device)
 for p in model.parameters():p.requires_grad=False
 # Tune last residual stage plus head, keep earlier stages and BN running statistics fixed.
 for p in model.net.resnet.encoder.stages[-1].parameters():p.requires_grad=True
 for p in model.net.classifier.parameters():p.requires_grad=True
 opt=torch.optim.AdamW([dict(params=model.net.resnet.encoder.stages[-1].parameters(),lr=2e-5),dict(params=model.net.classifier.parameters(),lr=2e-4)],weight_decay=.001)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=PLAN['species_epochs'],eta_min=1e-6)
 train=species_data(rows,classes,cache,'train',True);valid=species_data(rows,classes,cache,'val_select');plain=species_data(rows,classes,cache,'train')
 loader=DataLoader(train,batch_size=PLAN['species_batch'],shuffle=True,num_workers=2,pin_memory=True)
 val_loader=DataLoader(valid,batch_size=48,num_workers=2);diagnostic=DataLoader(Subset(plain,random.Random(42).sample(range(len(plain)),min(512,len(plain)))),batch_size=48,num_workers=2)
 counts=np.bincount([classes.index(b['species']) for _,b in train.items],minlength=len(classes));assert (counts>0).all()
 weights=np.sqrt(counts.sum()/counts);weights=weights/weights.mean()
 loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=.05)
 baseline_z,baseline_y=logits(model,val_loader,device);baseline=classification_metrics(baseline_z,baseline_y);save(dest/'baseline_validation.json',baseline);save(dest/'class_balance.json',dict(classes=classes,training_counts=counts.tolist(),loss_weights=weights.tolist(),label_smoothing=.05))
 best=baseline['macro_f1'];bad=0;start=0;history=[]
 def save_weights():
  state={k:v.detach().cpu() for k,v in model.state_dict().items()};payload={**ck,'state_dict':state,'regime':'full','version':'optimization_v3','source_checkpoint_sha256':sha(source)}
  tmp=dest/'best.tmp';torch.save(payload,tmp);tmp.replace(dest/'best.pt')
 if (dest/'resume.pt').exists():
  state=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False);assert state['source_sha']==sha(source)
  model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start=state['epoch']+1;best=state['best'];bad=state['bad'];history=state['history']
  torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
 else:save_weights()
 for epoch in range(start,PLAN['species_epochs']):
  if bad>=PLAN['species_patience']:break
  began=time.perf_counter();model.train()
  for m in model.modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
  total=0.;n=0
  for x,y in loader:
   opt.zero_grad(set_to_none=True);x,y=x.to(device),y.to(device);loss=loss_fn(model(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(y);n+=len(y)
  scheduler.step();z,y=logits(model,val_loader,device);m=classification_metrics(z,y);tz,ty=logits(model,diagnostic,device);tm=classification_metrics(tz,ty)
  history.append(dict(epoch=epoch+1,train_loss=total/n,train_subset_accuracy=tm['accuracy'],val_accuracy=m['accuracy'],val_macro_f1=m['macro_f1'],val_ece=m['ece'],accuracy_gap=tm['accuracy']-m['accuracy'],seconds=time.perf_counter()-began))
  if m['macro_f1']>best+1e-4:best=m['macro_f1'];bad=0;save_weights()
  else:bad+=1
  state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,bad=bad,history=history,source_sha=sha(source),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
  torch.save(state,dest/'resume.tmp');(dest/'resume.tmp').replace(dest/'resume.pt');write_csv(dest/'history.csv',history)
  print('Species:',history[-1],flush=True)
 model.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True)['state_dict']);z,y=logits(model,val_loader,device)
 cal=DataLoader(species_data(rows,classes,cache,'val_calibrate'),batch_size=48,num_workers=2);cz,cy=logits(model,cal,device)
 from scipy.optimize import minimize_scalar
 from scipy.special import logsumexp
 objective=lambda t:float(np.mean(logsumexp(cz/math.exp(t),axis=1)-(cz/math.exp(t))[np.arange(len(cy)),cy]))
 fit=minimize_scalar(objective,bounds=(-3,3),method='bounded');assert fit.success;temperature=math.exp(fit.x)
 save(dest/'policy.json',dict(temperature=temperature,calibration_split='val_calibrate',test_used_for_calibration=False,checkpoint_sha256=sha(dest/'best.pt'),species_review_confidence=.6,version='optimization_v3'))
 save(dest/'validation.json',classification_metrics(z,y,temperature));np.savez_compressed(dest/'validation_logits.npz',logits=z,labels=y)
 save(dest/'calibration.json',dict(before=classification_metrics(cz,cy),after=classification_metrics(cz,cy,temperature)))
 save(dest/'completed.json',dict(best_sha256=sha(dest/'best.pt'),baseline_retained=best<=baseline['macro_f1']+1e-4,selection='best macro-F1 vs recovery baseline on val_select',fitting_diagnostic='history contains unaugmented 512-training-crop accuracy and full validation accuracy; gap is diagnostic, not proof of zero overfitting'))

def box_matches(gt,pred):
 from scipy.optimize import linear_sum_assignment
 a=np.asarray(gt,float).reshape(-1,4);b=np.asarray(pred,float).reshape(-1,4)
 if not len(a) or not len(b):return []
 inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
 iou=inter/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None,:]-inter,1e-9)
 x,y=linear_sum_assignment(-(iou>=.5).astype(float)*1000-iou)
 return [(int(i),int(j)) for i,j in zip(x,y) if iou[i,j]>=.5]

def detection_metrics(records):
 tp=sum(r['tp'] for r in records);fp=sum(r['fp'] for r in records);fn=sum(r['fn'] for r in records)
 animal=[r for r in records if r['animal']];empty=[r for r in records if not r['animal']]
 return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),animal_recall=sum(r['count']>0 for r in animal)/max(len(animal),1),empty_removal=sum(r['count']==0 for r in empty)/max(len(empty),1),n=len(records))

def evaluate_detector(model,rows,cache,conf,nms,size):
 records=[]
 for r in rows:
  result=model.predict(str(Path(cache)/'images'/(r['id']+'.jpg')),conf=conf,iou=nms,imgsz=size,rect=False,device=0,verbose=False)[0]
  gt=[b['xyxy'] for b in r['boxes']];pred=result.boxes.xyxyn.cpu().numpy().tolist();pairs=box_matches(gt,pred)
  records.append(dict(id=r['id'],animal=bool(r['animal']),count=len(pred),tp=len(pairs),fp=len(pred)-len(pairs),fn=len(gt)-len(pairs)))
 return detection_metrics(records)

def raster(polygons,w,h):
 im=Image.new('1',(w,h));d=ImageDraw.Draw(im)
 for p in polygons:
  if len(p)>=3:d.polygon([(float(x)*w,float(y)*h) for x,y in p],fill=1)
 return np.asarray(im,dtype=bool)

def evaluate_masks(model,paths,cache,conf,size):
 records=[]
 for path in paths:
  im_path=Path(cache)/'images'/(path.stem+'.jpg')
  with Image.open(im_path) as im:w,h=im.size
  polygons=[np.asarray([float(x) for x in line.split()[1:]]).reshape(-1,2) for line in path.read_text().splitlines() if line.strip()]
  gt=raster(polygons,w,h);result=model.predict(str(im_path),conf=conf,iou=.7,imgsz=size,rect=False,device=0,verbose=False)[0]
  pred=raster(result.masks.xyn if result.masks is not None else [],w,h);inter=int((gt&pred).sum());union=int((gt|pred).sum());den=int(gt.sum()+pred.sum())
  records.append(dict(id=path.stem,iou=inter/union if union else 1.,dice=2*inter/den if den else 1.,area_error=abs(float(gt.mean())-float(pred.mean()))))
 return dict(n=len(records),mean_iou=float(np.mean([r['iou'] for r in records])),mean_dice=float(np.mean([r['dice'] for r in records])),area_mae_fraction=float(np.mean([r['area_error'] for r in records]))),records

def select(root,cache):
 from ultralytics import YOLO
 root,run,rows,classes=init(root);setup_device();cache=Path(cache)
 dest=run/'selection';dest.mkdir(exist_ok=True)
 if (dest/'release_candidate.json').exists():print('Selection already frozen:',dest/'release_candidate.json');return
 valid=[r for r in rows if r['split']=='val_select' and not(r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes']))];assert len(valid)==1208
 detector_paths=[root/'releases/Segmentation_M4_Guided_Recovery_v2/detector.pt',run/'detection/weights/best.pt']
 sweep=[];aps={}
 # Low-confidence predictions once per NMS; confidence thresholds filtered without repeating inference.
 for path in detector_paths:
  assert path.is_file(),path
  model=YOLO(str(path));name='baseline' if path==detector_paths[0] else 'candidate';size=640 if name=='baseline' else PLAN['imgsz']
  a=model.val(data=str(cache/'detection/data.yaml'),split='val',imgsz=size,conf=.001,iou=.7,device=0,batch=4,project=str(dest),name=name+'_detector_ap',plots=True,verbose=False)
  aps[name]=dict(map50=float(a.box.map50),map50_95=float(a.box.map),checkpoint_sha256=sha(path))
  for nms in (.3,.5):
   cache_path=dest/f'{name}_nms_{nms}.json'
   if cache_path.exists():cached=read(cache_path);assert cached['sha']==sha(path)
   else:
    predictions=[]
    for i,r in enumerate(valid,1):
     result=model.predict(str(cache/'images'/(r['id']+'.jpg')),conf=.005,iou=nms,imgsz=size,rect=False,device=0,verbose=False)[0]
     predictions.append(dict(id=r['id'],boxes=result.boxes.xyxyn.cpu().numpy().tolist(),scores=result.boxes.conf.cpu().numpy().tolist()))
     if i%200==0:print('Detection validation',name,nms,i,'/1208',flush=True)
    cached=dict(sha=sha(path),predictions=predictions);save(cache_path,cached)
   by_id={r['id']:r for r in cached['predictions']}
   for conf in (.005,.01,.02,.03,.04,.045,.05,.075,.1,.15,.25,.4,.5):
    records=[]
    for r in valid:
     p=by_id[r['id']];pred=[b for b,s in zip(p['boxes'],p['scores']) if s>conf];gt=[b['xyxy'] for b in r['boxes']];m=box_matches(gt,pred)
     records.append(dict(animal=bool(r['animal']),count=len(pred),tp=len(m),fp=len(pred)-len(m),fn=len(gt)-len(m)))
    sweep.append(dict(model=name,path=str(path),size=size,conf=conf,nms=nms,**detection_metrics(records)))
 write_csv(dest/'detector_threshold_sweep.csv',sweep);save(dest/'detector_ap.json',aps)
 feasible=[r for r in sweep if r['animal_recall']>=.99 and r['empty_removal']>=.2]
 winner=max(feasible,key=lambda r:(r['empty_removal'],r['f1'])) if feasible else max(sweep,key=lambda r:(r['animal_recall'],r['empty_removal']))
 winner['constraints_met']=bool(feasible)
 train_rows=[r for r in rows if r['split']=='train' and not(r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes']))]
 sample=random.Random(42).sample(train_rows,min(512,len(train_rows)))
 train_metrics=evaluate_detector(YOLO(winner['path']),sample,cache,winner['conf'],winner['nms'],winner['size'])
 save(dest/'detector_fitting_diagnostic.json',dict(train_sample=train_metrics,validation=winner,scope='Unaugmented fixed train subset vs full validation; no guarantee against over/underfitting'))
 paths=sorted((cache/'segmentation/labels/val_select').glob('*.txt'));assert len(paths)==40
 seg_paths=[root/'releases/Segmentation_M4_Guided_Recovery_v2/selected.pt',run/'segmentation/weights/best.pt',run/'segmentation/weights/last.pt'];comparisons=[]
 for index,path in enumerate(seg_paths):
  assert path.is_file(),path
  model=YOLO(str(path));size=640 if index==0 else PLAN['imgsz']
  for conf in (.05,.1,.2,.35,.5):
   metric,per=evaluate_masks(model,paths,cache,conf,size);comparisons.append(dict(path=str(path),checkpoint_sha256=sha(path),conf=conf,size=size,**metric))
   write_csv(dest/f'masks_{index}_{conf}.csv',per);print('Mask validation:',index,conf,metric,flush=True)
 write_csv(dest/'mask_validation_comparison.csv',comparisons);mask=max(comparisons,key=lambda r:r['mean_iou'])
 m=YOLO(mask['path']);a=m.val(data=str(cache/'segmentation/data.yaml'),split='val',imgsz=mask['size'],conf=.001,iou=.7,device=0,batch=4,project=str(dest),name='selected_mask_ap',plots=True,verbose=False)
 mask['native_mask_map50']=float(a.seg.map50);mask['native_mask_map50_95']=float(a.seg.map)
 train_paths=random.Random(42).sample(sorted((cache/'segmentation/labels/train').glob('*.txt')),40)
 train_m,_=evaluate_masks(m,train_paths,cache,mask['conf'],mask['size']);mask['train_subset_iou']=train_m['mean_iou'];mask['train_validation_iou_gap']=train_m['mean_iou']-mask['mean_iou']
 species=read(run/'species/validation.json');policy=read(run/'species/policy.json')
 candidate=dict(version='optimization_v3',detector=winner,segmentation=mask,species=dict(path=str(run/'species/best.pt'),policy=str(run/'species/policy.json'),**species),
  target_results=dict(species_accuracy=species['accuracy']>.9,species_macro_f1=species['macro_f1']>.9,detector_map50=aps[winner['model']]['map50']>.9,mask_mean_iou=mask['mean_iou']>.9),
  mask_quality_matches_archived_full_pipeline=mask['mean_iou']>=PLAN['baseline_full_pipeline_validation_iou'],
  scope='Validation-selected experimental candidate; masks are direct learned masks, not Recovery-v2 SAM masks. Fresh deployment evaluation required.',deployment_status='candidate',test_evaluated=False)
 save(dest/'release_candidate.json',candidate);save(dest/'freeze.json',dict(candidate_sha256=sha(dest/'release_candidate.json'),detector_sha256=sha(winner['path']),segmenter_sha256=sha(mask['path']),species_sha256=sha(run/'species/best.pt'),species_policy_sha256=sha(run/'species/policy.json')))
 print('CANDIDATE:',json.dumps(candidate,indent=2),flush=True)

def report(root):
 root,run,rows,classes=init(root);candidate=read(run/'selection/release_candidate.json');freeze=read(run/'selection/freeze.json');assert freeze['candidate_sha256']==sha(run/'selection/release_candidate.json')
 lines=['# Phase 2 v3 measured validation results','','| Component | Metric | Value |','| --- | --- | --- |',f"| Species | Accuracy | {candidate['species']['accuracy']:.6f} |",f"| Species | Macro-F1 | {candidate['species']['macro_f1']:.6f} |",f"| Detector | Animal-frame recall | {candidate['detector']['animal_recall']:.6f} |",f"| Detector | Empty removal | {candidate['detector']['empty_removal']:.6f} |",f"| Direct segmentation | Mean IoU | {candidate['segmentation']['mean_iou']:.6f} |",'',candidate['scope'],'','The released v2 report remains the measured deployment baseline until candidate HTTP execution, count/latency evaluation, ONNX parity and acceptance results are available.']
 (run/'VALIDATION_REPORT.md').write_text('\n'.join(lines));print('Saved:',run/'VALIDATION_REPORT.md')

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--cache',default='/content/phase2_v3_cache');parser.add_argument('--stage',required=True,choices=['prepare','detection','segmentation','species','select','report']);args=parser.parse_args()
 if args.stage=='prepare':prepare(args.root,args.cache)
 elif args.stage in ('detection','segmentation'):train_yolo(args.root,args.cache,args.stage)
 elif args.stage=='species':train_species(args.root,args.cache)
 elif args.stage=='select':select(args.root,args.cache)
 else:report(args.root)
