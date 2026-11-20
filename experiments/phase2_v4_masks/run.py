"""Train on 56 approved training-mask updates; compare all models on unchanged validation."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,zipfile,time,platform
import numpy as np
from PIL import Image
MANIFEST='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'
WARMSTART='60dfc13188fec981424105452db51e44a00ca5fd3530b8d6a2397653d460b4b2'
PLAN=dict(version='mask_training_v4',seed=42,epochs=40,patience=8,imgsz=960,batch=4,lr0=.0001,selection='unchanged original 40-image val_select masks',approved_training_updates=56,approved_validation_updates_used=0,stretch_iou_target=.9,test_used=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def init(root):
 root=Path(root);run=root/'phase2_results/mask_training_v4';run.mkdir(parents=True,exist_ok=True)
 if (run/'plan.json').exists():assert read(run/'plan.json')==PLAN,'Use a new version for changed settings'
 else:save(run/'plan.json',PLAN)
 manifest=root/'phase1_source/data/cct20_team_v2/manifest.json';assert sha(manifest)==MANIFEST
 return root,run,{r['id']:r for r in read(manifest)}
def prepare(root,cache,overrides):
 root,run,rows=init(root);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
 original_release=read(root/'segmentation/reference_release.json');derived={r['id']:r for r in read(root/'releases/Segmentation_M4_Guided_Recovery_v2/derived_training_manifest.json')};audit=[]
 with zipfile.ZipFile(overrides) as z:
  assert z.testzip() is None;changes=json.loads(z.read('override_manifest.json'))['approved_overrides'];assert sum(r['split']=='train' for r in changes.values())==56
  for split,expected in [('train',150),('val_select',40)]:
   labels=sorted((root/'segmentation/labels'/split).glob('*.txt'));assert len(labels)==expected
   for p in labels:
    ident=p.stem;r=rows[ident];assert r['split']==split and sha(p)==original_release['mask_hashes'][ident]
    mode='original_validation_reference' if split=='val_select' else 'baseline_training_reference'
    content=p.read_bytes()
    if split=='train':
     if ident in changes:
      change=changes[ident];assert change['source_label_sha256']==sha(p) and change['image_pixel_sha256']==r['pixel_sha256'];content=z.read(f'labels/train/{ident}.txt');assert hashlib.sha256(content).hexdigest()==change['label_sha256'];mode='named_approved_training_update'
     else:
      old=derived[ident];assert old['source_label_sha256']==sha(p);lines=[s for i,s in enumerate(p.read_text().splitlines()) if i not in set(old['removed_indices'])];content=('\n'.join(lines)+'\n').encode();assert hashlib.sha256(content).hexdigest()==old['derived_label_sha256']
    # Numeric/raster-compatible YOLO labels; disclosed touching contours remain permitted.
    assert content.strip()
    for line in content.decode().splitlines():
     a=[float(v) for v in line.split()];assert a[0]==0 and len(a)>=7 and len(a)%2==1 and all(np.isfinite(v) and 0<=v<=1 for v in a[1:])
    image=root/'images'/(ident+'.jpg');fast=Path('/content/phase2_v3_cache/images')/image.name
    if fast.exists():image=fast
    with Image.open(image) as im:
     rgb=im.convert('RGB');assert rgb.size==(r['width'],r['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256']
    for folder in [cache/'images',cache/'dataset/images'/split,cache/'dataset/labels'/split]:folder.mkdir(parents=True,exist_ok=True)
    target=cache/'images'/image.name;shutil.copyfile(image,target);shutil.copyfile(target,cache/'dataset/images'/split/image.name);(cache/'dataset/labels'/split/p.name).write_bytes(content)
    audit.append(dict(id=ident,split=split,mode=mode,original_label_sha256=sha(p),used_label_sha256=hashlib.sha256(content).hexdigest(),image_sha256=sha(target)))
    if len(audit)%25==0:print('Prepared:',len(audit),'/190',flush=True)
 save(cache/'dataset/data.yaml',dict(path=str((cache/'dataset').resolve()),train='images/train',val='images/val_select',names={0:'animal'}));save(run/'data_audit.json',audit)
 assert sum(r['mode']=='named_approved_training_update' for r in audit)==56
 assert all(r['original_label_sha256']==r['used_label_sha256'] for r in audit if r['split']=='val_select')
 save(run/'prepared.json',dict(override_archive_sha256=sha(overrides),train=150,validation=40,training_updates=56,validation_references='original unchanged; one unresolved reference remains tracked',test_used=False))
 print('Prepared 56 training updates; original validation kept unchanged.',flush=True)
def train(root,cache):
 import torch
 from ultralytics import YOLO
 root,run,rows=init(root);assert torch.cuda.is_available(),'GPU required for training; use a Colab GPU runtime.'
 assert (run/'prepared.json').exists() and (Path(cache)/'dataset/data.yaml').exists(),'Run prepare first'
 start=root/'phase2_results/optimization_v3/segmentation/weights/best.pt';assert sha(start)==WARMSTART
 folder=run/'segmenter'
 if (folder/'completed.json').exists():assert sha(folder/'weights/best.pt')==read(folder/'completed.json')['checkpoint_sha256'];print('Already trained.');return
 last=folder/'weights/last.pt'
 if last.exists():
  checkpoint=torch.load(last,map_location='cpu',weights_only=False);assert checkpoint.get('epoch',-1)>=0 and checkpoint.get('optimizer') is not None,'Inspect this existing finished/stripped run before continuing';YOLO(str(last)).train(resume=True,device=0)
 else:
  YOLO(str(start)).train(data=str(Path(cache)/'dataset/data.yaml'),project=str(run),name='segmenter',exist_ok=True,epochs=40,patience=8,imgsz=960,batch=4,device=0,seed=42,optimizer='AdamW',lr0=.0001,weight_decay=.001,cos_lr=True,warmup_epochs=1,workers=2,amp=True,mosaic=0.,mixup=0.,fliplr=.5,flipud=0.,hsv_h=.01,hsv_s=.15,hsv_v=.15,scale=.2,translate=.05,save_period=5,plots=True)
 save(folder/'completed.json',dict(checkpoint_sha256=sha(folder/'weights/best.pt'),warmstart_sha256=WARMSTART,test_used=False))
def compare(root,cache,code):
 import torch
 from ultralytics import YOLO,SAM
 root,run,rows=init(root);assert torch.cuda.is_available(),'GPU recommended for the full comparison; connect the training GPU.';cache=Path(cache)
 v3=module(Path(code)/'optimize_v3.py','v3_metrics');release=root/'releases/Segmentation_M4_Guided_Recovery_v2';policy=read(release/'refinement_policy.json');assert sha(release/'predict_refined.py')==read(release/'artifact_checksums.json')['predict_refined.py'];helper=module(release/'predict_refined.py','v2_helper')
 labels=sorted((cache/'dataset/labels/val_select').glob('*.txt'));assert len(labels)==40
 dest=run/'comparison';dest.mkdir(exist_ok=True)
 if (dest/'selection.json').exists():print('Selection already frozen:',dest/'selection.json');return
 histories=[];models=[('v3',root/'phase2_results/optimization_v3/segmentation/weights/best.pt'),('v4_best',run/'segmenter/weights/best.pt'),('v4_last',run/'segmenter/weights/last.pt')]
 for name,path in models:
  m=YOLO(str(path))
  for confidence in (.05,.1,.2,.35,.5):
   metric,records=v3.evaluate_masks(m,labels,cache,confidence,960);histories.append(dict(model=name,path=str(path),conf=confidence,**metric));v3.write_csv(dest/f'{name}_{confidence}.csv',records);print('Raw masks:',name,confidence,metric,flush=True)
 v3.write_csv(dest/'raw_mask_comparison.csv',histories);selected=max(histories,key=lambda x:x['mean_iou']);save(dest/'raw_selection.json',selected)
 # Measure the old complete deployed mask policy again on the exact same original references.
 assert sha(release/policy['detector_file'])==policy['detector_sha256'] and sha(release/policy['model_file'])==policy['model_sha256']
 detector=YOLO(str(release/policy['detector_file']));old_segmenter=YOLO(str(release/policy['model_file']));sam_b=release/policy['sam_file']
 if not sam_b.exists():sam_b=root/'releases/Segmentation_M4_Guided'/policy['sam_file']
 assert sha(sam_b)==policy['sam_sha256'];sam_old=SAM(str(sam_b))
 sam_t=Path(cache)/'sam2.1_t.pt';model_sha='3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3'
 if not sam_t.exists():
  import urllib.request
  urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt',sam_t)
 assert sha(sam_t)==model_sha;sam_new=SAM(str(sam_t));learned=YOLO(selected['path']);pipelines=[]
 configurations=[('v2_baseline',old_segmenter,sam_old,dict(policy))]
 for size in (640,1024):
  for quality in (0.,.75):
   p=dict(policy);p.update(sam_input=size,segmenter_input=960,sam_quality=quality);configurations.append((f'v4_sam2_{size}_q{quality}',learned,sam_new,p))
 for name,segmenter,sam,p in configurations:
  records=[];predictions=[]
  for i,label in enumerate(labels,1):
   r=rows[label.stem];path=cache/'images'/(r['id']+'.jpg');start=time.perf_counter();prediction=helper.predict_image(detector,segmenter,sam,path,p,0);seconds=time.perf_counter()-start
   gtpolys=[np.array(list(map(float,s.split()[1:]))).reshape(-1,2) for s in label.read_text().splitlines() if s.strip()];gt=v3.raster(gtpolys,r['width'],r['height']);pred=v3.raster([o['polygon_normalized'] for o in prediction['objects']],r['width'],r['height']);intersection=(gt&pred).sum();union=(gt|pred).sum();den=gt.sum()+pred.sum()
   records.append(dict(id=r['id'],iou=float(intersection/union) if union else 1.,dice=float(2*intersection/den) if den else 1.,seconds=seconds));predictions.append(prediction)
   if i%10==0:print(name,i,'/40',flush=True)
  score=dict(name=name,mean_iou=float(np.mean([r['iou'] for r in records])),mean_dice=float(np.mean([r['dice'] for r in records])),gpu_image_seconds_mean=float(np.mean([r['seconds'] for r in records])),n=40,scope='Original unchanged validation references. Timing includes all mask-helper operations on GPU, not full CPU species/HTTP pipeline or production benchmark.',policy=p)
  pipelines.append(score);v3.write_csv(dest/(name+'.csv'),records);save(dest/(name+'_predictions.json'),predictions);print(score['name'],score['mean_iou'],flush=True)
 best=max(pipelines,key=lambda x:x['mean_iou']);v3.write_csv(dest/'pipeline_comparison.csv',[{k:v for k,v in r.items() if k!='policy'} for r in pipelines]);save(dest/'selection.json',dict(selected=best,candidates=pipelines,raw_selected=selected,target_met=best['mean_iou']>.9,selection_split='val_select',test_evaluated=False,independent_test=False,training_updates=56,validation_unchanged=True,new_deployment_installed=False))
 print('Saved comparison:',dest/'selection.json',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cache',required=True);p.add_argument('--code',required=True);p.add_argument('--stage',choices=['prepare','train','compare'],required=True);a=p.parse_args()
 if a.stage=='prepare':prepare(a.root,a.cache,Path(a.code)/'Reviewed_Mask_Overrides_V4.zip')
 elif a.stage=='train':train(a.root,a.cache)
 else:compare(a.root,a.cache,a.code)
