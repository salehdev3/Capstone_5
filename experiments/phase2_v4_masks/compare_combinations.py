"""Nine fixed validation configurations; no training and no human-box inference prompts."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,importlib.util,time,csv
import numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def render(polygons,w,h):
 im=Image.new('1',(w,h));d=ImageDraw.Draw(im)
 for p in polygons:
  if len(p)>=3:d.polygon([(x*w,y*h) for x,y in p],fill=1)
 return np.asarray(im,dtype=bool)
def run(root,cache):
 import torch
 from ultralytics import YOLO,SAM
 root,cache=Path(root),Path(cache);assert torch.cuda.is_available(),'Connect GPU for this comparison; no training is needed'
 release=root/'releases/Segmentation_M4_Guided_Recovery_v2';policy=read(release/'refinement_policy.json');assert sha(release/'predict_refined.py')==read(release/'artifact_checksums.json')['predict_refined.py'];helper=module(release/'predict_refined.py','unchanged_v2_mask_helper')
 manifest=root/'phase1_source/data/cct20_team_v2/manifest.json';assert sha(manifest)=='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53';byid={r['id']:r for r in read(manifest)}
 labels=sorted((cache/'dataset/labels/val_select').glob('*.txt'));assert len(labels)==40,'Run the V4 prepare cell again to restore the local cache'
 for p in labels:assert sha(p)==policy['reference_label_sha256'][p.stem]
 specs={
  'd_old':(release/policy['detector_file'],policy['detector_sha256'],640),
  'd_new':(root/'phase2_results/optimization_v3/detection/weights/best.pt','d9403728c5b53ca0355f369e0ffcf55d6f3f13e9f711d48d59215a05a86d1601',960),
  'm_old':(release/policy['model_file'],policy['model_sha256'],640),
  'm_v3':(root/'phase2_results/optimization_v3/segmentation/weights/best.pt','60dfc13188fec981424105452db51e44a00ca5fd3530b8d6a2397653d460b4b2',960),
  'm_v4':(root/'phase2_results/mask_training_v4/segmenter/weights/best.pt','bb89f9a6ec1b19a48c60de266f0cf08171ca5798bdd34e75996f1b0eed3eaaa0',960)}
 old=release/policy['sam_file']
 if not old.exists():old=root/'releases/Segmentation_M4_Guided'/policy['sam_file']
 specs['sam_b']=(old,policy['sam_sha256'],640)
 tiny=cache/'sam2.1_t.pt'
 if not tiny.exists():
  import urllib.request
  urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt',tiny)
 specs['sam2_t']=(tiny,'3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3',1024)
 for key,(p,expected,size) in specs.items():assert p.is_file() and sha(p)==expected,key+' checkpoint differs'
 models={k:(SAM(str(p)) if k.startswith('sam') else YOLO(str(p))) for k,(p,_,_) in specs.items()}
 configurations=[('baseline','d_old','m_old','sam_b')]+[(d+'_'+m+'_'+s,d,m,s) for d in ['d_old','d_new'] for m in ['m_v3','m_v4'] for s in ['sam_b','sam2_t']]
 dest=root/'phase2_results/mask_training_v4/combinations';dest.mkdir(exist_ok=True)
 plan=dict(configurations=configurations,model_hashes={k:v[1] for k,v in specs.items()},reference_sha256={p.stem:sha(p) for p in labels},scope='Fixed validation architecture comparison only; no training, test rerun, GT-box prompts, acceptance decision or deployment.',sam_quality={'sam_b':.75,'sam2_t':0.},detector_confidence=.045,primary_mask_detector_confidence=.4)
 if (dest/'plan.json').exists():assert read(dest/'plan.json')==json.loads(json.dumps(plan))
 else:save(dest/'plan.json',plan)
 scores=[]
 for name,d,m,s in configurations:
  identity=dict(detector_sha256=specs[d][1],segmenter_sha256=specs[m][1],sam_sha256=specs[s][1],detector_input=specs[d][2],segmenter_input=specs[m][2],sam_input=specs[s][2])
  score_file=dest/(name+'_metrics.json')
  if score_file.exists():
   score=read(score_file);assert score['identity']==identity and score['plan_sha256']==sha(dest/'plan.json');scores.append(score);print('Cached',name,score['mean_iou'],flush=True);continue
  p=dict(policy);p.pop('validation',None);p.update(identity);p.update(version='mask_combinations_development_v1',sam_url='https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt' if s=='sam2_t' else policy['sam_url'],sam_quality=.75 if s=='sam_b' else 0.,detector_file=str(specs[d][0]),model_file=str(specs[m][0]),model_sha256=specs[m][1],sam_file=str(specs[s][0]))
  records=[];predictions=[]
  for i,label in enumerate(labels,1):
   r=byid[label.stem];path=cache/'images'/(r['id']+'.jpg')
   with Image.open(path) as im:assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==r['pixel_sha256']
   start=time.perf_counter();prediction=helper.predict_image(models[d],models[m],models[s],path,p,0);seconds=time.perf_counter()-start
   polygons=[np.array(list(map(float,line.split()[1:]))).reshape(-1,2) for line in label.read_text().splitlines() if line.strip()];gt=render(polygons,r['width'],r['height']);pred=render([o['polygon_normalized'] for o in prediction['objects']],r['width'],r['height']);inter=(gt&pred).sum();union=(gt|pred).sum();den=gt.sum()+pred.sum()
   records.append(dict(id=r['id'],iou=float(inter/union) if union else 1.,dice=float(2*inter/den) if den else 1.,seconds=seconds));predictions.append(prediction)
   if i%10==0:print(name,i,'/40',flush=True)
  with (dest/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  score=dict(name=name,n=40,mean_iou=float(np.mean([r['iou'] for r in records])),mean_dice=float(np.mean([r['dice'] for r in records])),gpu_mask_helper_seconds_mean=float(np.mean([r['seconds'] for r in records])),identity=identity,policy=p,plan_sha256=sha(dest/'plan.json'),scope='GPU mask helper only; excludes species/HTTP; validation development evidence, not full CPU p95 or independent test')
  save(dest/(name+'_predictions.json'),predictions);save(score_file,score);scores.append(score);print('Completed',name,score['mean_iou'],flush=True)
 best=max(scores,key=lambda x:x['mean_iou']);save(dest/'selection.json',dict(selected=best,candidates=scores,all_original_validation_references_unchanged=True,target_met=best['mean_iou']>.9,test_evaluated=False,new_deployment_installed=False));print('Selected:',best['name'],'IoU:',best['mean_iou'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cache',default='/content/mask_training_v4_cache');a=p.parse_args();run(a.root,a.cache)
