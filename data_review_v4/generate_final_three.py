"""Generate unapproved SAM2.1-t annotation alternatives for the three rejected references."""
from pathlib import Path
import json,hashlib,importlib.util,time
import numpy as np
import cv2,torch
from ultralytics import SAM

def generate(repo):
 repo=Path(repo);spec=importlib.util.spec_from_file_location('workflow',repo/'data_review_v4/workflow.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
 rows={r['id']:r for r in w.load_manifest(repo/'runs/phase1_recovered/CameraTrap_Phase1_Team_Mixed_v2/data/cct20_team_v2/manifest.json')};ids=['585dac58-23d2-11e8-a6a3-ec086b02610b','59ffbbf9-23d2-11e8-a6a3-ec086b02610b','59e77d63-23d2-11e8-a6a3-ec086b02610b']
 out=repo/'results/phase2/data_review_v4/final_three_drafts';out.mkdir(exist_ok=True)
 model_path=repo/'runs/assistant_segmentation/sam2.1_t.pt';torch.set_num_threads(4);sam=SAM(str(model_path));config=dict(version='reference_drafts_round3',model='SAM2.1-t',model_sha256=w.sha(model_path),model_url='https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt',input_size=1024,prompt='Human box plus five distributed positive points in original reference foreground inside the box',variants=['original_pixels','CLAHE_L_clip2_grid8'],source_scope='Train/validation annotation assistance only; no test or model score; every variant requires human approval',device='cpu',torch=torch.__version__)
 w.save(out/'config.json',config)
 for ident in ids:
  row=rows[ident];item=json.loads((repo/'runs/v4_round2_uploaded/items'/(ident+'.json')).read_text());path=repo/'runs/assistant_segmentation/transfer/dataset/images'/row['split']/(ident+'.jpg');im=cv2.imread(str(path));assert hashlib.sha256(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==item['pixel_sha256'];h,width=im.shape[:2]
  boxes=np.asarray([b['xyxy'] for b in row['boxes']])*[width,h,width,h];fg=w.raster(item['original'],width,h);all_points=[]
  for box in boxes:
   left,top,right,bottom=max(0,int(box[0])),max(0,int(box[1])),min(width,int(np.ceil(box[2]))),min(h,int(np.ceil(box[3])));clip=np.zeros_like(fg);clip[top:bottom,left:right]=fg[top:bottom,left:right];distance=cv2.distanceTransform(clip,cv2.DIST_L2,3)
   ys,xs=np.where(distance>=max(1.,float(distance.max())*.1));coords=np.column_stack([xs,ys]).astype(float)
   if not len(coords):all_points.append([[float((box[0]+box[2])/2),float((box[1]+box[3])/2)]]);continue
   y,x=np.unravel_index(distance.argmax(),distance.shape);chosen=[[float(x),float(y)]]
   for _ in range(min(4,len(coords)-1)):
    d=np.min(np.sum((coords[:,None,:]-np.asarray(chosen)[None,:,:])**2,axis=2),axis=1);chosen.append(coords[int(d.argmax())].tolist())
   all_points.append(chosen)
  lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0]);enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
  for label,photo in [('A',im),('B',enhanced)]:
   target=out/(ident+'_'+label+'.json')
   if target.exists():continue
   print('Generating',ident,label,flush=True);start=time.perf_counter()
   r=sam.predict(photo,bboxes=boxes.tolist(),points=all_points,labels=[[1]*len(p) for p in all_points],imgsz=1024,conf=0.,device='cpu',verbose=False)[0]
   selected={}
   if r.masks is not None:
    for polygon,quality,index in zip(r.masks.xyn,r.boxes.conf.cpu().numpy().tolist(),r.boxes.cls.cpu().numpy().astype(int).tolist()):
     if 0<=index<len(boxes) and (index not in selected or quality>selected[index][0]):selected[index]=(quality,polygon.tolist())
   polys=[selected[j][1] for j in range(len(boxes)) if j in selected]
   record=dict(id=ident,split=row['split'],variant=label,polygons=polys,qualities=[selected[j][0] for j in range(len(boxes)) if j in selected],expected_animals=len(boxes),mask_count=len(polys),approval_enabled=len(polys)==len(boxes) and all(w.raster([p],width,h).any() for p in polys),warnings=['Unapproved SAM2.1-t annotation draft; inspect visible anatomy before approval.'],source_label_sha256=item['label_sha256'],pixel_sha256=item['pixel_sha256'],config_sha256=w.sha(out/'config.json'),status='draft_requires_human_approval',prompt_points=all_points,seconds=time.perf_counter()-start)
   w.save(target,record);print('Draft saved',label,round(record['seconds'],2),'seconds',flush=True)
 w.save(out/'artifact_checksums.json',{p.name:w.sha(p) for p in out.glob('*.json') if p.name!='artifact_checksums.json'})
 return out
if __name__=='__main__':generate(Path(__file__).resolve().parents[1])
