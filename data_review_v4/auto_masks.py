"""Automatic annotation drafts from human boxes + SAM; explicit per-image human approval."""
from pathlib import Path
from datetime import datetime,timezone
import base64,io,json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw

def touching_topology_safe(polygons,w,h,workflow):
 """Check the displayed pixel masks; SAM's merged contour paths need not be simple polygons.

 This does not certify polygon topology or anatomical correctness. Explicit human
 approval applies to the unchanged rendered foreground; geometry findings remain recorded.
 """
 for points in polygons:
  a=np.asarray(points,float)
  if a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all() or (a<0).any() or (a>1).any():return False
  if len(np.unique(a,axis=0))<3 or not workflow.raster([points],w,h).any():return False
 return True

def refresh_cached_topology(path,item,workflow):
 path=Path(path);record=json.loads(path.read_text());polygons=record['polygons']
 if record.get('mask_count')!=record.get('expected_animals') or not polygons:return record
 findings=workflow.geometric_findings(polygons)
 if not findings or not touching_topology_safe(polygons,item['width'],item['height'],workflow):return record
 if record.get('raster_topology_validated'):return record
 # Retain the exact mask foreground; record a distinct reviewable topology exception.
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 workflow.save(path.parent/'draft_history'/record['id']/(stamp+'.json'),record)
 record['raster_topology_validated']=True;record['topology_review_required']=True;record['geometry_findings']=findings;record['approval_enabled']=True
 record['warnings']=[x for x in record['warnings'] if not x.startswith('Unresolved geometry for animal')]
 record['warnings'].append('SAM merged-contour topology is non-simple; the displayed pixel mask is unchanged. Approve accepts the visible animal boundaries after your inspection; geometry findings remain recorded.')
 workflow.save(path,record);return record

def save_approved_draft(review,item,proposal,workflow,name,path):
 data=dict(reviewer=name,observation='Explicitly approved automatic SAM proposal after visual comparison with the image.',decision='verified',polygons=proposal['polygons'],complete_animals_confirmed=True,method='SAM_human_boxes_explicit_human_approval',topology_exception_confirmed=False,automatic_proposal=proposal,automatic_proposal_config=json.loads((review/'auto_proposals/config.json').read_text()),automatic_proposal_sha256=workflow.sha(path))
 if not workflow.geometric_findings(proposal['polygons']):return workflow.save_review(review,item,data)
 assert proposal.get('raster_topology_validated') and touching_topology_safe(proposal['polygons'],item['width'],item['height'],workflow),'Empty/out-of-bounds or non-finite mask remains; reject this draft'
 assert workflow.sha(item['source_label'])==item['label_sha256'] and name.strip()
 text='\n'.join('0 '+' '.join(f'{v:.17g}' for point in polygon for v in point) for polygon in proposal['polygons'])+'\n'
 assert np.array_equal(workflow.raster(workflow.parse_labels(text),item['width'],item['height']),workflow.raster(proposal['polygons'],item['width'],item['height']))
 data.update(topology_exception_confirmed=True,raster_topology_human_approved=True,observation='Explicitly approved automatic SAM mask including its disclosed touching/retraced raster topology after visual inspection.',id=item['id'],split=item['split'],source_label_sha256=item['label_sha256'],image_pixel_sha256=item['pixel_sha256'],timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
 dest=review/'labels'/item['split']/(item['id']+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(text);tmp.replace(dest)
 workflow.save(review/'history'/item['id']/(data['timestamp']+'.json'),data);workflow.save(review/'reviews'/(item['id']+'.json'),data);return data

def generate(root,review,workflow,limit=None,only_id=None,only_ids=None):
 import torch
 from ultralytics import SAM
 root,review=Path(root),Path(review);rows=workflow.load_manifest(root/'phase1_source/data/cct20_team_v2/manifest.json');byid={r['id']:r for r in rows}
 policy=json.loads((root/'releases/Segmentation_M4_Guided_Recovery_v2/refinement_policy.json').read_text())
 sam_path=root/'releases/Segmentation_M4_Guided_Recovery_v2'/policy['sam_file']
 if not sam_path.exists():
  alternate=root/'releases/Segmentation_M4_Guided'/policy['sam_file']
  assert alternate.exists(),'Install the SAM weight from the existing project release first.';sam_path=alternate
 assert workflow.sha(sam_path)==policy['sam_sha256'],'SAM weight differs'
 device=0 if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
 print('Automatic mask proposals | device:',device,'| input: 640 | no training',flush=True)
 global _SAM_CACHE
 key=(str(sam_path),str(device))
 if globals().get('_SAM_CACHE',{}).get('key')==key:sam=_SAM_CACHE['model']
 else:
  sam=SAM(str(sam_path));_SAM_CACHE=dict(key=key,model=sam)
 queue=list(__import__('csv').DictReader((review/'queue.csv').open()))
 if only_id is not None:
  queue=[q for q in queue if q['id']==only_id];assert len(queue)==1
 if only_ids is not None:
  ids=set(only_ids);queue=[q for q in queue if q['id'] in ids];assert {q['id'] for q in queue}==ids
 dest=review/'auto_proposals';dest.mkdir(exist_ok=True)
 config=dict(version='sam_human_box_drafts_v1',sam_sha256=policy['sam_sha256'],imgsz=640,prompts='Human reference boxes and their center points; not detector predictions or prior-mask IoU selection',test_used=False)
 if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==config
 else:workflow.save(dest/'config.json',config)
 if limit is not None:queue=queue[:limit]
 start=time.perf_counter();created=0
 for i,q in enumerate(queue,1):
  item=json.loads((review/'items'/(q['id']+'.json')).read_text());r=byid[q['id']];assert r['split'] in ('train','val_select')
  target=dest/(q['id']+'.json')
  if target.exists():
   old=json.loads(target.read_text());assert old['source_label_sha256']==item['label_sha256'] and old['config_sha256']==workflow.sha(dest/'config.json');print('Cached',i,'/',len(queue),flush=True);continue
  print('Generating',i,'/',len(queue),q['id'],flush=True)
  with Image.open(item['image']) as im:
   im=im.convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==item['pixel_sha256'];w,h=im.size
  boxes=np.array([b['xyxy'] for b in r['boxes']],float)*[w,h,w,h];assert len(boxes)>0
  points=[[[float((b[0]+b[2])/2),float((b[1]+b[3])/2)]] for b in boxes]
  result=sam.predict(item['image'],bboxes=boxes.tolist(),points=points,labels=[[1] for _ in points],imgsz=640,conf=0.,device=device,verbose=False)[0]
  selected={};warnings=[]
  if result.masks is not None:
   ids=result.boxes.cls.cpu().numpy().astype(int).tolist();scores=result.boxes.conf.cpu().numpy().tolist()
   for polygon,score,index in zip(result.masks.xyn,scores,ids):
    if 0<=index<len(boxes) and (index not in selected or score>selected[index][0]):selected[index]=(float(score),polygon.tolist())
  polygons=[];qualities=[];proposal_complete=True
  for index in range(len(boxes)):
   if index not in selected:warnings.append(f'No SAM mask for animal {index+1}');proposal_complete=False;continue
   score,p=selected[index]
   if workflow.geometric_findings([p]):
    repaired,_=workflow.proposal([p],w,h)
    if repaired:p=repaired[0]
   if workflow.geometric_findings([p]):warnings.append(f'Unresolved geometry for animal {index+1}');proposal_complete=False
   polygons.append(p);qualities.append(score)
   mask=workflow.raster([p],w,h)
   if mask.mean()>.8:warnings.append(f'Animal {index+1} mask covers over 80% of image; inspect carefully')
   if score<.75:warnings.append(f'Low predicted SAM quality for animal {index+1}; inspect carefully')
  original=workflow.raster(item['original'],w,h);proposed=workflow.raster(polygons,w,h) if polygons else np.zeros((h,w),np.uint8);union=np.logical_or(original,proposed).sum();overlap=float(np.logical_and(original,proposed).sum()/union) if union else 1.
  if overlap<.5:warnings.append('Large change from old reference; review actual animal boundaries')
  record=dict(id=q['id'],split=r['split'],polygons=polygons,qualities=qualities,expected_animals=len(boxes),mask_count=len(polygons),approval_enabled=proposal_complete and len(polygons)==len(boxes),warnings=warnings,old_new_raster_iou=overlap,original_area=float(original.mean()),proposed_area=float(proposed.mean()),source_label_sha256=item['label_sha256'],pixel_sha256=item['pixel_sha256'],config_sha256=workflow.sha(dest/'config.json'),status='draft_requires_human_approval',generation_device=str(device),timestamp=datetime.now(timezone.utc).isoformat())
  workflow.save(target,record);created+=1;elapsed=time.perf_counter()-start
  print('Saved draft',i,'/',len(queue),'| elapsed',round(elapsed/60,1),'min | approximate remaining',round(elapsed/max(created,1)*(len(queue)-i)/60,1),'min',flush=True)
 print('Automatic proposal generation complete. Drafts require Approve/Reject.',flush=True);return dest

def approve_next(review,workflow):
 from google.colab.output import eval_js
 from IPython.display import display,Javascript
 review=Path(review);queue=list(__import__('csv').DictReader((review/'queue.csv').open()));chosen=None
 for q in queue:
  record=review/'auto_approvals'/(q['id']+'.json')
  if not record.exists():chosen=q;break
 if chosen is None:print('All available queue images have an approval decision.');return
 proposal_path=review/'auto_proposals'/(chosen['id']+'.json');assert proposal_path.exists(),'Generate automatic proposals first (step 3).'
 item=json.loads((review/'items'/(chosen['id']+'.json')).read_text());proposal=refresh_cached_topology(proposal_path,item,workflow);assert proposal['source_label_sha256']==item['label_sha256']
 assert proposal['config_sha256']==workflow.sha(review/'auto_proposals/config.json')
 print('Opening',chosen['id'],'— compare the images and click Approve or Reject.',flush=True)
 with Image.open(item['image']) as original:
  original=original.convert('RGB');assert hashlib.sha256(original.tobytes()).hexdigest()==item['pixel_sha256']
  images=[]
  for title,polygons,color in [('CURRENT REFERENCE',item['original'],(255,120,0,95)),('AUTOMATIC PROPOSAL',proposal['polygons'],(0,200,70,95))]:
   im=original.convert('RGBA');overlay=Image.new('RGBA',im.size);draw=ImageDraw.Draw(overlay)
   for p in polygons:draw.polygon([(x*im.width,y*im.height) for x,y in p],fill=color,outline=color[:3]+(255,))
   im=Image.alpha_composite(im,overlay).convert('RGB');im.thumbnail((600,520));panel=Image.new('RGB',(600,560),'white');panel.paste(im,(0,32));ImageDraw.Draw(panel).text((10,10),title,fill='black');images.append(panel)
  comparison=Image.new('RGB',(1200,560),'white');comparison.paste(images[0],(0,0));comparison.paste(images[1],(600,0));buf=io.BytesIO();comparison.save(buf,format='PNG')
 name=(review/'reviewer_name.txt').read_text() if (review/'reviewer_name.txt').exists() else ''
 payload=dict(id=item['id'],split=item['split'],reviewer=name,image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),enabled=proposal['approval_enabled'],warnings=proposal['warnings'],mask_count=proposal['mask_count'],expected=proposal['expected_animals'])
 js=Path(__file__).with_name('approve_masks.js').read_text();display(Javascript('window.approveAutoMask = '+js+';'));result=eval_js('window.approveAutoMask('+json.dumps(payload)+')')
 if result is None:print('Closed without a decision.');return
 (review/'reviewer_name.txt').write_text(result['reviewer'])
 if result['decision']=='approve':
  assert proposal['approval_enabled'],'Incomplete or invalid draft cannot be approved'
  save_approved_draft(review,item,proposal,workflow,result['reviewer'],proposal_path)
 else:
  data=dict(reviewer=result['reviewer'],observation='Explicitly rejected automatic SAM proposal after visual comparison.',decision='needs_correction',polygons=item['original'],complete_animals_confirmed=False,method='SAM_human_boxes_human_rejected',automatic_proposal_sha256=workflow.sha(proposal_path));workflow.save_review(review,item,data)
 workflow.save(review/'auto_approvals'/(item['id']+'.json'),dict(**result,id=item['id'],proposal_sha256=workflow.sha(proposal_path),timestamp=datetime.now(timezone.utc).isoformat()));print('Saved:',result['decision'],item['id']);return result

def remaining_geometry(review):
 import csv
 review=Path(review);queue=list(csv.DictReader((review/'queue.csv').open()));result=[]
 for r in queue:
  if int(r['geometry_findings'])<=0 or (review/'auto_approvals'/(r['id']+'.json')).exists():continue
  prior=review/'reviews'/(r['id']+'.json')
  if prior.exists() and json.loads(prior.read_text()).get('decision')=='verified':continue
  result.append(r)
 return result

def approve_geometry_batch(root,review,workflow):
 from IPython.display import display,Javascript
 from google.colab.output import eval_js
 review=Path(review);queue=remaining_geometry(review)
 if not queue:print('All geometry-queue images already have saved decisions.');return
 pending=[r['id'] for r in queue if not (review/'auto_proposals'/(r['id']+'.json')).exists()]
 print('Remaining geometry-review images:',len(queue),'| drafts to generate:',len(pending),flush=True)
 if pending:generate(root,review,workflow,only_ids=pending)
 cards=[];evidence={}
 for i,r in enumerate(queue,1):
  ident=r['id'];item=json.loads((review/'items'/(ident+'.json')).read_text());p=review/'auto_proposals'/(ident+'.json');proposal=refresh_cached_topology(p,item,workflow)
  assert proposal['source_label_sha256']==item['label_sha256'] and proposal['config_sha256']==workflow.sha(review/'auto_proposals/config.json')
  with Image.open(item['image']) as im:
   im=im.convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==item['pixel_sha256']
   comparison=Image.new('RGB',(1200,530),'white')
   for x,title,polygons,color in [(0,'CURRENT / ORANGE',item['original'],(255,120,0,95)),(600,'PROPOSAL / GREEN',proposal['polygons'],(0,200,70,95))]:
    base=im.convert('RGBA');layer=Image.new('RGBA',base.size);draw=ImageDraw.Draw(layer)
    for poly in polygons:draw.polygon([(a*base.width,b*base.height) for a,b in poly],fill=color,outline=color[:3]+(255,))
    view=Image.alpha_composite(base,layer).convert('RGB');view.thumbnail((600,495));comparison.paste(view,(x,30));ImageDraw.Draw(comparison).text((x+10,10),title,fill='black')
   buffer=io.BytesIO();comparison.save(buffer,format='JPEG',quality=75)
  cards.append(dict(id=ident,index=i,split=r['split'],image='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode(),enabled=proposal['approval_enabled'],warnings=proposal['warnings'],masks=proposal['mask_count'],expected=proposal['expected_animals']))
  evidence[ident]=dict(item=item,path=p,sha=workflow.sha(p),proposal=proposal)
  if i%10==0:print('Gallery prepared:',i,'/',len(queue),flush=True)
 reviewer=(review/'reviewer_name.txt').read_text() if (review/'reviewer_name.txt').exists() else ''
 print('Gallery ready. Review all comparisons, choose decisions, then use the batch approval button.',flush=True)
 script=Path(__file__).with_name('approve_batch.js').read_text();display(Javascript('window.approveMaskBatch = '+script+';'))
 result=eval_js('window.approveMaskBatch('+json.dumps(dict(cards=cards,reviewer=reviewer))+')')
 if result is None:print('Closed without saving batch decisions.');return
 assert result['reviewer'].strip() and result.get('inspection_confirmed') is True
 chosen=result['decisions'];assert chosen and len({r['id'] for r in chosen})==len(chosen)
 # Validate the complete submitted decision list before writing any approval.
 for r in chosen:
  assert r['id'] in evidence and r['decision'] in ('approve','reject');e=evidence[r['id']]
  assert workflow.sha(e['path'])==e['sha'] and workflow.sha(e['item']['source_label'])==e['item']['label_sha256']
  if r['decision']=='approve':assert e['proposal']['approval_enabled']
 workflow.save(review/'batch_intents'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),result)
 for i,r in enumerate(chosen,1):
  e=evidence[r['id']];item=e['item'];proposal=e['proposal']
  if r['decision']=='approve':save_approved_draft(review,item,proposal,workflow,result['reviewer'],e['path'])
  else:workflow.save_review(review,item,dict(reviewer=result['reviewer'],observation='Explicitly rejected in automatic-proposal gallery batch review.',decision='needs_correction',polygons=item['original'],complete_animals_confirmed=False,method='SAM_human_boxes_batch_human_rejected',automatic_proposal_sha256=e['sha']))
  workflow.save(review/'auto_approvals'/(r['id']+'.json'),dict(id=r['id'],decision=r['decision'],reviewer=result['reviewer'],proposal_sha256=e['sha'],batch_review=True,timestamp=datetime.now(timezone.utc).isoformat()))
  print('Saved',i,'/',len(chosen),r['decision'],r['id'],flush=True)
 (review/'reviewer_name.txt').write_text(result['reviewer'])
 print('Batch saved. Approved:',sum(r['decision']=='approve' for r in chosen),'| rejected:',sum(r['decision']=='reject' for r in chosen),'| left pending:',len(cards)-len(chosen))
 return result
