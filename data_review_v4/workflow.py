"""Immutable source audit and human-reviewed data corrections; no model predictions as labels."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,hashlib,io,json,math,zipfile
import numpy as np
from PIL import Image,ImageDraw
MANIFEST_SHA='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def csv_write(p,rows):
 if not rows:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_manifest(path):
 assert sha(path)==MANIFEST_SHA,'Source manifest changed'
 rows=json.loads(Path(path).read_text());assert len(rows)==6037
 for key in ('camera','event','pixel_sha256'):
  groups=defaultdict(set)
  for r in rows:groups[r[key]].add(r['split'])
  assert all(len(v)==1 for v in groups.values()),key
 return rows

def inventory(manifest,classes,out):
 rows=load_manifest(manifest);out=Path(out);classes=list(classes);stats=[];by_camera=[]
 for species in classes:
  result=dict(species=species)
  for split in ('train','val_select','val_calibrate','test'):
   rr=[r for r in rows if r['split']==split and any(b['species']==species for b in r['boxes'])]
   result[split+'_crops']=sum(b['species']==species for r in rr for b in r['boxes']);result[split+'_events']=len({r['event'] for r in rr});result[split+'_cameras']=len({r['camera'] for r in rr})
  result['train_additional_to_planning_floor_100']=max(0,100-result['train_crops']);result['validation_coverage_required']=result['val_select_crops']==0;stats.append(result)
  for split,camera in sorted({(r['split'],r['camera']) for r in rows}):
   rr=[r for r in rows if r['split']==split and r['camera']==camera]
   n=sum(b['species']==species for r in rr for b in r['boxes'])
   if n:by_camera.append(dict(species=species,split=split,camera=camera,crops=n,events=len({r['event'] for r in rr if any(b['species']==species for b in r['boxes'])})))
 csv_write(out/'species_coverage.csv',stats);csv_write(out/'species_by_camera.csv',by_camera)
 missing=[r['species'] for r in stats if r['validation_coverage_required']]
 summary=dict(manifest_sha256=MANIFEST_SHA,frames=len(rows),classes=classes,missing_validation_classes=missing,all_class_validation_macro_f1_ceiling=(len(classes)-len(missing))/len(classes),planning_floor=100,planning_floor_scope='Planning target, not an adequacy guarantee or training requirement.',scope='Counts are human-box known-class crop counts. No split reassignment or new collection has occurred.')
 save(out/'coverage_summary.json',summary);return stats,summary

def parse_labels(text):
 polygons=[]
 for line in text.splitlines():
  if not line.strip():continue
  v=list(map(float,line.split()));assert len(v)>=7 and len(v)%2==1 and v[0]==0
  assert all(math.isfinite(x) and 0<=x<=1 for x in v[1:]);polygons.append(np.array(v[1:]).reshape(-1,2).tolist())
 return polygons

def raster(polys,w,h):
 im=Image.new('1',(w,h));d=ImageDraw.Draw(im)
 for poly in polys:d.polygon([(x*w,y*h) for x,y in poly],fill=1)
 return np.asarray(im,dtype=np.uint8)

def geometric_findings(polys):
 from shapely.geometry import Polygon
 from shapely.validation import explain_validity
 return [dict(polygon=i,reason=explain_validity(Polygon(p))) for i,p in enumerate(polys) if not Polygon(p).is_valid or Polygon(p).area<=0]

def proposal(polys,w,h):
 import cv2
 from shapely.geometry import Polygon
 result=[];changes=0
 for poly in polys:
  if Polygon(poly).is_valid and Polygon(poly).area>0:result.append(poly);continue
  original=raster([poly],w,h)
  repaired=Polygon(poly).buffer(0)
  if repaired.geom_type=='Polygon' and repaired.is_valid and len(repaired.interiors)==0 and repaired.area>0:
   candidate=np.asarray(repaired.exterior.coords)[:-1].tolist()
   proposed_mask=raster([candidate],w,h);union=np.logical_or(original,proposed_mask).sum();overlap=float(np.logical_and(original,proposed_mask).sum()/union) if union else 1.
   if overlap>=.995:
    result.append(candidate);changes+=1;continue
  contours,_=cv2.findContours(original,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  if len(contours)!=1:return None,None
  found=None
  for epsilon in (.25,.5,1.):
   points=cv2.approxPolyDP(contours[0],epsilon,True).reshape(-1,2).astype(float);points[:,0]/=w;points[:,1]/=h;p=points.tolist()
   if len(p)<3 or not Polygon(p).is_valid:continue
   mask=raster([p],w,h);union=np.logical_or(original,mask).sum();iou=float(np.logical_and(original,mask).sum()/union) if union else 1.
   if iou>=.995:found=p;break
  if found is None:return None,None
  result.append(found);changes+=1
 if not changes:return None,None
 a=raster(polys,w,h);b=raster(result,w,h);union=np.logical_or(a,b).sum();iou=float(np.logical_and(a,b).sum()/union) if union else 1.
 return result,iou

def initialize(root,manifest=None):
 root=Path(root);manifest=Path(manifest or root/'phase1_source/data/cct20_team_v2/manifest.json');rows=load_manifest(manifest);byid={r['id']:r for r in rows}
 dest=root/'data_review_v4';dest.mkdir(exist_ok=True)
 classes=json.loads((root/'phase1_source/data/cct20_team_v2/species.json').read_text());inventory(manifest,classes,dest/'inventory')
 release=root/'releases/Segmentation_M4_Guided_Recovery_v2';derived={r['id']:r for r in json.loads((release/'derived_training_manifest.json').read_text())};queue=[]
 previous=json.loads((dest/'source_snapshot.json').read_text()) if (dest/'source_snapshot.json').exists() else None;snapshot={}
 for split,expected in [('train',150),('val_select',40)]:
  labels=sorted((root/'segmentation/labels'/split).glob('*.txt'));assert len(labels)==expected
  for label in labels:
   r=byid[label.stem];assert r['split']==split
   lines=label.read_text().splitlines();source_sha=sha(label)
   if split=='train':
    source=derived[r['id']];assert source['source_label_sha256']==source_sha
    lines=[line for i,line in enumerate(lines) if i not in set(source['removed_indices'])]
    assert hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()==source['derived_label_sha256']
   polys=parse_labels('\n'.join(lines));assert polys;findings=geometric_findings(polys);proposed,iou=proposal(polys,r['width'],r['height'])
   source_record=dict(id=r['id'],split=split,label_sha256=source_sha,pixel_sha256=r['pixel_sha256']);snapshot[r['id']]=source_record
   item=dict(**source_record,source_label=str(label),image=str(root/'images'/(r['id']+'.jpg')),width=r['width'],height=r['height'],original=polys,proposal=proposed,proposal_raster_iou=iou,findings=findings,review_required=True)
   target=dest/'items'/(r['id']+'.json')
   if target.exists():assert json.loads(target.read_text())==item,'Source changed; start a new review version'
   else:save(target,item)
   queue.append(dict(id=r['id'],split=split,geometry_findings=len(findings),proposal_available=proposed is not None,proposal_raster_iou=iou))
 if previous is not None:assert previous==snapshot,'Source snapshot changed'
 else:save(dest/'source_snapshot.json',snapshot)
 queue.sort(key=lambda r:(not bool(r['geometry_findings']),r['split']!='train',r['id']));csv_write(dest/'queue.csv',queue)
 save(dest/'review_protocol.json',dict(version='data_review_v4',manifest_sha256=MANIFEST_SHA,images=190,order='Geometry findings first; then all remaining train/validation references. No model-output ranking.',reference_scope='New reviewed reference version; original sources untouched. Validation edits cannot be used to claim an untouched independent test.',proposal='Geometry/raster-contour suggestions preserve raster IoU >= .995; suggestions require explicit named human approval and are not verified ground truth by default.',test_images_used=False))
 print('Queue ready:',len(queue),'images;',sum(bool(r['geometry_findings']) for r in queue),'with geometry findings;',sum(r['proposal_available'] for r in queue),'have conservative suggestions.');return dest

def save_review(dest,item,result):
 dest=Path(dest);assert sha(item['source_label'])==item['label_sha256'],'Original source changed during review'
 assert result['reviewer'].strip() and result['observation'].strip()
 assert result['decision'] in ('verified','needs_correction')
 polys=result['polygons'];assert polys and all(len(p)>=3 for p in polys)
 for p in polys:assert all(len(x)==2 and all(math.isfinite(v) and 0<=v<=1 for v in x) for x in p)
 if result['decision']=='verified':
  findings=geometric_findings(polys)
  if findings:
   assert result.get('topology_exception_confirmed') is True and polys==item['original'] and len(result['observation'].strip())>=24,'Geometry finding: edit, defer, or explicitly justify retaining the unchanged visually correct reference topology'
  assert result.get('complete_animals_confirmed') is True
 text=None
 if result['decision']=='verified':
  text='\n'.join('0 '+' '.join(f'{v:.17g}' for point in p for v in point) for p in polys)+'\n'
  parsed=parse_labels(text)
  assert not geometric_findings(parsed) or result.get('topology_exception_confirmed') is True,'Export geometry differs; adjust vertices'
  if result.get('topology_exception_confirmed'):
   assert np.array_equal(raster(parsed,item['width'],item['height']),raster(item['original'],item['width'],item['height'])),'Retained topology must preserve original foreground exactly'
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');record=dict(**result,id=item['id'],split=item['split'],source_label_sha256=item['label_sha256'],image_pixel_sha256=item['pixel_sha256'],timestamp=stamp)
 save(dest/'history'/item['id']/(stamp+'.json'),record);save(dest/'reviews'/(item['id']+'.json'),record)
 if result['decision']=='verified':
  path=dest/'labels'/item['split']/(item['id']+'.txt');path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(text);tmp.replace(path)
 return record

def review_next(dest):
 from google.colab.output import eval_js
 import base64
 dest=Path(dest);queue=list(csv.DictReader((dest/'queue.csv').open()));selected=None
 for r in queue:
  path=dest/'reviews'/(r['id']+'.json')
  if not path.exists():selected=r;break
 if selected is None:print('First pass complete. For outstanding corrections run review_flagged().');return
 return open_item(dest,selected['id'],eval_js)

def review_flagged(dest):
 from google.colab.output import eval_js
 dest=Path(dest)
 for path in sorted((dest/'reviews').glob('*.json')):
  if json.loads(path.read_text())['decision']=='needs_correction':return open_item(dest,path.stem,eval_js)
 print('No flagged reviews remain.')

def open_item(dest,ident,eval_js):
 import base64
 dest=Path(dest);item=json.loads((dest/'items'/(ident+'.json')).read_text());image=Path(item['image']);assert image.exists(),str(image)
 with Image.open(image) as im:
  im=im.convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==item['pixel_sha256'];buf=io.BytesIO();im.save(buf,format='PNG')
 prior=dest/'reviews'/(ident+'.json');draft=dest/'drafts'/(ident+'.json');review=json.loads(draft.read_text()) if draft.exists() else json.loads(prior.read_text()) if prior.exists() else None
 payload=dict(id=ident,split=item['split'],reference=item['original'],image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),polygons=review['polygons'] if review else item['original'],proposal=item['proposal'],proposal_iou=item['proposal_raster_iou'],reviewer=review['reviewer'] if review else (dest/'reviewer_name.txt').read_text() if (dest/'reviewer_name.txt').exists() else '',findings=item['findings'])
 html=Path(__file__).with_name('mask_editor.js').read_text();result=eval_js('('+html+')('+json.dumps(payload)+')')
 if result is None:print('Closed without saving.');return
 try:record=save_review(dest,item,result)
 except AssertionError as error:
  save(draft,dict(**result,status='draft_not_verified'));print('Draft preserved; edit this image again:',str(error));return None
 if draft.exists():draft.unlink()
 (dest/'reviewer_name.txt').write_text(result['reviewer'].strip())
 print('Saved:',record['id'],record['decision']);return record

def export_review(dest):
 dest=Path(dest);queue=list(csv.DictReader((dest/'queue.csv').open()));reviews=[json.loads(p.read_text()) for p in (dest/'reviews').glob('*.json')];verified={r['id']:r for r in reviews if r['decision']=='verified'}
 missing=[r['id'] for r in queue if r['id'] not in verified];receipt=dict(images=len(queue),verified=len(verified),requires_review=len(missing),verified_label_sha256={},source_manifest_sha256=MANIFEST_SHA,original_sources_changed=False,reviewed_unchanged_topology_exceptions=[])
 for ident,r in verified.items():
  path=dest/'labels'/r['split']/(ident+'.txt');assert path.is_file()
  findings=geometric_findings(parse_labels(path.read_text()))
  if findings:
   assert r.get('topology_exception_confirmed') is True;receipt['reviewed_unchanged_topology_exceptions'].append(ident)
  receipt['verified_label_sha256'][ident]=sha(path)
 save(dest/'review_status.json',receipt)
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');archive=dest/f'Mask_Review_V4_{stamp}.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for folder in ('items','reviews','history','labels','inventory','drafts'):
   for p in sorted((dest/folder).rglob('*')):
    if p.is_file():z.write(p,p.relative_to(dest))
  for name in ('queue.csv','source_snapshot.json','review_protocol.json','review_status.json'):z.write(dest/name,name)
  assert z.testzip() is None
 print('Saved:',archive,'| Verified:',len(verified),'/',len(queue));return archive
