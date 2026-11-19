"""Stage extra training candidates from unused CCT20 events; never alter split files."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,io,json,tarfile,time,urllib.request,urllib.parse
from PIL import Image
META='https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz'
BASES=['https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/','https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/']

def fetch(url,path):
 path=Path(path)
 if path.exists():return
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.part')
 with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as f:
  while chunk:=response.read(1024*1024):f.write(chunk)
 tmp.replace(path)

def plan(root,workflow):
 root=Path(root);dest=root/'data_review_v4/collection';dest.mkdir(parents=True,exist_ok=True)
 manifest=root/'phase1_source/data/cct20_team_v2/manifest.json';rows=workflow.load_manifest(manifest);classes=json.loads((root/'phase1_source/data/cct20_team_v2/species.json').read_text())
 train_counts=Counter(b['species'] for r in rows if r['split']=='train' for b in r['boxes']);weak={c for c in classes if train_counts[c]<100}
 archive=dest/'metadata/eccv_18_annotations.tar.gz';print('Reading public annotation catalogue (no image download yet).',flush=True);fetch(META,archive)
 categories={};images={};annotations={}
 with tarfile.open(archive) as tar:
  for entry in tar.getmembers():
   if not entry.isfile() or not entry.name.endswith('.json'):continue
   data=json.load(tar.extractfile(entry))
   if not all(k in data for k in ('images','annotations','categories')):continue
   for c in data['categories']:
    if c['id'] in categories:assert categories[c['id']]==c['name']
    categories[c['id']]=c['name']
   for image in data['images']:
    if image['id'] in images:assert images[image['id']]==image
    images[image['id']]=image
   for a in data['annotations']:annotations[(a['image_id'],a['category_id'],tuple(a.get('bbox',[])))]=a
 grouped=defaultdict(list)
 for a in annotations.values():grouped[a['image_id']].append(a)
 used_ids={r['id'] for r in rows};used_events={r['event'] for r in rows};camera_split={str(r['camera']):r['split'] for r in rows if r['source']=='CCT20'};candidates=[];availability=Counter();all_cameras=set()
 for ident,image in sorted(images.items()):
  assert isinstance(ident,str) and Path(ident).name==ident,'Unsafe image identifier'
  camera=str(image.get('location',image.get('camera','')));event=str(image.get('seq_id',image.get('event','')));all_cameras.add(camera)
  if ident in used_ids or not event or event in used_events or camera_split.get(camera)!='train':continue
  labels=[];bad=False
  for a in grouped[ident]:
   species=categories[a['category_id']]
   if species not in classes or not a.get('bbox'):bad=True;break
   x,y,w,h=map(float,a['bbox']);width,height=image['width'],image['height'];box=[x/width,y/height,(x+w)/width,(y+h)/height]
   if not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):bad=True;break
   labels.append(dict(species=species,xyxy=box,annotation_source='human_CCT20_catalog_team_review_pending'))
  species={b['species'] for b in labels}
  if bad or not labels or not (species&weak):continue
  for c in species&weak:availability[c]+=1
  candidates.append(dict(id=ident,camera=camera,event=event,file_name=image['file_name'],original_width=image['width'],original_height=image['height'],boxes=labels,proposed_split='train',source='CCT20_catalog_candidate',review_status='pending',priority=max((100-train_counts[c])/max(train_counts[c],1) for c in species&weak)))
 candidates.sort(key=lambda r:(-r['priority'],r['id']));selected=[];events=Counter();new_counts=Counter()
 for r in candidates:
  species={b['species'] for b in r['boxes']}&weak
  if events[r['event']]>=2 or not any(train_counts[c]+new_counts[c]<100 for c in species):continue
  selected.append(r);events[r['event']]+=1
  new_counts.update(b['species'] for b in r['boxes'])
  if len(selected)>=150:break
 frozen=dest/'candidate_plan.json'
 if frozen.exists():assert json.loads(frozen.read_text())==selected,'Source candidate plan changed; use a new collection version'
 else:workflow.save(frozen,selected)
 workflow.save(dest/'catalogue_summary.json',dict(metadata_url=META,metadata_sha256=workflow.sha(archive),available_unused_training_images_by_species=dict(availability),planned_images=len(selected),additional_crops_planned=dict(new_counts),cameras_unseen_in_original=sorted(all_cameras-set(camera_split)),scope='Only unused events from original training cameras. No validation/test/calibration reassignment. Labels require named team verification. If class coverage is unavailable here, use additional licensed data or new team camera locations.'))
 print('Training candidates planned:',len(selected));print('Unused-event availability:',dict(availability));return selected

def download(root,workflow):
 import imagehash
 root=Path(root);dest=root/'data_review_v4/collection';plan_rows=json.loads((dest/'candidate_plan.json').read_text());original=workflow.load_manifest(root/'phase1_source/data/cct20_team_v2/manifest.json')
 pixels={r['pixel_sha256'] for r in original};phashes=[int(r['phash'],16) for r in original if r.get('phash')];records=[]
 for i,r in enumerate(plan_rows,1):
  print('Checking training candidate',i,'/',len(plan_rows),flush=True)
  path=dest/'images'/(r['id']+'.jpg')
  if not path.exists():
   errors=[];raw=None
   for base in BASES:
    try:
     with urllib.request.urlopen(base+urllib.parse.quote(r['file_name']),timeout=45) as response:raw=response.read()
     break
    except Exception as e:errors.append(str(e))
   if raw is None:records.append(dict(id=r['id'],status='download_error',details='; '.join(errors)));continue
   with Image.open(io.BytesIO(raw)) as im:
    rgb=im.convert('RGB');assert rgb.size==(r['original_width'],r['original_height']);rgb.thumbnail((1024,1024),Image.Resampling.LANCZOS);path.parent.mkdir(parents=True,exist_ok=True);rgb.save(path,quality=95)
  with Image.open(path) as im:rgb=im.convert('RGB');pixel=hashlib.sha256(rgb.tobytes()).hexdigest();phash=int(str(imagehash.phash(rgb)),16)
  duplicate=pixel in pixels or any((phash^p).bit_count()<=4 for p in phashes)
  if duplicate:records.append(dict(id=r['id'],status='duplicate_quarantine',details='Pixel identity or perceptual distance <=4 versus frozen or already accepted candidate images'));continue
  pixels.add(pixel);phashes.append(phash)
  record=dict(**r,status='team_review_required',path=str(path),width=rgb.width,height=rgb.height,pixel_sha256=pixel,file_sha256=workflow.sha(path),phash=f'{phash:016x}');records.append(record)
  workflow.save(dest/'download_records.json',records)
  print('Staged:',i,'/',len(plan_rows),r['id'],flush=True)
 workflow.save(dest/'download_records.json',records);print('Staged for team review:',sum(r['status']=='team_review_required' for r in records));return records

def review_next(root,workflow):
 import ipywidgets as widgets
 from IPython.display import display
 from PIL import ImageDraw
 root=Path(root);dest=root/'data_review_v4/collection';records=json.loads((dest/'download_records.json').read_text());reviews=dest/'reviews'
 pending=[r for r in records if r['status']=='team_review_required' and not (reviews/(r['id']+'.json')).exists()]
 if not pending:print('Candidate first-pass review complete.');return
 r=pending[0]
 with Image.open(r['path']) as source:
  im=source.convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256'];im.thumbnail((850,650));draw=ImageDraw.Draw(im)
  for b in r['boxes']:
   x1,y1,x2,y2=b['xyxy'];draw.rectangle([x1*im.width,y1*im.height,x2*im.width,y2*im.height],outline='red',width=2);draw.text((x1*im.width,y1*im.height),b['species'],fill='yellow')
 buffer=io.BytesIO();im.save(buffer,format='PNG');view=widgets.Image(value=buffer.getvalue(),format='png');name=widgets.Text(description='Reviewer:');note=widgets.Textarea(description='Observation:');decision=widgets.Dropdown(options=[('Choose',''),('Verified labels and boxes','verified'),('Needs correction','needs_correction')]);status=widgets.HTML();button=widgets.Button(description='Save review',button_style='success')
 def store(_):
  if not name.value.strip() or not note.value.strip() or not decision.value:status.value='Complete reviewer, decision and observation.';return
  from datetime import datetime,timezone
  workflow.save(reviews/(r['id']+'.json'),dict(id=r['id'],reviewer=name.value.strip(),observation=note.value.strip(),decision=decision.value,boxes=r['boxes'],image_sha256=r['file_sha256'],timestamp=datetime.now(timezone.utc).isoformat(),scope='Candidate training data; no automatic training inclusion'))
  button.disabled=True;status.value='Saved. Run this cell again for the next candidate.'
 button.on_click(store);display(widgets.HTML(value=f'<b>{r["id"]} — training candidate — {len(pending)} remaining</b>'),view,name,decision,note,button,status)
