"""Discover actual additional class coverage without images, training, or split changes."""
from pathlib import Path
from collections import defaultdict,Counter
import json,csv,tarfile,urllib.request,hashlib
MANIFEST_SHA='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'
URL='https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz'
def run(root):
 root=Path(root);out=root/'data_review_v4/species_coverage_plan';out.mkdir(parents=True,exist_ok=True)
 manifest=root/'phase1_source/data/cct20_team_v2/manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==MANIFEST_SHA;rows=json.loads(manifest.read_text());classes=json.loads((root/'phase1_source/data/cct20_team_v2/species.json').read_text())
 old_ids={r['id'] for r in rows};old_events={r['event'] for r in rows};camera_splits=defaultdict(set)
 for r in rows:
  if r['source']=='CCT20':camera_splits[str(r['camera'])].add(r['split'])
 assert all(len(x)==1 for x in camera_splits.values());camera_split={k:next(iter(v)) for k,v in camera_splits.items()}
 archive=root/'data_review_v4/collection/metadata/eccv_18_annotations.tar.gz'
 if not archive.exists():
  archive=out/'eccv_18_annotations.tar.gz'
  if not archive.exists():
   print('Downloading public annotation metadata only...',flush=True)
   temp=archive.with_suffix('.download')
   with urllib.request.urlopen(URL,timeout=60) as response,temp.open('wb') as f:
    while chunk:=response.read(1024*1024):f.write(chunk)
   temp.replace(archive)
 print('Reading catalogue and excluding existing events/test cameras...',flush=True)
 images={};annotations={};categories={}
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():
   if not member.isfile() or not member.name.endswith('.json'):continue
   d=json.load(tar.extractfile(member))
   if not all(k in d for k in ('images','annotations','categories')):continue
   for c in d['categories']:
    if c['id'] in categories:assert categories[c['id']]==c['name']
    categories[c['id']]=c['name']
   for im in d['images']:
    if im['id'] in images:assert images[im['id']]==im
    images[im['id']]=im
   for a in d['annotations']:annotations[(a['image_id'],a['category_id'],tuple(a.get('bbox',[])))]=a
 grouped=defaultdict(list)
 for a in annotations.values():grouped[a['image_id']].append(a)
 available=defaultdict(set);events=defaultdict(set);examples=defaultdict(list)
 for ident,im in sorted(images.items()):
  event=str(im.get('seq_id',im.get('event','')));camera=str(im.get('location',im.get('camera','')));split=camera_split.get(camera,'new_camera')
  if ident in old_ids or not event or event in old_events or split=='test':continue
  for a in grouped[ident]:
   species=categories[a['category_id']]
   if species not in classes:continue
   box=a.get('bbox');valid=False
   if box and len(box)==4:
    x,y,w,h=map(float,box);valid=0<=x<x+w<=im['width'] and 0<=y<y+h<=im['height']
   kind='human_box' if valid else 'box_review_needed';key=(species,split,kind);available[key].add(ident);events[key].add(event)
   if len(examples[key])<10:examples[key].append(dict(id=ident,species=species,camera=camera,event=event,original_camera_split=split,label_kind=kind,file_name=im['file_name'],scope='Uncollected proposal; original labels and groups require versioned review before use'))
 result=[]
 for species in classes:
  row=dict(species=species)
  for split in ['train','val_select','val_calibrate']:
   row[split+'_current_crops']=sum(b['species']==species for r in rows if r['split']==split for b in r['boxes'])
   for kind in ['human_box','box_review_needed']:
    row[split+'_additional_'+kind+'_images']=len(available[(species,split,kind)]);row[split+'_additional_'+kind+'_events']=len(events[(species,split,kind)])
  row['new_camera_additional_human_box_images']=len(available[(species,'new_camera','human_box')]);row['missing_current_validation']=row['val_select_current_crops']==0;result.append(row)
 with (out/'coverage_availability.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 all_examples=[r for group in examples.values() for r in group]
 (out/'candidate_examples.json').write_text(json.dumps(all_examples,indent=2));(out/'protocol.json').write_text(json.dumps(dict(metadata_url=URL,metadata_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest_sha256=MANIFEST_SHA,scope='Availability discovery only. No images downloaded, labels approved, new splits installed, or model scores claimed. Existing test cameras excluded. New validation images, if later selected, define versioned development data; new independent assessment requires a frozen unseen-camera protocol.'),indent=2))
 print('Species | current train / validation | additional train / validation boxed images')
 for r in result:print(r['species'],r['train_current_crops'],r['val_select_current_crops'],r['train_additional_human_box_images'],r['val_select_additional_human_box_images'],flush=True)
 print('SAVED:',out,flush=True);return out
