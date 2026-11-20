"""Review precomputed SAM2.1-t alternatives or retain a visually correct original reference."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,io,base64,csv
import numpy as np
from PIL import Image,ImageDraw

def review(root,workflow,auto,code):
 from IPython.display import display,Javascript
 from google.colab.output import eval_js
 import cv2,shutil
 root,code=Path(root),Path(code);base=root/'data_review_v4';dest=base/'round3';dest.mkdir(exist_ok=True);models=code/'drafts';config=json.loads((models/'config.json').read_text());checks=json.loads((models/'artifact_checksums.json').read_text())
 for name,expected in checks.items():assert workflow.sha(models/name)==expected
 ids=['585dac58-23d2-11e8-a6a3-ec086b02610b','59ffbbf9-23d2-11e8-a6a3-ec086b02610b','59e77d63-23d2-11e8-a6a3-ec086b02610b'];manifest={r['id']:r for r in workflow.load_manifest(root/'phase1_source/data/cct20_team_v2/manifest.json')}
 with (base/'queue.csv').open() as f:queue=[r for r in csv.DictReader(f) if r['id'] in ids]
 assert len(queue)==3;workflow.csv_write(dest/'queue.csv',queue)
 snapshot=json.loads((base/'source_snapshot.json').read_text());workflow.save(dest/'source_snapshot.json',{i:snapshot[i] for i in ids});workflow.save(dest/'review_protocol.json',dict(version='data_review_v4_round3',images=3,manifest_sha256=workflow.MANIFEST_SHA,test_images_used=False,parent_review=str(base),proposal='Precomputed SAM2.1-t alternatives; original pixels and CLAHE inference; retain original only with explicit human approval. Display contrast enhanced; original image files unchanged.'))
 cards=[];records={}
 for ident in ids:
  if (dest/'decisions'/(ident+'.json')).exists():continue
  item=json.loads((base/'items'/(ident+'.json')).read_text());workflow.save(dest/'items'/(ident+'.json'),item);row=manifest[ident]
  proposals={}
  for variant in ['A','B']:
   src=models/(ident+'_'+variant+'.json');p=json.loads(src.read_text());assert p['source_label_sha256']==item['label_sha256'] and p['pixel_sha256']==item['pixel_sha256'];assert p['config_sha256']==workflow.sha(models/'config.json')
   target=dest/'auto_proposals'/(ident+'_'+variant+'.json');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
   if auto.touching_topology_safe(p['polygons'],item['width'],item['height'],workflow):
    p['raster_topology_validated']=bool(workflow.geometric_findings(p['polygons']));p['approval_enabled']=p['mask_count']==p['expected_animals'];workflow.save(target,p)
   proposals[variant]=(p,target)
  workflow.save(dest/'auto_proposals/config.json',config)
  with Image.open(item['image']) as source:
   im=source.convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==item['pixel_sha256'];rgb=np.asarray(im);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0]);view=Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
  # Include the full extent of every candidate and the human boxes; never hide background errors by cropping.
  points=[v for b in row['boxes'] for v in [[b['xyxy'][0],b['xyxy'][1]],[b['xyxy'][2],b['xyxy'][3]]]]
  for polys in [item['original'],*[p['polygons'] for p,_ in proposals.values()]]:points.extend(v for poly in polys for v in poly)
  coords=np.array(points);lo=np.maximum(0,coords.min(0)-.04);hi=np.minimum(1,coords.max(0)+.04);box=(int(lo[0]*im.width),int(lo[1]*im.height),max(int(lo[0]*im.width)+1,int(np.ceil(hi[0]*im.width))),max(int(lo[1]*im.height)+1,int(np.ceil(hi[1]*im.height))))
  panels=[]
  for title,polys,color in [('ORIGINAL',item['original'],(255,120,0,100)),('A: SAM2.1',proposals['A'][0]['polygons'],(0,200,70,100)),('B: SAM2.1 + CONTRAST',proposals['B'][0]['polygons'],(0,180,255,100))]:
   layer=Image.new('RGBA',im.size);draw=ImageDraw.Draw(layer)
   for poly in polys:draw.polygon([(x*im.width,y*im.height) for x,y in poly],fill=color,outline=color[:3]+(255,))
   panel=Image.alpha_composite(view.convert('RGBA'),layer).convert('RGB').crop(box);factor=min(600/panel.width,500/panel.height);panel=panel.resize((max(1,round(panel.width*factor)),max(1,round(panel.height*factor))),Image.Resampling.LANCZOS);canvas=Image.new('RGB',(600,530),'white');canvas.paste(panel,(0,30));ImageDraw.Draw(canvas).text((10,8),title,fill='black');panels.append(canvas)
  combined=Image.new('RGB',(1800,530),'white')
  for i,panel in enumerate(panels):combined.paste(panel,(600*i,0))
  buffer=io.BytesIO();combined.save(buffer,format='JPEG',quality=82)
  cards.append(dict(id=ident,split=item['split'],image='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode(),A=proposals['A'][0]['approval_enabled'],B=proposals['B'][0]['approval_enabled']))
  records[ident]=(item,proposals)
 if not cards:print('All three already have saved decisions.');return workflow.export_review(dest)
 name=(base/'reviewer_name.txt').read_text() if (base/'reviewer_name.txt').exists() else ''
 display(Javascript('window.finalThreeReview = '+(code/'final_three.js').read_text()+';'))
 result=eval_js('window.finalThreeReview('+json.dumps(dict(cards=cards,reviewer=name))+')')
 if result is None:print('Closed without saving.');return
 assert result['reviewer'].strip() and result['confirmed']
 for decision in result['decisions']:
  ident=decision['id'];choice=decision['choice'];assert ident in records and choice in ('A','B','original','reject');item,proposals=records[ident]
  if choice in ('A','B'):
   p,path=proposals[choice];assert p['approval_enabled'];auto.save_approved_draft(dest,item,p,workflow,result['reviewer'],path)
  elif choice=='original':
   workflow.save_review(dest,item,dict(reviewer=result['reviewer'],observation='Explicitly re-approved original visible mask after inspecting original and new alternatives with contrast-enhanced display.',decision='verified',polygons=item['original'],complete_animals_confirmed=True,method='original_reference_explicitly_reapproved',topology_exception_confirmed=bool(workflow.geometric_findings(item['original']))))
  else:workflow.save_review(dest,item,dict(reviewer=result['reviewer'],observation='Explicitly rejected round-three alternatives after visual inspection.',decision='needs_correction',polygons=item['original'],complete_animals_confirmed=False,method='round3_human_rejected'))
  workflow.save(dest/'decisions'/(ident+'.json'),dict(**decision,reviewer=result['reviewer'],timestamp=datetime.now(timezone.utc).isoformat()));print('Saved',ident,choice,flush=True)
 return workflow.export_review(dest)
