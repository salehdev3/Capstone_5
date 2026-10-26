
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ImageHash', 'pandas', 'scipy', 'matplotlib', 'Pillow', 'PyYAML', 'ipywidgets', 'scikit-learn', 'opencv-python-headless', 'tqdm', 'wandb', 'transformers==4.49.0'])


from pathlib import Path
import hashlib
import sys
import urllib.request
import importlib

module_candidates = [Path.cwd(), *Path.cwd().parents]
if '__file__' in globals():
    module_candidates.insert(0, Path(__file__).resolve().parent)
module_root = next((p for p in module_candidates if (p/'preprocessing.py').is_file()), None)
if module_root is None:
    module_root = Path.cwd()/'project_modules'
    module_root.mkdir(exist_ok=True)
    module_path = module_root/'preprocessing.py'
    source_url = 'https://raw.githubusercontent.com/salehdev3/Capstone_5/ed75540a1f57ad08d0300480f2ea922e136c5cda/preprocessing.py'
    expected_sha256 = '87e3366dd7946e566c3a9238095748eb698ed0b57e2a9a5699d49817c440a5eb'
    if not module_path.exists():
        with urllib.request.urlopen(source_url, timeout=60) as response:
            module_bytes = response.read()
        assert hashlib.sha256(module_bytes).hexdigest() == expected_sha256, 'Preprocessing download checksum mismatch'
        module_path.write_bytes(module_bytes)
    assert hashlib.sha256(module_path.read_bytes()).hexdigest() == expected_sha256, 'Cached preprocessing checksum mismatch'
module_root = module_root.resolve()
sys.path.insert(0, str(module_root))
if 'preprocessing' in sys.modules:
    assert Path(sys.modules['preprocessing'].__file__).resolve() == module_root/'preprocessing.py', 'Restart the runtime to load the selected preprocessing module'
import preprocessing
importlib.reload(preprocessing)
print('Preprocessing module:', preprocessing.__file__)


from pathlib import Path
import os, sys, subprocess
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import collections
import hashlib
import html
import io
import json
from pathlib import Path
import random
import tarfile
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import imagehash


META_URL = 'https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz'
IMAGE_BASES = [
    'https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/',
    'https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/']
NON_ANIMALS = {'empty', 'car', 'vehicle', 'person', 'human', 'people'}



SEED = 42
MAX_IMAGES = 6000
AUDIT_IMAGES = 100
PROJECT = Path.cwd() if Path.cwd().name == 'CameraTrap_Phase1_Team' else Path.cwd()/'CameraTrap_Phase1_Team'
PROJECT.mkdir(exist_ok=True)
os.chdir(PROJECT)
DATA = Path('data/cct20_v1')
DATA.mkdir(parents=True, exist_ok=True)
Path('runs/data').mkdir(parents=True, exist_ok=True)
Path('handoff').mkdir(exist_ok=True)

def read_json(path):
    return json.loads(Path(path).read_text())
def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

cfg = {'seed': SEED, 'max_images': MAX_IMAGES, 'max_image_side': 1024,
       'audit_events': AUDIT_IMAGES, 'data_dir': str(DATA), 'output_dir': 'runs/data'}
assert not Path('handoff/dataset_release.json').exists(), 'This release is immutable after training begins.'
print('Project folder:', PROJECT)


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return
    tmp = path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as r, tmp.open('wb') as f:
                while chunk := r.read(1024*1024):
                    f.write(chunk)
            tmp.replace(path)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt+1)

def read_metadata(root):
    archive = root/'metadata'/'eccv_18_annotations.tar.gz'
    fetch(META_URL, archive)
    images, annotations, categories = {}, {}, {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.json') or not member.isfile():
                continue

            d = json.load(tar.extractfile(member))
            for c in d['categories']:
                if c['id'] in categories and categories[c['id']] != c['name']:
                    raise ValueError('Conflicting category maps')
                categories[c['id']] = c['name']
            for im in d['images']:
                if im['id'] in images and images[im['id']] != im:
                    raise ValueError('Conflicting image metadata')
                images[im['id']] = im
            for a in d['annotations']:
                key = (a['image_id'], a['category_id'], tuple(a.get('bbox', [])))
                annotations[key] = a
    by_image = collections.defaultdict(list)
    for a in annotations.values():
        by_image[a['image_id']].append(a)
    rows, excluded = [], []
    for ident, im in sorted(images.items()):
        aa = by_image[ident]
        labels = sorted({categories[a['category_id']] for a in aa})
        if not labels or any(c in NON_ANIMALS-{'empty'} for c in labels):
            excluded.append({'id': ident, 'reason': 'non-animal-category-or-unlabelled', 'labels': labels})
            continue
        animal_labels = [s for s in labels if s not in NON_ANIMALS]
        if 'empty' in labels and animal_labels:
            excluded.append({'id': ident, 'reason': 'conflicting-empty-label'})
            continue
        boxes = []
        incomplete = False
        for a in aa:
            name = categories[a['category_id']]
            if name in NON_ANIMALS:
                continue
            if 'bbox' not in a:
                incomplete = True
                continue
            x, y, w, h = map(float, a['bbox'])
            W, H = im['width'], im['height']
            x1, y1 = max(0.,x/W), max(0.,y/H)
            x2, y2 = min(1.,(x+w)/W), min(1.,(y+h)/H)
            if x2 <= x1 or y2 <= y1:
                incomplete = True
                continue
            boxes.append({'species': name, 'xyxy': [x1,y1,x2,y2], 'annotation_source': 'human_CCT20'})
        rows.append(dict(id=ident, file_name=Path(im['file_name']).name,
                         camera=str(im['location']), event=str(im.get('seq_id') or ident),
                         timestamp=im.get('date_captured',im.get('datetime')),
                         original_width=im['width'], original_height=im['height'],
                         seq_num_frames=im.get('seq_num_frames'), labels=animal_labels,
                         animal=int(bool(animal_labels)), boxes=boxes,
                         incomplete_boxes=incomplete))
    summary = dict(source=META_URL, sha256=digest(archive), total_images=len(images),
                   unique_annotations=len(annotations), human_boxes=sum('bbox' in a for a in annotations.values()),
                   cameras=len({str(i['location']) for i in images.values()}),
                   categories=categories, eligible_frames=len(rows), excluded_frames=len(excluded),
                   empty_frames=sum(r['animal']==0 for r in rows),
                   note='CCT20 is curated and is not the full 243k-image dataset. Percent empty is recomputed.')
    write_json(root/'metadata_summary.json', summary)
    write_json(root/'metadata_exclusions.json', excluded)
    return rows

rows = read_metadata(DATA)
print("Eligible frames:", len(rows))
display(pd.DataFrame(rows)[["id", "camera", "event", "animal", "labels"]].head())


from sklearn.model_selection import StratifiedGroupKFold

def assign_sites(rows, seed):

    y=np.array(['|'.join(sorted(r['labels'])) or 'empty' for r in rows])
    groups=np.array([r['camera'] for r in rows])
    def first_fold(indices, folds):
        splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
        a,b=next(splitter.split(np.zeros(len(indices)),y[indices],groups[indices]))
        return indices[a],indices[b]
    remaining,test=first_fold(np.arange(len(rows)),5)
    train,validation=first_fold(remaining,4)
    select,calibrate=first_fold(validation,2)
    site_map={}
    for name,ids in [('train',train),('val_select',select),('val_calibrate',calibrate),('test',test)]:
        for camera in set(groups[ids]):
            assert camera not in site_map
            site_map[camera]=name
    return site_map

def assert_disjoint(rows):
    for field in ['id', 'camera', 'event', 'pixel_sha256']:
        seen = {}
        for r in rows:
            if field not in r:
                continue
            key = r[field]
            if key in seen and seen[key] != r['split']:
                raise AssertionError(f'{field} leakage across splits')
            seen[key] = r['split']

site_map=assign_sites(rows,SEED)
for r in rows:r['split']=site_map[r['camera']]
assert_disjoint(rows)
write_json(DATA/'site_split.json',site_map)
display(pd.DataFrame(rows).groupby('split').agg(images=('id','count'),cameras=('camera','nunique')))


def sample_events(rows, limit, seed):
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    selected = []
    for camera in sorted({r['camera'] for r in rows}):
        rr = [r for r in rows if r['camera']==camera]
        groups = collections.defaultdict(list)
        for r in rr:
            groups[r['event']].append(r)
        events = sorted(groups)
        rng.shuffle(events)
        budget = max(3, round(limit*len(rr)/len(rows)))
        chosen = []
        for event in events:
            chosen.extend(groups[event])
            if len(chosen) >= budget:
                break
        selected.extend(chosen)
    return selected

chosen = sample_events(rows, MAX_IMAGES, SEED)
write_json(DATA/'selected_ids.json', [r['id'] for r in chosen])
print('Selected images:', len(chosen))


def download_one(row, root, max_side):
    path = root/'images'/(row['id']+'.jpg')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        err = None
        for base in IMAGE_BASES:
            try:
                with urllib.request.urlopen(base+urllib.parse.quote(row['file_name']),timeout=40) as r:
                    raw = r.read()
                im = Image.open(io.BytesIO(raw)).convert('RGB')
                im.thumbnail((max_side,max_side), Image.Resampling.LANCZOS)
                tmp = path.with_suffix('.tmp')
                im.save(tmp, format='JPEG',quality=95)
                tmp.replace(path)
                break
            except Exception as e:
                err = str(e)
        else:
            return None, {'id':row['id'],'event':row['event'],'reason':err}
    try:
        with Image.open(path) as src:
            im = src.convert('RGB'); im.load()
            small = np.array(im.resize((64,64)),dtype=float)
            row = dict(row, path=str(path.resolve()),width=im.width,height=im.height,
                       processed_file_md5=hashlib.md5(path.read_bytes()).hexdigest(),
                       pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                       phash=str(imagehash.phash(im)),
                       grayscale_proxy=bool(np.mean(np.max(small,2)-np.min(small,2))<6))
            return row, None
    except Exception as e:
        return None, {'id':row['id'],'event':row['event'],'reason':str(e)}

from tqdm.auto import tqdm
good, failures = [], []
for row in tqdm(chosen):
    image_row, error = download_one(row, DATA, cfg['max_image_side'])
    if image_row is not None:
        image_row['path'] = str(DATA/'images'/(image_row['id']+'.jpg'))
        good.append(image_row)
    if error is not None: failures.append(error)
write_json(DATA/'download_failures.json', failures)
print('Readable:', len(good), '| Failed:', len(failures))
assert len(failures) <= 0.02*len(chosen), 'More than 2% of selected images failed to download or validate.'
failed_events = {e['event'] for e in failures}
good = sorted([r for r in good if r['event'] not in failed_events], key=lambda r:r['id'])


import cv2
inventory=[]
for r in good:
    im=Image.open(r['path']);rgb=np.asarray(im.convert('RGB'))
    small=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)
    blur=float(cv2.Laplacian(cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
    inventory.append({'id':r['id'],'split':r['split'],'mode':im.mode,'width':im.width,'height':im.height,
      'brightness':float(small.mean()),'blur_score':blur,'path':r['path']})
inventory=pd.DataFrame(inventory);inventory.to_csv(DATA/'inventory.csv',index=False)
display(inventory.describe(include='all'))
lowest=inventory.query("split == 'train'").nsmallest(8,'blur_score').copy()
lowest['reviewer']='';lowest['decision']='';lowest['notes']=''
if not (DATA/'blur_review.csv').exists():lowest.to_csv(DATA/'blur_review.csv',index=False)


import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image

def blur_review_widget(data_dir):
    data_dir = Path(data_dir)
    review_path = data_dir / "blur_review.csv"
    review_rows = pd.read_csv(review_path).fillna("")
    if review_rows.empty:
        raise ValueError("The image-quality review contains no images.")
    reviewer = widgets.Text(description="Reviewer:", layout=widgets.Layout(width="75%"))
    names = review_rows["reviewer"].astype(str).str.strip()
    if names.ne("").all() and names.nunique() == 1:
        reviewer.value = names.iloc[0]
    display(reviewer)
    review_widgets = []
    for index, row in review_rows.iterrows():
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        image.thumbnail((500, 350))
        decision = widgets.Dropdown(
            options=[("Select decision", ""), ("keep", "keep"), ("flag", "flag")],
            value=row["decision"] if row["decision"] in ("keep", "flag") else "",
            description="Decision:")
        notes = widgets.Text(value=str(row["notes"]), description="Observation:",
                             layout=widgets.Layout(width="75%"))
        print("Image:", row["id"], "| Blur score:", round(row["blur_score"], 2))
        display(image, decision, notes)
        review_widgets.append((index, decision, notes))
    save_button = widgets.Button(description="Save review", button_style="success")
    save_output = widgets.Output()
    def save_review(_):
        with save_output:
            clear_output()
            if Path("handoff/dataset_release.json").exists():
                print("The dataset release is frozen. Create a new release before changing its review.")
                return
            if not reviewer.value.strip():
                print("Enter the reviewer name before saving.")
                return
            if any(decision.value not in ("keep", "flag") or not notes.value.strip()
                   for _, decision, notes in review_widgets):
                print("Select a decision and add an observation for every image before saving.")
                return
            for index, decision, notes in review_widgets:
                review_rows.loc[index, "reviewer"] = reviewer.value.strip()
                review_rows.loc[index, "decision"] = decision.value
                review_rows.loc[index, "notes"] = notes.value.strip()
            temporary = review_path.with_suffix(".csv.tmp")
            review_rows.to_csv(temporary, index=False)
            temporary.replace(review_path)
            print("Review saved to:", review_path)
    save_button.on_click(save_review)
    display(save_button, save_output)
    return reviewer, review_widgets, save_button, save_output

blur_review_controls = blur_review_widget(DATA)


def near_duplicate_quarantine(rows, distance=2):

    buckets = collections.defaultdict(list)
    bad, pairs = set(), []
    for r in rows:
        h = int(r['phash'],16)
        candidates = {}
        for part in range(4):
            for old in buckets[(part,(h>>(part*16)) & 65535)]:
                candidates[old['id']] = old
        for old in candidates.values():
            if old['split']==r['split']:
                continue
            dist = (h ^ int(old['phash'],16)).bit_count()
            if dist <= distance:
                bad.update([r['event'],old['event']])
                pairs.append({'id_a':old['id'],'id_b':r['id'],'distance':dist})
        for part in range(4):
            buckets[(part,(h>>(part*16)) & 65535)].append(r)
    return [r for r in rows if r['event'] not in bad], pairs, sorted(bad)


hashes=collections.defaultdict(list)
for r in good:hashes[r['pixel_sha256']].append(r)
excluded_events=set(); duplicate_ids=[];conflicting_duplicate_groups=[]
for group in hashes.values():
    if len(group)>1:
        if len({tuple(r['labels']) for r in group})>1:
            conflicting_duplicate_groups.append([r['id'] for r in group])
            excluded_events.update(r['event'] for r in group)
        elif len({r['split'] for r in group})>1:
            excluded_events.update(r['event'] for r in group)
        else:duplicate_ids.extend(r['id'] for r in group[1:])
cleaned=[r for r in good if r['event'] not in excluded_events and r['id'] not in set(duplicate_ids)]
cleaned,pairs,near_events=near_duplicate_quarantine(cleaned)
assert_disjoint(cleaned)
for split in ['train','val_select','val_calibrate','test']:
    if {r['animal'] for r in cleaned if r['split']==split}!={0,1}:
        raise ValueError(f'{split} lacks empty/animal examples after sampling. Increase max_images; do not adjust test based on results.')
labels=sorted({b['species'] for r in cleaned if r['split']=='train' for b in r['boxes']})
if len(labels)<2:raise ValueError('Need at least two training species')
write_json(DATA/'species.json',labels)
write_json(DATA/'manifest.json',cleaned)
write_json(DATA/'preparation_config.json',cfg)
write_json(DATA/'cleaning_report.json',dict(selected=len(chosen),downloaded=len(good),kept=len(cleaned),
    failures=len(failures),excluded_failed_events=len(failed_events),duplicate_ids=duplicate_ids,
    exact_cross_split_events=sorted(excluded_events),near_duplicate_pairs=pairs,near_quarantined_events=near_events,
    conflicting_duplicate_label_groups=conflicting_duplicate_groups,
    near_policy='Conservative pHash <=2 across partitions: both events quarantined; not asserted to be true duplicates.',
    corrupted_or_failed_files=failures,grayscale_proxy='RGB channel difference <6; proxy only, not verified infrared ground truth'))

print('Clean frames:', len(cleaned), '| Training species:', labels)


def gallery(path, rows, draw_boxes=False, titles=None):
    path=Path(path);thumbs=path.parent/(path.stem+'_images');thumbs.mkdir(exist_ok=True)
    cards=[]
    for i,r in enumerate(rows):
        im=Image.open(r['path']).convert('RGB')
        if draw_boxes:
            draw=ImageDraw.Draw(im)
            for b in r['boxes']:
                x1,y1,x2,y2=b['xyxy'];box=[x1*im.width,y1*im.height,x2*im.width,y2*im.height]
                draw.rectangle(box,outline='lime',width=3);draw.text((box[0],box[1]),b['species'],fill='yellow')
        im.thumbnail((480,360));name=f'{i:04d}.jpg';im.save(thumbs/name)
        title=titles[i] if titles else f'{r["id"]} | {r["split"]} | {r["labels"] or ["empty"]}'
        cards.append(f'<article><img src="{thumbs.name}/{name}"><p>{html.escape(str(title))}</p></article>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>Local review gallery</title><style>body{font:14px sans-serif;background:#132127;color:white}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}article{background:#22343b;padding:10px}img{max-width:100%}p{overflow-wrap:anywhere}</style><h1>Private local review gallery</h1><p>Review before sharing; exclude or blur any identifiable people. Do not redistribute dataset images.</p><main>'+''.join(cards)+'</main>',encoding='utf-8')

def audit_templates(cfg, rows):
    root=Path(cfg['data_dir'])

    rng=random.Random(cfg['seed'])
    events=collections.defaultdict(list)
    for r in rows:
        if r['split']=='train':events[r['event']].append(r)
    selected=rng.sample(sorted(events),min(cfg['audit_events'],len(events)))
    audit=[]
    for event in selected:
        r=rng.choice(events[event]);audit.append(dict(id=r['id'],event=event,split=r['split'],
          expected_species=';'.join(r['labels']) or 'empty',label_correct='',boxes_correct='',
          identifiable_person='',reviewer='',notes=''))
    pd.DataFrame(audit).to_csv(root/'label_audit.csv',index=False)
    gallery(root/'audit_gallery.html',[next(r for r in rows if r['id']==a['id']) for a in audit],draw_boxes=True)

    count_rows=[dict(split=r['split'],label=s) for r in rows for s in (r['labels'] or ['empty'])]
    pd.crosstab(pd.DataFrame(count_rows).label,pd.DataFrame(count_rows).split).to_csv(root/'class_counts.csv')
    crop_rows=[dict(split=r['split'],label=b['species']) for r in rows for b in r['boxes']]
    pd.crosstab(pd.DataFrame(crop_rows).label,pd.DataFrame(crop_rows).split).to_csv(root/'crop_counts.csv')
    gallery(root/'annotation_gallery.html',rng.sample(rows,min(48,len(rows))),draw_boxes=True)

if not (DATA/'label_audit.csv').exists():
    audit_templates(cfg, cleaned)
counts = pd.read_csv(DATA/'class_counts.csv', index_col=0)
display(counts)
counts.plot.bar(stacked=True, figsize=(12,4))
plt.title('Frame labels by split | source images, maximum side 1024 px | threshold N/A')
plt.ylabel('Label occurrences'); plt.tight_layout(); plt.show()
for row in [r for r in cleaned if r['split']=='train'][:4]:
    print(row['id'], row['labels'] or ['empty'])
    im=Image.open(row['path']); im.thumbnail((600,400)); display(im)


def audit_widget(cfg):
    import ipywidgets as w
    from IPython.display import display,clear_output
    from PIL import Image,ImageDraw
    root=Path(cfg['data_dir']);path=root/'label_audit.csv'
    df=pd.read_csv(path).fillna('');rows={r['id']:r for r in read_json(root/'manifest.json')}
    index=w.IntSlider(value=0,min=0,max=len(df)-1,description='Sample')
    correct=w.Dropdown(options=['','yes','no'],description='Label correct')
    boxes=w.Dropdown(options=['','yes','no','not applicable'],description='Boxes correct')
    person=w.Dropdown(options=['','yes','no'],description='Person visible')
    reviewer=w.Text(description='Reviewer');notes=w.Textarea(description='Notes')
    save=w.Button(description='Save and next',button_style='success');output=w.Output();status=w.Output()
    def show(change=None):
        row=df.iloc[index.value];r=rows[row.id]
        correct.value=str(row.label_correct);boxes.value=str(row.boxes_correct);person.value=str(row.identifiable_person)
        if row.reviewer:reviewer.value=str(row.reviewer)
        notes.value=str(row.notes)
        with output:
            clear_output(wait=True);print('Expected:',row.expected_species,'| sample:',row.id)
            im=Image.open(r['path']).convert('RGB');draw=ImageDraw.Draw(im)
            for b in r['boxes']:
                x1,y1,x2,y2=b['xyxy'];draw.rectangle([x1*im.width,y1*im.height,x2*im.width,y2*im.height],outline='lime',width=3)
            im.thumbnail((900,600));display(im)
    def persist(_):
        with status:
            clear_output()
            if not reviewer.value.strip() or not correct.value or not person.value or not boxes.value:
                print('Fill every decision and the reviewer name before saving.');return
        i=index.value
        for key,value in [('label_correct',correct.value),('boxes_correct',boxes.value),('identifiable_person',person.value),
                          ('reviewer',reviewer.value.strip()),('notes',notes.value)]:df.loc[i,key]=value
        df.to_csv(path,index=False)
        with status:print('Saved. Corrections/privacy flags need resolution before final training/report.')
        if i<len(df)-1:index.value=i+1
    index.observe(show,names='value');save.on_click(persist)
    display(w.VBox([index,output,correct,boxes,person,reviewer,notes,save,status]));show()

audit_widget(cfg)


from scipy.stats import beta

def audit_report(cfg):
    root=Path(cfg['data_dir']);df=pd.read_csv(root/'label_audit.csv').fillna('')
    valid=(df.label_correct.astype(str).str.lower().isin(['yes','no']) &
           df.boxes_correct.astype(str).str.lower().isin(['yes','no','not applicable']) &
           df.identifiable_person.astype(str).str.lower().isin(['yes','no']) &
           df.reviewer.astype(str).str.strip().ne(''))
    reviewed=df[valid];n=len(reviewed);errors=int((reviewed.label_correct.astype(str).str.lower()=='no').sum())
    rows=read_json(root/'manifest.json');present_events={r['event'] for r in rows}
    flagged=reviewed[(reviewed.label_correct.astype(str).str.lower()=='no') |
                     (reviewed.boxes_correct.astype(str).str.lower()=='no') |
                     (reviewed.identifiable_person.astype(str).str.lower()=='yes')]
    upper=float(beta.ppf(.95,errors+1,n-errors)) if n and errors<n else (1. if n else None)
    result=dict(status='complete' if n==len(df) else 'pending',requested=len(df),reviewed=n,label_errors=errors,
      estimated_error_rate=errors/n if n else None,one_sided_95_upper_bound=upper,
      sampling_unit='random train event, then one random eligible frame; not all dataset frames',
      limitations='Binomial interval assumes independent sampled events. Same-camera correlation may make it optimistic.',
      no_review_is_not_zero_errors=True,
      privacy_flags=int(((reviewed.identifiable_person.astype(str).str.lower()=='yes') & reviewed.event.isin(present_events)).sum()),
      unresolved_flagged_events=len(set(flagged.event)&present_events))
    write_json(root/'audit_report.json',result);return result

def apply_audit(cfg):
    root=Path(cfg['data_dir']);out=Path(cfg['output_dir'])
    df=pd.read_csv(root/'label_audit.csv').fillna('')
    reviewed=df[df.reviewer.astype(str).str.strip().ne('')]
    flagged=reviewed[(reviewed.label_correct.astype(str).str.lower()=='no') |
                     (reviewed.boxes_correct.astype(str).str.lower()=='no') |
                     (reviewed.identifiable_person.astype(str).str.lower()=='yes')]
    bad=set(flagged.event);rows=read_json(root/'manifest.json')
    if not (bad & {r['event'] for r in rows}):
        print('No unresolved human-flagged events to quarantine.');return
    if list(out.glob('*/*/best.pt')):
        raise RuntimeError('Checkpoints already exist. Set a NEW output_dir in config before applying training-data corrections.')
    if any(r['split']!='train' and r['event'] in bad for r in rows):
        raise ValueError('Only train audit changes are allowed; held-out protocol is immutable.')
    cleaned=[r for r in rows if r['event'] not in bad]
    labels=sorted({b['species'] for r in cleaned if r['split']=='train' for b in r['boxes']})
    if len(labels)<2 or {r['animal'] for r in cleaned if r['split']=='train'}!={0,1}:
        raise ValueError('Audit exclusions remove required training support; collect more training events.')
    write_json(root/'audit_quarantine.json',dict(events=sorted(bad),frames_removed=len(rows)-len(cleaned),
        policy='Human-flagged events excluded. No automatic corrected species labels.'))
    write_json(root/'manifest.json',cleaned);write_json(root/'species.json',labels);assert_disjoint(cleaned)
    for name,values in [('class_counts',[dict(split=r['split'],label=s) for r in cleaned for s in (r['labels'] or ['empty'])]),
                        ('crop_counts',[dict(split=r['split'],label=b['species']) for r in cleaned for b in r['boxes']])]:
        frame=pd.DataFrame(values);pd.crosstab(frame.label,frame.split).to_csv(root/(name+'.csv'))
    print('Quarantined',len(rows)-len(cleaned),'training frames across',len(bad),'flagged events.')

audit = audit_report(cfg)
assert audit['status']=='complete', 'Complete the audit before continuing.'
apply_audit(cfg)
audit = audit_report(cfg)
assert audit['unresolved_flagged_events']==0
rows = read_json(DATA/'manifest.json')
print(audit)


splits=['train','val_select','val_calibrate','test']
all_labels=sorted({label for r in rows for label in (r['labels'] or ['empty'])})
balance=[]
for label in all_labels:
    overall=sum(label in (r['labels'] or ['empty']) for r in rows)/len(rows)
    for split in splits:
        subset=[r for r in rows if r['split']==split]
        count=sum(label in (r['labels'] or ['empty']) for r in subset)
        rate=count/len(subset) if subset else 0
        balance.append({'label':label,'split':split,'count':count,'frames':len(subset),
                        'percent':100*rate,'overall_percent':100*overall,'difference_pp':100*(rate-overall)})
balance_report=pd.DataFrame(balance);balance_report.to_csv(DATA/'class_balance_proportions.csv',index=False)
display(balance_report)
balance_report.pivot(index='label',columns='split',values='percent').plot.bar(figsize=(12,5))
plt.ylabel('Frames containing label (%)');plt.tight_layout();plt.show()
counts=pd.DataFrame([{'species':b['species'],'split':r['split']} for r in rows for b in r['boxes']])
support=pd.crosstab(counts.species,counts.split).reindex(columns=splits,fill_value=0)
support.to_csv(DATA/'species_crop_support.csv');display(support)
print('Missing class/split combinations:');display(balance_report.query('count == 0')[['label','split']])
assert_disjoint(rows)
print('PASS: cameras, events and exact pixel duplicates are disjoint.')


import cv2, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import preprocessing
from preprocessing import Preprocessor, FrameDataset, load_rgb_sample


train_rows=[dict(r,label=r['animal'],crop=None) for r in rows if r['split']=='train']
train_dataset=FrameDataset(train_rows,224,train=True)
eval_dataset=FrameDataset(train_rows,224,train=False)
train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(SEED))
images,targets,_=next(iter(train_loader))
print('N,C,H,W:',images.shape,'| labels:',targets.tolist())
fig,axes=plt.subplots(2,4,figsize=(12,6))
for x,y,ax in zip(images,targets,axes.flat):
    rgb=x.permute(1,2,0).numpy()*[.229,.224,.225]+[.485,.456,.406]
    ax.imshow(np.clip(rgb,0,1));ax.set_title(['empty','animal'][int(y)]);ax.axis('off')
fig.suptitle('Training augmentation | train | input 224 x 224 | threshold N/A')
plt.tight_layout();plt.show()


sample=next(r for r in rows if r['split']=='train' and r['animal'])
bgr=cv2.imread(sample['path']);pre=Preprocessor(224)
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);prepared=pre.prepare(bgr)
augmented=np.array(train_dataset.tf(Image.fromarray(prepared)))
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,img,title in zip(axes,[rgb,prepared,augmented],['Downloaded RGB','Resize 224','Train augmentation']):
    ax.imshow(img);ax.set_title(title);ax.axis('off')
fig.suptitle('Preprocessing | train | input 224 x 224 | threshold N/A')
plt.show()
for ch,color in enumerate(['red','green','blue']):
    plt.hist(prepared[:,:,ch].ravel(),bins=32,alpha=.35,color=color,label=color)
plt.legend();plt.title('Channel histogram | train | input 224 x 224 | threshold N/A');plt.show()


row = next(r for r in rows if r['split']=='train' and r['boxes'])
box = row['boxes'][0]
im = Image.open(row['path']).convert('RGB')
x1,y1,x2,y2 = box['xyxy']
dx,dy=(x2-x1)*.05,(y2-y1)*.05
crop=im.crop((max(0,int((x1-dx)*im.width)),max(0,int((y1-dy)*im.height)),
              min(im.width,int((x2+dx)*im.width)),min(im.height,int((y2+dy)*im.height))))
print('Species:', box['species'], '| Split: train | Crop:', crop.size, '| Model input: 224 x 224 | Threshold: N/A'); display(crop)


TEAM_NOTES = ""
counts = pd.read_csv(DATA/'class_counts.csv', index_col=0)
cleaning = read_json(DATA/'cleaning_report.json')
summary = pd.DataFrame(rows).groupby('split').agg(frames=('id','count'),cameras=('camera','nunique'))
card = f"""# Dataset Card — CCT20 / Phase 1
Source: https://lila.science/datasets/caltech-camera-traps
Citation: Beery, Van Horn and Perona, Recognition in Terra Incognita, ECCV 2018.
Licence: CDLA-Permissive; follow course policy: do not redistribute images or commit them to Git.

## Tasks
Full-frame empty/animal classification; species classification on human-box crops.
No predicted boxes are used as ground truth. Positive frames without boxes remain in triage only.
Human/vehicle-labelled, conflicting or unlabelled frames are excluded by metadata.

## Acquisition and cleaning
Selected: {cleaning['selected']}; retained after human audit: {len(rows)}.
Download failures: {cleaning['failures']}; exact duplicate IDs removed: {len(cleaning['duplicate_ids'])}.
Near-duplicate pairs across splits: {len(cleaning['near_duplicate_pairs'])}.
Image maximum side: 1024; model input: 224 x 224; seed: {SEED}.
Corrupt/failed-event and duplicate exclusions are recorded in cleaning_report.json.

## Split
StratifiedGroupKFold by camera, with joint image-label strata. Camera counts and class proportions below are measured, not assumed.
Whole bursts stay within one split; exact duplicate and event leakage checks pass.
Group constraints approximate class stratification; this is not the official CCT benchmark split.
{summary.to_string()}

## Long-tailed class distribution (frame label occurrences)
{counts.to_string()}
Species crop counts: crop_counts.csv. Rare species may have little or zero support in held-out locations.
Training vocabulary is built from train crops only; unseen held-out species are reported by Member 2.

## Final stratification check
{balance_report.to_string(index=False)}
Rates count label presence per frame; multi-label class percentages need not sum to 100.
See class_balance_proportions.csv and species_crop_support.csv. Camera constraints can prevent exact balance.

## Label audit
Reviewed: {audit['reviewed']} / {audit['requested']}; observed label errors: {audit['label_errors']}.
One-sided 95% upper error bound: {audit['one_sided_95_upper_bound']}.
Sampling: one frame per randomly sampled train event. Same-camera dependence may make this bound optimistic.
Flagged training events are quarantined; original audit decisions remain recorded.

## Preprocessing
RGB, resize 224, ImageNet normalization. Random flip and brightness/contrast changes on train only.
All sections import image preprocessing and augmentation from preprocessing.py.

## Limitations and privacy
Curated US wildlife data; no claim of Saudi generalisation. The empty fraction must be measured on this subset.
Grayscale is an infrared proxy, not verified nighttime. Blur is a review flag; no image is discarded solely for low Laplacian variance. No GPS publication; keep photos and review galleries private.
Test accuracy confidence intervals belong to Member 2 after training, not this untrained data notebook.

## Team observations
{TEAM_NOTES or 'Required observations: add actual observations before submission.'}
"""
Path('runs/data/DATASET_CARD.md').write_text(card)
display(Markdown(card))


blur_check=pd.read_csv(DATA/'blur_review.csv').fillna('')
assert len(blur_check) and blur_check['reviewer'].str.strip().ne('').all() and blur_check['decision'].isin(['keep','flag']).all() and blur_check['notes'].str.strip().ne('').all(), 'Save a reviewer name, keep/flag decision and observation for every image in the image-quality review.'
assert TEAM_NOTES.strip(), 'Complete the dataset observations before continuing.'
training_defaults = {'project': 'camera-trap-phase1', 'backbone': 'microsoft/resnet-50', 'seed': 42, 'profile': 'full', 'data_dir': 'data/cct20_v1', 'output_dir': 'runs/data', 'max_images': 6000, 'download_workers': 12, 'max_image_side': 1024, 'image_size': 224, 'batch_size': 32, 'workers': 2, 'epochs': 12, 'patience': 4, 'warmup_epochs': 0, 'lr_scratch': 0.001, 'lr_frozen': 0.001, 'lr_full': 0.0001, 'weight_decay': 0.0, 'label_smoothing': 0.0, 'class_weighted_loss': False, 'wandb_mode': 'offline', 'target_animal_recall': 0.99, 'species_review_confidence': 0.6, 'bootstrap_repeats': 1000, 'audit_events': 100, 'acceptance': {'animal_recall_min': 0.99, 'empty_removal_min': 0.2, 'species_macro_f1_min': 0.65, 'species_ece_max': 0.1}}
training_defaults.update(cfg)
training_defaults.update(contract_version=2,preprocessing_sha256=digest(Path(preprocessing.__file__)),optimizer='Adam')
write_json('handoff/preprocessing.json',{'sha256':digest(Path(preprocessing.__file__)),'input_size':224,'library':'OpenCV + torchvision','train_only':['horizontal_flip','brightness_contrast']})
write_json('handoff/data_config.json', training_defaults)
write_json(DATA/'preparation_config.json', training_defaults)
Path('handoff/requirements_member1.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
write_json('handoff/runtime_member1.json', {'python':sys.version, 'platform':sys.platform})
files=[DATA/name for name in ['manifest.json','species.json','site_split.json','audit_report.json',
 'label_audit.csv','preparation_config.json','cleaning_report.json','class_counts.csv','crop_counts.csv','metadata_summary.json','class_balance_proportions.csv','species_crop_support.csv','inventory.csv','blur_review.csv']]
files += [Path('handoff/data_config.json'),Path('runs/data/DATASET_CARD.md'),Path('handoff/preprocessing.json')]
write_json('handoff/dataset_release.json', {'schema':2,'scope':'Phase 1 M1-M3',
 'data_dir':str(DATA),'audit_complete':True,'files':{str(f):digest(f) for f in files},
 'owner':'member1','source':'CCT20 human annotations','dataset_card':'runs/data/DATASET_CARD.md'})
print('Member 1 outputs are ready in:', PROJECT)


from pathlib import Path
import os,json,hashlib,sys,subprocess,platform,random
import numpy as np,pandas as pd
from IPython.display import display,Markdown
ROOT=Path.cwd() if Path.cwd().name=='CameraTrap_Phase1_Team' else Path.cwd()/'CameraTrap_Phase1_Team'
ROOT.mkdir(exist_ok=True);os.chdir(ROOT)
for name in ['handoff','runs','review','submission','docs']:Path(name).mkdir(exist_ok=True)
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def environment(member):
    Path(f'handoff/requirements_member{member}.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write_json(f'handoff/runtime_member{member}.json',{'python':sys.version,'platform':platform.platform()})
print('Workspace:',ROOT)


release=read_json('handoff/dataset_release.json')
assert release['schema']==2 and release['audit_complete'], 'Use the completed dataset release.'
for p,h in release['files'].items():assert Path(p).is_file() and digest(p)==h, f'Changed/missing: {p}'
DATA=Path(release['data_dir']);rows=read_json(DATA/'manifest.json')
assert rows and all(Path(r['path']).exists() for r in rows)
DATA_HASH=digest(DATA/'manifest.json')
CFG=read_json('handoff/data_config.json')
assert CFG['preprocessing_sha256']==digest(Path(preprocessing.__file__))
print('Frames:',len(rows),'| split sizes:',pd.Series([r['split'] for r in rows]).value_counts().to_dict())


import cv2, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import preprocessing
from preprocessing import Preprocessor, FrameDataset, load_rgb_sample


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def artifact_context(cfg, task):
    return dict(training_implementation=2, task=task, profile=cfg['profile'], image_size=cfg['image_size'],
                seed=cfg['seed'], manifest_sha256=digest(Path(cfg['data_dir'])/'manifest.json'),
                split_policy='camera-disjoint train / val_select / val_calibrate / test',
                preprocessing_sha256=cfg['preprocessing_sha256'],
                training_config={k:cfg[k] for k in ['optimizer','backbone','batch_size','epochs','patience','warmup_epochs',
                    'lr_scratch','lr_frozen','lr_full','weight_decay','label_smoothing','class_weighted_loss']})

def samples_for_task(rows,task,classes):
    samples=[]
    for r in rows:
        if task=='triage':
            samples.append(dict(r,label=r['animal'],sample_id=r['id'],crop=None))
        else:
            for i,b in enumerate(r['boxes']):
                samples.append(dict(r,label=classes.index(b['species']) if b['species'] in classes else -1,
                                    sample_id=f'{r["id"]}_{i}',crop=b['xyxy'],species=b['species']))
    return samples


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print('Device:',DEVICE)
if DEVICE=='cpu':print('Training device: CPU. GPU acceleration is unavailable in this runtime.')
WANDB_MODE=CFG['wandb_mode']
import wandb
if WANDB_MODE=='online':wandb.login()
REGIMES=['scratch']
OUT=Path('runs/member2');OUT.mkdir(parents=True,exist_ok=True)
CFG['output_dir']=str(OUT)
assert CFG['optimizer']=='Adam'
write_json(OUT/'config.json',CFG)
environment(2)


from torch import nn

class ConvNN(nn.Module):
    def __init__(self,num_classes,size=224,num_filters=16,num_neurons=100):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
        self.block_1=block(3,num_filters)
        self.block_2=block(num_filters,2*num_filters)
        self.block_3=block(2*num_filters,4*num_filters)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2)
        self.dropout=nn.Dropout(.2);self.relu=nn.ReLU();self.flat=nn.Flatten()
        h=size
        for _ in range(3):h=(h-3)//2+1
        self.fc1=nn.Linear(4*num_filters*h*h,num_neurons)
        self.fc2=nn.Linear(num_neurons,num_neurons)
        self.output=nn.Linear(num_neurons,num_classes)
    def forward(self,x):
        x=self.maxpool(self.block_1(x));x=self.maxpool(self.block_2(x));x=self.maxpool(self.block_3(x))
        x=self.flat(self.dropout(x));x=self.relu(self.fc1(x));x=self.relu(self.fc2(x))
        return self.output(x)

class TransferNet(nn.Module):
    def __init__(self,classes,pretrained=True):
        super().__init__()
        from transformers import AutoModelForImageClassification,ResNetConfig,ResNetForImageClassification
        self.net=(AutoModelForImageClassification.from_pretrained('microsoft/resnet-50') if pretrained
                  else ResNetForImageClassification(ResNetConfig()))
        self.net.classifier=nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(self.net.config.hidden_sizes[-1],classes))
        self.net.config.num_labels=classes
    def forward(self,x):return self.net(x).logits
    @property
    def classifier(self):return self.net.classifier

def make_model(regime,classes,pretrained=True,size=224):
    return ConvNN(classes,size) if regime=='scratch' else TransferNet(classes,pretrained)

def freeze_backbone(model,freeze):
    for p in model.net.base_model.parameters():p.requires_grad=not freeze
    for p in model.classifier.parameters():p.requires_grad=True


classes=['empty','animal']
example_model=make_model(REGIMES[0],len(classes),size=CFG['image_size'])
if REGIMES[0]!='scratch':freeze_backbone(example_model,True)
print(example_model)
print('Trainable parameters:',sum(p.numel() for p in example_model.parameters() if p.requires_grad))
example_model.eval()
with torch.no_grad():print('Forward shape:',example_model(torch.zeros(2,3,CFG['image_size'],CFG['image_size'])).shape)
del example_model


import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def ece(y, probabilities, bins=15):
    y=np.asarray(y);p=np.asarray(probabilities)
    if len(y)==0:return None
    conf=p.max(1);correct=p.argmax(1)==y
    indices=np.minimum((conf*bins).astype(int),bins-1)
    return float(sum(np.mean(indices==i)*abs(correct[indices==i].mean()-conf[indices==i].mean())
                     for i in range(bins) if np.any(indices==i)))


def fit_temperature(logits,y):
    logits=np.asarray(logits);y=np.asarray(y,dtype=int)
    if not len(y):raise ValueError('No known-class calibration examples')
    def nll(log_t):
        z=logits/np.exp(log_t)
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
    result=minimize_scalar(nll,bounds=(np.log(0.25),np.log(10)),method='bounded')
    return float(np.exp(result.x))


def wilson(k,n,z=1.959963984540054):
    if n==0:return [None,None]
    p=k/n;d=1+z*z/n
    centre=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,centre-half)),float(min(1,centre+half))]


def triage_metrics(y, animal_scores, threshold):
    y=np.asarray(y);pred=np.asarray(animal_scores)>=threshold
    pos=y==1;neg=y==0
    tp=int(np.sum(pred & pos));fn=int(np.sum(~pred & pos));tn=int(np.sum(~pred & neg));fp=int(np.sum(pred & neg))
    return dict(threshold=float(threshold),animal_recall=float(tp/pos.sum()) if pos.any() else None,
                empty_removal_rate=float(tn/neg.sum()) if neg.any() else None,
                animal_frames_lost=fn,animal_frames=int(pos.sum()),empty_frames=int(neg.sum()),
                removed_frames=tn+fn,tp=tp,tn=tn,fp=fp,fn=fn,
                animal_recall_wilson95_independence_assumption=wilson(tp,int(pos.sum())))


def choose_threshold(y, scores, target=0.99):
    if set(np.asarray(y).tolist())!={0,1}:raise ValueError('Calibration needs empty and animal labels')
    candidates=np.unique(np.r_[0.,scores,1.])
    feasible=[triage_metrics(y,scores,t) for t in candidates]
    feasible=[r for r in feasible if r['animal_recall']>=target]
    best=max(feasible,key=lambda r:(r['empty_removal_rate'],r['threshold']))
    best['selection_split']='val_calibrate'
    best['recall_target_is_empirical_not_guaranteed']=True
    return best


def classification_metrics(y,probs,classes):
    y=np.asarray(y);probs=np.asarray(probs);pred=probs.argmax(1);n=len(classes)
    support=np.bincount(y,minlength=n);supported=np.flatnonzero(support)
    report=classification_report(y,pred,labels=list(range(n)),target_names=classes,output_dict=True,zero_division=0)
    k=min(3,n)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),
       macro_f1_all_training_classes=float(f1_score(y,pred,labels=list(range(n)),average='macro',zero_division=0)),
       macro_f1_supported_classes=float(f1_score(y,pred,labels=supported,average='macro',zero_division=0)),
       top_k=k,top_k_accuracy=float(np.mean([label in indices for label,indices in zip(y,np.argsort(probs,axis=1)[:,-k:])])),
       ece=ece(y,probs),per_class=report,confusion_matrix=confusion_matrix(y,pred,labels=list(range(n))).tolist(),
       classes_absent_from_this_split=[classes[i] for i in range(n) if support[i]==0])


def camera_bootstrap(y,probs,cameras,repeats=1000,seed=42,threshold=None):
    y=np.asarray(y);p=np.asarray(probs);c=np.asarray(cameras);sites=np.unique(c)
    if len(sites)<2:return {'status':'not_estimable_fewer_than_two_cameras'}
    rng=np.random.default_rng(seed);acc=[];f1=[];rec=[];empty=[]
    for _ in range(repeats):
        idx=np.concatenate([np.flatnonzero(c==site) for site in rng.choice(sites,len(sites),replace=True)])
        yy=y[idx];pp=p[idx]
        acc.append(float(np.mean(pp.argmax(1)==yy)))
        f1.append(float(f1_score(yy,pp.argmax(1),labels=range(p.shape[1]),average='macro',zero_division=0)))
        if threshold is not None:
            m=triage_metrics(yy,pp[:,1],threshold)
            if m['animal_recall'] is not None:rec.append(m['animal_recall'])
            if m['empty_removal_rate'] is not None:empty.append(m['empty_removal_rate'])
    ci=lambda v:np.quantile(v,[.025,.975]).tolist() if v else [None,None]
    out=dict(unit='camera',cameras=len(sites),repeats=repeats,accuracy95=ci(acc),macro_f1_95=ci(f1),
             caveat='Intervals are descriptive and unstable when there are few held-out cameras.')
    if threshold is not None:out.update(animal_recall95=ci(rec),empty_removal95=ci(empty))
    return out


from scipy.special import softmax
import time,wandb

def loader(samples,train=False):
    gen=torch.Generator().manual_seed(CFG['seed'])
    return DataLoader(FrameDataset(samples,CFG['image_size'],train),batch_size=CFG['batch_size'],
                      shuffle=train,num_workers=0,generator=gen)

def predict(model,data):
    model.eval();all_logits=[];all_y=[]
    with torch.no_grad():
        for images,labels,_ in data:
            all_logits.append(model(images.to(DEVICE)).cpu().numpy());all_y.extend(labels.tolist())
    assert all_logits, 'No known-class samples in this split'
    return np.concatenate(all_logits),np.array(all_y,dtype=int)

def train_model(task,regime):
    seed_everything(CFG['seed'])
    classes=['empty','animal'] if task=='triage' else read_json(DATA/'species.json')
    samples=samples_for_task(rows,task,classes)
    train=[r for r in samples if r['split']=='train' and r['label']>=0]
    val=[r for r in samples if r['split']=='val_select' and r['label']>=0]
    calibrate=[r for r in samples if r['split']=='val_calibrate' and r['label']>=0]
    assert train and val and calibrate
    folder=OUT/task/regime;folder.mkdir(parents=True,exist_ok=True)
    context=artifact_context(CFG,task)
    if (folder/'summary.json').exists():
        saved=read_json(folder/'summary.json');assert saved['context']==context
        assert saved['checkpoint_sha256']==digest(folder/'best.pt')
        print('Reusing completed run:',task,regime);return saved
    assert not (folder/'best.pt').exists(), 'An unfinished checkpoint exists. Preserve it and use a new output directory for a fresh run.'
    model=make_model(regime,len(classes),size=CFG['image_size']).to(DEVICE)
    if regime!='scratch':freeze_backbone(model,regime=='frozen')
    assert CFG['optimizer']=='Adam', 'Only Adam is implemented'
    weights=None
    if CFG['class_weighted_loss']:
        counts=np.bincount([r['label'] for r in train],minlength=len(classes))
        assert (counts>0).all(), 'Every training class must have samples'
        weights=torch.tensor(len(train)/(len(classes)*counts),dtype=torch.float32,device=DEVICE)
    criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=CFG['label_smoothing'])
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=CFG['lr_'+regime],weight_decay=CFG['weight_decay'])
    run=wandb.init(project=CFG['project'],name=f'{task}-{regime}',group=DATA_HASH[:12],
                   mode=WANDB_MODE,dir=str(folder.resolve()),config=dict(CFG,task=task,regime=regime),reinit='finish_previous')
    best=-1;stale=0;history=[]
    for epoch in range(CFG['epochs']):
        warmup=max(0,int(CFG['warmup_epochs']))
        for group in optimizer.param_groups:
            group['lr']=CFG['lr_'+regime]*min(1.,(epoch+1)/max(1,warmup))
        model.train()
        if regime=='frozen':model.eval();model.classifier.train()
        total_loss=0;correct=0;total=0
        for images,labels,_ in loader(train,True):
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            logits=model(images)
            loss=criterion(logits,labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*len(labels)
            correct+=int((logits.argmax(1)==labels).sum());total+=len(labels)
        logits,y=predict(model,loader(val))
        m=classification_metrics(y,softmax(logits,axis=1),classes)
        val_loss=float(nn.functional.cross_entropy(torch.tensor(logits),torch.tensor(y)))
        row={'epoch':epoch+1,'learning_rate':optimizer.param_groups[0]['lr'],'train_loss':total_loss/total,'train_accuracy':correct/total,
             'val_loss':val_loss,'val_accuracy':m['accuracy'],'val_macro_f1':m['macro_f1_all_training_classes']}
        history.append(row);run.log(row);print(task,regime,row)
        if row['val_macro_f1']>best:
            best=row['val_macro_f1'];stale=0
            torch.save({'state_dict':model.state_dict(),'regime':regime,'task':task,'classes':classes,
              'image_size':CFG['image_size'],'epoch':epoch+1,'config':CFG,'context':context},folder/'best.pt')
        else:stale+=1
        if stale>=CFG['patience']:break
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    checkpoint=torch.load(folder/'best.pt',map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    z,y=predict(model,loader(calibrate));temperature=fit_temperature(z,y)
    before=softmax(z,axis=1);after=softmax(z/temperature,axis=1)
    policy={'temperature':temperature,'calibration_split':'val_calibrate',
            'species_review_confidence':CFG['species_review_confidence']}
    if task=='triage':
        policy.update(choose_threshold(y,after[:,1],CFG['target_animal_recall']))
        sweep=[triage_metrics(y,after[:,1],t) for t in np.linspace(0,1,101)]
        pd.DataFrame(sweep).to_csv(folder/'validation_threshold_sweep.csv',index=False)
    write_json(folder/'policy.json',policy)
    result={'task':task,'regime':regime,'context':context,'selection_score':best,
      'best_epoch':checkpoint['epoch'],'classes':classes,'checkpoint_sha256':digest(folder/'best.pt'),
      'wandb_run_id':run.id,'wandb_url':run.url if WANDB_MODE=='online' else None,
      'wandb_mode':WANDB_MODE,'calibration_ece_before':ece(y,before),'calibration_ece_after':ece(y,after),
      'train_n':len(train),'val_select_n':len(val),'val_calibrate_n':len(calibrate)}
    run.summary.update({'best_val_macro_f1':best,'calibration_ece_before':ece(y,before),'calibration_ece_after':ece(y,after)})
    run.finish();write_json(folder/'summary.json',result)
    return result


triage_runs=[train_model('triage',regime) for regime in REGIMES]
display(pd.DataFrame(triage_runs)[['task','regime','selection_score','best_epoch']])


species_runs=[train_model('species',regime) for regime in REGIMES]
display(pd.DataFrame(species_runs)[['task','regime','selection_score','best_epoch']])


selection={task:max([r for r in triage_runs+species_runs if r['task']==task],key=lambda r:r['selection_score'])['regime'] for task in ['triage','species']}
write_json(OUT/'selection.json',{'selected':selection,'criterion':'val_select macro-F1','dataset_hash':DATA_HASH})
print(selection)


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(task,regime):
    folder=OUT/task/regime;summary=read_json(folder/'summary.json')
    signature={'data':DATA_HASH,'checkpoint':digest(folder/'best.pt'),'policy':digest(folder/'policy.json')}
    if (folder/'test_metrics.json').exists():
        saved=read_json(folder/'test_metrics.json');assert saved['signature']==signature
        return saved
    checkpoint=torch.load(folder/'best.pt',map_location=DEVICE,weights_only=False)
    assert checkpoint['context']==artifact_context(CFG,task)
    classes=checkpoint['classes'];policy=read_json(folder/'policy.json')
    model=make_model(regime,len(classes),pretrained=False,size=CFG['image_size']).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    all_test=[r for r in samples_for_task(rows,task,classes) if r['split']=='test']
    known=[r for r in all_test if r['label']>=0];unknown=[r for r in all_test if r['label']<0]
    logits,y=predict(model,loader(known));raw=softmax(logits,axis=1);p=softmax(logits/policy['temperature'],axis=1)
    m=classification_metrics(y,p,classes)
    m.update(task=task,regime=regime,split='test',image_size=CFG['image_size'],policy=policy,
       signature=signature,context=checkpoint['context'],uncalibrated_ece=ece(y,raw),
       known_count=len(known),unknown_count=len(unknown),coverage=len(known)/len(all_test),
       unknown_species=sorted({r['species'] for r in unknown}),
       accuracy_counting_unknown_as_errors=float((p.argmax(1)==y).sum()/len(all_test)),
       accuracy_wilson95_independence_assumption=wilson(int((p.argmax(1)==y).sum()),len(y)),
       camera_bootstrap=camera_bootstrap(y,p,[r['camera'] for r in known],CFG['bootstrap_repeats'],CFG['seed'],
                                      policy['threshold'] if task=='triage' else None))
    if task=='triage':m['operating_point']=triage_metrics(y,p[:,1],policy['threshold'])
    m['per_camera']={camera:classification_metrics(y[idx],p[idx],classes)
      for camera in sorted({r['camera'] for r in known})
      for idx in [[i for i,r in enumerate(known) if r['camera']==camera]]}
    np.savez_compressed(folder/'test_probabilities.npz',y=y,probabilities=p,raw=raw)
    pred=p.argmax(1)
    pd.DataFrame([{'sample_id':r['sample_id'],'camera':r['camera'],'true':classes[y[i]],
      'prediction':classes[pred[i]],'confidence':float(p[i].max())} for i,r in enumerate(known)]).to_csv(folder/'test_predictions.csv',index=False)

    decisions=(p[:,1]>=policy['threshold']).astype(int) if task=='triage' else pred
    errors=np.flatnonzero(decisions!=y)
    chosen=sorted(errors,key=lambda i:float(p[i].max()),reverse=True)[:12]
    if chosen:
        fig,axes=plt.subplots((len(chosen)+3)//4,4,figsize=(12,3*((len(chosen)+3)//4)))
        for ax in np.asarray(axes).flat:ax.axis('off')
        for ax,i in zip(np.asarray(axes).flat,chosen):
            rgb=cv2.cvtColor(load_rgb_sample(known[i]),cv2.COLOR_BGR2RGB)
            ax.imshow(rgb);ax.set_title(f"{known[i]['sample_id'][:12]}\n{classes[y[i]]} -> {classes[decisions[i]]}",fontsize=8)
        fig.suptitle(f'{task}/{regime}: test errors, input {CFG["image_size"]}, threshold '+str(policy.get('threshold','argmax')))
        fig.tight_layout();fig.savefig(folder/'private_errors.jpg');plt.close(fig)
    run=wandb.init(project=CFG['project'],name=f'{task}-{regime}-test',group=DATA_HASH[:12],
      mode=WANDB_MODE,dir=str(folder.resolve()),config=dict(CFG,task=task,regime=regime,split='locked_test'),reinit='finish_previous')
    run.log({'test/accuracy':m['accuracy'],'test/macro_f1':m['macro_f1_all_training_classes'],
             'test/ece':m['ece'],'test/top_k_accuracy':m['top_k_accuracy']})
    run.log({'test/per_class':wandb.Table(dataframe=pd.DataFrame(m['per_class']).T.reset_index())})
    m['wandb_test_run_id']=run.id;m['wandb_test_url']=run.url if WANDB_MODE=='online' else None
    run.finish();write_json(folder/'test_metrics.json',m)
    return m

def show_results(task,regime):
    folder=OUT/task/regime;m=read_json(folder/'test_metrics.json')
    display(pd.DataFrame(m['per_class']).T)
    print('Accuracy:',m['accuracy'],'Top-k:',m['top_k_accuracy'],'ECE:',m['ece'])
    print('Camera bootstrap:',m['camera_bootstrap'],'Unknown crops:',m['unknown_count'])
    h=pd.read_csv(folder/'history.csv');h.plot(x='epoch',y=['train_loss','val_loss'],title=f'{task}/{regime}: train/val_select loss, {CFG["image_size"]}px, threshold N/A');plt.tight_layout();plt.savefig(folder/'learning_loss.png');plt.show()
    h.plot(x='epoch',y=['train_accuracy','val_accuracy'],title=f'{task}/{regime}: train/val_select accuracy, {CFG["image_size"]}px, argmax');plt.tight_layout();plt.savefig(folder/'learning_accuracy.png');plt.show()
    classes=read_json(folder/'summary.json')['classes']
    ConfusionMatrixDisplay(np.array(m['confusion_matrix']),display_labels=classes).plot(xticks_rotation=90)
    plt.title(f'{task}/{regime}: test, {CFG["image_size"]}px, argmax');plt.tight_layout();plt.savefig(folder/'confusion.png');plt.show()
    z=np.load(folder/'test_probabilities.npz');y=z['y']
    plt.plot([0,1],[0,1],'--',color='gray')
    for key,label in [('raw','Before calibration'),('probabilities','After calibration')]:
        p=z[key];conf=p.max(1);correct=p.argmax(1)==y;bins=np.minimum((conf*10).astype(int),9)
        points=[(conf[bins==b].mean(),correct[bins==b].mean()) for b in range(10) if np.any(bins==b)]
        plt.plot(*zip(*points),'o-',label=label)
    plt.xlabel('Confidence');plt.ylabel('Accuracy');plt.legend();plt.title(f'{task}/{regime}: test reliability, {CFG["image_size"]}px, argmax');plt.tight_layout();plt.savefig(folder/'reliability.png');plt.show()
    if (folder/'private_errors.jpg').exists():display(Image.open(folder/'private_errors.jpg'))
    if task=='triage':print('Operating point:',m['operating_point'])


for task in ['triage','species']:
    for regime in REGIMES:
        evaluate_model(task,regime)
        show_results(task,regime)


analysis=OUT/'ERROR_ANALYSIS.md'
if not analysis.exists():analysis.write_text('# Error analysis\n\nRequired evidence is missing: actual sample IDs, rare-class recall/support, false empty removals, calibration, plausible causes, and data improvement plan.\n')
SYNC_WANDB=False
if SYNC_WANDB:
    wandb.login()
    for p in OUT.glob('*/*/wandb/offline-run-*'):
        subprocess.run([sys.executable,'-m','wandb','sync',str(p)],check=True)


required=[OUT/'config.json',OUT/'selection.json']
for task in ['triage','species']:
    for regime in REGIMES:
        required += [OUT/task/regime/n for n in ['best.pt','summary.json','history.csv','policy.json','test_metrics.json','test_probabilities.npz','test_predictions.csv']]
write_json('handoff/member2_release.json',{'schema':2,'dataset_hash':DATA_HASH,'output_dir':str(OUT),'member':2,'files':{str(p):digest(p) for p in required}})
print('Ready: member2_release.json')


from pathlib import Path
import os,json,hashlib,sys,subprocess,platform,random
import numpy as np,pandas as pd
from IPython.display import display,Markdown
ROOT=Path.cwd() if Path.cwd().name=='CameraTrap_Phase1_Team' else Path.cwd()/'CameraTrap_Phase1_Team'
ROOT.mkdir(exist_ok=True);os.chdir(ROOT)
for name in ['handoff','runs','review','submission','docs']:Path(name).mkdir(exist_ok=True)
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def environment(member):
    Path(f'handoff/requirements_member{member}.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write_json(f'handoff/runtime_member{member}.json',{'python':sys.version,'platform':platform.platform()})
print('Workspace:',ROOT)


release=read_json('handoff/dataset_release.json')
assert release['schema']==2 and release['audit_complete'], 'Use the completed dataset release.'
for p,h in release['files'].items():assert Path(p).is_file() and digest(p)==h, f'Changed/missing: {p}'
DATA=Path(release['data_dir']);rows=read_json(DATA/'manifest.json')
assert rows and all(Path(r['path']).exists() for r in rows)
DATA_HASH=digest(DATA/'manifest.json')
CFG=read_json('handoff/data_config.json')
assert CFG['preprocessing_sha256']==digest(Path(preprocessing.__file__))
print('Frames:',len(rows),'| split sizes:',pd.Series([r['split'] for r in rows]).value_counts().to_dict())


import cv2, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import preprocessing
from preprocessing import Preprocessor, FrameDataset, load_rgb_sample


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def artifact_context(cfg, task):
    return dict(training_implementation=2, task=task, profile=cfg['profile'], image_size=cfg['image_size'],
                seed=cfg['seed'], manifest_sha256=digest(Path(cfg['data_dir'])/'manifest.json'),
                split_policy='camera-disjoint train / val_select / val_calibrate / test',
                preprocessing_sha256=cfg['preprocessing_sha256'],
                training_config={k:cfg[k] for k in ['optimizer','backbone','batch_size','epochs','patience','warmup_epochs',
                    'lr_scratch','lr_frozen','lr_full','weight_decay','label_smoothing','class_weighted_loss']})

def samples_for_task(rows,task,classes):
    samples=[]
    for r in rows:
        if task=='triage':
            samples.append(dict(r,label=r['animal'],sample_id=r['id'],crop=None))
        else:
            for i,b in enumerate(r['boxes']):
                samples.append(dict(r,label=classes.index(b['species']) if b['species'] in classes else -1,
                                    sample_id=f'{r["id"]}_{i}',crop=b['xyxy'],species=b['species']))
    return samples


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print('Device:',DEVICE)
if DEVICE=='cpu':print('Training device: CPU. GPU acceleration is unavailable in this runtime.')
WANDB_MODE=CFG['wandb_mode']
import wandb
if WANDB_MODE=='online':wandb.login()
REGIMES=['frozen', 'full']
OUT=Path('runs/member3');OUT.mkdir(parents=True,exist_ok=True)
CFG['output_dir']=str(OUT)
assert CFG['optimizer']=='Adam'
write_json(OUT/'config.json',CFG)
environment(3)


from torch import nn

class ConvNN(nn.Module):
    def __init__(self,num_classes,size=224,num_filters=16,num_neurons=100):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
        self.block_1=block(3,num_filters)
        self.block_2=block(num_filters,2*num_filters)
        self.block_3=block(2*num_filters,4*num_filters)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2)
        self.dropout=nn.Dropout(.2);self.relu=nn.ReLU();self.flat=nn.Flatten()
        h=size
        for _ in range(3):h=(h-3)//2+1
        self.fc1=nn.Linear(4*num_filters*h*h,num_neurons)
        self.fc2=nn.Linear(num_neurons,num_neurons)
        self.output=nn.Linear(num_neurons,num_classes)
    def forward(self,x):
        x=self.maxpool(self.block_1(x));x=self.maxpool(self.block_2(x));x=self.maxpool(self.block_3(x))
        x=self.flat(self.dropout(x));x=self.relu(self.fc1(x));x=self.relu(self.fc2(x))
        return self.output(x)

class TransferNet(nn.Module):
    def __init__(self,classes,pretrained=True):
        super().__init__()
        from transformers import AutoModelForImageClassification,ResNetConfig,ResNetForImageClassification
        self.net=(AutoModelForImageClassification.from_pretrained('microsoft/resnet-50') if pretrained
                  else ResNetForImageClassification(ResNetConfig()))
        self.net.classifier=nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(self.net.config.hidden_sizes[-1],classes))
        self.net.config.num_labels=classes
    def forward(self,x):return self.net(x).logits
    @property
    def classifier(self):return self.net.classifier

def make_model(regime,classes,pretrained=True,size=224):
    return ConvNN(classes,size) if regime=='scratch' else TransferNet(classes,pretrained)

def freeze_backbone(model,freeze):
    for p in model.net.base_model.parameters():p.requires_grad=not freeze
    for p in model.classifier.parameters():p.requires_grad=True


classes=['empty','animal']
example_model=make_model(REGIMES[0],len(classes),size=CFG['image_size'])
if REGIMES[0]!='scratch':freeze_backbone(example_model,True)
print(example_model)
print('Trainable parameters:',sum(p.numel() for p in example_model.parameters() if p.requires_grad))
example_model.eval()
with torch.no_grad():print('Forward shape:',example_model(torch.zeros(2,3,CFG['image_size'],CFG['image_size'])).shape)
del example_model


import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def ece(y, probabilities, bins=15):
    y=np.asarray(y);p=np.asarray(probabilities)
    if len(y)==0:return None
    conf=p.max(1);correct=p.argmax(1)==y
    indices=np.minimum((conf*bins).astype(int),bins-1)
    return float(sum(np.mean(indices==i)*abs(correct[indices==i].mean()-conf[indices==i].mean())
                     for i in range(bins) if np.any(indices==i)))


def fit_temperature(logits,y):
    logits=np.asarray(logits);y=np.asarray(y,dtype=int)
    if not len(y):raise ValueError('No known-class calibration examples')
    def nll(log_t):
        z=logits/np.exp(log_t)
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
    result=minimize_scalar(nll,bounds=(np.log(0.25),np.log(10)),method='bounded')
    return float(np.exp(result.x))


def wilson(k,n,z=1.959963984540054):
    if n==0:return [None,None]
    p=k/n;d=1+z*z/n
    centre=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,centre-half)),float(min(1,centre+half))]


def triage_metrics(y, animal_scores, threshold):
    y=np.asarray(y);pred=np.asarray(animal_scores)>=threshold
    pos=y==1;neg=y==0
    tp=int(np.sum(pred & pos));fn=int(np.sum(~pred & pos));tn=int(np.sum(~pred & neg));fp=int(np.sum(pred & neg))
    return dict(threshold=float(threshold),animal_recall=float(tp/pos.sum()) if pos.any() else None,
                empty_removal_rate=float(tn/neg.sum()) if neg.any() else None,
                animal_frames_lost=fn,animal_frames=int(pos.sum()),empty_frames=int(neg.sum()),
                removed_frames=tn+fn,tp=tp,tn=tn,fp=fp,fn=fn,
                animal_recall_wilson95_independence_assumption=wilson(tp,int(pos.sum())))


def choose_threshold(y, scores, target=0.99):
    if set(np.asarray(y).tolist())!={0,1}:raise ValueError('Calibration needs empty and animal labels')
    candidates=np.unique(np.r_[0.,scores,1.])
    feasible=[triage_metrics(y,scores,t) for t in candidates]
    feasible=[r for r in feasible if r['animal_recall']>=target]
    best=max(feasible,key=lambda r:(r['empty_removal_rate'],r['threshold']))
    best['selection_split']='val_calibrate'
    best['recall_target_is_empirical_not_guaranteed']=True
    return best


def classification_metrics(y,probs,classes):
    y=np.asarray(y);probs=np.asarray(probs);pred=probs.argmax(1);n=len(classes)
    support=np.bincount(y,minlength=n);supported=np.flatnonzero(support)
    report=classification_report(y,pred,labels=list(range(n)),target_names=classes,output_dict=True,zero_division=0)
    k=min(3,n)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),
       macro_f1_all_training_classes=float(f1_score(y,pred,labels=list(range(n)),average='macro',zero_division=0)),
       macro_f1_supported_classes=float(f1_score(y,pred,labels=supported,average='macro',zero_division=0)),
       top_k=k,top_k_accuracy=float(np.mean([label in indices for label,indices in zip(y,np.argsort(probs,axis=1)[:,-k:])])),
       ece=ece(y,probs),per_class=report,confusion_matrix=confusion_matrix(y,pred,labels=list(range(n))).tolist(),
       classes_absent_from_this_split=[classes[i] for i in range(n) if support[i]==0])


def camera_bootstrap(y,probs,cameras,repeats=1000,seed=42,threshold=None):
    y=np.asarray(y);p=np.asarray(probs);c=np.asarray(cameras);sites=np.unique(c)
    if len(sites)<2:return {'status':'not_estimable_fewer_than_two_cameras'}
    rng=np.random.default_rng(seed);acc=[];f1=[];rec=[];empty=[]
    for _ in range(repeats):
        idx=np.concatenate([np.flatnonzero(c==site) for site in rng.choice(sites,len(sites),replace=True)])
        yy=y[idx];pp=p[idx]
        acc.append(float(np.mean(pp.argmax(1)==yy)))
        f1.append(float(f1_score(yy,pp.argmax(1),labels=range(p.shape[1]),average='macro',zero_division=0)))
        if threshold is not None:
            m=triage_metrics(yy,pp[:,1],threshold)
            if m['animal_recall'] is not None:rec.append(m['animal_recall'])
            if m['empty_removal_rate'] is not None:empty.append(m['empty_removal_rate'])
    ci=lambda v:np.quantile(v,[.025,.975]).tolist() if v else [None,None]
    out=dict(unit='camera',cameras=len(sites),repeats=repeats,accuracy95=ci(acc),macro_f1_95=ci(f1),
             caveat='Intervals are descriptive and unstable when there are few held-out cameras.')
    if threshold is not None:out.update(animal_recall95=ci(rec),empty_removal95=ci(empty))
    return out


from scipy.special import softmax
import time,wandb

def loader(samples,train=False):
    gen=torch.Generator().manual_seed(CFG['seed'])
    return DataLoader(FrameDataset(samples,CFG['image_size'],train),batch_size=CFG['batch_size'],
                      shuffle=train,num_workers=0,generator=gen)

def predict(model,data):
    model.eval();all_logits=[];all_y=[]
    with torch.no_grad():
        for images,labels,_ in data:
            all_logits.append(model(images.to(DEVICE)).cpu().numpy());all_y.extend(labels.tolist())
    assert all_logits, 'No known-class samples in this split'
    return np.concatenate(all_logits),np.array(all_y,dtype=int)

def train_model(task,regime):
    seed_everything(CFG['seed'])
    classes=['empty','animal'] if task=='triage' else read_json(DATA/'species.json')
    samples=samples_for_task(rows,task,classes)
    train=[r for r in samples if r['split']=='train' and r['label']>=0]
    val=[r for r in samples if r['split']=='val_select' and r['label']>=0]
    calibrate=[r for r in samples if r['split']=='val_calibrate' and r['label']>=0]
    assert train and val and calibrate
    folder=OUT/task/regime;folder.mkdir(parents=True,exist_ok=True)
    context=artifact_context(CFG,task)
    if (folder/'summary.json').exists():
        saved=read_json(folder/'summary.json');assert saved['context']==context
        assert saved['checkpoint_sha256']==digest(folder/'best.pt')
        print('Reusing completed run:',task,regime);return saved
    assert not (folder/'best.pt').exists(), 'An unfinished checkpoint exists. Preserve it and use a new output directory for a fresh run.'
    model=make_model(regime,len(classes),size=CFG['image_size']).to(DEVICE)
    if regime!='scratch':freeze_backbone(model,regime=='frozen')
    assert CFG['optimizer']=='Adam', 'Only Adam is implemented'
    weights=None
    if CFG['class_weighted_loss']:
        counts=np.bincount([r['label'] for r in train],minlength=len(classes))
        assert (counts>0).all(), 'Every training class must have samples'
        weights=torch.tensor(len(train)/(len(classes)*counts),dtype=torch.float32,device=DEVICE)
    criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=CFG['label_smoothing'])
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=CFG['lr_'+regime],weight_decay=CFG['weight_decay'])
    run=wandb.init(project=CFG['project'],name=f'{task}-{regime}',group=DATA_HASH[:12],
                   mode=WANDB_MODE,dir=str(folder.resolve()),config=dict(CFG,task=task,regime=regime),reinit='finish_previous')
    best=-1;stale=0;history=[]
    for epoch in range(CFG['epochs']):
        warmup=max(0,int(CFG['warmup_epochs']))
        for group in optimizer.param_groups:
            group['lr']=CFG['lr_'+regime]*min(1.,(epoch+1)/max(1,warmup))
        model.train()
        if regime=='frozen':model.eval();model.classifier.train()
        total_loss=0;correct=0;total=0
        for images,labels,_ in loader(train,True):
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            logits=model(images)
            loss=criterion(logits,labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*len(labels)
            correct+=int((logits.argmax(1)==labels).sum());total+=len(labels)
        logits,y=predict(model,loader(val))
        m=classification_metrics(y,softmax(logits,axis=1),classes)
        val_loss=float(nn.functional.cross_entropy(torch.tensor(logits),torch.tensor(y)))
        row={'epoch':epoch+1,'learning_rate':optimizer.param_groups[0]['lr'],'train_loss':total_loss/total,'train_accuracy':correct/total,
             'val_loss':val_loss,'val_accuracy':m['accuracy'],'val_macro_f1':m['macro_f1_all_training_classes']}
        history.append(row);run.log(row);print(task,regime,row)
        if row['val_macro_f1']>best:
            best=row['val_macro_f1'];stale=0
            torch.save({'state_dict':model.state_dict(),'regime':regime,'task':task,'classes':classes,
              'image_size':CFG['image_size'],'epoch':epoch+1,'config':CFG,'context':context},folder/'best.pt')
        else:stale+=1
        if stale>=CFG['patience']:break
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    checkpoint=torch.load(folder/'best.pt',map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    z,y=predict(model,loader(calibrate));temperature=fit_temperature(z,y)
    before=softmax(z,axis=1);after=softmax(z/temperature,axis=1)
    policy={'temperature':temperature,'calibration_split':'val_calibrate',
            'species_review_confidence':CFG['species_review_confidence']}
    if task=='triage':
        policy.update(choose_threshold(y,after[:,1],CFG['target_animal_recall']))
        sweep=[triage_metrics(y,after[:,1],t) for t in np.linspace(0,1,101)]
        pd.DataFrame(sweep).to_csv(folder/'validation_threshold_sweep.csv',index=False)
    write_json(folder/'policy.json',policy)
    result={'task':task,'regime':regime,'context':context,'selection_score':best,
      'best_epoch':checkpoint['epoch'],'classes':classes,'checkpoint_sha256':digest(folder/'best.pt'),
      'wandb_run_id':run.id,'wandb_url':run.url if WANDB_MODE=='online' else None,
      'wandb_mode':WANDB_MODE,'calibration_ece_before':ece(y,before),'calibration_ece_after':ece(y,after),
      'train_n':len(train),'val_select_n':len(val),'val_calibrate_n':len(calibrate)}
    run.summary.update({'best_val_macro_f1':best,'calibration_ece_before':ece(y,before),'calibration_ece_after':ece(y,after)})
    run.finish();write_json(folder/'summary.json',result)
    return result


triage_runs=[train_model('triage',regime) for regime in REGIMES]
display(pd.DataFrame(triage_runs)[['task','regime','selection_score','best_epoch']])


species_runs=[train_model('species',regime) for regime in REGIMES]
display(pd.DataFrame(species_runs)[['task','regime','selection_score','best_epoch']])


selection={task:max([r for r in triage_runs+species_runs if r['task']==task],key=lambda r:r['selection_score'])['regime'] for task in ['triage','species']}
write_json(OUT/'selection.json',{'selected':selection,'criterion':'val_select macro-F1','dataset_hash':DATA_HASH})
print(selection)


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(task,regime):
    folder=OUT/task/regime;summary=read_json(folder/'summary.json')
    signature={'data':DATA_HASH,'checkpoint':digest(folder/'best.pt'),'policy':digest(folder/'policy.json')}
    if (folder/'test_metrics.json').exists():
        saved=read_json(folder/'test_metrics.json');assert saved['signature']==signature
        return saved
    checkpoint=torch.load(folder/'best.pt',map_location=DEVICE,weights_only=False)
    assert checkpoint['context']==artifact_context(CFG,task)
    classes=checkpoint['classes'];policy=read_json(folder/'policy.json')
    model=make_model(regime,len(classes),pretrained=False,size=CFG['image_size']).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    all_test=[r for r in samples_for_task(rows,task,classes) if r['split']=='test']
    known=[r for r in all_test if r['label']>=0];unknown=[r for r in all_test if r['label']<0]
    logits,y=predict(model,loader(known));raw=softmax(logits,axis=1);p=softmax(logits/policy['temperature'],axis=1)
    m=classification_metrics(y,p,classes)
    m.update(task=task,regime=regime,split='test',image_size=CFG['image_size'],policy=policy,
       signature=signature,context=checkpoint['context'],uncalibrated_ece=ece(y,raw),
       known_count=len(known),unknown_count=len(unknown),coverage=len(known)/len(all_test),
       unknown_species=sorted({r['species'] for r in unknown}),
       accuracy_counting_unknown_as_errors=float((p.argmax(1)==y).sum()/len(all_test)),
       accuracy_wilson95_independence_assumption=wilson(int((p.argmax(1)==y).sum()),len(y)),
       camera_bootstrap=camera_bootstrap(y,p,[r['camera'] for r in known],CFG['bootstrap_repeats'],CFG['seed'],
                                      policy['threshold'] if task=='triage' else None))
    if task=='triage':m['operating_point']=triage_metrics(y,p[:,1],policy['threshold'])
    m['per_camera']={camera:classification_metrics(y[idx],p[idx],classes)
      for camera in sorted({r['camera'] for r in known})
      for idx in [[i for i,r in enumerate(known) if r['camera']==camera]]}
    np.savez_compressed(folder/'test_probabilities.npz',y=y,probabilities=p,raw=raw)
    pred=p.argmax(1)
    pd.DataFrame([{'sample_id':r['sample_id'],'camera':r['camera'],'true':classes[y[i]],
      'prediction':classes[pred[i]],'confidence':float(p[i].max())} for i,r in enumerate(known)]).to_csv(folder/'test_predictions.csv',index=False)

    decisions=(p[:,1]>=policy['threshold']).astype(int) if task=='triage' else pred
    errors=np.flatnonzero(decisions!=y)
    chosen=sorted(errors,key=lambda i:float(p[i].max()),reverse=True)[:12]
    if chosen:
        fig,axes=plt.subplots((len(chosen)+3)//4,4,figsize=(12,3*((len(chosen)+3)//4)))
        for ax in np.asarray(axes).flat:ax.axis('off')
        for ax,i in zip(np.asarray(axes).flat,chosen):
            rgb=cv2.cvtColor(load_rgb_sample(known[i]),cv2.COLOR_BGR2RGB)
            ax.imshow(rgb);ax.set_title(f"{known[i]['sample_id'][:12]}\n{classes[y[i]]} -> {classes[decisions[i]]}",fontsize=8)
        fig.suptitle(f'{task}/{regime}: test errors, input {CFG["image_size"]}, threshold '+str(policy.get('threshold','argmax')))
        fig.tight_layout();fig.savefig(folder/'private_errors.jpg');plt.close(fig)
    run=wandb.init(project=CFG['project'],name=f'{task}-{regime}-test',group=DATA_HASH[:12],
      mode=WANDB_MODE,dir=str(folder.resolve()),config=dict(CFG,task=task,regime=regime,split='locked_test'),reinit='finish_previous')
    run.log({'test/accuracy':m['accuracy'],'test/macro_f1':m['macro_f1_all_training_classes'],
             'test/ece':m['ece'],'test/top_k_accuracy':m['top_k_accuracy']})
    run.log({'test/per_class':wandb.Table(dataframe=pd.DataFrame(m['per_class']).T.reset_index())})
    m['wandb_test_run_id']=run.id;m['wandb_test_url']=run.url if WANDB_MODE=='online' else None
    run.finish();write_json(folder/'test_metrics.json',m)
    return m

def show_results(task,regime):
    folder=OUT/task/regime;m=read_json(folder/'test_metrics.json')
    display(pd.DataFrame(m['per_class']).T)
    print('Accuracy:',m['accuracy'],'Top-k:',m['top_k_accuracy'],'ECE:',m['ece'])
    print('Camera bootstrap:',m['camera_bootstrap'],'Unknown crops:',m['unknown_count'])
    h=pd.read_csv(folder/'history.csv');h.plot(x='epoch',y=['train_loss','val_loss'],title=f'{task}/{regime}: train/val_select loss, {CFG["image_size"]}px, threshold N/A');plt.tight_layout();plt.savefig(folder/'learning_loss.png');plt.show()
    h.plot(x='epoch',y=['train_accuracy','val_accuracy'],title=f'{task}/{regime}: train/val_select accuracy, {CFG["image_size"]}px, argmax');plt.tight_layout();plt.savefig(folder/'learning_accuracy.png');plt.show()
    classes=read_json(folder/'summary.json')['classes']
    ConfusionMatrixDisplay(np.array(m['confusion_matrix']),display_labels=classes).plot(xticks_rotation=90)
    plt.title(f'{task}/{regime}: test, {CFG["image_size"]}px, argmax');plt.tight_layout();plt.savefig(folder/'confusion.png');plt.show()
    z=np.load(folder/'test_probabilities.npz');y=z['y']
    plt.plot([0,1],[0,1],'--',color='gray')
    for key,label in [('raw','Before calibration'),('probabilities','After calibration')]:
        p=z[key];conf=p.max(1);correct=p.argmax(1)==y;bins=np.minimum((conf*10).astype(int),9)
        points=[(conf[bins==b].mean(),correct[bins==b].mean()) for b in range(10) if np.any(bins==b)]
        plt.plot(*zip(*points),'o-',label=label)
    plt.xlabel('Confidence');plt.ylabel('Accuracy');plt.legend();plt.title(f'{task}/{regime}: test reliability, {CFG["image_size"]}px, argmax');plt.tight_layout();plt.savefig(folder/'reliability.png');plt.show()
    if (folder/'private_errors.jpg').exists():display(Image.open(folder/'private_errors.jpg'))
    if task=='triage':print('Operating point:',m['operating_point'])


for task in ['triage','species']:
    for regime in REGIMES:
        evaluate_model(task,regime)
        show_results(task,regime)


analysis=OUT/'ERROR_ANALYSIS.md'
if not analysis.exists():analysis.write_text('# Error analysis\n\nRequired evidence is missing: actual sample IDs, rare-class recall/support, false empty removals, calibration, plausible causes, and data improvement plan.\n')
SYNC_WANDB=False
if SYNC_WANDB:
    wandb.login()
    for p in OUT.glob('*/*/wandb/offline-run-*'):
        subprocess.run([sys.executable,'-m','wandb','sync',str(p)],check=True)


required=[OUT/'config.json',OUT/'selection.json']
for task in ['triage','species']:
    for regime in REGIMES:
        required += [OUT/task/regime/n for n in ['best.pt','summary.json','history.csv','policy.json','test_metrics.json','test_probabilities.npz','test_predictions.csv']]
write_json('handoff/member3_release.json',{'schema':2,'dataset_hash':DATA_HASH,'output_dir':str(OUT),'member':3,'files':{str(p):digest(p) for p in required}})
print('Ready: member3_release.json')


from pathlib import Path
import os,json,hashlib,sys,subprocess,platform,random
import numpy as np,pandas as pd
from IPython.display import display,Markdown
ROOT=Path.cwd() if Path.cwd().name=='CameraTrap_Phase1_Team' else Path.cwd()/'CameraTrap_Phase1_Team'
ROOT.mkdir(exist_ok=True);os.chdir(ROOT)
for name in ['handoff','runs','review','submission','docs']:Path(name).mkdir(exist_ok=True)
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def environment(member):
    Path(f'handoff/requirements_member{member}.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write_json(f'handoff/runtime_member{member}.json',{'python':sys.version,'platform':platform.platform()})
print('Workspace:',ROOT)


release=read_json('handoff/dataset_release.json')
assert release['schema']==2 and release['audit_complete'], 'Use the completed dataset release.'
for p,h in release['files'].items():assert Path(p).is_file() and digest(p)==h, f'Changed/missing: {p}'
DATA=Path(release['data_dir']);rows=read_json(DATA/'manifest.json')
assert rows and all(Path(r['path']).exists() for r in rows)
DATA_HASH=digest(DATA/'manifest.json')
CFG=read_json('handoff/data_config.json')
assert CFG['preprocessing_sha256']==digest(Path(preprocessing.__file__))
print('Frames:',len(rows),'| split sizes:',pd.Series([r['split'] for r in rows]).value_counts().to_dict())


environment(4)


from collections import defaultdict
from sklearn.metrics import accuracy_score,f1_score
checks=[]
def check(name,passed,details=''):
    checks.append({'check':name,'passed':bool(passed),'details':str(details)})
for key in ['id','camera','event','pixel_sha256']:
    groups=defaultdict(set)
    for r in rows:groups[r[key]].add(r['split'])
    bad=[k for k,v in groups.items() if len(v)>1]
    check(key+'_disjoint',not bad,len(bad))
check('unique_ids',len(rows)==len({r['id'] for r in rows}))
changed=[]
for r in rows:
    if hashlib.md5(Path(r['path']).read_bytes()).hexdigest()!=r['processed_file_md5']:changed.append(r['id'])
check('image_integrity',not changed,len(changed))
audit=read_json(DATA/'audit_report.json')
check('human_audit',audit['status']=='complete' and audit['unresolved_flagged_events']==0)
for split in ['train','val_select','val_calibrate','test']:
    check('empty_animal_support_'+split,{r['animal'] for r in rows if r['split']==split}=={0,1})
records=[];contexts=[]
for member in [2,3]:
    rel=read_json(f'handoff/member{member}_release.json')
    assert rel['schema']==2 and rel['dataset_hash']==DATA_HASH
    for p,h in rel['files'].items():assert Path(p).is_file() and digest(p)==h, f'Changed run artifact: {p}'
    out=Path(rel['output_dir'])
    for task in ['triage','species']:
        for regime in (['scratch'] if member==2 else ['frozen','full']):
            folder=out/task/regime;s=read_json(folder/'summary.json');m=read_json(folder/'test_metrics.json')
            z=np.load(folder/'test_probabilities.npz');y=z['y'];p=z['probabilities'];pred=p.argmax(1)
            confidence=p.max(1);correct=pred==y;bins=np.minimum((confidence*15).astype(int),14)
            ece_value=sum(np.mean(bins==i)*abs(correct[bins==i].mean()-confidence[bins==i].mean()) for i in range(15) if np.any(bins==i))
            f1=f1_score(y,pred,labels=list(range(p.shape[1])),average='macro',zero_division=0)
            check(f'metrics_{task}_{regime}',np.isclose(m['accuracy'],accuracy_score(y,pred)) and
                  np.isclose(m['macro_f1_all_training_classes'],f1) and np.isclose(m['ece'],ece_value))
            check(f'provenance_{task}_{regime}',m['context']==s['context'] and
                  m['signature']=={'data':DATA_HASH,'checkpoint':digest(folder/'best.pt'),'policy':digest(folder/'policy.json')})
            contexts.append({k:v for k,v in s['context'].items() if k!='task'})
            records.append({'member':member,'task':task,'regime':regime,'folder':str(folder),
              'val_macro_f1':s['selection_score'],'test_accuracy':m['accuracy'],
              'test_macro_f1':m['macro_f1_all_training_classes'],'top_k':m['top_k'],
              'top_k_accuracy':m['top_k_accuracy'],'ece':m['ece'],'raw_ece':m['uncalibrated_ece'],
              'coverage':m['coverage'],'unknown_crops':m['unknown_count'],'input_size':m['image_size'],
              'threshold':m['policy'].get('threshold','argmax'),
              'animal_recall':m.get('operating_point',{}).get('animal_recall'),
              'empty_removal':m.get('operating_point',{}).get('empty_removal_rate'),
              'accuracy_ci_camera':str(m['camera_bootstrap'].get('accuracy95')),
              'train_run_id':s['wandb_run_id'],'train_url':s.get('wandb_url') or '',
              'test_run_id':m['wandb_test_run_id'],'test_url':m.get('wandb_test_url') or ''})
check('common_training_configuration',all(c==contexts[0] for c in contexts))
comparison=pd.DataFrame(records)
comparison['selected_on_validation']=False
for task in ['triage','species']:
    idx=comparison.loc[comparison.task==task,'val_macro_f1'].idxmax()
    comparison.loc[idx,'selected_on_validation']=True

acceptance=[]
for row in comparison[comparison.selected_on_validation].to_dict('records'):
    targets=([('animal_recall','animal_recall_min','min'),('empty_removal','empty_removal_min','min')]
             if row['task']=='triage' else [('test_macro_f1','species_macro_f1_min','min'),('ece','species_ece_max','max')])
    for metric,key,direction in targets:
        value=row[metric];target=CFG['acceptance'][key]
        acceptance.append({'task':row['task'],'regime':row['regime'],'metric':metric,'value':value,'target':target,
                           'direction':direction,'met':bool(value>=target if direction=='min' else value<=target)})
pd.DataFrame(acceptance).to_csv('submission/acceptance.csv',index=False)
display(pd.DataFrame(acceptance))
comparison.to_csv('submission/comparison.csv',index=False)
pd.DataFrame(checks).to_csv('review/automatic_checks.csv',index=False)
display(pd.DataFrame(checks));display(comparison)
print('Model selection uses validation data only.')


import matplotlib.pyplot as plt
from PIL import Image
for task in ['triage','species']:
    table=comparison[comparison.task==task]
    table.plot.bar(x='regime',y=['test_accuracy','test_macro_f1','ece'],title=f'{task}: held-out test, {CFG["image_size"]}px, argmax')
    plt.tight_layout();plt.savefig('submission/'+task+'_comparison.png');plt.show()
for row in comparison[comparison.selected_on_validation].to_dict('records'):
    folder=Path(row['folder']);m=read_json(folder/'test_metrics.json')
    print(row['task'],row['regime'],'| Split:',m['split'],'Input:',m['image_size'],'Policy:',m['policy'])
    display(pd.DataFrame(m['per_class']).T)
    if (folder/'private_errors.jpg').exists():display(Image.open(folder/'private_errors.jpg'))


audit_path=Path('review/independent_audit.csv')
if not audit_path.exists():
    events={}
    for r in rows:
        if r['split']=='train':events.setdefault(r['event'],[]).append(r)
    rng=random.Random(2026)
    sample=[rng.choice(events[e]) for e in rng.sample(sorted(events),min(20,len(events)))]
    pd.DataFrame([{'id':r['id'],'label_correct':'','boxes_correct':'','person_visible':'','reviewer':'','notes':''} for r in sample]).to_csv(audit_path,index=False)
by_id={r['id']:r for r in rows}
from PIL import ImageDraw
for ident in pd.read_csv(audit_path).id:
    r=by_id[ident];im=Image.open(r['path']).convert('RGB');draw=ImageDraw.Draw(im)
    for b in r['boxes']:
        x1,y1,x2,y2=b['xyxy'];draw.rectangle((x1*im.width,y1*im.height,x2*im.width,y2*im.height),outline='lime',width=3)
    im.thumbnail((650,420));print(ident,r['labels'] or ['empty']);display(im)
issues=Path('review/issues.csv')
if not issues.exists():pd.DataFrame(columns=['issue_id','owner','problem','status','resolution','evidence']).to_csv(issues,index=False)
print('CSV decisions: label_correct=yes/no, boxes_correct=yes/no/not applicable, person_visible=yes/no.')


PROBE_SOURCE = 'import sys\nfrom pathlib import Path\nimport hashlib\nsys.path.insert(0, sys.argv[3])\nimport cv2, numpy as np, torch\nfrom PIL import Image\nfrom torchvision import transforms\nfrom torch.utils.data import DataLoader\nimport preprocessing\nfrom preprocessing import Preprocessor, FrameDataset, load_rgb_sample\n\nfrom torch import nn\n\nclass ConvNN(nn.Module):\n    ""\n    def __init__(self,num_classes,size=224,num_filters=16,num_neurons=100):\n        super().__init__()\n        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())\n        self.block_1=block(3,num_filters)\n        self.block_2=block(num_filters,2*num_filters)\n        self.block_3=block(2*num_filters,4*num_filters)\n        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2)\n        self.dropout=nn.Dropout(.2);self.relu=nn.ReLU();self.flat=nn.Flatten()\n        h=size\n        for _ in range(3):h=(h-3)//2+1\n        self.fc1=nn.Linear(4*num_filters*h*h,num_neurons)\n        self.fc2=nn.Linear(num_neurons,num_neurons)\n        self.output=nn.Linear(num_neurons,num_classes)\n    def forward(self,x):\n        x=self.maxpool(self.block_1(x));x=self.maxpool(self.block_2(x));x=self.maxpool(self.block_3(x))\n        x=self.flat(self.dropout(x));x=self.relu(self.fc1(x));x=self.relu(self.fc2(x))\n        return self.output(x)\n\nclass TransferNet(nn.Module):\n    ""\n    def __init__(self,classes,pretrained=True):\n        super().__init__()\n        from transformers import AutoModelForImageClassification,ResNetConfig,ResNetForImageClassification\n        self.net=(AutoModelForImageClassification.from_pretrained(\'microsoft/resnet-50\') if pretrained\n                  else ResNetForImageClassification(ResNetConfig()))\n        self.net.classifier=nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(self.net.config.hidden_sizes[-1],classes))\n        self.net.config.num_labels=classes\n    def forward(self,x):return self.net(x).logits\n    @property\n    def classifier(self):return self.net.classifier\n\ndef make_model(regime,classes,pretrained=True,size=224):\n    return ConvNN(classes,size) if regime==\'scratch\' else TransferNet(classes,pretrained)\n\ndef freeze_backbone(model,freeze):\n    for p in model.net.base_model.parameters():p.requires_grad=not freeze\n    for p in model.classifier.parameters():p.requires_grad=True\nif __name__==\'__main__\':\n    import sys,json\n    checkpoint,image=sys.argv[1:3]\n    torch.set_num_threads(2)\n    ckpt=torch.load(checkpoint,map_location=\'cpu\',weights_only=False)\n    assert hashlib.sha256(Path(preprocessing.__file__).read_bytes()).hexdigest()==ckpt[\'config\'][\'preprocessing_sha256\']\n    model=make_model(ckpt[\'regime\'],len(ckpt[\'classes\']),pretrained=False,size=ckpt[\'image_size\']).eval()\n    model.load_state_dict(ckpt[\'state_dict\'])\n    x,_,_=FrameDataset([{\'path\':image,\'label\':0,\'crop\':None}],ckpt[\'image_size\'])[0]\n    with torch.no_grad():y=model(x.unsqueeze(0))\n    assert y.shape==(1,len(ckpt[\'classes\'])) and torch.isfinite(y).all()\n    print(json.dumps({\'shape\':list(y.shape),\'finite\':True}))\n'
fresh=[]
sample=next(r['path'] for r in rows if r['split']=='train')
for row in comparison.to_dict('records'):
    checkpoint=Path(row['folder'])/'best.pt'
    try:
        proc=subprocess.run([sys.executable,'-c',PROBE_SOURCE,str(checkpoint),sample,str(Path(preprocessing.__file__).resolve().parent)],capture_output=True,text=True,timeout=180)
        fresh.append({'checkpoint':str(checkpoint),'sha256':digest(checkpoint),'passed':proc.returncode==0,
                      'stdout':proc.stdout[-1000:],'stderr':proc.stderr[-2000:]})
    except subprocess.TimeoutExpired:
        fresh.append({'checkpoint':str(checkpoint),'sha256':digest(checkpoint),'passed':False,'stderr':'timeout'})
write_json('review/fresh_process_results.json',fresh);display(pd.DataFrame(fresh)[['checkpoint','passed']])


links_path=Path('submission/wandb_links.csv')
if not links_path.exists():comparison[['task','regime','train_run_id','train_url','test_run_id','test_url']].to_csv(links_path,index=False)
TEAM={'member1':'','member2':'','member3':'','member4':''}
NOTES={'motivation':'','data_findings':'','model_comparison':'','error_analysis':'','data_improvement_plan':'','limitations':'','conclusion':''}
REPLAY={'reviewer':'','date':'','notebook':'','cells_executed':'','evidence':''}
REVIEWER=''
VISUAL_REVIEW_COMPLETE=False
REPOSITORY_URL='https://github.com/salehdev3/Capstone_5'
TEAM_NAME=''
WANDB_PROJECT_URL=''
DATASET_CARD_URL=''
write_json('submission/report_inputs.json',{'team':TEAM,'notes':NOTES,'replay':REPLAY,'repository':REPOSITORY_URL,'team_name':TEAM_NAME,'wandb_project':WANDB_PROJECT_URL,'dataset_card_url':DATASET_CARD_URL})


gaps=[]
automatic=pd.read_csv('review/automatic_checks.csv')
if not automatic.passed.all():gaps.append('Automatic integrity/metrics checks failed')
review=pd.read_csv('review/independent_audit.csv').fillna('')
valid=review.label_correct.eq('yes') & review.boxes_correct.isin(['yes','not applicable']) & review.person_visible.eq('no') & review.reviewer.str.strip().ne('')
if not valid.all():gaps.append('Independent audit pending or flagged')
issues=pd.read_csv('review/issues.csv').fillna('')
if not issues.empty and not (issues.status.eq('resolved') & issues.resolution.str.strip().ne('') & issues.evidence.str.strip().ne('')).all():gaps.append('Unresolved issues')
if len(fresh)!=6 or not all(r['passed'] for r in fresh):gaps.append('Fresh-process inference')
if not REVIEWER.strip() or not VISUAL_REVIEW_COMPLETE:gaps.append('Named visual review signoff')
for k,v in TEAM.items():
    if not v.strip():gaps.append('Team name: '+k)
for k,v in NOTES.items():
    if not v.strip():gaps.append('Report section: '+k)
if not all(v.strip() for v in REPLAY.values()):gaps.append('Clean notebook replay evidence')
for member in [2,3]:
    p=Path(f'runs/member{member}/ERROR_ANALYSIS.md')
    if not p.exists() or 'Required evidence is missing' in p.read_text():gaps.append(f'Actual error analysis member {member}')
card=Path(release['dataset_card']).read_text()
if 'Required observations:' in card:gaps.append('Dataset Card team observations')
blur=pd.read_csv(DATA/'blur_review.csv').fillna('')
if not (blur.reviewer.str.strip().ne('') & blur.decision.isin(['keep','flag']) & blur.notes.str.strip().ne('')).all():gaps.append('Blur review notes')
links=pd.read_csv(links_path).fillna('')
expected={(t,r) for t in ['triage','species'] for r in ['scratch','frozen','full']}
if len(links)!=6 or set(zip(links.task,links.regime))!=expected:gaps.append('Six W&B experiment rows')
for field in ['train_url','test_url']:
    if not links[field].str.match(r'^https://wandb\.ai/[^/]+/[^/]+/runs/[^/]+').all():gaps.append('W&B '+field)
for i in range(1,5):
    if not Path(f'handoff/requirements_member{i}.txt').exists():gaps.append(f'Environment member {i}')
NAMES=['Phase1_Final_Presentation.ipynb']
for name in NAMES:
    if not (Path('notebooks')/name).exists():gaps.append('Notebook '+name)
if not REPOSITORY_URL.startswith('https://github.com/'):gaps.append('GitHub repository URL')
if not TEAM_NAME.strip():gaps.append('Submission team name')
if not WANDB_PROJECT_URL.startswith('https://wandb.ai/'):gaps.append('W&B project URL for report Section 1')
if not DATASET_CARD_URL.startswith('https://'):gaps.append('Published Dataset Card URL for report Section 1')
STATUS='READY_FOR_TEAM_REVIEW' if not gaps else 'INCOMPLETE'
print(STATUS)
for gap in gaps:print('-',gap)
write_json('handoff/qa_release.json',{'schema':2,'dataset_hash':DATA_HASH,'status':STATUS,'gaps':gaps,
 'reviewer':REVIEWER,'input_hashes':{f'member{i}':digest(f'handoff/member{i}_release.json') for i in [2,3]},
 'scope':'data/metrics checks and fresh-process inference; full replay scope documented separately'})


def table(df):
    df=df.fillna('');clean=lambda v:str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(clean(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
sections=['# Project 5 — Phase 1 report','Status: '+STATUS,
 '## Section 1 links', 'Repository: '+REPOSITORY_URL,
 'W&B project: '+(WANDB_PROJECT_URL or '[pending]'),
 'Dataset Card: '+(DATASET_CARD_URL or '[pending]'),
 '## Submission', 'Compile the completed official Overleaf template as '+(TEAM_NAME or 'TeamName')+'_Phase1.pdf. Submit one PDF per team. Replace every grey prompt and complete every section and table. Verify instructor access to all three links. Every figure must state its split, input resolution and threshold (or argmax / not applicable). This Markdown file and the evidence archive are report inputs, not the final PDF.',
 'Scope: M1–M3 only. Frame empty/animal classification and species human-box crop classification.',
 '## Contributions',str(TEAM),'## Motivation',NOTES['motivation'] or '[pending]',
 '## Dataset Card',Path(release['dataset_card']).read_text(),NOTES['data_findings'] or '[pending]',
 '## Acceptance criteria',table(pd.DataFrame(acceptance)),
 '## Model comparison',table(comparison.drop(columns=['folder','train_url','test_url'])),
 'Selections above use validation macro-F1 only. Test is held out. Per-class reports and confidence intervals are included in results/.',
 NOTES['model_comparison'] or '[pending]','## Error analysis',NOTES['error_analysis'] or '[pending]']
for member in [2,3]:
    p=Path(f'runs/member{member}/ERROR_ANALYSIS.md')
    sections.append(p.read_text() if p.exists() else '[missing analysis]')
sections+=['## Data improvement plan',NOTES['data_improvement_plan'] or '[pending]',
 '## Limitations',NOTES['limitations'] or '[pending]',
 'The species task uses human-box crops. This does not evaluate a detector-classifier cascade. Camera-based confidence intervals use few sites and are descriptive.',
 '## Reproducibility',str(REPLAY),'Repository: '+(REPOSITORY_URL or '[pending]'),
 '## W&B',table(links),'## Conclusion',NOTES['conclusion'] or '[pending]',
 '## Outstanding evidence','\n'.join('- '+g for g in gaps) or 'No automated completeness gaps.']
Path('submission/REPORT_DRAFT.md').write_text('\n\n'.join(sections))
display(Markdown(Path('submission/REPORT_DRAFT.md').read_text()))


LOCAL_DATA_ROOT = Path(preprocessing.__file__).resolve().parent / 'data/team_collected_riyadh'
LOCAL_OUTPUT = Path('submission/local_riyadh')
LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)

def evaluate_local_riyadh(root):
    from PIL import Image
    import imagehash
    metadata = pd.read_csv(root / 'metadata_review.csv').fillna('')
    annotations = read_json(root / 'draft_boxes.json')['annotations']
    required = ['site_id', 'session_id', 'reviewer']
    ready = metadata.status.eq('verified')
    for field in required:
        ready &= metadata[field].astype(str).str.strip().ne('')
    if not len(metadata) or not ready.all():
        write_json(LOCAL_OUTPUT / 'status.json', {'status': 'PENDING_HUMAN_REVIEW',
                   'pending_images': metadata.loc[~ready, 'image_id'].tolist()})
        print('Local evaluation pending: verify labels and boxes, record reviewer, site and session in metadata_review.csv.')
        return
    assert all(a['status'] == 'verified' for a in annotations), 'Verify every animal box before evaluation.'
    assert len(metadata.image_id.unique()) == len(metadata)
    local_rows = []
    excluded = []
    seen = set()
    reference_hashes = {r['pixel_sha256'] for r in rows if 'pixel_sha256' in r}
    reference_phashes = [imagehash.hex_to_hash(r['phash']) for r in rows if r.get('phash')]
    local_phashes = []
    for row in metadata.to_dict('records'):
        path = root / row['filename']
        assert digest(path) == row['sha256'], 'Image changed after annotation: ' + row['image_id']
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            width, height = rgb.size
            pixels = hashlib.sha256(rgb.tobytes()).hexdigest()
            phash = imagehash.phash(rgb)
        if pixels in seen or pixels in reference_hashes or any(phash - old <= 2 for old in reference_phashes + local_phashes):
            excluded.append({'image_id': row['image_id'], 'reason': 'exact or near duplicate / potential overlap'})
            continue
        seen.add(pixels)
        local_phashes.append(phash)
        assert row['proposed_label'] in ['empty', 'animal']
        boxes = []
        for a in annotations:
            if a['image_id'] != row['image_id']:
                continue
            x1,y1,x2,y2 = a['xyxy_pixels']
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            boxes.append({'species': a['species'], 'xyxy': [x1/width,y1/height,x2/width,y2/height]})
        assert bool(boxes) == (row['proposed_label'] == 'animal'), 'Review missing or inconsistent boxes: ' + row['image_id']
        local_rows.append({'id': row['image_id'], 'path': str(path), 'camera': row['site_id'],
                           'event': row['site_id'] + '/' + row['session_id'], 'split': 'local_external_test',
                           'animal': int(row['proposed_label'] == 'animal'), 'boxes': boxes})
    write_json(LOCAL_OUTPUT / 'cleaning_report.json', {'submitted': len(metadata), 'retained': len(local_rows), 'excluded': excluded})
    assert local_rows, 'No local images remain after cleaning.'
    results = []
    for selected in comparison[comparison.selected_on_validation].to_dict('records'):
        folder = Path(selected['folder'])
        checkpoint = torch.load(folder/'best.pt', map_location=DEVICE, weights_only=False)
        assert checkpoint['config']['preprocessing_sha256'] == digest(preprocessing.__file__)
        classes = checkpoint['classes']
        task, regime = checkpoint['task'], checkpoint['regime']
        samples = samples_for_task(local_rows, task, classes)
        known = [r for r in samples if r['label'] >= 0]
        unknown = [r for r in samples if r['label'] < 0]
        policy = read_json(folder/'policy.json')
        result = {'task': task, 'regime': regime, 'split': 'local_external_test',
                  'input_resolution': checkpoint['image_size'], 'policy': policy,
                  'known_count': len(known), 'unknown_count': len(unknown),
                  'unknown_species': sorted({r['species'] for r in unknown}),
                  'checkpoint_sha256': digest(folder/'best.pt')}
        if known:
            model = make_model(regime, len(classes), pretrained=False, size=checkpoint['image_size']).to(DEVICE)
            model.load_state_dict(checkpoint['state_dict'])
            batches = DataLoader(FrameDataset(known, checkpoint['image_size'], False), batch_size=32, shuffle=False)
            logits, labels = predict(model, batches)
            probabilities = softmax(logits/policy['temperature'], axis=1)
            result['metrics_argmax'] = classification_metrics(labels, probabilities, classes)
            if task == 'triage':
                result['operating_point'] = triage_metrics(labels, probabilities[:,1], policy['threshold'])
            pd.DataFrame([{'image_id': r['sample_id'], 'true': classes[labels[i]],
                          'prediction': classes[probabilities[i].argmax()], 'confidence': float(probabilities[i].max())}
                         for i,r in enumerate(known)]).to_csv(LOCAL_OUTPUT/(task+'_predictions.csv'), index=False)
            del model
        run = wandb.init(project=CFG['project'], name=f'{task}-{regime}-local-riyadh',
                         mode=WANDB_MODE, dir=str(LOCAL_OUTPUT.resolve()),
                         config={'split':'local_external_test','task':task,'regime':regime,
                                 'input_resolution':checkpoint['image_size']}, reinit='finish_previous')
        run.log({'local/known_count':len(known),'local/unknown_count':len(unknown)})
        if known:
            run.log({'local/accuracy':result['metrics_argmax']['accuracy'],
                     'local/macro_f1':result['metrics_argmax']['macro_f1_all_training_classes']})
        result['wandb_run_id'] = run.id
        result['wandb_url'] = run.url if WANDB_MODE == 'online' else None
        run.finish()
        results.append(result)
    write_json(LOCAL_OUTPUT/'metrics.json', results)
    metadata.to_csv(LOCAL_OUTPUT/'collection_metadata.csv', index=False)
    (LOCAL_OUTPUT/'DATASET_CARD.md').write_text('# Team-collected Riyadh images\n\nPurpose: separate exploratory external evaluation. Models and thresholds were selected using CCT20 validation only. No local image is used for training or model selection.\n\nSource: team-provided photographs from Riyadh. Capture dates/times and site/session codes are recorded in collection_metadata.csv; blank times are unknown. Labels and boxes were proposed with assistance and verified by the named human reviewers.\n\nSubmitted images: '+str(len(metadata))+'; retained after duplicate screening: '+str(len(local_rows))+'. Exact and perceptual duplicates are quarantined. See cleaning_report.json.\n\nAll local sites and sessions belong exclusively to local_external_test. Image groups are correlated; this small, cat-dominated convenience sample is not representative of wildlife or camera-trap deployment. Unknown species are reported separately and excluded from known-species metrics.\n\nImages remain in private team storage. Confirm permission and anonymize any identifiable people before sharing. Bonus credit is assessed by the instructor.\n')
    write_json(LOCAL_OUTPUT/'status.json', {'status':'EVALUATED', 'split':'local_external_test'})
    display(pd.DataFrame([{k:v for k,v in r.items() if k not in ['metrics_argmax','policy','operating_point']} for r in results]))

if (LOCAL_DATA_ROOT/'metadata_review.csv').exists():
    evaluate_local_riyadh(LOCAL_DATA_ROOT)
else:
    write_json(LOCAL_OUTPUT/'status.json', {'status':'DATA_NOT_AVAILABLE'})
    print('Set LOCAL_DATA_ROOT to the private repository data/team_collected_riyadh folder, then rerun this cell.')


import shutil
repo=Path('submission/repository');repo.mkdir(exist_ok=True)
def collect(source,destination):
    source=Path(source)
    if source.exists():
        target=repo/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
collect(preprocessing.__file__,'preprocessing.py')
for name in NAMES:
    p=Path('notebooks')/name
    if p.exists():
        n=read_json(p)
        for cell in n['cells']:
            if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
        write_json(repo/'notebooks'/name,n)
for p in Path('handoff').glob('requirements_member*.txt'):collect(p,Path('environments')/p.name)
for p in Path('handoff').glob('runtime_member*.json'):collect(p,Path('environments')/p.name)
collect('handoff/data_config.json','config.json')
collect(release['dataset_card'],'DATASET_CARD.md')
for name in ['comparison.csv','acceptance.csv','REPORT_DRAFT.md','wandb_links.csv','report_inputs.json','triage_comparison.png','species_comparison.png']:
    collect(Path('submission')/name,name)
for name in ['automatic_checks.csv','independent_audit.csv','issues.csv','fresh_process_results.json']:
    collect(Path('review')/name,Path('review')/name)
for name in ['class_balance_proportions.csv','species_crop_support.csv','audit_report.json','blur_review.csv']:
    collect(DATA/name,Path('data_evidence')/name)
weights=[]
for row in comparison.to_dict('records'):
    folder=Path(row['folder']);destination=Path('results')/row['task']/row['regime']
    for name in ['summary.json','test_metrics.json','history.csv','policy.json','confusion.png','learning_loss.png','learning_accuracy.png','reliability.png','validation_threshold_sweep.csv']:
        collect(folder/name,destination/name)
    weights.append({'path':str(folder/'best.pt'),'sha256':digest(folder/'best.pt'),'access':'private team storage'})
write_json(repo/'weights_manifest.json',weights)
(repo/'.gitignore').write_text('data/\nruns/\nwandb/\n*.pt\n*.pth\n*.jpg\n*.jpeg\n.env\n__pycache__/\n.ipynb_checkpoints/\n')
(repo/'README.md').write_text("# Project 5 — Phase 1\n\nRun Phase1_Final_Presentation.ipynb in section order. Use the same Member 1 dataset release for all classification experiments. All sections run in the unified notebook workspace. Internet and Colab's torch/torchvision are required; use GPU for training.\n\nMember1 uses CCT20 from https://lila.science/datasets/caltech-camera-traps. Follow dataset licence and course non-redistribution policy. Do not commit images or raw archives. The Dataset Card records filtering and limitations.\n\nAll sections import the shared preprocessing.py module. Split is stratified by camera groups; bursts stay together. One shared config defines all six experiments. Validation selects models; calibration chooses temperature/triage threshold; test is evaluated once.\n\nSee environments/ for the actual resolved package versions and Python/platform of each member. Install one role's environment in a compatible runtime. No single cross-platform lock is claimed. The initial install cells describe required packages; actual resolved pins are captured after installation.\n\nReport and metrics are from actual execution, with pending sections visible. Test accuracy intervals include camera bootstrap and Wilson (the latter assumes independent samples). Human audit, error discussion, clean replay and W&B sync require team action. Fresh-process inference alone is not full training reproduction.\n\nWeights remain private; weights_manifest.json provides hashes and paths. Repository notebooks have outputs stripped; keep executed copies privately for assessment. This repository covers M1–M3 only, without detector training or deployment.\n")
assert not list(repo.rglob('*.pt')) and not list(repo.rglob('*.jpg'))
write_json('handoff/integration_release.json',{'schema':2,'scope':'Phase 1','status':STATUS,'gaps':gaps,
 'dataset_hash':DATA_HASH,'report_sha256':digest(repo/'REPORT_DRAFT.md'),'repository':REPOSITORY_URL})
archive=shutil.make_archive(str(ROOT.parent/'Phase1_Submission'),'zip',root_dir=repo)
print('Prepared:',repo.resolve(),'|',archive)
print('Submission archive:', archive)

for evidence in Path('submission/local_riyadh').glob('*'):
    if evidence.is_file():collect(evidence,Path('local_riyadh')/evidence.name)
archive=shutil.make_archive(str(ROOT.parent/'Phase1_Submission'),'zip',root_dir=repo)

