"""CPU integration: original M4 triage, Phase-1 species crops and frozen Recovery v2 masks."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import shutil
import time
import urllib.request
import cv2
import numpy as np
import torch

PREPROCESSING_SHA256 = '87e3366dd7946e566c3a9238095748eb698ed0b57e2a9a5699d49817c440a5eb'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_tensor(bgr, box, size):
    """Match the Phase-1 preprocessing.py crop, resize and ImageNet normalization."""
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('Invalid normalized crop box')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('Crop box must be normalized and nonempty')
    h, w = bgr.shape[:2]
    dx, dy = (x2-x1)*.05, (y2-y1)*.05
    left, top = max(0, int((x1-dx)*w)), max(0, int((y1-dy)*h))
    right = min(w, max(left+1, int((x2+dx)*w)))
    bottom = min(h, max(top+1, int((y2+dy)*h)))
    rgb = cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    ih, iw = rgb.shape[:2]
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA if ih > size or iw > size else cv2.INTER_LINEAR)
    arr = (rgb.astype(np.float32)/255.-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2,0,1).float()


def match_masks(boxes, mask_boxes, threshold=.5):
    """One mask per detection, matched by predicted box IoU; no reference input."""
    if not boxes or not mask_boxes:
        return []
    a, b = np.asarray(boxes), np.asarray(mask_boxes)
    lo = np.maximum(a[:,None,:2], b[None,:,:2])
    hi = np.minimum(a[:,None,2:], b[None,:,2:])
    inter = np.maximum(hi-lo, 0).prod(-1)
    area_a = np.maximum(a[:,2:]-a[:,:2], 0).prod(-1)
    area_b = np.maximum(b[:,2:]-b[:,:2], 0).prod(-1)
    overlap = inter/np.maximum(area_a[:,None]+area_b[None,:]-inter, 1e-12)
    proposals = sorted([(float(overlap[i,j]),i,j) for i,j in zip(*np.where(overlap>=threshold))],reverse=True)
    used_a, used_b, matches = set(), set(), []
    for _,i,j in proposals:
        if i not in used_a and j not in used_b:
            matches.append((i,j));used_a.add(i);used_b.add(j)
    return matches


def aggregate_events(frames):
    events=[]
    for event in sorted({str(r['event']) for r in frames}):
        group=[r for r in frames if str(r['event'])==event]
        if not any(r['count'] for r in group):
            continue
        species_counts={}
        for frame in group:
            counts={}
            for obj in frame['objects']:
                counts[obj['species']]=counts.get(obj['species'],0)+1
            for species,count in counts.items():
                species_counts[species]=max(count,species_counts.get(species,0))
        timestamps=[str(r['timestamp']) for r in group if r.get('timestamp')]
        events.append(dict(event=event,time=min(timestamps,default=None),
            count=max(r['count'] for r in group),species_counts=species_counts,
            count_definition='maximum simultaneous detections; not unique individuals',
            species_counts_definition='per-species maximum over frames; not additive to event count',
            review=any(o['review'] for r in group for o in r['objects'])))
    return events


class IntegratedPipeline:
    def __init__(self, release, species_weights, species_policy, output_dir, threads=4):
        self.release, self.output_dir = Path(release), Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        species_weights, species_policy = Path(species_weights), Path(species_policy)
        missing=[str(p) for p in (species_weights,species_policy) if not p.is_file()]
        if missing:
            status=dict(ready=False,missing_files=missing,reason='A trained Phase-1 species classifier and its matching calibration policy are required. Detector/SAM weights cannot substitute for this classifier.')
            (self.output_dir/'integration_readiness.json').write_text(json.dumps(status,indent=2))
            raise FileNotFoundError(status['reason']+'\nMissing:\n'+'\n'.join(missing))
        # Validate classifier before downloading/loading SAM.
        checkpoint=torch.load(species_weights,map_location='cpu',weights_only=True)
        if checkpoint.get('task')!='species' or checkpoint.get('regime') not in ('frozen','full'):
            raise ValueError('Expected a Phase-1 frozen/full species checkpoint, not a YOLO detector')
        self.classes=checkpoint['classes']
        if not isinstance(self.classes,list) or len(self.classes)<2 or not all(isinstance(c,str) for c in self.classes):
            raise ValueError('Checkpoint must provide its original ordered species classes')
        if checkpoint.get('config',{}).get('preprocessing_sha256')!=PREPROCESSING_SHA256:
            raise ValueError('Checkpoint preprocessing differs or is unspecified; supply its original preprocessing before integration')
        self.size=int(checkpoint['image_size'])
        if self.size<=0:
            raise ValueError('Invalid classifier image size')
        self.species_policy=json.loads(species_policy.read_text())
        self.temperature=float(self.species_policy['temperature'])
        self.review_confidence=float(self.species_policy.get('species_review_confidence',.6))
        if not math.isfinite(self.temperature) or self.temperature<=0 or not 0<=self.review_confidence<=1:
            raise ValueError('Invalid species calibration policy')
        from transformers import ResNetConfig, ResNetForImageClassification
        class TransferNet(torch.nn.Module):
            def __init__(self,count):
                super().__init__()
                self.net=ResNetForImageClassification(ResNetConfig())
                self.net.classifier=torch.nn.Sequential(torch.nn.Flatten(start_dim=1),torch.nn.Linear(self.net.config.hidden_sizes[-1],count))
                self.net.config.num_labels=count
            def forward(self,x):
                return self.net(x).logits
        torch.set_num_threads(threads)
        self.classifier=TransferNet(len(self.classes))
        self.classifier.load_state_dict(checkpoint['state_dict'],strict=True)
        self.classifier.float().cpu().eval()
        policy_path=self.release/'refinement_policy.json'
        self.policy=json.loads(policy_path.read_text())
        if self.policy.get('version')!='recovery_v2':
            raise ValueError('Install Recovery v2 before integration')
        for file_key,sha_key in [('detector_file','detector_sha256'),('model_file','model_sha256')]:
            if digest(self.release/self.policy[file_key])!=self.policy[sha_key]:
                raise ValueError('Released model checksum mismatch: '+file_key)
        # Verify the exact published inference helper as well as model weights.
        checks=json.loads((self.release/'artifact_checksums.json').read_text())
        helper=self.release/'predict_refined.py'
        if digest(helper)!=checks['predict_refined.py']:
            raise ValueError('Released inference helper checksum mismatch')
        spec=importlib.util.spec_from_file_location('recovery_v2_runtime',helper)
        self.runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.runtime)
        sam_path=self.release/self.policy['sam_file']
        if not sam_path.exists():
            for candidate in (self.release.parent/'Segmentation_M4_Guided'/self.policy['sam_file'],Path.cwd()/self.policy['sam_file'],Path('/content')/self.policy['sam_file']):
                if candidate.is_file() and digest(candidate)==self.policy['sam_sha256']:
                    shutil.copyfile(candidate,sam_path);break
        if not sam_path.exists():
            temporary=sam_path.with_suffix('.download')
            urllib.request.urlretrieve(self.policy['sam_url'],temporary)
            if digest(temporary)!=self.policy['sam_sha256']:
                raise ValueError('Downloaded SAM checksum mismatch')
            temporary.replace(sam_path)
        if digest(sam_path)!=self.policy['sam_sha256']:
            raise ValueError('SAM checksum mismatch')
        from ultralytics import YOLO,SAM
        self.detector=YOLO(str(self.release/self.policy['detector_file']))
        self.segmenter=YOLO(str(self.release/self.policy['model_file']))
        self.sam=SAM(str(sam_path))
        self.policy_sha256=digest(policy_path)
        provenance=dict(ready=True,backend='pytorch_cpu',classifier_sha256=digest(species_weights),
            species_policy_sha256=digest(species_policy),species_regime=checkpoint['regime'],
            preprocessing_sha256=PREPROCESSING_SHA256,classes=self.classes,segmentation_policy_sha256=self.policy_sha256,
            detector_conf=self.policy['prompt_floor'],detector_nms=self.policy['detector_nms'],
            count_scope='all detector outputs at the original prompt floor; independent of accepted segmentation mask count',
            execution_scope='Detection plus classification plus unchanged recovery-v2 inference. The segmentation helper reruns M4 detection; total timing includes that cost.',
            validation_scope='Full integrated species/count/latency performance has not been measured by module initialization.')
        (self.output_dir/'integration_readiness.json').write_text(json.dumps(provenance,indent=2))
        (self.output_dir/'integration_provenance.json').write_text(json.dumps(provenance,indent=2))

    def classify_crop(self,bgr,box):
        x=crop_tensor(bgr,box,self.size)
        with torch.inference_mode():
            prob=torch.softmax(self.classifier(x.unsqueeze(0))/self.temperature,dim=1)[0]
        if not torch.isfinite(prob).all():
            raise ValueError('Classifier produced nonfinite probabilities')
        score,idx=prob.max(0)
        return self.classes[int(idx)],float(score)

    def process_frame(self,path,metadata):
        if not metadata.get('id') or not metadata.get('event'):
            raise ValueError('Each frame requires id and event')
        started=time.perf_counter();bgr=cv2.imread(str(path))
        if bgr is None:
            raise ValueError('Unreadable image: '+str(path))
        det=self.detector.predict(bgr,conf=self.policy['prompt_floor'],iou=self.policy['detector_nms'],
            imgsz=self.policy['detector_input'],device='cpu',verbose=False)[0]
        boxes=det.boxes.xyxyn.cpu().numpy().tolist()
        confidences=det.boxes.conf.cpu().numpy().tolist()
        detected=time.perf_counter();objects=[]
        for box,confidence in zip(boxes,confidences):
            species,score=self.classify_crop(bgr,box)
            objects.append(dict(box=box,species=species,confidence=score,detector_confidence=float(confidence),
                review=score<self.review_confidence or confidence<self.policy['conf'],
                relative_body_area=None,mask_found=False))
        classified=time.perf_counter()
        segmentation=self.runtime.predict_image(self.detector,self.segmenter,self.sam,path,self.policy,'cpu')
        pairs=match_masks(boxes,[o['box_xyxy_normalized'] for o in segmentation['objects']])
        for i,j in pairs:
            mask_obj=segmentation['objects'][j]
            mask=self.runtime.render([mask_obj['polygon_normalized']],bgr.shape[1],bgr.shape[0])
            objects[i].update(relative_body_area=float(mask.mean()),mask_found=True,
                polygon_normalized=mask_obj['polygon_normalized'],sam_quality=mask_obj['sam_quality'])
        for obj in objects:
            if not obj['mask_found']:obj['review']=True
        ended=time.perf_counter()
        return dict(id=metadata['id'],event=str(metadata['event']),timestamp=metadata.get('timestamp'),objects=objects,
            count=len(objects),segmentation=dict(mask_count=segmentation['mask_count'],matched_masks=len(pairs),
            unmatched_masks=segmentation['mask_count']-len(pairs),fallback_used=segmentation['fallback_used'],
            foreground_area_fraction=segmentation['foreground_area_fraction'],policy_sha256=self.policy_sha256),
            timing=dict(detection=detected-started,classification=classified-detected,segmentation=ended-classified,total=ended-started))

    def process_batch(self,paths,metadata):
        paths,metadata=list(paths),list(metadata)
        if len(paths)!=len(metadata):
            raise ValueError('One metadata record is required per image')
        if not all(isinstance(m,dict) and m.get('id') and m.get('event') for m in metadata):
            raise ValueError('Each metadata record requires id and event')
        frames=[self.process_frame(path,meta) for path,meta in zip(paths,metadata)]
        return dict(events=aggregate_events(frames),frames=frames)
