"""Fixed-shape FP32 detector equivalence; evaluation only, never threshold tuning."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, platform
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression

MANIFEST_SHA = 'd3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'
DETECTOR_SHA = 'ac38aaffe735bf7a6666866d2410f3601a93c6f9829ae6f68b7cd0117d1135e8'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))

def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1,4), np.asarray(b, dtype=float).reshape(-1,4)
    inter = np.maximum(0, np.minimum(a[:,None,2:], b[None,:,2:]) - np.maximum(a[:,None,:2], b[None,:,:2])).prod(2)
    aa = np.maximum(0,a[:,2:]-a[:,:2]).prod(1)
    bb = np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    return inter / np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def score_frame(pred, gt):
    # Confidence-ordered greedy one-to-one matching, IoU >= .5, one animal class.
    overlap=iou_matrix(pred[:,:4],gt);used=set();tp=0
    for index in np.argsort(-pred[:,4],kind='stable'):
        candidates=[j for j in range(len(gt)) if j not in used and overlap[index,j]>=.5]
        if candidates:
            chosen=max(candidates,key=lambda j:overlap[index,j]);used.add(chosen);tp+=1
    return tp,len(pred)-tp,len(gt)-tp

def aggregate(records, backend):
    tp=sum(r[backend+'_tp'] for r in records);fp=sum(r[backend+'_fp'] for r in records);fn=sum(r[backend+'_fn'] for r in records)
    animal=[r for r in records if r['animal']];empty=[r for r in records if not r['animal']]
    p=tp/max(tp+fp,1);rec=tp/max(tp+fn,1)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=rec,f1=2*p*rec/max(p+rec,1e-12),
        animal_recall=sum(r[backend+'_count']>0 for r in animal)/len(animal) if animal else None,
        empty_removal=sum(r[backend+'_count']==0 for r in empty)/len(empty) if empty else None,
        animal_frames=len(animal),empty_frames=len(empty),frames=len(records))

def run(weights, onnx_path, manifest, images, output, threads=4, subset_ids=None):
    weights,onnx_path,manifest,images,output=map(Path,(weights,onnx_path,manifest,images,output))
    assert digest(weights)==DETECTOR_SHA,'Detector checkpoint differs'
    assert digest(manifest)==MANIFEST_SHA,'Frozen manifest differs'
    rows=[r for r in json.loads(manifest.read_text()) if r['split']=='val_select' and not(r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes']))]
    assert len(rows)==1208 and len({r['id'] for r in rows})==1208
    full_count=len(rows)
    if subset_ids is not None:
        chosen=json.loads(Path(subset_ids).read_text());assert len(chosen)==len(set(chosen)) and set(chosen)<={r['id'] for r in rows}
        rows=[r for r in rows if r['id'] in set(chosen)]
        assert rows
    missing=[r['id'] for r in rows if not (images/(r['id']+'.jpg')).is_file()]
    if missing:raise FileNotFoundError(f'{len(missing)} original validation images missing; first: {missing[:3]}')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(threads)
    native=YOLO(str(weights));assert native.names=={0:'animal'} and native.task=='detect'
    native.model.fuse(verbose=False);native.model.float().cpu().eval()
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,3,640,640]
    settings=dict(split='val_select',frames=len(rows),full_retained_validation_frames=full_count,full_validation_evaluated=len(rows)==full_count,imgsz=640,batch=1,precision='float32',confidence=.045,nms_iou=.3,max_det=300,
        preprocessing='Identical BGR decode; LetterBox 640x640 auto=False scaleup=True; RGB CHW contiguous float32 /255. Equivalent to rect=False, not default native rectangular inference.',
        metric_matching='Confidence-ordered greedy one-to-one animal-box IoU >= 0.5; same function and references for both backends.',
        decision_tolerance=dict(same_counts_and_classes=True,matched_iou_min=.999,box_max_abs_pixels=.1,confidence_max_abs=.0001),
        metric_absolute_tolerance=1e-6,independent_test=False,
        scope='Export equivalence on the explicitly recorded selection-validation IDs (see per_image.csv and full_validation_evaluated); images are pixel-verified; no model, threshold or acceptance-target changes. Not a new test score, timing benchmark, or segmentation/classifier parity.',
        manifest_sha256=digest(manifest),detector_sha256=digest(weights),onnx_sha256=digest(onnx_path))
    save(output/'protocol.json',settings)
    letterbox=LetterBox(new_shape=(640,640),auto=False,scaleup=True,stride=32)
    records=[];predictions=[];raw_max=0.;raw_all=True
    for i,r in enumerate(rows,1):
        path=images/(r['id']+'.jpg');raw=path.read_bytes();bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        assert bgr is not None and bgr.shape[:2]==(r['height'],r['width'])
        # Frozen decoded pixels, not merely filenames, bind the evaluation dataset.
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256'],'Image pixels differ: '+r['id']
        padded=letterbox(image=bgr)
        x=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
        with torch.inference_mode():
            result=native.model(torch.from_numpy(x));raw_native=result[0] if isinstance(result,(tuple,list)) else result
        raw_onnx=session.run(None,{session.get_inputs()[0].name:x})[0]
        assert tuple(raw_native.shape)==raw_onnx.shape and np.isfinite(raw_onnx).all()
        rn=raw_native.cpu().numpy();error=float(np.abs(rn-raw_onnx).max());raw_max=max(raw_max,error)
        raw_ok=bool(np.allclose(rn,raw_onnx,rtol=1e-3,atol=1e-3));raw_all &= raw_ok
        predictions_here=[]
        for prediction in (raw_native,torch.from_numpy(raw_onnx)):
            predictions_here.append(non_max_suppression(prediction.clone(),conf_thres=.045,iou_thres=.3,classes=None,agnostic=False,max_det=300,nc=1)[0].cpu().numpy())
        a,b=predictions_here
        overlap=iou_matrix(a[:,:4],b[:,:4]);ia,ib=linear_sum_assignment(-overlap)
        matched_ious=overlap[ia,ib];box_error=float(np.abs(a[ia,:4]-b[ib,:4]).max()) if len(ia) else 0.
        conf_error=float(np.abs(a[ia,4]-b[ib,4]).max()) if len(ia) else 0.
        matched=int(((matched_ious>=.999)&(a[ia,5]==b[ib,5])).sum())
        passed=bool(len(a)==len(b)==matched and box_error<=.1 and conf_error<=.0001)
        # Convert frozen normalized GT to the same square-letterbox coordinate space.
        gain=min(640/r['width'],640/r['height']);padx=round((640-round(r['width']*gain))/2-.1);pady=round((640-round(r['height']*gain))/2-.1)
        gt=np.array([v['xyxy'] for v in r['boxes']],dtype=float).reshape(-1,4)
        gt=gt*np.array([r['width'],r['height'],r['width'],r['height']])*gain+np.array([padx,pady,padx,pady])
        record=dict(id=r['id'],animal=bool(r['animal']),gt=len(gt),pytorch_count=len(a),onnx_count=len(b),matched=matched,
            min_matched_iou=float(matched_ious.min()) if len(ia) else None,box_max_abs_pixels=box_error,confidence_max_abs=conf_error,raw_max_abs=error,raw_pass=raw_ok,decision_pass=passed)
        for backend,pred in zip(('pytorch','onnx'),(a,b)):
            tp,fp,fn=score_frame(pred,gt)
            record.update({backend+'_tp':tp,backend+'_fp':fp,backend+'_fn':fn})
        records.append(record);predictions.append(dict(id=r['id'],image_sha256=hashlib.sha256(raw).hexdigest(),pytorch=a.tolist(),onnx=b.tolist()))
        if i%50==0 or i==len(rows):
            pd.DataFrame(records).to_csv(output/'per_image.csv',index=False)
            print(f'Compared: {i}/{len(rows)} | decision mismatches: {sum(not v["decision_pass"] for v in records)}',flush=True)
    metrics={name:aggregate(records,name) for name in ('pytorch','onnx')}
    keys=['precision','recall','f1','animal_recall','empty_removal']
    deltas={k:abs(metrics['pytorch'][k]-metrics['onnx'][k]) for k in keys if metrics['pytorch'][k] is not None and metrics['onnx'][k] is not None}
    matched=sum(r['matched'] for r in records);total=sum(r['pytorch_count'] for r in records)
    summary=dict(protocol=settings,raw_pass=raw_all,raw_max_abs=raw_max,frames=len(records),frames_decision_pass=sum(r['decision_pass'] for r in records),
        decision_pass=all(r['decision_pass'] for r in records),native_detections=total,onnx_detections=sum(r['onnx_count'] for r in records),matched_detections=matched,
        matched_fraction=matched/max(total,1),max_box_error_pixels=max(r['box_max_abs_pixels'] for r in records),max_confidence_error=max(r['confidence_max_abs'] for r in records),
        metrics=metrics,metric_absolute_differences=deltas,metric_pass=all(v<=1e-6 for v in deltas.values()))
    summary['passed']=bool(summary['raw_pass'] and summary['decision_pass'] and summary['metric_pass'])
    save(output/'predictions.json',predictions);save(output/'onnx_decision_parity.json',summary)
    pd.DataFrame([dict(backend=k,**v) for k,v in metrics.items()]).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame([r for r in records if not r['decision_pass']]).to_csv(output/'mismatches.csv',index=False)
    save(output/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,onnxruntime=ort.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__,numpy=np.__version__,scipy=scipy.__version__,device='cpu',threads=threads))
    print('Parity:', 'PASS' if summary['passed'] else 'FAIL',flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k!='protocol'},indent=2),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('weights','onnx','manifest','images','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--subset-ids');args=parser.parse_args()
    summary=run(args.weights,args.onnx,args.manifest,args.images,args.output,args.threads,args.subset_ids)
    raise SystemExit(0 if summary['passed'] else 2)
