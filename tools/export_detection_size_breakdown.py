"""Derive size recall from fixed detector caches; no inference or threshold changes."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math
import numpy as np
from scipy.optimize import linear_sum_assignment

MANIFEST_SHA='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'
DETECTOR_SHA='ac38aaffe735bf7a6666866d2410f3601a93c6f9829ae6f68b7cd0117d1135e8'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def matches(gt,pred):
    a=np.asarray(gt,dtype=float).reshape(-1,4);b=np.asarray(pred,dtype=float).reshape(-1,4)
    if not len(a) or not len(b):return []
    inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    iou=inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)
    x,y=linear_sum_assignment(-(iou>=.5).astype(float)*1000-iou)
    return [(int(i),int(j)) for i,j in zip(x,y) if iou[i,j]>=.5]

def run(root):
    root=Path(root);out=root/'phase2_results'
    manifest=root/'phase1_source/data/cct20_team_v2/manifest.json';assert digest(manifest)==MANIFEST_SHA
    rows=json.loads(manifest.read_text());rows=[r for r in rows if r['split']=='test' and not(r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes']))]
    assert len(rows)==554
    identity=dict(detector_sha256=DETECTOR_SHA,confidence=.045,nms=.3,imgsz=640,manifest_sha256=MANIFEST_SHA,count_definition='all M4 outputs at fixed floor; classification and mask filtering do not change pipeline count')
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=out/'member4_recovery_v2/count_prediction_cache'/identity_hash[:16]
    assert cache.is_dir(),f'Fixed operating-point cache required: {cache}. Do not rerun training or the historical test to obtain this table.'
    items=[];sources={};tp=fp=fn=animal_hit=empty_removed=0
    for r in rows:
        path=cache/(r['id']+'.json');entry=json.loads(path.read_text());sources[str(path.relative_to(root))]=digest(path)
        assert entry['id']==r['id'] and entry['identity_sha256']==identity_hash
        boxes=entry['boxes'];assert len(boxes)==entry['count']
        assert all(len(b)==4 and all(math.isfinite(v) and 0<=v<=1 for v in b) and b[0]<b[2] and b[1]<b[3] for b in boxes)
        gt=[v['xyxy'] for v in r['boxes']];pairs=matches(gt,boxes);hit={i for i,j in pairs}
        tp+=len(pairs);fp+=len(boxes)-len(pairs);fn+=len(gt)-len(pairs)
        if r['animal']:animal_hit+=bool(boxes)
        else:empty_removed+=not boxes
        for i,b in enumerate(gt):
            area=(b[2]-b[0])*(b[3]-b[1])*640**2
            size='Small' if area<32**2 else 'Medium' if area<96**2 else 'Large'
            items.append(dict(id=r['id'],gt_index=i,size=size,area_proxy_pixels=area,detected=i in hit))
    assert len(items)==508 and (tp,fp,fn)==(450,170,58),'Cache does not reproduce the archived detector operating counts; preserve it for review.'
    assert animal_hit==489 and empty_removed==28
    table=[]
    for size in ('Small','Medium','Large'):
        group=[r for r in items if r['size']==size];detected=sum(r['detected'] for r in group)
        table.append(dict(size=size,instances=len(group),matched=detected,recall=detected/len(group) if group else None))
    assert table[0]['instances']==5 and table[0]['matched']==0
    destination=out/'member4_recovery_v2/size_breakdown'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');destination.mkdir(parents=True,exist_ok=False)
    for name,values in [('detection_size_recall.csv',table),('per_instance.csv',items)]:
        with (destination/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    protocol=dict(identity=identity,identity_sha256=identity_hash,split='test',frames=554,reference_instances=508,
        definition='Normalized reference box area times 640 squared; Small <32 squared, Medium >=32 squared and <96 squared, Large >=96 squared. This preserves the archived proxy, not exact post-letterbox COCO sizes.',
        matching='Original notebook Hungarian matching: maximize number of IoU >= .5 pairs, then total IoU; one-to-one.',
        source='Previously saved fixed-detector count cache; no new inference, threshold selection or training.',
        scope='Historical retained supervised test frames; size detail derived from unchanged predictions, not an independent test.',
        tp=tp,fp=fp,fn=fn,animal_hit=animal_hit,empty_removed=empty_removed,source_checksums=sources)
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Size recall from saved fixed predictions:')
    for r in table:print(r)
    print('SAVED:',destination)
    print('Upload detection_size_recall.csv and protocol.json.')
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);args=parser.parse_args();run(args.root)
