"""Versioned development evaluation for the integrated Recovery-v2 application."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,time,resource,io
import numpy as np
import pandas as pd

MANIFEST_SHA='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def event_scores(rows,predictions):
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='test':groups[r['event']].append(r)
    scores=[];excluded=[]
    for event,group in sorted(groups.items()):
        invalid=[r['id'] for r in group if r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes'])]
        if invalid:
            excluded.append(dict(event=event,reason='incomplete positive box supervision',frames=invalid));continue
        assert all(r['id'] in predictions for r in group), 'Missing predictions in a fully supervised event'
        expected=max(len(r['boxes']) for r in group);predicted=max(predictions[r['id']]['count'] for r in group)
        scores.append(dict(event=event,frames=len(group),expected=expected,predicted=predicted,absolute_error=abs(expected-predicted)))
    if not scores:raise ValueError('No complete-box retained test events available')
    return scores,excluded


def evaluate(pipeline,root,output,demo):
    root,output=Path(root),Path(output)
    manifest=root/'phase1_source/data/cct20_team_v2/manifest.json'
    assert digest(manifest)==MANIFEST_SHA
    rows=json.loads(manifest.read_text());by_id={r['id']:r for r in rows};images=root/'images'
    # Load the actually measured v2 development test, never the old segmentation_metrics.json.
    release=pipeline.release
    policy_hash=digest(release/'refinement_policy.json')
    assert policy_hash==pipeline.policy_sha256
    seg_candidates=[]
    for p in release.glob('evaluations/*/metrics.json'):
        m=json.loads(p.read_text())
        if m.get('policy_sha256')==policy_hash and m.get('split')=='test' and m.get('n')==40:seg_candidates.append((p,m))
    if not seg_candidates:raise FileNotFoundError('Matching Recovery-v2 40-image test metrics missing under '+str(release/'evaluations'))
    seg_path,seg_metrics=sorted(seg_candidates,key=lambda item:str(item[0]))[-1]
    species_run=root/'phase2_results/species_recovery_v1'
    species_path=species_run/'historical_test/metrics.json'
    species_metrics=json.loads(species_path.read_text())
    provenance=json.loads((Path(pipeline.output_dir)/'integration_provenance.json').read_text())
    assert species_metrics['checkpoint_sha256']==provenance['classifier_sha256']
    assert species_metrics['policy_sha256']==provenance['species_policy_sha256']
    targets_path=output/'protocol.json'
    if not targets_path.is_file():raise FileNotFoundError('Original acceptance protocol missing: '+str(targets_path))
    targets=json.loads(targets_path.read_text())['acceptance']
    phase1_config=json.loads((root/'phase1_source/handoff/data_config.json').read_text())
    species_targets=phase1_config['acceptance']
    folder=output/'member4_recovery_v2/evaluations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    identity=dict(detector_sha256=pipeline.policy['detector_sha256'],confidence=pipeline.policy['prompt_floor'],
        nms=pipeline.policy['detector_nms'],imgsz=pipeline.policy['detector_input'],manifest_sha256=MANIFEST_SHA,
        count_definition='all M4 outputs at fixed floor; classification and mask filtering do not change pipeline count')
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=output/'member4_recovery_v2/count_prediction_cache'/identity_hash[:16];cache.mkdir(parents=True,exist_ok=True)
    retained=[r for r in rows if r['split']=='test' and not (r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes']))]
    predictions={}
    print(f'Counting {len(retained)} supervised test frames with the fixed M4 detector; no classifier/SAM rerun is needed for the count definition.',flush=True)
    for i,r in enumerate(retained,1):
        path=images/(r['id']+'.jpg')
        if not path.is_file():raise FileNotFoundError('Missing original test image: '+str(path))
        raw=path.read_bytes()
        if hashlib.md5(raw).hexdigest()!=r['processed_file_md5']:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as im:
                rgb=im.convert('RGB')
                assert rgb.size==(r['width'],r['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256'], 'Test pixels differ from original manifest: '+r['id']
        image_sha=hashlib.sha256(raw).hexdigest();cached=cache/(r['id']+'.json')
        entry=json.loads(cached.read_text()) if cached.is_file() else None
        if entry is None or entry.get('image_sha256')!=image_sha or entry.get('identity_sha256')!=identity_hash:
            result=pipeline.detector.predict(str(path),conf=identity['confidence'],iou=identity['nms'],imgsz=identity['imgsz'],device='cpu',verbose=False)[0]
            entry=dict(id=r['id'],count=len(result.boxes),boxes=result.boxes.xyxyn.cpu().numpy().tolist(),image_sha256=image_sha,identity_sha256=identity_hash)
            save_json(cached,entry)
        predictions[r['id']]=entry
        if i%50==0:print('Counted:',i,'/',len(retained),flush=True)
    scores,excluded=event_scores(rows,predictions)
    pd.DataFrame(scores).to_csv(folder/'event_counts.csv',index=False)
    save_json(folder/'excluded_events.json',excluded)
    event_metrics=dict(mae=float(np.mean([r['absolute_error'] for r in scores])),events=len(scores),excluded_events=len(excluded),
        scope='All test events whose every retained frame has complete box supervision; counts are maxima over retained frames, not unique individuals or complete original source sequences.',independent_test=False,identity=identity)
    save_json(folder/'event_metrics.json',event_metrics)
    positives=[r for r in retained if r['animal']];negatives=[r for r in retained if not r['animal']]
    assert positives and negatives
    detector_metrics=dict(animal_recall=sum(predictions[r['id']]['count']>0 for r in positives)/len(positives),
        empty_removal=sum(predictions[r['id']]['count']==0 for r in negatives)/len(negatives),
        animal_frames=len(positives),empty_frames=len(negatives),frames=len(retained),identity=identity,
        scope='Frame-level triage on all retained complete-box test frames, not bounding-box recall or AP.',independent_test=False)
    save_json(folder/'detection_operating_metrics.json',detector_metrics)
    chosen=[by_id[r['id']] for r in demo]
    assert chosen and all(r['split']=='val_select' for r in chosen)
    print('Warming up the full detection + species + M5/SAM pipeline (3 images).',flush=True)
    for i in range(3):pipeline.process_frame(images/(chosen[i%len(chosen)]['id']+'.jpg'),chosen[i%len(chosen)])
    latency=[]
    for i in range(20):
        r=chosen[i%len(chosen)];record=pipeline.process_frame(images/(r['id']+'.jpg'),r)
        latency.append(dict(id=r['id'],iteration=i+1,count=record['count'],matched_masks=record['segmentation']['matched_masks'],**record['timing']))
        pd.DataFrame(latency).to_csv(folder/'pipeline_latencies.csv',index=False)
        print('Pipeline timing:',i+1,'/20',round(record['timing']['total'],3),'seconds',flush=True)
    seconds=[r['total'] for r in latency]
    pipeline_metrics=dict(p50=float(np.quantile(seconds,.5)),p95=float(np.quantile(seconds,.95)),p99=float(np.quantile(seconds,.99)),
        throughput=1/float(np.mean(seconds)),warmup=3,repeats=20,batch=1,backend='pytorch_cpu',
        process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scope='Sequential full inference, image read included, HTTP excluded, repeated fixed validation branch-coverage cases; not representative production traffic. Segmentation helper repeats M4 inference.',
        memory_scope='cumulative process peak RSS, not incremental model memory')
    save_json(folder/'pipeline_benchmark.json',pipeline_metrics)
    entries=[('animal_recall',detector_metrics['animal_recall'],targets['animal_recall_min'],'min','retained test frames'),
        ('empty_removal',detector_metrics['empty_removal'],targets['empty_removal_min'],'min','retained test frames'),
        ('mask_iou',seg_metrics['mean_iou'],targets['mask_iou_min'],'min','40-image recovery-v2 development test'),
        ('event_count_mae',event_metrics['mae'],targets['event_count_mae_max'],'max','complete-box retained test events'),
        ('pipeline_p95_seconds',pipeline_metrics['p95'],targets['p95_seconds_max'],'max','fixed validation functional-case mix, 20 repeats'),
        ('species_macro_f1',species_metrics['macro_f1_all_training_classes'],species_targets['species_macro_f1_min'],'min','508 known-class historical test crops'),
        ('species_ece',species_metrics['ece'],species_targets['species_ece_max'],'max','508 known-class historical test crops')]
    acceptance=[dict(metric=name,value=value,target=target,direction=direction,met=bool(value>=target if direction=='min' else value<=target),scope=scope) for name,value,target,direction,scope in entries]
    pd.DataFrame(acceptance).to_csv(folder/'acceptance.csv',index=False)
    decision=dict(performance_decision='GO' if all(r['met'] for r in acceptance) else 'NO-GO',
        scope='Preset measured performance criteria for unattended deployment; not a course-completion grade. Research prototype remains subject to human review.',
        independent_generalization_validated=False,acceptance=acceptance)
    save_json(folder/'decision.json',decision)
    save_json(folder/'metric_sources.json',dict(segmentation=dict(path=str(seg_path),sha256=digest(seg_path)),species=dict(path=str(species_path),sha256=digest(species_path)),
        protocol=dict(path=str(targets_path),sha256=digest(targets_path)),classifier_sha256=provenance['classifier_sha256'],segmentation_policy_sha256=policy_hash,
        test_scope='Exposed historical/development data; separate unseen evaluation required for independent generalization claims.'))
    card=f'''# Camera-Trap Event Triage — Recovery v2\n\nIntended use: research review of camera-trap events with human oversight.\n\nModels: PyTorch M4 YOLO11n detector; newly retrained frozen ResNet-50 species classifier and matching calibration; unchanged M4 + fine-tuned M5 + pretrained SAM ViT-B Recovery v2 masks. The species checkpoint is not the original Phase-1 model.\n\nPerformance decision: **{decision['performance_decision']}** against unchanged preset criteria. See acceptance.csv for all passes and failures. This is not a course-completion decision.\n\nMeasured scopes differ: detection frame triage uses retained complete-box test frames, counts use complete-box events over retained frames, segmentation uses 40 exposed development-test images, species uses 508 known-class historical test crops. These are not a single new independent end-to-end accuracy test. Pipeline AP is unmeasured.\n\nLatency includes full image inference and repeated detector execution inside the segmentation helper. Benchmark uses 20 repeats of fixed validation functional cases; p95/p99 are descriptive, not a production SLO certification. Memory is cumulative process peak RSS.\n\nLimitations: camera/domain shift, rare classes, small or obscured animals, missing detections, large false-positive masks and SAM-assisted reference bias. Grayscale is not verified infrared. Relative mask area is not physical size or distance. Event count is maximum simultaneous detections, not unique individuals. Per-species event maxima need not sum to the event total. Low-confidence or mask-missing outputs require review.\n\nOut of scope: unattended ecological population estimates and identifying people. Preserve the original Phase-1 report and all previous results separately.\n'''
    (folder/'MODEL_CARD.md').write_text(card)
    for filename in ['event_counts.csv','event_metrics.json','pipeline_benchmark.json','acceptance.csv','decision.json','MODEL_CARD.md']:
        # Legacy report paths point to the current evaluation; detailed old runs stay timestamped.
        import shutil
        shutil.copyfile(folder/filename,output/filename)
    print('Saved evaluation:',folder,flush=True)
    print(pd.DataFrame(acceptance).to_string(index=False),flush=True)
    print('Performance decision:',decision['performance_decision'],flush=True)
    return folder,acceptance
