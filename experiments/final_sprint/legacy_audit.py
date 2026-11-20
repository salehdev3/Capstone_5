"""Version-aware evidence checks and a curated, checkpoint-containing submission ZIP."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,csv,shutil,tempfile,zipfile,math
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def polygons_mask(polygons,w,h):
    image=Image.new('1',(w,h));draw=ImageDraw.Draw(image)
    for polygon in polygons:
        if len(polygon)>=3:draw.polygon([(float(x*w),float(y*h)) for x,y in polygon],fill=1)
    return np.asarray(image,dtype=np.uint8)


def operating_mask_ap(predictions,rows,label_folder,protocol):
    """COCO mask AP of accepted deployed-policy outputs, not a low-floor native model sweep."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    dataset=dict(info={},images=[],categories=[dict(id=1,name='animal')],annotations=[])
    detections=[];ann_id=1
    expected=protocol.get('reference_sha256',{})
    for image_id,pred in enumerate(predictions['results'],1):
        key=Path(pred['image']).stem;r=rows[key];w,h=r['width'],r['height']
        label=Path(label_folder)/(key+'.txt')
        if expected:assert digest(label)==expected[key],'Changed reference: '+key
        dataset['images'].append(dict(id=image_id,width=w,height=h))
        for line in label.read_text().splitlines():
            if not line.strip():continue
            vals=[float(x) for x in line.split()]
            assert vals[0]==0 and len(vals)>=7 and len(vals)%2==1 and all(math.isfinite(v) and 0<=v<=1 for v in vals[1:])
            polygon=np.asarray(vals[1:]).reshape(-1,2)
            mask=polygons_mask([polygon],w,h);rle=mask_utils.encode(np.asfortranarray(mask))
            dataset['annotations'].append(dict(id=ann_id,image_id=image_id,category_id=1,segmentation=rle,
                area=float(mask_utils.area(rle)),bbox=mask_utils.toBbox(rle).tolist(),iscrowd=0));ann_id+=1
        for obj in pred['objects']:
            mask=polygons_mask([obj['polygon_normalized']],w,h)
            if not mask.any():continue
            rle=mask_utils.encode(np.asfortranarray(mask))
            detections.append(dict(image_id=image_id,category_id=1,segmentation=rle,score=float(obj['confidence'])))
    gt=COCO();gt.dataset=dataset;gt.createIndex()
    if detections:dt=gt.loadRes(detections)
    else:
        dt=COCO();dt.dataset={**dataset,'annotations':[]};dt.createIndex()
    evaluation=COCOeval(gt,dt,'segm');evaluation.params.imgIds=[r['id'] for r in dataset['images']]
    evaluation.evaluate();evaluation.accumulate();evaluation.summarize()
    return dict(mask_map50_95=float(evaluation.stats[0]),mask_map50=float(evaluation.stats[1]),n=len(dataset['images']),
        instances=ann_id-1,predicted_instances=len(detections),score='M4 detector confidence of accepted masks',
        scope='COCO instance-mask AP of deployed operating-policy outputs including fallback and quality filtering. Confidence range is truncated by the deployed policy; not a native low-floor YOLO/SAM AP sweep.',
        independent_test=False,reference_hashes_checked=bool(expected))


def audit(pipeline,root,output):
    root,output=Path(root),Path(output)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=output/'member4_recovery_v2/audits'/stamp;folder.mkdir(parents=True,exist_ok=False)
    policy_path=pipeline.release/'refinement_policy.json';policy=read(policy_path);policy_sha=digest(policy_path)
    provenance=read(Path(pipeline.output_dir)/'integration_provenance.json')
    source_items={};checks=[]
    def check(name,path,validator=None,archive_name=None):
        path=Path(path);present=path.is_file();valid=False;details='Missing file'
        if present:
            try:
                if validator:validator(path)
                valid=True;details='File present; specified semantic checks passed' if validator else 'File present; content requires human review'
            except Exception as exc:details=str(exc)[:500]
            source_items[archive_name or ('evidence/'+name+'/'+path.name)]=path
        checks.append(dict(requirement=name,evidence=str(path),present=present,verified=valid,details=details))
        return valid
    def matches(path,expected):assert digest(path)==expected,'SHA256 mismatch'
    check('M4_current_detector',pipeline.release/policy['detector_file'],lambda p:matches(p,policy['detector_sha256']),'releases/Segmentation_M4_Guided_Recovery_v2/detector.pt')
    check('M5_current_segmenter',pipeline.release/policy['model_file'],lambda p:matches(p,policy['model_sha256']),'releases/Segmentation_M4_Guided_Recovery_v2/selected.pt')
    check('species_current_checkpoint',root/'models/species_recovery_v1/best.pt',lambda p:matches(p,provenance['classifier_sha256']),'models/species_recovery_v1/best.pt')
    check('species_current_policy',root/'models/species_recovery_v1/policy.json',lambda p:matches(p,provenance['species_policy_sha256']),'models/species_recovery_v1/policy.json')
    check('M5_current_policy',policy_path,lambda p:matches(p,pipeline.policy_sha256),'releases/Segmentation_M4_Guided_Recovery_v2/refinement_policy.json')
    check('M5_inference_helper',pipeline.release/'predict_refined.py',lambda p:matches(p,read(pipeline.release/'artifact_checksums.json')['predict_refined.py']),'releases/Segmentation_M4_Guided_Recovery_v2/predict_refined.py')
    source_items['releases/Segmentation_M4_Guided_Recovery_v2/SAM_DOWNLOAD.json']=policy_path
    for name in ('member4_integration_v2.py','member4_service_v2.py','member4_evaluation_v2.py'):
        source=output/'member4_recovery_v2'/name
        if source.is_file():source_items['source/'+name]=source
    source=root/'models/species_recovery_v1/summary.json'
    if source.is_file():source_items['models/species_recovery_v1/summary.json']=source
    for name,filename in [('M4_thresholds','threshold_sweep.csv'),('M4_AP','detection_ap.json'),('M4_data_protocol','protocol.json'),('M8_detector_benchmark','benchmark.csv')]:check(name,output/filename)
    check('species_history',output/'species_recovery_v1/history.csv')
    check('species_reload',output/'species_recovery_v1/checkpoint_reload.json',lambda p:(_ for _ in ()).throw(AssertionError('Reload check did not pass')) if not read(p).get('passed') else None)
    species_test=output/'species_recovery_v1/historical_test/metrics.json'
    def species_bound(p):
        m=read(p);assert m['checkpoint_sha256']==provenance['classifier_sha256'] and m['policy_sha256']==provenance['species_policy_sha256']
    check('species_historical_test',species_test,species_bound)
    ref=root/'segmentation/reference_release.json';review=root/'segmentation_masks/review.csv'
    def verified_references(p):
        release=read(p);assert len(release['mask_hashes'])==230
        frame=pd.read_csv(review).fillna('');assert len(frame)==230 and frame.id.is_unique
        assert set(frame.id)==set(release['mask_hashes'])
        for col in ('reviewer','reference_source'):assert frame[col].astype(str).str.strip().ne('').all(), 'Missing '+col
        assert frame.decision.eq('verified').all(),'Unverified reference masks'
        assert digest(review)==release['review_sha256'],'Review CSV differs from reference release'
        for key,sha in release['mask_hashes'].items():assert digest(root/'segmentation_masks'/(key+'.verified.txt'))==sha,key
    check('M5_verified_230_reference_masks',ref,verified_references);check('M5_reference_review',review)
    evaluations=[]
    for p in (output/'member4_recovery_v2/evaluations').glob('*/metric_sources.json'):
        data=read(p)
        if data.get('classifier_sha256')==provenance['classifier_sha256'] and data.get('segmentation_policy_sha256')==policy_sha:evaluations.append(p.parent)
    application=sorted(evaluations)[-1] if evaluations else output/'member4_recovery_v2/evaluations/MISSING_CURRENT_EVALUATION'
    for name in ['acceptance.csv','decision.json','MODEL_CARD.md','event_counts.csv','event_metrics.json','pipeline_benchmark.json','pipeline_latencies.csv','metric_sources.json','detection_operating_metrics.json']:
        check('M8_'+name,application/name,archive_name='application/'+name)
    if (application/'decision.json').is_file() and (application/'acceptance.csv').is_file():
        d=read(application/'decision.json');frame=pd.read_csv(application/'acceptance.csv')
        computed=[float(r.value)>=float(r.target) if r.direction=='min' else float(r.value)<=float(r.target) for r in frame.itertuples()]
        assert d['performance_decision']==('GO' if all(computed) else 'NO-GO')
        assert computed==frame.met.tolist(),'Acceptance outcomes differ from thresholds'
    parity=output/'member4_recovery_v2/onnx_parity.json'
    def parity_bound(p):
        m=read(p);assert m['passed'] and m['source_detector_sha256']==policy['detector_sha256']
        assert m['segmentation_policy_sha256']==policy_sha
    check('M8_ONNX_detector_parity',parity,parity_bound)
    http=None
    for p in sorted((output/'member4_recovery_v2/http_demos').glob('*/demo_record.json'),reverse=True):
        record=read(p)
        if record.get('segmentation_policy_sha256')==policy_sha and record.get('coverage',{}).get('all_stages_exercised'):http=p.parent;break
    http=http or output/'member4_recovery_v2/http_demos/MISSING_FULL_STAGE_DEMO'
    def http_check(p):
        m=read(p);record=read(http/'demo_record.json');frames=m['result']['frames']
        assert m['status']==200 and [f['id'] for f in frames]==record['frame_ids']
        assert sum(len(f['objects']) for f in frames)>0 and sum(f['segmentation']['matched_masks'] for f in frames)>0
        assert all(f['segmentation']['policy_sha256']==policy_sha for f in frames)
    check('M8_HTTP_full_stage_execution',http/'service_test.json',http_check,archive_name='http_demo/service_test.json')
    check('M8_HTTP_demo_record',http/'demo_record.json',archive_name='http_demo/demo_record.json')
    for p in http.glob('*_overlay.png'):source_items['http_demo/'+p.name]=p
    seg_candidates=[]
    for p in pipeline.release.glob('evaluations/*/metrics.json'):
        if read(p).get('policy_sha256')==policy_sha:seg_candidates.append(p.parent)
    seg=sorted(seg_candidates)[-1] if seg_candidates else pipeline.release/'evaluations/MISSING_CURRENT_TEST'
    def seg_bound(p):
        m=read(p);assert m['policy_sha256']==policy_sha and m['split']=='test' and m['n']==40
    check('M5_v2_test_overlap',seg/'metrics.json',seg_bound,archive_name='segmentation/metrics.json')
    for name in ['per_image.csv','predictions.json','protocol.json']:check('M5_v2_'+name,seg/name,archive_name='segmentation/'+name)
    # Generate a new version-bound review template, without inventing human review.
    failure=output/'member4_recovery_v2/segmentation_failure_analysis_v2.csv'
    if not failure.exists() and (seg/'per_image.csv').is_file():
        worst=pd.read_csv(seg/'per_image.csv').sort_values('iou').head(12).copy()
        worst['policy_sha256']=policy_sha;worst['cause']='';worst['improvement']='';worst['reviewer']=''
        worst.to_csv(failure,index=False)
    def reviewed_failure(p,bound=False):
        frame=pd.read_csv(p).fillna('');assert len(frame)>0,'Empty failure analysis'
        for col in ('cause','improvement','reviewer'):assert col in frame and frame[col].astype(str).str.strip().ne('').all(),'Complete human review field: '+col
        if bound:assert frame.policy_sha256.eq(policy_sha).all(),'Failure analysis references another segmentation version'
    check('M4_reviewed_failure_analysis',output/'detection_failure_analysis.csv',reviewed_failure)
    check('M5_v2_reviewed_failure_analysis',failure,lambda p:reviewed_failure(p,True))
    ap_path=seg/'operating_mask_ap.json'
    if not ap_path.exists() and all((seg/name).is_file() for name in ['predictions.json','protocol.json']):
        try:
            by_id={r['id']:r for r in read(root/'phase1_source/data/cct20_team_v2/manifest.json')}
            predictions=read(seg/'predictions.json');assert predictions['policy_sha256']==policy_sha
            assert len(predictions['results'])==40 and len({Path(r['image']).stem for r in predictions['results']})==40
            ap=operating_mask_ap(predictions,by_id,root/'segmentation/labels/test',read(seg/'protocol.json'))
            ap['policy_sha256']=policy_sha;write(ap_path,ap)
        except Exception as exc:write(folder/'mask_AP_pending.json',dict(reason=str(exc),scope='No model score was fabricated. See missing evidence.'))
    check('M5_operating_policy_mask_AP',ap_path,lambda p:(_ for _ in ()).throw(AssertionError('Wrong AP policy')) if read(p).get('policy_sha256')!=policy_sha else None,archive_name='segmentation/operating_mask_ap.json')
    # Human deliverables are checked separately; a file cannot prove its content or approval.
    for name,path in [('Phase2_report_PDF',root/'submission/Phase2_Report.pdf'),('Final_presentation_PDF',root/'submission/Final_Presentation.pdf'),('Recorded_demo_video',root/'submission/Demo.mp4')]:
        check(name,path)
    check('M8_WandB_evidence_link',output/'member4_recovery_v2/wandb_link.json',lambda p:(_ for _ in ()).throw(AssertionError('No uploaded run URL')) if not (read(p).get('url') and read(p).get('status')=='uploaded' and read(p).get('artifact_digest')) else None)
    frame=pd.DataFrame(checks);frame.to_csv(folder/'completion.csv',index=False)
    pending=frame.loc[~frame.verified];pending.to_csv(folder/'missing_or_unverified.csv',index=False)
    status=dict(machine_evidence_complete=bool(frame.verified.all()),status='FILES PRESENT — human content review still required' if frame.verified.all() else 'INCOMPLETE — see missing_or_unverified.csv',
        scope='Artifact/identity audit, not a course grade or a claim that performance targets passed.',
        deployment_decision=read(application/'decision.json').get('performance_decision') if (application/'decision.json').is_file() else 'MISSING',
        current_application_evaluation=str(application),current_http_demo=str(http),current_segmentation_evaluation=str(seg),
        missing_or_unverified=pending.requirement.tolist(),human_content_review_required=True,
        original_phase1_report_changed=False,trained_weights_included_in_zip=True,pretrained_SAM_in_zip=False)
    write(folder/'audit_status.json',status)
    for key,p in {'completion.csv':folder/'completion.csv','missing_or_unverified.csv':folder/'missing_or_unverified.csv','audit_status.json':folder/'audit_status.json'}.items():source_items['audit/'+key]=p
    # Curated ZIP includes the three trained checkpoints; avoid archiving all old runs or secrets.
    packages=root/'submission_packages';packages.mkdir(exist_ok=True)
    archive=packages/f'Phase2_Evidence_{stamp}.zip'
    with tempfile.TemporaryDirectory() as tmp:
        stage=Path(tmp)/'Phase2_Evidence';stage.mkdir()
        manifest={}
        for relative,source in source_items.items():
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);manifest[relative]=digest(target)
        readme='''# Phase 2 evidence package\n\nRead audit/audit_status.json and audit/missing_or_unverified.csv first. This archive can be incomplete; packaging does not establish submission readiness. Performance NO-GO is distinct from artifact completeness.\n\nContains actual trained detector, fine-tuned M5 and newly retrained species checkpoint/policy, current-version evidence and functional demo outputs. No old Phase-1 numbers are relabelled as new results. Original images/datasets are not bundled. The public pretrained SAM weight is downloaded using the official URL and SHA256 recorded in the segmentation policy; it is not duplicated here.\n\nThe mask AP file, when present, evaluates accepted deployed-policy masks; it is not a low-confidence native-model AP sweep. Scores have different documented scopes and are not a new independent end-to-end test. HTTP images are explicit validation branch-coverage examples including prior misses, not an accuracy sample. A real presentation, final report and recorded fallback video still need human preparation/review.\n\nKeep all checkpoints in durable storage and upload a separate model backup to W&B. No credentials are included.\n'''
        (stage/'README.md').write_text(readme);manifest['README.md']=digest(stage/'README.md')
        write(stage/'package_checksums.json',manifest)
        temporary=Path(tmp)/'evidence.zip'
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in stage.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(stage))
        with zipfile.ZipFile(temporary) as z:
            assert z.testzip() is None
            for name,expected in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
        drive_temp=archive.with_suffix('.uploading');shutil.copyfile(temporary,drive_temp);drive_temp.replace(archive)
    write(folder/'package_receipt.json',dict(path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,files=len(manifest)+1))
    print(frame.to_string(index=False),flush=True)
    print(status['status'],flush=True);print('Package saved:',archive,flush=True)
    print('Current v2 failure-review CSV:',failure,flush=True)
    return folder,archive,status
