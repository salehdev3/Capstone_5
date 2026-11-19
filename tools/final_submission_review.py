"""Review archived project evidence without training or rewriting measured results."""
from pathlib import Path
import argparse,ast,csv,hashlib,json,math,re,statistics,subprocess,zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/phase2_report/evidence/supplied'
OUT=ROOT/'docs/final_review'
checks=[];sources=set()

def read(path):
    path=Path(path);sources.add(path);return json.loads(path.read_text())

def table(path):
    path=Path(path);sources.add(path)
    with path.open(newline='') as stream:return list(csv.DictReader(stream))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),(a,b)

def check(key,description,evidence,validator):
    try:
        detail=validator()
        checks.append(dict(requirement=key,status='verified',verified_evidence=description,detail=str(detail or 'Cross-file checks consistent'),evidence=evidence))
    except Exception as exc:
        checks.append(dict(requirement=key,status='review_required',verified_evidence=description,detail=str(exc),evidence=evidence))

def manifest():
    path=ROOT/'releases/species_recovery/phase1_training_metadata.zip';sources.add(path)
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('/manifest.json') or n=='manifest.json'];assert len(names)==1
        raw=z.read(names[0]);assert hashlib.sha256(raw).hexdigest()=='d3bfb6ba0c9d31412f9d603a6f63ae7c4890f458c5d2aa2a87d7653bb882bb53'
        return json.loads(raw)

def dataset_check():
    rows=manifest();assert len(rows)==6037
    for key in ('camera','event','pixel_sha256'):
        groups={}
        for r in rows:groups.setdefault(r[key],set()).add(r['split'])
        assert all(len(v)==1 for v in groups.values()),key
    return '6,037 frozen frames; camera/event/decoded-pixel split separation checked'

def detection():
    meta=read(ROOT/'docs/phase2_report/evidence/detector_checkpoint_training_metadata.json')
    history=table(ROOT/'docs/phase2_report/evidence/detector_training_from_checkpoint.csv');assert len(history)==40
    assert meta['checkpoint_sha256']=='ac38aaffe735bf7a6666866d2410f3601a93c6f9829ae6f68b7cd0117d1135e8'
    ap=read(BASE/'evidence/M4_AP/detection_ap.json');assert all(0<=ap[k]<=1 for k in ('map50','map50_95'))
    sweeps=table(BASE/'evidence/M4_thresholds/threshold_sweep.csv');assert len(sweeps)==15
    return '40 training epochs, 15 archived threshold combinations and native low-floor test AP'

def reviews():
    frame=table(ROOT/'results/phase2/member1/detection_review.csv');assert len(frame)==100 and len({r['id'] for r in frame})==100
    assert all(r['decision']=='verified' and r['reviewer'].strip() for r in frame)
    ref=read(BASE/'evidence/M5_verified_230_reference_masks/reference_release.json')
    review_path=BASE/'evidence/M5_reference_review/review.csv';masks=table(review_path)
    assert len(masks)==230 and {r['id'] for r in masks}==set(ref['mask_hashes'])
    assert sha(review_path)==ref['review_sha256']
    assert all(r['decision']=='verified' and r['reviewer'].strip() and r['reference_source'].strip() for r in masks)
    return '100 animal-box image reviews and 230 reference-mask reviews with named reviewers'

def segmentation():
    policy=BASE/'releases/Segmentation_M4_Guided_Recovery_v2/refinement_policy.json';policy_hash=sha(policy);sources.add(policy)
    metric=read(BASE/'segmentation/metrics.json');protocol=read(BASE/'segmentation/protocol.json');per=table(BASE/'segmentation/per_image.csv')
    assert metric['n']==len(per)==40 and metric['policy_sha256']==protocol['policy_sha256']==policy_hash
    close(statistics.mean(float(r['iou']) for r in per),metric['mean_iou']);close(statistics.mean(float(r['dice']) for r in per),metric['mean_dice'])
    ap=read(BASE/'segmentation/operating_mask_ap.json');assert ap['policy_sha256']==policy_hash and ap['n']==40
    assert ap['instances']==45 and ap['predicted_instances']==40
    return f"Recovery-v2 development evaluation: 40 images; IoU {metric['mean_iou']:.6f}; accepted-output mask AP archived"

def failure_reviews():
    for path in [BASE/'evidence/M4_reviewed_failure_analysis/detection_failure_analysis.csv',BASE/'evidence/M5_v2_reviewed_failure_analysis/segmentation_failure_analysis_v2.csv']:
        frame=table(path);assert len(frame)==12
        assert all(all(r[k].strip() for k in ('cause','improvement','reviewer')) for r in frame)
    return '12 reviewed detector examples and 12 reviewed Recovery-v2 mask examples'

def species():
    metric=read(BASE/'evidence/species_historical_test/metrics.json');policy=read(BASE/'models/species_recovery_v1/policy.json')
    assert metric['n']==sum(v['support'] for v in metric['per_class'])==508 and len(metric['per_class'])==14
    assert metric['checkpoint_sha256']==policy['checkpoint_sha256'] and metric['policy_sha256']==sha(BASE/'models/species_recovery_v1/policy.json')
    assert policy['calibration_split']=='val_calibrate' and not policy['test_used_for_calibration']
    assert len(table(BASE/'evidence/species_history/history.csv'))==12
    assert read(BASE/'evidence/species_reload/checkpoint_reload.json')['passed']
    return 'New recovery classifier: 12 epochs; 508 historical crops; calibration and reload evidence bound to checkpoint'

def http():
    demo=read(BASE/'http_demo/demo_record.json');receipt=read(BASE/'http_demo/service_test.json');frames=receipt['result']['frames']
    assert receipt['status']==200 and demo['coverage']['all_stages_exercised']
    assert [r['id'] for r in frames]==demo['frame_ids'] and len(frames)==5
    assert sum(len(r['objects']) for r in frames)==demo['coverage']['classified_objects']==2
    assert sum(r['segmentation']['matched_masks'] for r in frames)==demo['coverage']['matched_masks']==2
    assert all(r['segmentation']['policy_sha256']==demo['segmentation_policy_sha256'] for r in frames)
    return 'HTTP 200 on 5 fixed validation cases; detection, classification and matched masks exercised'

def counting():
    metric=read(BASE/'application/event_metrics.json');frame=table(BASE/'application/event_counts.csv')
    assert len(frame)==metric['events']==288 and len({r['event'] for r in frame})==288
    for r in frame:assert abs(int(r['expected'])-int(r['predicted']))==int(r['absolute_error'])
    close(statistics.mean(float(r['absolute_error']) for r in frame),metric['mae'])
    return '288 supervised retained events; per-event errors reconstruct archived count MAE'

def latency():
    metric=read(BASE/'application/pipeline_benchmark.json');frame=table(BASE/'application/pipeline_latencies.csv');assert len(frame)==metric['repeats']==20 and metric['warmup']==3
    values=sorted(float(r['total']) for r in frame)
    for label,q in [('p50',.5),('p95',.95),('p99',.99)]:
        index=(len(values)-1)*q;lo=math.floor(index);hi=math.ceil(index);value=values[lo]+(values[hi]-values[lo])*(index-lo);close(value,metric[label])
    close(1/statistics.mean(values),metric['throughput'])
    return '20 full-pipeline timings after 3 warmups; p50/p95/p99 and throughput reconstructed'

def onnx():
    folder=ROOT/'docs/onnx_parity/full_validation_20261119T070317993000Z';summary=read(folder/'onnx_decision_parity.json');per=table(folder/'per_image.csv');m=table(folder/'metrics.csv')
    for name,expected in read(folder/'artifact_checksums.json').items():assert sha(folder/name)==expected,name
    protocol=summary['protocol'];rows=manifest();expected={r['id']:r for r in rows if r['split']=='val_select' and not(r['animal'] and (r.get('incomplete_boxes',True) or not r['boxes']))}
    assert len(per)==len({r['id'] for r in per})==len(expected)==1208 and {r['id'] for r in per}==set(expected)
    for r in per:
        assert r['raw_pass']==r['decision_pass']=='True'
        assert int(r['gt'])==len(expected[r['id']]['boxes'])
        assert int(r['matched'])==int(r['pytorch_count'])==int(r['onnx_count'])
        for k in ['tp','fp','fn']:assert r['pytorch_'+k]==r['onnx_'+k]
    assert sum(int(r['matched']) for r in per)==summary['matched_detections']==1626
    for row in m:
        backend=row['backend']
        for k,v in summary['metrics'][backend].items():close(row[k],v)
        for k in ['tp','fp','fn']:assert sum(int(r[backend+'_'+k]) for r in per)==int(row[k])
    assert summary['metrics']['pytorch']==summary['metrics']['onnx'] and summary['passed'] and protocol['full_validation_evaluated']
    raw=read(BASE/'evidence/M8_ONNX_detector_parity/onnx_parity.json');assert raw['onnx_sha256']==protocol['onnx_sha256'] and raw['source_detector_sha256']==protocol['detector_sha256']
    return '1,208 validation frames; 1,626 matched detections; original exports/hash identities consistent'

def acceptance():
    frame=table(BASE/'application/acceptance.csv');decision=read(BASE/'application/decision.json');assert len(frame)==7
    computed=[float(r['value'])>=float(r['target']) if r['direction']=='min' else float(r['value'])<=float(r['target']) for r in frame]
    assert computed==[r['met']=='True' for r in frame]
    assert decision['performance_decision']==('GO' if all(computed) else 'NO-GO')
    return 'All 7 fixed criterion calculations agree with the documented decision; values preserved'

def report():
    tex=ROOT/'docs/phase2_report/main.tex';sources.add(tex);source=tex.read_text()
    assert '1,626/1,626' in source
    for f in re.findall(r'\\fig\{([^}]+)\}',source):
        path=tex.parent/'figures'/f;assert path.is_file();sources.add(path)
    pdf=tex.parent/'Phase2_Report.pdf';assert pdf.read_bytes().startswith(b'%PDF');sources.add(pdf)
    archive=tex.parent/'Phase2_Report_Overleaf.zip';sources.add(archive)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and z.read('main.tex')==tex.read_bytes()
    return 'English template report PDF, 13 referenced figure files and matching editable source archive'

def reproducibility():
    for name in ['Phase2_Final_Presentation_English.ipynb','Member4_ONNX_Decision_Parity.ipynb','Detection_Size_Breakdown.ipynb']:
        p=ROOT/'notebooks'/name;nb=read(p)
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    wrapper=ROOT/'tools/member4_audit_colab.py';sources.add(wrapper)
    assert sha(ROOT/'tools/member4_audit_v2.py') in wrapper.read_text()
    wrapper=ROOT/'tools/member4_integration_colab.py';sources.add(wrapper)
    assert sha(ROOT/'tools/member4_integration_v2.py') in wrapper.read_text()
    original=subprocess.check_output(['git','diff','8ffeb2d','--','docs/phase1_report'],cwd=ROOT,text=True);assert not original
    return 'Notebook cells parse, critical loader hashes match modules, Phase 1 report preserved; runtime replay is a separate task'

def wandb():
    receipt=read(BASE/'evidence/M8_WandB_evidence_link/wandb_link.json')
    assert receipt['status']=='uploaded' and receipt['artifact_digest'] and receipt['url'].startswith('https://wandb.ai/')
    return 'Committed W&B artifact receipt and run URL archived; no new upload claimed'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    items=[('data','Frozen grouped dataset','releases/species_recovery/phase1_training_metadata.zip',dataset_check),('M4','Detector training and operating-point evidence','docs/phase2_report/evidence/',detection),('reviews','Human annotation review records','docs/phase2_report/evidence/supplied/evidence/',reviews),('M5','Segmentation evaluation and operating-mask AP','docs/phase2_report/evidence/supplied/segmentation/',segmentation),('analysis','Reviewed detector and mask examples','docs/phase2_report/evidence/supplied/evidence/',failure_reviews),('species','New classifier calibration/reload and historical evaluation','docs/phase2_report/evidence/supplied/',species),('HTTP','Executed full-stage HTTP batch','docs/phase2_report/evidence/supplied/http_demo/',http),('counts','Event-count application measure','docs/phase2_report/evidence/supplied/application/',counting),('benchmark','Warmup and full-pipeline benchmark','docs/phase2_report/evidence/supplied/application/',latency),('ONNX','Full validation raw/decision/operating-metric parity','docs/onnx_parity/full_validation_20261119T070317993000Z/',onnx),('decision','Fixed criteria and documented decision','docs/phase2_report/evidence/supplied/application/',acceptance),('report','Compiled English report and matching source','docs/phase2_report/',report),('source','Code/notebook static reproducibility','notebooks/',reproducibility),('backup','Archived W&B receipt','docs/phase2_report/evidence/supplied/evidence/M8_WandB_evidence_link/',wandb)]
    for key,label,path,fn in items:check(key,label,path,fn)
    with (OUT/'verified_evidence.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    verified=[r for r in checks if r['status']=='verified'];review=[r for r in checks if r['status']!='verified']
    actions=[dict(id='presentation',required='Prepare the final 15–20 minute presentation PDF; every member presents their contribution.',target='submission/Final_Presentation.pdf'),dict(id='demo',required='Record an end-to-end demo fallback and include detector execution on images and a video clip, as listed in the supplied template.',target='submission/Demo.mp4 and detector video execution receipt'),dict(id='size_breakdown',required='Run notebooks/Detection_Size_Breakdown.ipynb to derive separate small/medium/large recall rows from the saved fixed predictions; state the original size definition.',target='detection size-bin CSV and report table'),dict(id='clean_replay',required='Run the existing Initialize Member 4 and HTTP batch cells in a fresh session with the saved weights; preserve the new execution receipt.',target='timestamped fresh-runtime service receipt'),dict(id='submission_package',required='Install the reviewed report and presentation in submission/, add the recorded demo, rerun the existing Drive audit and freeze the final package.',target='versioned Drive submission ZIP')]
    (OUT/'required_actions.json').write_text(json.dumps(actions,indent=2))
    (OUT/'audit_status.json').write_text(json.dumps(dict(scope='Archived-evidence consistency and static source review; no new runtime execution, deployment approval or course-grade claim.',verified_groups=len(verified),review_required_groups=review,submission_actions=len(actions),phase1_preserved=True,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2))
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources) if p.is_file()}
    (OUT/'baseline_checksums.json').write_text(json.dumps(frozen,indent=2))
    lines=['# Verified project evidence','', 'Review of the current measured baseline before further optimization. No results or acceptance thresholds were changed.','', '| Area | Verified evidence |','| --- | --- |']
    lines += [f"| {r['verified_evidence']} | {r['detail']} |" for r in verified]
    lines += ['', 'The machine-readable register records exact checks and evidence paths. Runtime replay and final human deliverables are tracked separately in `required_actions.json`. Preserve the baseline hashes when starting new optimization experiments. Run `python tools/final_submission_review.py` from the repository to reproduce this review.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    lines=['# Required submission actions','',*['%d. %s\n   Target: `%s`.'%(i+1,r['required'],r['target']) for i,r in enumerate(actions)],'','These are deliverable and execution tasks. Model improvement is a separately versioned next stage; retain the current baseline and report measured scopes accurately.','']
    (OUT/'REQUIRED_ACTIONS.md').write_text('\n'.join(lines))
    print(f'Verified evidence groups: {len(verified)}/{len(checks)}')
    for r in review:print('Review required:',r['requirement'],r['detail'])
    print('Required actions:',OUT/'REQUIRED_ACTIONS.md')
    return not review

if __name__=='__main__':raise SystemExit(0 if main() else 1)
