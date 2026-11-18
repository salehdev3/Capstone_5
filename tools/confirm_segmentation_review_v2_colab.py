from pathlib import Path
from datetime import datetime,timezone
import urllib.request,hashlib,json,math,shutil
import pandas as pd
from IPython.display import display


def confirm_review(root,proposed_path):
    root=Path(root)
    current=root/'phase2_results/member4_recovery_v2/segmentation_failure_analysis_v2.csv'
    policy=root/'releases/Segmentation_M4_Guided_Recovery_v2/refinement_policy.json'
    assert current.is_file() and policy.is_file(), 'Run the current evidence audit first'
    live=pd.read_csv(current,dtype=str,keep_default_na=False)
    proposed=pd.read_csv(proposed_path,dtype=str,keep_default_na=False)
    policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    assert live.id.is_unique and proposed.id.is_unique and set(live.id)==set(proposed.id)
    assert live.policy_sha256.eq(policy_sha).all() and proposed.policy_sha256.eq(policy_sha).all()
    proposed=proposed.set_index('id').loc[live.id].reset_index()
    for before,after in zip(live.to_dict('records'),proposed.to_dict('records')):
        for field in ['iou','dice','area_mae_fraction']:
            assert math.isclose(float(before[field]),float(after[field]),rel_tol=0,abs_tol=1e-12), 'Metrics differ; do not replace the current review'
    with pd.option_context('display.max_colwidth',None,'display.max_rows',None):
        display(proposed[['id','cause','improvement','reference_concern']])
    print('Review all comments against your exported images. Reference concerns are hypotheses for future adjudication; original test labels and scores stay unchanged.')
    reviewer=input('Actual reviewer name: ').strip()
    assert reviewer, 'Reviewer name is required'
    confirmation=input('After reviewing all 12 cases, type APPROVE to save; anything else cancels: ').strip()
    if confirmation!='APPROVE':
        print('Cancelled; current review file was not changed.')
        return None
    before_sha=hashlib.sha256(current.read_bytes()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=current.parent/'review_backups'/stamp;backup.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(current,backup/current.name)
    updated=live.copy()
    updated['cause']=proposed.cause.tolist();updated['improvement']=proposed.improvement.tolist()
    updated['reviewer']=reviewer
    updated['analysis_status']='AI-assisted visual analysis confirmed by named reviewer'
    updated['reference_concern']=proposed.reference_concern.tolist()
    temporary=current.with_suffix('.csv.tmp');updated.to_csv(temporary,index=False);temporary.replace(current)
    record=dict(reviewer=reviewer,confirmed_at_utc=stamp,cases=len(updated),confirmation='APPROVE',
        analysis='AI-assisted visual review explicitly confirmed by the named reviewer',
        proposal_sha256=hashlib.sha256(Path(proposed_path).read_bytes()).hexdigest(),
        original_review_sha256=before_sha,confirmed_review_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
        model_policy_sha256=policy_sha,reference_masks_modified=False,metrics_modified=False,backup=str(backup/current.name))
    (current.parent/'segmentation_review_confirmation.json').write_text(json.dumps(record,indent=2))
    print('Confirmed review saved:',current)
    print('Rerun the evidence audit cell to create an updated package.')
    return current


if __name__=='__main__':
    ROOT=Path(globals().get('ROOT','/content/drive/MyDrive/CameraTrap_Phase2'))
    proposal=Path('/content/segmentation_failure_analysis_v2_proposed.csv')
    urllib.request.urlretrieve('https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/results/phase2/member4_recovery_v2/visual_failure_review/segmentation_failure_analysis_v2_proposed.csv',proposal)
    assert hashlib.sha256(proposal.read_bytes()).hexdigest()=='f5f052b835905fdadc2059ec4013d3e1a34b7cd3adefd74d2faa80b6a57bf603'
    confirm_review(ROOT,proposal)
