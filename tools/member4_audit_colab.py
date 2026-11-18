from pathlib import Path
import subprocess,sys,json,hashlib,urllib.request,importlib.util
assert 'pipeline' in globals(), 'Initialize Member 4 integration in this same runtime first.'
ROOT=Path(globals().get('ROOT','/content/drive/MyDrive/CameraTrap_Phase2'))
OUT=Path(globals().get('OUT',ROOT/'phase2_results'))
subprocess.run([sys.executable,'-m','pip','install','-q','pycocotools'],check=True)
MODULE=OUT/'member4_recovery_v2/member4_audit_v2.py'
MODULE.parent.mkdir(parents=True,exist_ok=True)
urllib.request.urlretrieve('https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/tools/member4_audit_v2.py',MODULE)
assert hashlib.sha256(MODULE.read_bytes()).hexdigest()=='bffdf56b7f3a1b6714b26acdd7293641426f8cdb3a99a165fda4b014ca3731aa'
spec=importlib.util.spec_from_file_location('member4_audit',MODULE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
AUDIT_DIR,ARCHIVE,AUDIT_STATUS=module.audit(pipeline,ROOT,OUT)
from IPython.display import display
import pandas as pd
display(pd.read_csv(AUDIT_DIR/'completion.csv'))
print('LOCAL PACKAGE SAVED:',ARCHIVE)
print('Send completion.csv and audit_status.json from:',AUDIT_DIR)

def upload_phase2_evidence_to_wandb():
    # Explicit optional upload after the local ZIP is safely written.
    subprocess.run([sys.executable,'-m','pip','install','-q','wandb'],check=True)
    import wandb
    wandb.login()
    with wandb.init(project='camera-trap-phase2',job_type='evidence-backup',
                    config={'segmentation_policy_sha256':pipeline.policy_sha256,'audit_status':AUDIT_STATUS['status']},
                    settings=wandb.Settings(init_timeout=30)) as run:
        artifact=wandb.Artifact('phase2-recovery-evidence',type='evaluation',metadata={'scope':'Versioned checkpoint/evidence package; inspect audit for incomplete items and deployment NO-GO.'})
        artifact.add_file(str(ARCHIVE),name=ARCHIVE.name)
        logged=run.log_artifact(artifact);logged.wait()
        link=dict(url=run.url,status='uploaded',artifact_name=logged.name,artifact_digest=logged.digest,
                  local_package_sha256=module.digest(ARCHIVE))
    module.write(OUT/'member4_recovery_v2/wandb_link.json',link)
    print('W&B backup uploaded. Rerun the audit cell to include this link in the next package.')

print('For a separate W&B package backup, run: upload_phase2_evidence_to_wandb()')
