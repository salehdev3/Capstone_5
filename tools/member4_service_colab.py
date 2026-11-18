from pathlib import Path
import subprocess,sys,json,hashlib,urllib.request,importlib.util

assert 'pipeline' in globals(), 'Initialize Member 4 integration in this same runtime first.'
ROOT=Path(globals().get('ROOT','/content/drive/MyDrive/CameraTrap_Phase2'))
OUT=Path(globals().get('OUT',ROOT/'phase2_results'))
subprocess.run([sys.executable,'-m','pip','install','-q','fastapi','uvicorn','python-multipart','requests'],check=True)
MODULE=OUT/'member4_recovery_v2/member4_service_v2.py'
MODULE.parent.mkdir(parents=True,exist_ok=True)
urllib.request.urlretrieve('https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/tools/member4_service_v2.py',MODULE)
EXPECTED_SHA256='3c351637ba5a7d313628a3aba93816e5bebb345a8db00063652153f951e1bf54'
assert hashlib.sha256(MODULE.read_bytes()).hexdigest()==EXPECTED_SHA256
spec=importlib.util.spec_from_file_location('member4_service',MODULE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Stop only the previous service started by this cell, if it exists.
if '_member4_service_state' in globals():
    _member4_service_state['server'].should_exit=True
    _member4_service_state['thread'].join(timeout=5)
_member4_service_state=module.record_demo(pipeline,ROOT,OUT)
server=_member4_service_state['server']
thread=_member4_service_state['thread']
app=_member4_service_state['app']
SERVICE_URL=_member4_service_state['base_url']
demo_output=_member4_service_state['demo_output']
demo=_member4_service_state['demo']
IMAGES=_member4_service_state['images']
from IPython.display import display
from PIL import Image
import pandas as pd
if not demo_output['events']:print('No animal events were retained in this demo.')
else:display(pd.DataFrame(demo_output['events']))
for r in demo:
    im=Image.open(_member4_service_state['evidence_dir']/(r['id']+'_overlay.png'))
    im.thumbnail((640,480));display(im)
print('Saved:',_member4_service_state['evidence_dir'])
