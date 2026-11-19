"""Complete fixed-shape detector decision/metric parity on all retained validation images."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,subprocess,sys,urllib.request

subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.3.221','onnx','onnxruntime','scipy','pandas'],check=True)
from google.colab import drive
mount=Path('/content/drive')
# Do not create fake Drive folders before mounting, or rename a live filesystem.
if not Path('/content/drive/MyDrive').is_dir():
    if mount.exists() and any(mount.iterdir()) and not os.path.ismount(mount):
        raise RuntimeError('Drive mount folder contains local files. Preserve them and mount Drive before rerunning this cell.')
    drive.mount(str(mount))
assert Path('/content/drive/MyDrive').is_dir(),'Drive is not mounted'
ROOT=Path(globals().get('ROOT','/content/drive/MyDrive/CameraTrap_Phase2'))
MANIFEST=Path(globals().get('MANIFEST',ROOT/'phase1_source/data/cct20_team_v2/manifest.json'))
IMAGES=Path(globals().get('IMAGES',ROOT/'images'))
RELEASE=ROOT/'releases/Segmentation_M4_Guided_Recovery_v2'
WEIGHTS=Path(globals().get('PARITY_WEIGHTS',RELEASE/'detector.pt'))
ONNX=Path(globals().get('PARITY_ONNX',ROOT/'phase2_results/member4_recovery_v2/detector.onnx'))
assert MANIFEST.is_file(),f'Frozen manifest missing: {MANIFEST}. Run Initialize Member 4 first.'
assert WEIGHTS.is_file(),f'Fixed detector missing: {WEIGHTS}'
assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()=='ac38aaffe735bf7a6666866d2410f3601a93c6f9829ae6f68b7cd0117d1135e8'
if not ONNX.is_file():
    from ultralytics import YOLO
    import shutil
    export_dir=ROOT/'phase2_results/member4_recovery_v2/onnx_decision_export';export_dir.mkdir(parents=True,exist_ok=True)
    working=export_dir/'detector.pt'
    if working.exists():assert hashlib.sha256(working.read_bytes()).hexdigest()==hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    else:shutil.copyfile(WEIGHTS,working)
    ONNX=Path(YOLO(str(working)).export(format='onnx',imgsz=640,dynamic=False,simplify=False,nms=False,opset=17,half=False,device='cpu'))
MODULE=Path('/content/member4_onnx_decision_parity.py')
MODULE_URL='https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/tools/member4_onnx_decision_parity.py'
MODULE_SHA='a1742aef7f36da87ac0f5cf6cdd51a08478c5a74e006ae767e92c5b2cbf8309f'
with urllib.request.urlopen(MODULE_URL,timeout=45) as response:source=response.read()
assert hashlib.sha256(source).hexdigest()==MODULE_SHA,'Downloaded parity module differs; use the matching notebook version.'
MODULE.write_bytes(source)
spec=importlib.util.spec_from_file_location('onnx_decision_parity',MODULE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
PARITY_OUT=ROOT/'phase2_results/member4_recovery_v2/onnx_decision_parity'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
print('Comparing all 1,208 retained validation images. CPU only; no training or test rerun.',flush=True)
print('Fixed square input 640, confidence .045, NMS .30. Progress is printed every 50 images.',flush=True)
summary=module.run(WEIGHTS,ONNX,MANIFEST,IMAGES,PARITY_OUT,threads=4)
print('SAVED:',PARITY_OUT,flush=True)
assert summary['protocol']['full_validation_evaluated']
assert summary['passed'],'Parity differs. Preserve the evidence; do not change thresholds to force a pass.'
print('Upload onnx_decision_parity.json, metrics.csv, per_image.csv and environment.json from this folder.',flush=True)
