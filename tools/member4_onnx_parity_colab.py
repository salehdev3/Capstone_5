from pathlib import Path
import subprocess, sys, json, hashlib, urllib.request, importlib.util, time

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'ultralytics==8.3.221', 'onnx', 'onnxruntime'], check=True)

ROOT = Path(globals().get('ROOT', '/content/drive/MyDrive/CameraTrap_Phase2'))
OUT = Path(globals().get('OUT', ROOT/'phase2_results'))
OUT.mkdir(parents=True, exist_ok=True)
RELEASE = ROOT/'releases/Segmentation_M4_Guided_Recovery_v2'
EXPORT_DIR = OUT/'member4_recovery_v2'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT = EXPORT_DIR/'member4_onnx_parity_v2.py'
URL = 'https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/tools/member4_onnx_parity_v2.py'
EXPECTED_SHA256 = 'bb9a00eec9bd77369fad4b75dad9a8064cad6d1570caea98d9cc52c3871a3370'
urllib.request.urlretrieve(URL, SCRIPT)
assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == EXPECTED_SHA256
spec = importlib.util.spec_from_file_location('member4_parity', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

detector, ONNX_PATH, session, x, parity = module.run_parity(RELEASE, EXPORT_DIR)
import numpy as np, torch
from ultralytics import YOLO
if 'CONFIG' not in globals():
    CONFIG = {'imgsz': parity['imgsz']}
assert CONFIG['imgsz'] == parity['imgsz'], 'Benchmark image size differs from export'
(OUT/'onnx_parity.json').write_text(json.dumps(parity, indent=2))
print('ONNX:', ONNX_PATH)
