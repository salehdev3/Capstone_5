from pathlib import Path
import subprocess, sys, json, hashlib, urllib.request, importlib.util, time

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'ultralytics==8.3.221', 'onnx', 'onnxruntime'], check=True)

try:
    from google.colab import drive
except ImportError:
    drive = None
if drive is not None:
    drive.mount('/content/drive')

ROOT = Path(globals().get('ROOT', '/content/drive/MyDrive/CameraTrap_Phase2'))
OUT = Path(globals().get('OUT', ROOT/'phase2_results'))
OUT.mkdir(parents=True, exist_ok=True)
RELEASE = ROOT/'releases/Segmentation_M4_Guided_Recovery_v2'
# Restore the exact release if this runtime is connected to a new Drive location.
if not all((RELEASE/name).is_file() for name in ('refinement_policy.json','detector.pt','selected.pt')):
    import zipfile
    print('Restoring the exact Recovery v2 release; no training.')
    expected_archive_sha = '0042209ee5f3d548175f80d000e2d13845fc616d992083d5af86b9a13ff306b5'
    archive = RELEASE.with_suffix('.zip')
    archive.parent.mkdir(parents=True, exist_ok=True)
    file_sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if archive.exists():
        assert file_sha(archive) == expected_archive_sha, 'Existing release archive differs; preserve it before retrying.'
    else:
        temporary = archive.with_suffix('.download')
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/releases/segmentation_m4_guided_recovery_v2/Segmentation_M4_Guided_Recovery_v2.zip',
            temporary)
        assert file_sha(temporary) == expected_archive_sha, 'Release download checksum mismatch'
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        checks = json.loads(z.read('artifact_checksums.json'))
        for name, expected in checks.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
        for entry in z.infolist():
            target = (RELEASE/entry.filename).resolve()
            assert target.is_relative_to(RELEASE.resolve()), 'Unsafe archive path'
            if target.exists() and not entry.is_dir():
                assert target.is_file() and file_sha(target) == hashlib.sha256(z.read(entry.filename)).hexdigest(), f'Existing release file differs: {target}'
        RELEASE.mkdir(parents=True, exist_ok=True)
        z.extractall(RELEASE)
    for name, expected in checks.items():
        assert file_sha(RELEASE/name) == expected, name
print('Using release:', RELEASE)

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
