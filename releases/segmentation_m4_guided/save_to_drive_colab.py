from google.colab import drive
from pathlib import Path
import urllib.request, hashlib, zipfile, json

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/CameraTrap_Phase2/releases')
ROOT.mkdir(parents=True, exist_ok=True)
URL = 'https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/releases/segmentation_m4_guided/Segmentation_M4_Guided.zip'
EXPECTED_SHA256 = '4026e9d8335a9d213a875d7045849e76bba5b180c02f7a2914c9dab247fc1133'
ARCHIVE = ROOT/'Segmentation_M4_Guided.zip'
RELEASE = ROOT/'Segmentation_M4_Guided'
if ARCHIVE.exists():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED_SHA256, 'Existing archive differs; preserve it before retrying.'
else:
    TEMP = ROOT/'Segmentation_M4_Guided.download'
    urllib.request.urlretrieve(URL, TEMP)
    assert hashlib.sha256(TEMP.read_bytes()).hexdigest() == EXPECTED_SHA256, 'Download integrity check failed'
    TEMP.replace(ARCHIVE)
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    checks = json.loads(z.read('artifact_checksums.json'))
    for name, expected in checks.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
    for entry in z.infolist():
        target = (RELEASE/entry.filename).resolve()
        assert target.is_relative_to(RELEASE.resolve()), 'Unsafe archive path'
        if target.exists() and not entry.is_dir():
            assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(z.read(entry.filename)).hexdigest(), f'Existing file differs: {target}'
    RELEASE.mkdir(parents=True, exist_ok=True)
    z.extractall(RELEASE)
for name, expected in checks.items():
    assert hashlib.sha256((RELEASE/name).read_bytes()).hexdigest() == expected, name
print('Archive saved:', ARCHIVE)
print('Release installed:', RELEASE)
print(json.loads((RELEASE/'validation_metrics.json').read_text()))
