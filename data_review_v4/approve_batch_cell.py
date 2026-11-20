from pathlib import Path
import urllib.request, hashlib, importlib.util, subprocess, sys
assert 'workflow' in globals() and 'REVIEW' in globals(), 'شغّل الخطوتين 1 و2 أولًا.'
if importlib.util.find_spec('ultralytics') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.3.221'], check=True)
CODE = Path('/content/data_review_v4_code'); CODE.mkdir(exist_ok=True)
CHECKS = {'auto_masks.py': '6a28cbfec1194bc3b7ba69d5d6fd6fbac916e5b85204420dcc6f336ba7923d9a', 'approve_batch.js': 'c162c27975f40da5e1b21360b61a6fa6b8dc450afd6880ffd8d9d6261ce9741d'}
for name, expected in CHECKS.items():
    path = CODE/name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        with urllib.request.urlopen('https://raw.githubusercontent.com/salehdev3/Capstone_5/c1cdf12/data_review_v4/'+name, timeout=60) as response:
            source = response.read()
        assert hashlib.sha256(source).hexdigest() == expected, name
        path.write_bytes(source)
if 'auto_masks' not in globals() or getattr(auto_masks, '_source_sha', '') != CHECKS['auto_masks.py']:
    spec = importlib.util.spec_from_file_location('auto_masks', CODE/'auto_masks.py')
    auto_masks = importlib.util.module_from_spec(spec); spec.loader.exec_module(auto_masks)
    auto_masks._source_sha = CHECKS['auto_masks.py']
print('سيجهّز المقترحات المتبقية ثم يعرضها معًا. القرارات السابقة محفوظة.', flush=True)
BATCH_RESULT = auto_masks.approve_geometry_batch(ROOT, Path(REVIEW), workflow)
