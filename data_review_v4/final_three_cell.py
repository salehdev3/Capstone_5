from pathlib import Path
import json, hashlib, urllib.request, importlib.util
assert 'ROOT' in globals() and 'workflow' in globals(), 'شغّل الخطوتين 1 و2 أولًا.'
CODE = Path('/content/data_review_v4_final_three'); CODE.mkdir(exist_ok=True)
BASE = 'https://raw.githubusercontent.com/salehdev3/Capstone_5/6fb4872/'
CHECKS = {'auto_masks.py': 'f1074e5ded4ccc8aba96b272cfdd7dea4f0eba354e2c6c558f2be7d0caa7fc89', 'final_three_review.py': '0479936276c0e4dc39cd71436e7f3c7d8df13f6f454c958d2a48e2ef097a780c', 'final_three.js': '6f9319ade0aadef31f8892be4bedde1916ca755ba593a11fc6cc5ef2d646751a'}
def fetch(url, path, expected):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read()
    assert hashlib.sha256(source).hexdigest() == expected, path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
for name, expected in CHECKS.items():
    fetch(BASE+'data_review_v4/'+name, CODE/name, expected)
drafts = CODE/'drafts'; drafts.mkdir(exist_ok=True)
fetch(BASE+'results/phase2/data_review_v4/final_three_drafts/artifact_checksums.json', drafts/'artifact_checksums.json', '3d065677a90db49c372c297e73eb6fbd664fb3dbe45ad9fc7985e6015eb853aa')
for name, expected in json.loads((drafts/'artifact_checksums.json').read_text()).items():
    assert Path(name).name == name
    fetch(BASE+'results/phase2/data_review_v4/final_three_drafts/'+name, drafts/name, expected)
def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE/(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
final_three = load('final_three_review')
auto = load('auto_masks')
print('المقترحات جاهزة؛ لا يوجد تدريب أو تشغيل GPU. اختر الصحيح لكل صورة ثم احفظ.', flush=True)
archive = final_three.review(ROOT, workflow, auto, CODE)
if archive is not None:
    print('أرسل هذه الحزمة:', archive)
    from google.colab import files
    files.download(str(archive))
