from pathlib import Path
import csv, hashlib, importlib.util, urllib.request

assert 'workflow' in globals() and 'REVIEW' in globals(), 'شغّل الخطوتين 1 و2 أولًا.'
CODE = Path('/content/data_review_v4_code')
CODE.mkdir(exist_ok=True)
checks = {
    'auto_masks.py': '2bda5f2a8590d6c43ac260697d17b1955fb23eaf3c63648a48b0e79a8067ea40',
    'approve_masks.js': '3da1b9ff305d01006a3e72634f4b08ce500eb2968ef0f9b4797cbad27a0c39e4'
}
for name, expected in checks.items():
    path = CODE/name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        with urllib.request.urlopen('https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/data_review_v4/'+name, timeout=60) as response:
            source = response.read()
        assert hashlib.sha256(source).hexdigest() == expected, name
        path.write_bytes(source)
if 'auto_masks' not in globals() or getattr(auto_masks, '_source_sha', '') != checks['auto_masks.py']:
    spec = importlib.util.spec_from_file_location('auto_masks', CODE/'auto_masks.py')
    auto_masks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto_masks)
    auto_masks._source_sha = checks['auto_masks.py']
REVIEW = Path(REVIEW)
with (REVIEW/'queue.csv').open() as stream:
    queue = list(csv.DictReader(stream))
pending = [r for r in queue if not (REVIEW/'auto_approvals'/(r['id']+'.json')).exists()]
if pending:
    if not (REVIEW/'auto_proposals'/(pending[0]['id']+'.json')).exists():
        auto_masks.generate(ROOT, REVIEW, workflow, only_id=pending[0]['id'])
    auto_masks.approve_next(REVIEW, workflow)
    print('أعد تشغيل الخلية للصورة التالية.')
else:
    print('انتهت مراجعة المقترحات؛ شغّل الخطوة 5 للتصدير.')
