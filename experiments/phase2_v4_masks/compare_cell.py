from pathlib import Path
import sys, subprocess, hashlib, urllib.request, zipfile
assert 'ROOT' in globals(), 'شغّل خلية الإعداد أولًا.'
script = Path('/content/compare_mask_combinations.py')
with urllib.request.urlopen('https://raw.githubusercontent.com/salehdev3/Capstone_5/6e17d61/experiments/phase2_v4_masks/compare_combinations.py', timeout=60) as response:
    source = response.read()
assert hashlib.sha256(source).hexdigest() == '9735b6c0d6905823005b40e29e7192442dd667c43bad4f52821b98dc905bc82c'
script.write_bytes(source)
cache = Path(globals().get('CACHE', '/content/mask_training_v4_cache'))
command = [sys.executable, '-u', str(script), '--root', str(ROOT), '--cache', str(cache)]
print('مقارنة 9 تركيبات على نفس 40 صورة تحقق. بدون تدريب أو إعادة اختبار.', flush=True)
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
lines = []
for line in process.stdout:
    print(line, end='', flush=True)
    lines.append(line)
result = process.wait()
folder = Path(ROOT)/'phase2_results/mask_training_v4/combinations'
folder.mkdir(parents=True, exist_ok=True)
(folder/'execution.log').write_text(''.join(lines))
if result:
    raise RuntimeError('توقفت المقارنة؛ السبب الأصلي ظاهر أعلاه. أرسل آخر سطور الخطأ.')
archive = folder.parent/'Mask_Combinations_Review.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(folder.iterdir()):
        if path.is_file():
            z.write(path,path.name)
print('Saved:',archive)
from google.colab import files
files.download(str(archive))
