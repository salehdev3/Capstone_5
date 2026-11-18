from pathlib import Path
import urllib.request,hashlib,importlib.util
ROOT=Path(globals().get('ROOT','/content/drive/MyDrive/CameraTrap_Phase2'))
script=Path('/content/export_segmentation_review_v2.py')
urllib.request.urlretrieve('https://raw.githubusercontent.com/salehdev3/Capstone_5/phase-2/tools/export_segmentation_review_v2.py',script)
assert hashlib.sha256(script.read_bytes()).hexdigest()=='d0efad1ba180390e99be8e62f3f519877a73d25c7b2604eed0ebc86835145b1a'
spec=importlib.util.spec_from_file_location('review_export',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
archive=module.export_review(ROOT)
from google.colab import files
files.download(str(archive))
