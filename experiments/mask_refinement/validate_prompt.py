"""Compare SAM crop context on identical saved detector outputs and validation labels."""
import sys
import json
import zipfile
from pathlib import Path
import torch
import cv2
import numpy as np
from ultralytics import SAM
from ultralytics.utils import torch_utils
from train_validate import raster
from crop_sam import crop_polygon

dataset, archive, weights, destination = map(Path, sys.argv[1:5])
torch_utils.NUM_THREADS = 1
torch.set_num_threads(1)
cv2.setNumThreads(1)
sam = SAM(str(weights))
with zipfile.ZipFile(archive) as z:
    frames = json.loads(z.read('validation_crop_float_192/masks/predictions.json'))
report = {}
use_bf16 = "--bf16" in sys.argv
if use_bf16:
    from bf16_encoder import enable_bf16_encoder
    enable_bf16_encoder(sam)
for margin in ([.45] if weights.name == "sam_b.pt" and "--bf16" in sys.argv else [.2,.3,.65]):
    scores = []
    for frame in frames:
        ident = frame['id']
        im = cv2.imread(str(dataset/'images/val_select'/f'{ident}.jpg'))
        h,w = im.shape[:2]
        truth = raster([np.fromstring(line,sep=' ')[1:].reshape(-1,2) for line in
                        (dataset/'labels/val_select'/f'{ident}.txt').read_text().splitlines()],w,h)
        objects = frame['objects']
        primary = [o for o in objects if o['detector_confidence'] >= .4] or sorted(objects,key=lambda o:-o['detector_confidence'])[:1]
        polygons = [crop_polygon(sam,im,o['box'],256 if use_bf16 else 192,margin) for o in primary]
        pred = raster([p for p in polygons if p],w,h)
        scores.append(float((pred&truth).sum()/(pred|truth).sum()))
    report[str(margin)] = float(np.mean(scores))
    print(margin,report[str(margin)],flush=True)
    destination.write_text(json.dumps(report,indent=2))
