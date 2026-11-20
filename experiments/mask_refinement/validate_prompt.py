"""Compare SAM crop context on identical saved detector outputs and validation labels."""
import argparse
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

parser = argparse.ArgumentParser()
for name in ('dataset','archive','weights','destination'):
    parser.add_argument(name,type=Path)
parser.add_argument('--bf16',action='store_true')
parser.add_argument('--size',type=int,default=192)
parser.add_argument('--margins',type=float,nargs='+',default=[.2,.3,.65])
args = parser.parse_args()
dataset,archive,weights,destination = args.dataset,args.archive,args.weights,args.destination
torch_utils.NUM_THREADS = 1
torch.set_num_threads(1)
cv2.setNumThreads(1)
sam = SAM(str(weights))
with zipfile.ZipFile(archive) as z:
    frames = json.loads(z.read('validation_crop_float_192/masks/predictions.json'))
report = {}
if args.bf16:
    from bf16_encoder import enable_bf16_encoder
    enable_bf16_encoder(sam)
for margin in args.margins:
    scores = []
    for frame in frames:
        ident = frame['id']
        im = cv2.imread(str(dataset/'images/val_select'/f'{ident}.jpg'))
        h,w = im.shape[:2]
        truth = raster([np.fromstring(line,sep=' ')[1:].reshape(-1,2) for line in
                        (dataset/'labels/val_select'/f'{ident}.txt').read_text().splitlines()],w,h)
        objects = frame['objects']
        primary = [o for o in objects if o['detector_confidence'] >= .4] or sorted(objects,key=lambda o:-o['detector_confidence'])[:1]
        polygons = [crop_polygon(sam,im,o['box'],args.size,margin) for o in primary]
        pred = raster([p for p in polygons if p],w,h)
        scores.append(float((pred&truth).sum()/(pred|truth).sum()))
    report[str(margin)] = float(np.mean(scores))
    print(margin,report[str(margin)],flush=True)
    destination.write_text(json.dumps(report,indent=2))
