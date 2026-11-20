import json
import sys
import zipfile
from pathlib import Path
import torch
from ultralytics import SAM
from ultralytics.utils import torch_utils
from guided import guided_polygon
from crop_model import CropMask, examples, raster
import numpy as np

dataset, archive, weights, out = map(Path, sys.argv[1:5])
size = int(sys.argv[5]) if len(sys.argv) > 5 else 224
torch_utils.NUM_THREADS = 1
torch.set_num_threads(1)
ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
model = CropMask().eval()
model.load_state_dict(ck["weights"])
sam = SAM(str(weights))
with zipfile.ZipFile(archive) as z:
    frames = {
        r["id"]: r
        for r in json.loads(z.read("validation_crop_float_192/masks/predictions.json"))
    }
scores = []
for ident, im, polys in examples(dataset, "val_select"):
    obj = frames[ident]["objects"]
    prompts = [o for o in obj if o["detector_confidence"] >= 0.4] or sorted(
        obj, key=lambda o: -o["detector_confidence"]
    )[:1]
    masks = [
        guided_polygon(model, ck["threshold"], sam, im, o["box"], size) for o in prompts
    ]
    h, w = im.shape[:2]
    p = raster([v for v in masks if v], w, h)
    truth = raster(polys, w, h)
    scores.append(float((truth & p).sum() / (truth | p).sum()))
assert len(scores) == 40
result = dict(size=size, mean_iou=float(np.mean(scores)), frames=len(scores))
print(result)
(out / f"guided_{size}.json").write_text(json.dumps(result, indent=2))
