"""Train a MobileNetV3 crop mask decoder with simulated detector-box error."""

import argparse
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def raster(polygons, width, height):
    image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([(x * width, y * height) for x, y in polygon], fill=1)
    return np.asarray(image, dtype=np.uint8)


class CropMask(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        layers = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        ).features
        self.shallow = layers[:2]
        self.middle = layers[2:4]
        self.deep = layers[4:9]
        self.decode = nn.Sequential(
            nn.Conv2d(16 + 24 + 48, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        a = self.shallow(x)
        b = self.middle(a)
        c = self.deep(b)
        skip = [
            F.interpolate(v, a.shape[-2:], mode="bilinear", align_corners=False)
            for v in (b, c)
        ]
        return F.interpolate(
            self.decode(torch.cat([a, *skip], 1)),
            x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def region(box, shape):
    h, w = shape[:2]
    box = np.asarray(box) * [w, h, w, h]
    pad = np.maximum(4, 0.25 * (box[2:] - box[:2]))
    x, y = np.maximum(0, np.floor(box[:2] - pad)).astype(int)
    r, b = np.minimum([w, h], np.ceil(box[2:] + pad)).astype(int)
    return np.s_[y:b, x:r]


def inputs(image):
    rgb = cv2.resize(image, (96, 96))[:, :, ::-1].astype("float32") / 255
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def examples(dataset, split):
    for label in sorted((dataset / "labels" / split).glob("*.txt")):
        im = cv2.imread(str(dataset / "images" / split / (label.stem + ".jpg")))
        polys = [
            np.fromstring(s, sep=" ")[1:].reshape(-1, 2)
            for s in label.read_text().splitlines()
        ]
        yield label.stem, im, polys


def run(dataset, archive, out):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    xs, ys = [], []
    for _, im, polys in examples(dataset, "train"):
        for p in polys:
            mask = raster([p], im.shape[1], im.shape[0])
            box = np.r_[p.min(0), p.max(0)]
            dims = np.tile(box[2:] - box[:2], 2)
            for i in range(5):
                noisy = (
                    box
                    if i == 0
                    else np.clip(box + dims * rng.uniform(-0.12, 0.12, 4), 0, 1)
                )
                roi = region(noisy, im.shape)
                x = inputs(im[roi])
                y = torch.from_numpy(
                    cv2.resize(mask[roi], (96, 96), interpolation=cv2.INTER_NEAREST)
                ).float()[None]
                xs.append(x.flip([-1]) if i % 2 else x)
                ys.append(y.flip([-1]) if i % 2 else y)
    x, y = torch.stack(xs), torch.stack(ys)
    model = CropMask(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0007)
    with zipfile.ZipFile(archive) as z:
        frames = json.loads(z.read("validation_crop_float_192/masks/predictions.json"))
    records = {r["id"]: r for r in frames}
    val = list(examples(dataset, "val_select"))
    assert len(val) == 40 and {r[0] for r in val} == set(records)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    best = -1
    for epoch in range(1, 13):
        model.train()
        for idx in torch.randperm(len(x)).split(24):
            logits = model(x[idx])
            prob = logits.sigmoid()
            dice = (2 * (prob * y[idx]).sum((1, 2, 3)) + 1) / (
                prob.sum((1, 2, 3)) + y[idx].sum((1, 2, 3)) + 1
            )
            loss = F.binary_cross_entropy_with_logits(logits, y[idx]) + 1 - dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch not in (4, 8, 12):
            continue
        model.eval()
        scores = {t: [] for t in (0.4, 0.5, 0.6)}
        with torch.inference_mode():
            for ident, im, polys in val:
                truth = raster(polys, im.shape[1], im.shape[0])
                pred = np.zeros(im.shape[:2], np.float32)
                objects = records[ident]["objects"]
                prompts = [
                    o for o in objects if o["detector_confidence"] >= 0.4
                ] or sorted(objects, key=lambda o: -o["detector_confidence"])[:1]
                for obj in prompts:
                    roi = region(obj["box"], im.shape)
                    prob = model(inputs(im[roi])[None]).sigmoid()[0, 0].numpy()
                    np.maximum(
                        pred[roi],
                        cv2.resize(prob, pred[roi].shape[::-1]),
                        out=pred[roi],
                    )
                for t in scores:
                    binary = pred >= t
                    scores[t].append(
                        float((binary & truth).sum() / (binary | truth).sum())
                    )
        for t, values in scores.items():
            iou = float(np.mean(values))
            report.append(dict(epoch=epoch, threshold=t, mean_iou=iou))
            if iou > best:
                best = iou
                torch.save(
                    dict(weights=model.state_dict(), threshold=t, epoch=epoch),
                    out / "best.pt",
                )
        print(epoch, report[-3:], flush=True)
        (out / "validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("dataset", "archive", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
