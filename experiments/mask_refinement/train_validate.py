"""Train a pixel classifier on boxed training crops and compare it with saved validation masks."""
import argparse
import json
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.ensemble import ExtraTreesClassifier


def raster(polygons, w, h):
    image = Image.new("1", (w, h))
    draw = ImageDraw.Draw(image)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon([(x*w, y*h) for x, y in poly], fill=1)
    return np.asarray(image, dtype=np.uint8)


def features(im, box):
    h, w = im.shape[:2]
    b = np.asarray(box)*[w,h,w,h]
    pad = np.maximum(2, .15*(b[2:]-b[:2]))
    x,y = np.maximum(0, np.floor(b[:2]-pad)).astype(int)
    r,t = np.minimum([w,h], np.ceil(b[2:]+pad)).astype(int)
    roi = np.s_[y:t,x:r]
    patch = cv2.resize(im[roi], (64,64))
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype('float32')/255
    yy,xx = np.mgrid[-1:1:64j,-1:1:64j].astype('float32')
    edge = lab[(abs(xx)>.85)|(abs(yy)>.85)]
    smooth = cv2.GaussianBlur(lab,(0,0),2)
    values = np.dstack([xx,yy,abs(xx),abs(yy),xx*xx+yy*yy,lab,lab-edge.mean(0),lab-smooth,smooth])
    return roi, values.reshape(4096,-1)


def main(dataset, archive, out):
    rng = np.random.default_rng(42)
    X, Y = [], []
    for label in sorted((dataset/'labels/train').glob('*.txt')):
        im = cv2.imread(str(dataset/'images/train'/f'{label.stem}.jpg'))
        h,w = im.shape[:2]
        for line in label.read_text().splitlines():
            poly = np.fromstring(line,sep=' ')[1:].reshape(-1,2)
            box = np.r_[poly.min(0),poly.max(0)]
            roi, feat = features(im,box)
            truth = cv2.resize(raster([poly],w,h)[roi],(64,64),interpolation=cv2.INTER_NEAREST)
            take = rng.choice(4096,768,replace=False)
            X.append(feat[take])
            Y.append(truth.ravel()[take])
    model = ExtraTreesClassifier(n_estimators=32,max_depth=12,min_samples_leaf=8,n_jobs=2,random_state=42)
    model.fit(np.concatenate(X),np.concatenate(Y))
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/'pixel_model.joblib',compress=3)
    with zipfile.ZipFile(archive) as z:
        frames = json.loads(z.read('validation_crop_float_192/masks/predictions.json'))
    thresholds = (.45,.5,.55,.60,.65)
    scores = {k:[] for k in ('baseline', *thresholds)}
    for f in frames:
        ident = f['id']
        im = cv2.imread(str(dataset/'images/val_select'/f'{ident}.jpg'))
        h,w=im.shape[:2]
        label=dataset/'labels/val_select'/f'{ident}.txt'
        truth=raster([np.fromstring(s,sep=' ')[1:].reshape(-1,2) for s in label.read_text().splitlines()],w,h)
        masks={k:np.zeros_like(truth) for k in scores}
        for obj in f['objects']:
            if 'polygon_normalized' not in obj:
                continue
            mask=raster([obj['polygon_normalized']],w,h)
            masks['baseline'] |= mask
            roi,feat=features(im,obj['box'])
            p=model.predict_proba(feat)[:,1].reshape(64,64)
            a=cv2.resize(mask[roi].astype('float32'),(64,64))
            prior=cv2.GaussianBlur(a,(0,0),3)
            for k in thresholds:
                result=mask.copy()
                result[roi]=cv2.resize((.7*prior+.3*p).astype('float32'),mask[roi].shape[::-1])>=k
                masks[k] |= result
        for k,pred in masks.items():
            scores[k].append(float((pred&truth).sum()/(pred|truth).sum()))
    assert len(frames)==40
    result=dict(training_images=len(list((dataset/'labels/train').glob('*.txt'))),instances=len(X),validation_images=len(frames),
                mean_iou={str(k):float(np.mean(v)) for k,v in scores.items()})
    print(result)
    (out/'validation.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['dataset','archive','out']:
        parser.add_argument(name,type=Path)
    main(**vars(parser.parse_args()))
