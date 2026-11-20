"""Use the trained animal mask to locate an interior point for SAM's box prompt."""

import numpy as np
import torch
import cv2
from crop_model import inputs, region


@torch.inference_mode()
def guided_polygon(student, threshold, sam, im, box, size):
    h, w = im.shape[:2]
    roi = region(box, im.shape)
    prob = student(inputs(im[roi])[None]).sigmoid()[0, 0].numpy()
    prompt_box = np.asarray(box) * [w, h, w, h]
    x1, y1, x2, y2 = prompt_box
    yy, xx = np.mgrid[:64, :64]
    rx, ry = roi[1], roi[0]
    px = rx.start + (xx + 0.5) * (rx.stop - rx.start) / 64
    py = ry.start + (yy + 0.5) * (ry.stop - ry.start) / 64
    binary = (
        (cv2.resize(prob, (64, 64)) >= threshold)
        & (px >= x1)
        & (px <= x2)
        & (py >= y1)
        & (py <= y2)
    )
    pad = np.maximum(8, 0.45 * (prompt_box[2:] - prompt_box[:2]))
    x, y = np.maximum(0, np.floor(prompt_box[:2] - pad)).astype(int)
    r, b = np.minimum([w, h], np.ceil(prompt_box[2:] + pad)).astype(int)
    kwargs = {}
    if binary.any():
        distance = cv2.distanceTransform(
            np.pad(binary.astype("uint8"), 1), cv2.DIST_L2, 3
        )[1:-1, 1:-1]
        j, i = np.unravel_index(distance.argmax(), distance.shape)
        kwargs = dict(
            points=[[[float(px[j, i] - x), float(py[j, i] - y)]]], labels=[[1]]
        )
    result = sam.predict(
        im[y:b, x:r],
        bboxes=[(prompt_box - [x, y, x, y]).tolist()],
        imgsz=size,
        device="cpu",
        conf=0.0,
        verbose=False,
        **kwargs
    )[0]
    if result.masks is None or not len(result.masks.xy):
        return None
    poly = result.masks.xy[0]
    if len(poly) < 3:
        return None
    return np.clip((poly + [x, y]) / [w, h], 0, 1).tolist()
