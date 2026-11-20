import numpy as np


def crop_polygon(sam, image, box, size):
    h, w = image.shape[:2]
    box = np.asarray(box) * [w, h, w, h]
    pad = np.maximum(8, .45 * (box[2:] - box[:2]))
    x, y = np.maximum(0, np.floor(box[:2] - pad)).astype(int)
    r, b = np.minimum([w, h], np.ceil(box[2:] + pad)).astype(int)
    if r <= x or b <= y:
        return None
    prompt = (box - [x, y, x, y]).tolist()
    result = sam.predict(image[y:b, x:r], bboxes=[prompt], imgsz=size,
                         device="cpu", conf=0., verbose=False)[0]
    if result.masks is None or not len(result.masks.xy):
        return None
    poly = result.masks.xy[0]
    if len(poly) < 3:
        return None
    return np.clip((poly + [x, y]) / [w, h], 0, 1).tolist()
