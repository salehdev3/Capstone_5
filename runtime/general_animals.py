"""Optional COCO animal instance segmentation for local photographs."""
from pathlib import Path
import hashlib
import time

import torch
from ultralytics import YOLO
from common import sha
from runtime import Pipeline


class GeneralAnimals(Pipeline):
    def __init__(self, weights="yolo11s-seg.pt"):
        self.model = YOLO(str(weights))
        digest = sha(Path(self.model.ckpt_path))
        if digest != "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3":
            raise ValueError("Unexpected YOLO11s segmentation checkpoint.")
        self.settings = dict(imgsz=960, conf=0.25, classes=list(range(14, 24)), device="cpu")
        self.policy_sha256 = hashlib.sha256((digest + repr(self.settings)).encode()).hexdigest()
        torch.set_num_threads(4)

    def process_frame(self, path, meta):
        start = time.perf_counter()
        prediction = self.model.predict(str(path), **self.settings, verbose=False)[0]
        objects = []
        polygons = prediction.masks.xyn if prediction.masks is not None else []
        for index, box in enumerate(prediction.boxes):
            item = dict(box=box.xyxyn[0].tolist(), species=prediction.names[int(box.cls.item())],
                        confidence=float(box.conf.item()))
            if index < len(polygons):
                item["polygon_normalized"] = polygons[index].tolist()
            objects.append(item)
        return dict(id=meta["id"], event=meta["event"], count=len(objects), objects=objects,
                    timing=dict(total=time.perf_counter()-start), model="COCO YOLO11s-seg")


class MultiScaleAnimals(GeneralAnimals):
    """Combine whole-frame predictions with four overlapping crops."""
    def __init__(self, weights="yolo11s-seg.pt"):
        super().__init__(weights)
        self.policy_sha256 = hashlib.sha256((self.policy_sha256 + ":tiles-v1").encode()).hexdigest()

    def process_frame(self, path, meta):
        import cv2
        from torchvision.ops import batched_nms

        start = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        h, w = image.shape[:2]
        ch, cw = max(1, round(h * .6)), max(1, round(w * .6))
        regions = [(0, 0, w, h)] + [(x, y, cw, ch) for y in (0, h-ch) for x in (0, w-cw)]
        objects, classes = [], []
        for x, y, width, height in regions:
            pred = self.model.predict(image[y:y+height, x:x+width], **self.settings, verbose=False)[0]
            for index, box in enumerate(pred.boxes):
                a, b, c, d = box.xyxy[0].tolist()
                # An object clipped by an interior crop edge has an incomplete outline.
                if ((x and a < 2) or (y and b < 2) or
                    (x+width < w and c > width-2) or (y+height < h and d > height-2)):
                    continue
                cls = int(box.cls.item())
                item = dict(box=[(a+x)/w,(b+y)/h,(c+x)/w,(d+y)/h],
                            species=pred.names[cls], confidence=float(box.conf.item()))
                if pred.masks is not None:
                    item["polygon_normalized"] = [[(px+x)/w,(py+y)/h] for px,py in pred.masks.xy[index].tolist()]
                objects.append(item)
                classes.append(cls)
        if objects:
            keep = batched_nms(torch.tensor([o["box"] for o in objects]),
                               torch.tensor([o["confidence"] for o in objects]), torch.tensor(classes), .5)
            objects = [objects[i] for i in keep.tolist()]
        return dict(id=meta["id"], event=meta["event"], count=len(objects), objects=objects,
                    timing=dict(total=time.perf_counter()-start), model="COCO YOLO11s-seg multiscale")
