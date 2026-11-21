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
