import numpy as np
import torch
import cv2
import time
from common import raster
from runtime import Pipeline
from crop_sam import crop_polygon


class FastPipeline(Pipeline):
    """Apply the species adjustment only after the existing animal gate."""
    def __init__(self, root, cache, policy):
        from ultralytics.utils import torch_utils
        torch_utils.NUM_THREADS = 1  # CPU predictor setup restores this value.
        super().__init__(root, cache, policy, device="cpu")
        assert self.animal is self.species
        if self.config["encoder_precision"] == "int8":
            from torch.ao.quantization import quantize_dynamic
            self.sam.model.image_encoder = quantize_dynamic(
                self.sam.model.image_encoder, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.offset = np.asarray(self.config["species_logit_offset"], dtype=np.float32)
        torch.set_num_threads(1)
        cv2.setNumThreads(1)

    def detect(self, image):
        result, logits = super().detect(image)
        logits = logits.copy()
        logits[:, :14] += self.offset
        return result, logits

    def predict_crop(self, image, box):
        return crop_polygon(self.sam, image, box, self.config["mask"]["sam"]["size"])

    def process_frame(self, path, meta, with_masks=True):
        if self.config["mask_mode"] == "frame":
            return super().process_frame(path, meta, with_masks)
        start = time.perf_counter()
        frame = super().process_frame(path, meta, with_masks=False)
        objects = frame["objects"]
        if with_masks and objects:
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            primary = [o for o in objects if o["detector_confidence"] >= self.config["mask"]["primary_conf"]]
            for obj in primary or [max(objects, key=lambda o: o["detector_confidence"])]:
                polygon = self.predict_crop(image, obj["box"])
                if polygon:
                    obj.update(polygon_normalized=polygon, relative_body_area=float(raster([polygon], w, h).mean()),
                               review=obj["confidence"] < 0.6)
        frame["segmentation"]["matched_masks"] = sum("polygon_normalized" in o for o in objects)
        total = time.perf_counter() - start
        t = frame["timing"]
        t.update(total=total, segmentation=total-t["classification"]-t["detection_and_animal_filter"])
        return frame
