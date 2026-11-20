import numpy as np
import torch
import cv2
from runtime import Pipeline


class FastPipeline(Pipeline):
    """Apply the species adjustment only after the existing animal gate."""
    def __init__(self, root, cache, policy):
        from ultralytics.utils import torch_utils
        torch_utils.NUM_THREADS = 1  # CPU predictor setup restores this value.
        super().__init__(root, cache, policy, device="cpu")
        assert self.animal is self.species
        self.offset = np.asarray(self.config["species_logit_offset"], dtype=np.float32)
        torch.set_num_threads(1)
        cv2.setNumThreads(1)

    def detect(self, image):
        result, logits = super().detect(image)
        logits = logits.copy()
        logits[:, :14] += self.offset
        return result, logits
