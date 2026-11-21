from pathlib import Path

import torch

from combined import CombinedPipeline
from common import sha
from crop_model import CropMask
from guided import guided_polygon


class GuidedPipeline(CombinedPipeline):
    def __init__(self, root, cache, policy):
        super().__init__(root, cache, policy)
        spec = self.config["crop_student"]
        assert sha(Path(spec["path"])) == spec["sha256"]
        checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=True)
        self.student = CropMask().eval()
        self.student.load_state_dict(checkpoint["weights"])
        self.threshold = checkpoint["threshold"]

    def predict_crop(self, image, box):
        return guided_polygon(self.student, self.threshold, self.sam, image, box,
                              self.config["mask"]["sam"]["size"])
