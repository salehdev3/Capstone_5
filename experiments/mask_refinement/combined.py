from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from common import sha
from candidate import FastPipeline
from bf16_encoder import enable_bf16_encoder


class CombinedPipeline(FastPipeline):
    def __init__(self, root, cache, policy):
        super().__init__(root, cache, policy)
        selection = self.config['species_head']
        assert sha(Path(selection['path'])) == selection['sha256']
        head = np.load(selection['path'])
        self.weight = head['weight']
        self.bias = head['bias']
        self.parts = []
        self.species.net.classifier[2].register_forward_hook(self.capture)
        if self.config['encoder_precision']=='bf16_encoder':
            enable_bf16_encoder(self.sam)

    def capture(self, layer, inputs, output):
        self.parts.append((inputs[0].detach().cpu().numpy(), output.detach().cpu().numpy()))

    def detect(self, image):
        self.parts.clear()
        result, logits = super().detect(image)
        if self.parts:
            features = np.concatenate([v[0] for v in self.parts])
            original = np.concatenate([v[1] for v in self.parts])
            keep = (1-softmax(original/self.config['animal_temperature'],axis=1)[:,14]
                    >= self.config['detector']['animal_min'])
            np.testing.assert_allclose(logits[:,14],original[keep,14])
            logits[:,:14] = features[keep] @ self.weight.T + self.bias
            self.parts.clear()
        return result, logits
