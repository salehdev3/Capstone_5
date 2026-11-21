import torch


def enable_bf16_encoder(sam):
    encoder = sam.model.image_encoder
    forward = encoder.forward
    def encode(image):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            features = forward(image)
        return features.float()
    encoder.forward = encode
