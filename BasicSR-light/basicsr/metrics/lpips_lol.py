import numpy as np
import torch
from basicsr.utils.registry import METRIC_REGISTRY

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

@METRIC_REGISTRY.register()
def calculate_lpips_lol(img, img2, device, model):
    tA = t(img).to(device)
    tB = t(img2).to(device)
    dist01 = model.forward(tA, tB).item()
    return dist01