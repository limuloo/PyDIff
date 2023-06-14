import torch
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_psnr_pytorch(img, img2):
    mse = (torch.abs(img - img2) ** 2).mean()
    psnr = 10 * torch.log10(1 * 1 / mse)
    return psnr